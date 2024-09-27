import torch, numpy as np
import transformers
from transformers.utils import ModelOutput

import logging
import re

from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


@dataclass
class BertScorerOutput(ModelOutput):
    loss: torch.FloatTensor = None
    acc: torch.FloatTensor = None
    relevance_score: torch.FloatTensor = None
    relevance_index: torch.LongTensor = None


class BertProjection(torch.nn.Module):
    def __init__(self, config, proj_size=None):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.proj_size)
        self.LayerNorm = torch.nn.LayerNorm(config.proj_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states


class M3Retriever(transformers.BertPreTrainedModel):
    def __init__(self, config, fit_passages, eval_passages, tokenizer):
        super().__init__(config)
        self.config = config                                                                    # 설정 값
        self.bert = transformers.BertModel(config)                                              # 인코더
        self.proj = BertProjection(config) if config.use_proj_layer else torch.nn.Identity()    # 차원 축소 layer
        self.tokenizer = tokenizer                                              # 토크나이저
        self.fit_passages = fit_passages                                        # 학습 외부 지식
        self.eval_passages = eval_passages                                      # 테스트 외부 지식
        self.tokenized_fit_passages = self._tokenize_passages(fit_passages)     # 토큰화된 학습 외부 지식
        self.tokenized_eval_passages = self._tokenize_passages(eval_passages)   # 토큰화된 테스트 외부 지식
        # 학습 외부 지식 임베딩
        self.register_buffer(
            "block_emb",
            torch.zeros(()).new_empty(
                size=(
                    len(fit_passages), 
                    (config.proj_size if config.use_proj_layer else config.hidden_size)
                ),
                dtype=torch.float32,
                device=torch.device(config.device),
            ),
        )
        # 테스트 외부 지식 임베딩
        self.register_buffer(
            "test_block_emb",
            torch.zeros(()).new_empty(
                size=(
                    len(eval_passages), 
                    (config.proj_size if config.use_proj_layer else config.hidden_size)
                ),
                dtype=torch.float32,
                device=torch.device(config.device),
            ),
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def loss_func(self, relevance_score, relevance_index, label_idxs):
        gen_label_mask = torch.zeros(relevance_score.size()).to(self.config.device)
        gen_label_mask[relevance_index == label_idxs.unsqueeze(1)] = 1
        
        loss = torch.nn.functional.cross_entropy(relevance_score, gen_label_mask)

        return loss

    @torch.no_grad()
    def acc_func(self, relevance_index, label_idxs):
        top_k = [1, 3, 5]
        total_acc = 0
        for k in top_k:
            topk_block_ids = relevance_index[:, :k]
            labels_broadcasted = label_idxs.view(-1, 1).expand_as(topk_block_ids)
            check_result = torch.where(labels_broadcasted == topk_block_ids, 1, 0)
            total_acc += torch.sum(check_result) / label_idxs.size(0)

        return total_acc / len(top_k)

    def get_emb(self, **kwargs):
        embeddings = self.bert(**kwargs)[1]
        embeddings = self.proj(embeddings)

        return embeddings

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            label_idxs=None,
            use_test_kb=False,
            is_eval=False, 
            **kwargs
        ):
        query_emb = self.get_emb(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        
        with torch.no_grad():
            # 학습 시에는 학습 외부 지식, 추론 시에는 테스트 외부 지식을 사용
            if not use_test_kb:
                block_emb = self.block_emb
                passages = self.fit_passages
                tokenized_passages = self.tokenized_fit_passages
            else:
                block_emb = self.test_block_emb
                passages = self.eval_passages
                tokenized_passages = self.tokenized_eval_passages
            
            # Query 임베딩은 텐서 연산 그래프에 계산 과정이 저장되어 있어서 backward 계산이 가능
            # KB 임베딩은 계산 과정이 저장되어 있지 않기 때문에 backward 계산이 불가능(외부 지식에 존재하는 모든 문서 임베딩의 계산 과정을 저장할 시 OutOfMemoryError 발생)
            # no_grad로 연산 과정을 저장하지 않고 벡터 비교 연산 수행
            if self.config.vector_similarity == "cosine":
                normalized_query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
                normalized_block_emb = torch.nn.functional.normalize(block_emb, dim=-1)
                stale_relevance_score = torch.einsum("bd,nd->bn", normalized_query_emb, normalized_block_emb)
            else:
                stale_relevance_score = torch.einsum("bd,nd->bn", query_emb, block_emb)
            
            # top k 문서 검색
            num_candidates = self.config.num_candidates
            stale_topk_score, stale_topk_index = torch.topk(
                stale_relevance_score, k=num_candidates, dim=1
            )
            
            acc = self.acc_func(stale_topk_index, label_idxs)
            
            # 추론 시 바로 결과를 반환
            if is_eval:
                return BertScorerOutput(
                    loss=None,
                    acc=acc,
                    relevance_score=stale_topk_score, 
                    relevance_index=stale_topk_index
                )
            
            # top k 안에 정답 문서가 없을 수 있으므로 학습 시에는 정답 문서를 포함(정답 문서가 top k안에 없을 시 학습이 안됨)

            # Option 1. 정답이 None인 경우만 top k에 None 문서를 포함시키는 코드
            #mask = (stale_topk_index == label_idxs.unsqueeze(1)) | (label_idxs.unsqueeze(1) != (len(passages) - 1))
            #replace_indices = ~mask.any(dim=1)
            #stale_topk_index[replace_indices, -1] = (len(passages) - 1)
            
            # Option 2. top k에 정답 문서를 포함시키는 코드
            mask = (stale_topk_index == label_idxs.unsqueeze(1))
            replace_indices = ~mask.any(dim=1)
            stale_topk_index[replace_indices, -1] = label_idxs[replace_indices]
            
            # Option 3. top k에 None 문서를 포함시키는 코드
            #mask = relevance_index == torch.full((relevance_index.size(0),1), (len(passages) - 1)).to(self.config.device)
            #replace_indices = ~mask.any(dim=1)
            #stale_topk_index[replace_indices, -1] = (len(passages) - 1)
        
        # 각 Query당 검색된 top k개의 후보 문서들의 임베딩을 계산
        # 동일한 후보 문서들은 임베딩을 한 번만 계산하기 위해서 중복 문서를 제거
        # ex) query 1 => 문서 1, 문서 2 검색, query 2 => 문서 1, 문서 3 검색 시 문서 1의 임베딩을 2번 계산하지 않기 위해 중복 제거

        # 1. 후보 문서 인덱스를 1차원으로 변환 후 중복되는 문서 인덱스 제거
        unique_index = torch.unique(stale_topk_index.flatten())
        
        # 2. unique한 후보 문서들의 input_ids, token_type_ids, attention_mask를 선택
        candidate_inputs = {
            key: [tokenized_passages[key][idx] for idx in unique_index] for key in tokenized_passages.keys()
        }
        for key in candidate_inputs:
            candidate_inputs[key] = torch.stack(candidate_inputs[key]).to(self.config.device)
        
        # 3. unique한 후보 문서들의 임베딩 계산
        unique_candidate_embs = self.get_emb(**candidate_inputs)
        
        # 4. unique 후보 문서 임베딩에서 각 Query에 해당하는 후보 문서 임베딩을 선택
        size = (
            query_emb.size(0), 
            stale_topk_index.size(1), 
            self.config.proj_size if self.config.use_proj_layer else self.config.hidden_size
        )
        candidate_embs = torch.zeros(size, dtype=unique_candidate_embs.dtype).to(self.config.device)
        for i in range(size[0]):
            for j in range(size[1]):
                index = stale_topk_index[i, j]
                indices = torch.where(unique_index == index)[0]
                candidate_embs[i, j] = unique_candidate_embs[indices.item()].clone()
        
        # Query 임베딩은 텐서 연산 그래프에 계산 과정이 저장되어 있어서 backward 계산이 가능
        # 후보 문서 임베딩 또한 계산 과정이 저장되어 있기 때문에 backward 계산이 가능(각 Query별 top k개에 대해서만 계산 과정을 저장)
        # no_grad를 하지 않고 벡터 유사도 계산
        if self.config.vector_similarity == "cosine":
            normalized_query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
            normalized_candidate_embs = torch.nn.functional.normalize(candidate_embs, dim=-1)
            refresh_relevance_score = torch.einsum("bd,bnd->bn", normalized_query_emb, normalized_candidate_embs)
        else:
            refresh_relevance_score = torch.einsum("bd,bnd->bn", query_emb, candidate_embs)
        retrieved_score, indices = torch.sort(refresh_relevance_score, dim=1, descending=True)
        retrieved_index = torch.gather(stale_topk_index, 1, indices)
        
        loss = self.loss_func(retrieved_score, retrieved_index, label_idxs)
        
        return BertScorerOutput(
            loss=loss,
            acc=acc,
            relevance_score=retrieved_score, 
            relevance_index=retrieved_index
        )

    @torch.no_grad()
    def set_fit_passage_embs(self):
        # 학습 외부 지식 문서의 임베딩을 계산하는 함수
        # 이 때 임베딩은 계산 과정을 저장하지 않고 생성(no_grad)
        batch_size = 1024
        def chunks(tokenized_data, n):
            for i in range(0, len(tokenized_data['input_ids']), n): 
                return_tokenized_data = {
                    'input_ids': tokenized_data['input_ids'][i:i+n],
                    'attention_mask': tokenized_data['attention_mask'][i:i+n],
                    'token_type_ids': tokenized_data['token_type_ids'][i:i+n]
                }
                yield return_tokenized_data
        
        for i, inputs in enumerate(chunks(self.tokenized_fit_passages, batch_size)):
            projected_doc_emb = self.get_emb(**inputs)
            self.block_emb[i*batch_size:(i+1)*batch_size] = projected_doc_emb

    @torch.no_grad()
    def set_eval_passage_embs(self):
        # 테스트 외부 지식 문서의 임베딩을 계산하는 함수
        # 이 때 임베딩은 계산 과정을 저장하지 않고 생성(no_grad)
        batch_size = 1024
        def chunks(tokenized_data, n):
            for i in range(0, len(tokenized_data['input_ids']), n): 
                return_tokenized_data = {
                    'input_ids': tokenized_data['input_ids'][i:i+n],
                    'attention_mask': tokenized_data['attention_mask'][i:i+n],
                    'token_type_ids': tokenized_data['token_type_ids'][i:i+n]
                }
                yield return_tokenized_data
        
        for i, inputs in enumerate(chunks(self.tokenized_eval_passages, batch_size)):
            projected_doc_emb = self.get_emb(**inputs)
            self.test_block_emb[i*batch_size:(i+1)*batch_size] = projected_doc_emb

    def _tokenize_passages(self, passages):
        # 외부 지식 문서들을 미리 토큰화 하는 함수
        batch_size = 1024
        tokenized_passages = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": []
        }

        def chunks(data, n):
            for i in range(0, len(data), n):
                yield data[i:i + n]

        for sub_passages in chunks(passages, batch_size):
            if not self.config.use_passage_body:
                max_len = 44
                text = [passage.split("[BODY]")[0] for passage in sub_passages]
            else:
                max_len = 85
                text = [passage.split("[CITY]")[0] for passage in sub_passages]
            
            tokenized_inputs = self.tokenizer(
                text=text,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                max_length=max_len
            )
            for key in tokenized_passages:
                tokenized_passages[key].append(tokenized_inputs[key])

        for key in tokenized_passages:
            tokenized_passages[key] = torch.cat(tokenized_passages[key], dim=0).to(self.config.device)

        max_token_length = torch.max(torch.sum(tokenized_passages['input_ids'] != 0, dim=1))
        logger.info(f"passage max_token_length : {max_token_length}")

        return tokenized_passages


class M3ICT(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = transformers.BertModel(config)
        self.proj = BertProjection(config) if config.use_proj_layer else torch.nn.Identity()
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def loss_func(self, relevance_score, relevance_index, label_idxs):
        gen_label_mask = torch.zeros(relevance_score.size()).to(self.config.device)
        gen_label_mask[relevance_index == label_idxs.unsqueeze(1)] = 1
        
        loss = torch.nn.functional.cross_entropy(relevance_score, gen_label_mask)

        return loss

    @torch.no_grad()
    def acc_func(self, relevance_index, label_idxs):
        top_k = [1, 3, 5]
        total_acc = 0
        for k in top_k:
            topk_block_ids = relevance_index[:, :k]
            labels_broadcasted = label_idxs.view(-1, 1).expand_as(topk_block_ids)
            check_result = torch.where(labels_broadcasted == topk_block_ids, 1, 0)
            total_acc += torch.sum(check_result) / label_idxs.size(0)

        return total_acc / len(top_k)

    def get_emb(self, **kwargs):
        embeddings = self.bert(**kwargs)[1]
        embeddings = self.proj(embeddings)

        return embeddings

    def forward(
            self,
            query_input_ids=None,
            query_token_type_ids=None,
            query_attention_mask=None,
            document_input_ids=None,
            document_token_type_ids=None,
            document_attention_mask=None,
            label_idxs=None,
            **kwargs
        ):
        
        query_emb = self.get_emb(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask,
        )
        
        document_emb = self.get_emb(
            input_ids=document_input_ids,
            token_type_ids=document_token_type_ids,
            attention_mask=document_attention_mask,
        )
        
        # 벡터 유사도 계산
        if self.config.vector_similarity == "cosine":
            normalized_query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
            normalized_document_emb = torch.nn.functional.normalize(document_emb, dim=-1)
            relevance_score = torch.einsum("ij,kj->ik", normalized_query_emb, normalized_document_emb)
        else:
            relevance_score = torch.einsum("ij,kj->ik", query_emb, document_emb)
        
        topk_score, topk_index = torch.topk(
            relevance_score, k=document_emb.size(0), dim=1
        )
        
        loss = self.loss_func(topk_score, topk_index, label_idxs)
        acc = self.acc_func(topk_index, label_idxs)
        
        return BertScorerOutput(
            loss=loss,
            acc=acc,
            relevance_score=topk_score,
            relevance_index=topk_index
        )
        
