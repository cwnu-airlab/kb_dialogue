from multiprocessing import pool
from operator import concat
import re
import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_utils import PreTrainedModel,apply_chunking_to_forward
from transformers.utils.doc import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import (
		BaseModelOutputWithPastAndCrossAttentions,
		BaseModelOutputWithPoolingAndCrossAttentions,
		MaskedLMOutput,
		ModelOutput,
)
from huggingface_hub import hf_hub_download
from transformers.activations import ACT2FN
import os
import torch, math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
from packaging import version
import sys
from nets.T5 import M3PreTrainedModel

#from transformers import RealmConfig, T5Config
import copy
#xxx
#torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=10000)


class _Retriever(M3PreTrainedModel):
	def __init__(self, config:dict, tokenizer=None, block_records=None, block_emb=None):
		super().__init__(config)

		self.config = config
		self.tokenizer = tokenizer
		self.block_records = block_records
		self.block_emb = block_emb
		concat_max_length = self.config.max_knowledge_length+self.config.max_source_length
		self.concat_max_length =concat_max_length if concat_max_length <=512 else 512
		self.post_init()


	@property
	def searcher_beam_size(self):
		
		return self.config.searcher_beam_size
		'''
		if self.training:
			return self.config.searcher_beam_size
		return self.config.reader_beam_size
		'''


	def forward( self, input_ids, input_projected_score, return_dict=None, b_index=None,):

		# embedder
		# return dict 모르겠음 
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		#print("\n block emb", self.block_emb.shape, self.block_emb) #  [kb_doc_length, model_size]: 512 은 proj_size의 값 
		#print("\n input_projected_score", input_projected_score.shape, input_projected_score) # [batch_size * 512 : 
		# [batch_size, block_emb_size]: block_emb & input_projected_score
		batch_scores = torch.einsum("BD,QD->QB", self.block_emb, input_projected_score.to(self.block_emb.device))
		#print("\nbatch scores", batch_scores.shape, batch_scores)

		# [batch_size, searcher_beam_size]: top k index추출
		_, retrieved_block_ids = torch.topk(batch_scores, k=self.searcher_beam_size, dim=1)
		retrieved_block_ids = retrieved_block_ids.squeeze(-1)
		#print("\n retrieved block id", retrieved_block_ids.shape, retrieved_block_ids, flush=True)

		# [batch_size, searcher_beam_size, projection_size]: top에 대한 임베딩 in block_emb
		retrieved_block_emb = torch.cat([torch.index_select(self.block_emb, dim=0, index=i).unsqueeze(0) for i in retrieved_block_ids])
		#print('retrieved_block_emb', retrieved_block_emb.shape, retrieved_block_emb)
		

		# [batch_size, searcher_beam_size] => byte text : top에 대한 원문장 추출l
		retrieved_blocks = np.take(self.block_records, indices=retrieved_block_ids.cpu(), axis=0)		
		#print('retrieved_blocks', retrieved_blocks.shape, retrieved_blocks)

		# input + retrieved
		# text= [bs:입력문장], text[bs:top1 문장]
		text = []; text_pair = []
		'''
		k=3 일 경우 
		12345 12345 12345
		'''
		for x in range(self.searcher_beam_size):
			for iidx in range(input_ids.size(0)):
				text.append(self.tokenizer.decode(input_ids[iidx], skip_special_tokens=True))
				if isinstance(retrieved_blocks[iidx], np.ndarray) is False:retrieved_blocks[iidx]= np.array([retrieved_blocks[iidx]], dtype=object)
				text_pair.append(retrieved_blocks[iidx][x].decode())

		# 중간 검사를 위해 필요함 
#		print("\n\n")
#		for t, p in zip(text, text_pair):
#			print("{} ::: {}".format(t, p))
#		exit()
		

		# decoder input을 위한 tokenize, input + retrieved
		# [batch_size, max_length]
		concat_inputs = self.tokenizer(text, text_pair, padding=True, truncation=True, \
					return_special_tokens_mask=True, max_length=self.concat_max_length)
		concat_inputs_tensors = concat_inputs.convert_to_tensors("pt")
		#print('-1-', concat_inputs_tensors.input_ids.dtype)
		#print('-2-', concat_inputs_tensors.special_tokens_mask.dtype)


		# retrieved_block_emb ==> (batch, searcher_beam_size,projection_size)
		# input_projected_score ==> (batch, hs)
		# [batch_size, searcher_beam_size]: top k score, relevance score
		#print(retrieved_block_emb.shape, input_projected_score.unsqueeze(2).shape)
		retrieved_logits = torch.einsum(
			"BDC,BCK->BDK", retrieved_block_emb.to(self.block_emb.device), input_projected_score.unsqueeze(2) 
		)
		#print('\n$$$$', retrieved_logits.shape, retrieved_logits, flush=True)
		
		return _RetrieverOutput(
				retrieved_scores=batch_scores,
				retrieved_block_emb=retrieved_block_emb,
				concat_inputs = concat_inputs_tensors,
				retrieved_logits = retrieved_logits,
				retrieved_blocks=retrieved_blocks,
				retrieved_block_ids=retrieved_block_ids
		)


@dataclass
class _RetrieverOutput(ModelOutput):
	retrieved_scores: torch.FloatTensor = None
	retrieved_block_emb: torch.FloatTensor = None
	retrieved_logits : torch.FloatTensor = None
	concat_inputs : torch.FloatTensor = None
	retrieved_blocks: torch.Tensor = None
	retrieved_block_ids : torch.Tensor = None


class M3RetrieverScorerProjection(nn.Module):
	''' Score projection for retriever '''
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.dense = nn.Linear(self.config.hidden_size, self.config.projected_size)
		self.LayerNorm = nn.LayerNorm(self.config.projected_size, eps=self.config.layer_norm_eps)

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.LayerNorm(hidden_states)
		return hidden_states




class M3RetrieverEmbedder(M3PreTrainedModel):
	def __init__(self, config, tokenizer=None):
		super().__init__(config)
		self.config = config
		self.embedder = transformers.T5ForConditionalGeneration(self.config).get_encoder()
		if bool(self.config.update_emb) :
			self.cls = M3RetrieverScorerProjection(self.config)

		self.post_init()

	def forward(
		self,
		input_ids,
		attention_mask=None,
		token_type_ids=None,
		answer_ids=None,
		return_dict=None,
		position_ids=None,
		tokenizer =None,
		labels=None,
		head_mask=None,
		inputs_embeds=None,
		output_attentions=None,
		output_hidden_states=None,
	):

		# embedder
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		embedder_outputs = self.embedder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		# [batch_size, seq_len, hidden_size] => # [batch_size, hidden_size]
		# pooler_output = embedder_outputs.last_hidden_state[:,0]
		#pooler_output=(embedder_outputs.last_hidden_state * embedder_outputs.attentions.unsqueeze(-1)).sum(dim=-2) / embedder_outputs.attentions.sum(dim=-1) 
		
		# 마지막 hidden state 하나만으로 검색 	
		pooler_output= embedder_outputs.last_hidden_state[:,0]
		#pooler_output=((embedder_outputs.last_hidden_state *attention_mask.unsqueeze(-1)).sum(dim=-2))/ (attention_mask.sum(dim=-1)).unsqueeze(1).expand((attention_mask.sum(dim=-1)).unsqueeze(1).size(0),embedder_outputs.last_hidden_state.size(-1))
		'''
		if attention_mask is not None:
			pooler_output=((embedder_outputs.last_hidden_state *attention_mask.unsqueeze(-1)).sum(dim=-2))/ (attention_mask.sum(dim=-1)).unsqueeze(1).expand((attention_mask.sum(dim=-1)).unsqueeze(1).size(0),embedder_outputs.last_hidden_state.size(-1))
			
		else: pooler_output = torch.mean(embedder_outputs.last_hidden_state,dim=1)
		'''
		
		#print("####", pooler_output, self.config.update_emb, flush=True)
		# [batch_size, retriever_proj_size]
		if bool(self.config.update_emb) : 
			projected_score = self.cls(pooler_output)		
		else: projected_score = pooler_output

		if not return_dict:	return (projected_score,) 
		else: return M3RetrieverEmbedderOutput(projected_score=projected_score,)

@dataclass
class M3RetrieverEmbedderOutput(ModelOutput):
	projected_score: torch.FloatTensor = None


class M3Retriever(M3PreTrainedModel):
	def __init__(self, config, tokenizer=None, block_records=None):
		super().__init__(config)
		self.config = config
		self.tokenizer=tokenizer
		self.block_records=block_records
		self.embedder = M3RetrieverEmbedder(self.config)
		self.register_buffer(
				"block_emb",
				torch.zeros(()).new_empty(
					size=(self.config.num_block_records, self.config.projected_size),
					dtype=torch.float32,
					device=torch.device(self.config.device),
				),
			)
		self._retriever = _Retriever(self.config, tokenizer=self.tokenizer, block_records=self.block_records, block_emb=self.block_emb)
		self.post_init()   # ??


	def forward(
		self,
		input_ids,		# ......
		kb_answers,
		attention_mask=None,	#.......
		token_type_ids=None,
		answer_ids=None,
		return_dict=None,
		position_ids=None,
		tokenizer =None,
		labels=None,
		head_mask=None,
		inputs_embeds=None,
		output_attentions=None,
		output_hidden_states=None,
		b_index=None,				#.........
	):

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict


		#query embedder
		embedder_outputs= self.embedder(
			input_ids,
			attention_mask=attention_mask,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		
		# query [batch_size, d_model]
		embedder_projected_score = embedder_outputs.projected_score

		# retriever
		retriever_outputs = self._retriever(input_ids=input_ids, input_projected_score=embedder_projected_score, b_index=b_index)

		return retriever_outputs

