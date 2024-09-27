import os, re
import json
import logging

import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Agent():
    def __init__(
            self,
            args,
            train_dataloader,
            valid_dataloader,
            test_dataloader,
            tokenizer,
            model,
            optimizer=None
        ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer

        if args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
            )

    def fit(self): ## TRAINING
        earlystop_threshold = self.args.patience
        patience = 0
        min_loss = 99999

        for epoch in range(self.args.max_epochs):
            # KB 임베딩 업데이트(초기화)
            if self.args.model_type == "retriever":
                self.model.eval()
                if self.args.distributed:
                    self.model.module.set_fit_passage_embs()
                else:
                    self.model.set_fit_passage_embs()

            # training step
            tr_loss = tr_acc = 0
            epoch_iterator = tqdm(self.train_dataloader, disable=self.args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model_inputs = dict()
                for key in batch:
                    if not isinstance(batch[key], torch.Tensor): continue
                    model_inputs[key] = batch[key].to(self.args.device)

                self.model.train()
                outputs = self.model(**model_inputs)

                if self.args.model_type in ['ict', 'retriever']:
                    loss = outputs[0]
                    acc = outputs[1]
                else:
                    loss = outputs[0]
                    logits = outputs[1]
                    acc = self.acc_func(logits, batch['labels'].to(self.args.device))

                # loss를 누적 단계 수로 나눔
                if (step + 1) >= len(epoch_iterator) - (len(epoch_iterator) % self.args.gradient_accumulation_steps):
                    iters_to_accumulate = self.args.gradient_accumulation_steps
                else:
                    iters_to_accumulate = len(epoch_iterator) % self.args.gradient_accumulation_steps
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / iters_to_accumulate

                # 기울기 누적
                loss.backward()

                # 누적된 기울기를 사용하여 가중치 업데이트
                if (step+1) % self.args.gradient_accumulation_steps == 0 or (step+1) == len(epoch_iterator):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                tr_loss += loss.item()
                tr_acc += acc.item()

                # KB 임베딩 업데이트
                if self.args.model_type == 'retriever' and self.args.refresh_frequency > 0 and \
                        (step+1) % (self.args.refresh_frequency*self.args.gradient_accumulation_steps) == 0:
                    self.model.eval()
                    if self.args.distributed:
                        self.model.module.set_fit_passage_embs()
                    else:
                        self.model.set_fit_passage_embs()

                epoch_iterator.set_description(f"[TRAIN] Epoch{epoch}-L{tr_loss/(step+1):.3f}-A{tr_acc/(step+1):.3f}")

            logger.info(f"[TRAIN] Epoch_{epoch}-L_{tr_loss/(step+1):.3f}-A_{tr_acc/(step+1):.3f}")

            # KB 임베딩 업데이트
            if self.args.model_type == 'retriever':
                self.model.eval()
                if self.args.distributed:
                    self.model.module.set_fit_passage_embs()
                    self.model.module.set_eval_passage_embs()
                else:
                    self.model.set_fit_passage_embs()
                    self.model.set_eval_passage_embs()

            # 모델 저장
            if self.args.model_all_save and self.args.local_rank in [-1, 0]:
                train_path = os.path.join(
                    self.args.checkpoint,
                    f"train_{self.args.task_name}" +\
                    f"_lr_{self.args.learning_rate:.0E}" +\
                    f"_per_gpu_batch_{self.args.per_gpu_batch_size}" +\
                    f"_pat_{self.args.patience}" +\
                    f"_epoch_{epoch:07d}" +\
                    f"_loss_{tr_loss/(step+1):.4f}" +\
                    f"_acc_{tr_acc/(step+1):.4f}"
                )
                self.save_model(train_path)

            # validation step
            self.model.eval()
            val_loss = val_acc = 0
            epoch_iterator = tqdm(self.valid_dataloader, disable=self.args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model_inputs = dict()
                for key in batch:
                    if not isinstance(batch[key], torch.Tensor): continue
                    model_inputs[key] = batch[key].to(self.args.device)

                with torch.no_grad():
                    outputs = self.model(**model_inputs)

                if self.args.model_type in ['ict', 'retriever']:
                    loss = outputs[0]
                    acc = outputs[1]
                else:
                    loss = outputs[0]
                    logits = outputs[1]
                    acc = self.acc_func(logits, batch['labels'].to(self.args.device))

                val_loss += loss.item()
                val_acc += acc.item()

                epoch_iterator.set_description(f"[VALID] Epoch{epoch}-L{val_loss/(step+1):.3f}-A{val_acc/(step+1):.3f}")

            logger.info(f"[VALID] Epoch_{epoch}-L_{val_loss/(step+1):.3f}-A_{val_acc/(step+1):.3f}")

            # earlystop
            init_patience = False
            if val_loss < min_loss:
                init_patience = True
                min_loss = val_loss

            if init_patience:
                patience = 0
                if self.args.local_rank in [-1, 0]:
                    valid_path = os.path.join(
                        self.args.checkpoint,
                        f"valid_{self.args.task_name}" +\
                        f"_lr_{self.args.learning_rate:.0E}" +\
                        f"_per_gpu_batch_{self.args.per_gpu_batch_size}" +\
                        f"_pat_{self.args.patience}" +\
                        f"_epoch_{epoch:07d}" +\
                        f"_loss_{val_loss/(step+1):.4f}"+\
                        f"_acc_{val_acc/(step+1):.4f}"
                    )
                    self.save_model(valid_path)
                    checkpoint = os.path.join(self.args.checkpoint, "trained_model")
                    self.save_model(checkpoint)
            else:
                patience += 1
                if patience > earlystop_threshold:
                    logger.info('Ran out of patience.')
                    break   # STOP training

    @torch.no_grad()
    def predict(self):
        ofp_name = os.path.join(self.args.predict_model_path, "prediction_test.jsonl")
        with open(ofp_name, 'w', encoding='utf-8') as ofp:
            logger.info(f"WRITE {ofp_name}")

            self.model.eval()
            epoch_iterator = tqdm(self.test_dataloader)
            for index, batch in enumerate(epoch_iterator):
                model_inputs = dict()
                for key in batch:
                    if not isinstance(batch[key], torch.Tensor): continue
                    model_inputs[key] = batch[key].to(self.args.device)

                if self.args.model_type in ['ict', 'retriever']:
                    outputs = self.model(use_test_kb=True, is_eval=True, **model_inputs)
                    pred_responses = [None for _ in range(self.args.predict_batch_size)]
                else:
                    outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=self.args.max_new_tokens,
                        do_sample=True,
                        num_beams=self.args.num_beams,
                        temperature=self.args.temperature,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    pred_responses = self.tokenizer.batch_decode(outputs)

                # Predict 결과 출력
                # query : 모델 입력
                # document : 입력과 쌍이 되는 문서(ICT때만 사용)
                # source : test sub dataset type(multiwoz, sf_written, sf_spoken)
                # label_idx : Retriever 정답
                # relevance_index : Retriever top-5 추측 결과
                # relevance_score : Retriever top-5 벡터 유사도 값
                # label_response : Generator 정답 답변
                # pred_response : Generator 생성 답변

                for query, document, label_idx, pred_idx, relevance_score, label_response, pred_response in zip(
                        batch.get('querys', [None for _ in range(self.args.predict_batch_size)]),
                        batch.get('documents', [None for _ in range(self.args.predict_batch_size)]),
                        batch['label_idxs'],
                        outputs['relevance_index'][:, :5].tolist() if isinstance(outputs, dict) else \
                            batch.get('pred_idxs', [None for _ in range(self.args.predict_batch_size)]),
                        outputs['relevance_score'][:, :5].tolist() if isinstance(outputs, dict) else \
                            [None for _ in range(self.args.predict_batch_size)],
                        batch.get('responses', [None for _ in range(self.args.predict_batch_size)]),
                        pred_responses):

                    if pred_response is not None:
                        pred_response = pred_response.split(f'{self.tokenizer.sep_token}')[1]
                        pred_response = pred_response.split(f'{self.tokenizer.eos_token}')[0]
                        pred_response = pred_response.replace(f'{self.tokenizer.pad_token}', "")
                        pred_response = pred_response.replace('[SYSTEM]', "")

                    result = {
                        'query': query,
                        'document': document,
                        'label_idx': int(label_idx),
                        "pred_idx": pred_idx,
                        "relevance_score": relevance_score,
                        "label_response": label_response,
                        "pred_response": pred_response,
                    }
                    ofp.write(json.dumps(result, ensure_ascii=False)+'\n')
                epoch_iterator.set_description(f"[PREDICT]")

            logger.info(f"SAVE PREDICT FILE : {ofp_name}")

    @torch.no_grad()
    def acc_func(self, preds, labels):
        preds = torch.argmax(preds, dim=-1)
        shift_preds = preds[..., :-1].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 값이 ignore_index(-100)인 부분은 학습을 하지 않기 때문에 무시(torch cross entropy loss func 확인)
        mask = (shift_labels != -100)
        shift_preds = shift_preds[mask]
        shift_labels = shift_labels[mask]

        correct_preds = (shift_preds == shift_labels).sum()
        total_samples = mask.sum()
        acc = correct_preds / total_samples

        return acc

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)

        # DDP로 감싼 부분을 풀어줌
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        logger.info(f"SAVE MODEL PATH : {path}")
