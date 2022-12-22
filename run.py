#-*-coding:utf-8-*-
# Copyright 2019 HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Adaptive Intelligence Research Lab(https://air.changwon.ac.kr/)., 2020. 01. ~

import sys, torch, logging, os
import json
from time import time
import datetime, pickle
from os.path import exists
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from transformers import RealmConfig, T5Config
import transformers
transformers.logging.set_verbosity_error()
import models as m
from utils import *

from torch.nn.parallel import DistributedDataParallel
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.getLogger().setLevel(logging.INFO)
#np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile="full")

torch.manual_seed(100)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Trainer():

    def __init__(self, args):
        self.args = args

        self.model = None
        self.tokenizer = None

        self.optimizer = None
        self.loss_func = None
        self.acc_func = None

        self.tensorboard = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.label_list = None

        if not self.args['trainer']['label_path'] :
            # 분류 task인데 label_path 옵션이 없는 경우
            if self.args['trainer']['model'] in ['classifier'] : raise AttributeError('Classifier model need "label_path".')
        else :
            with open(self.args['trainer']['label_path'], 'r') as f : self.label_list = [d.strip() for d in f]

    def save_model(self, model, path, args=None, extra_info=None):
        logging.info('SAVE: {}.'.format(path))
        os.makedirs(path, exist_ok=True)
        # save block records
        np.save(os.path.join(path, "block_records.npy"), self.block_records)
        # save tokenizer
        self.tokenizer.save_pretrained(path)
        model.save_pretrained(path)
        if args:
            with open(os.path.join(path,'training_args.json'), 'w') as f:
                merged_dict = {**config['t5'], **config['trainer']}
                args_dict = dict(merged_dict)
                args_dict['extra_info'] = {} if not extra_info else extra_info
                json.dump(args_dict, f, ensure_ascii=False, indent=4)

    def set_tensorboard(self, path):
        self.tensorboard = SummaryWriter(log_dir=path)

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01, eps=1e-6,)
    
    def set_lr_scheduler(self, optimizer, num_training_steps):
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                 num_warmup_steps=min(10000, max(100,int(num_training_steps / 10))),
                 num_training_steps=num_training_steps,)

    def set_tokenizer(self, opt, path):
        if 't5-' in opt: self.tokenizer = transformers.T5Tokenizer.from_pretrained(path)
        else: raise NotImplementedError('OPTION "{}" is not supported in set_tokenizer fuction.'.format(opt))

    def set_knowledge_base(self, path=None):
        if bool(path):
            key = os.path.splitext(path)[-1]
            if key in ['.npy']: self.block_records = np.load(path, allow_pickle=True)
            elif key in ['.tsv','.txt']:
                with open(path, 'r') as ifp: data = [bytes(d.strip().split("\t")[-1], 'utf-8') for d in ifp]
                #with open(path, 'r') as ifp: data = [bytes(d.strip(),'utf-8') for d in ifp]
                self.block_records = np.array(data, dtype=object)
            else: raise KeyError('No rule for {} filetype.'.format(key))
        else:
            if exists('../data/enwiki_block_record.npy') is False:
                os.system("scp -r odin.nlp.wo.tc:/mnt/odin3/share/Projects/M3/enwiki_documents/block_records.npy ../data/enwiki_block_record.npy")
            block_records_path = "../data/enwiki_block_record.npy"
            self.block_records = np.load(block_records_path, allow_pickle=True)
        

    def set_knowledge_base_in_model(self, tokenizer=None, model=None, projected_size=None, batch_size=None, predict=None):
        if predict==False: 
            documents = [doc.decode() for doc in self.block_records]
            def chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]

            for bidx, each in enumerate(chunks(documents, batch_size)):
                inputs = tokenizer(each, padding=True, truncation=True, return_tensors="pt").to(torch.cuda.current_device())
                with torch.no_grad():
                    projected_score = model.m3_t5_encoder.embedder(**inputs, return_dict=True).projected_score
                    model.m3_t5_encoder.block_emb[bidx*batch_size: bidx*batch_size+projected_score.size(0)] = projected_score
        else:
            pass

    def set_model(self, t5_opt, trainer_opt):
        ## set config
        if 'realm-t5-' in trainer_opt["model"]:
            config = transformers.RealmConfig()
            config.num_block_records = len(self.block_records) # knowledge size?
            config.searcher_beam_size = trainer_opt['searcher_beam_size']
            config.reader_beam_size = trainer_opt['reader_beam_size']
            config.projected_size = trainer_opt['projected_size']
            config.hidden_size = t5_opt['d_model']
            config.update(T5Config.from_pretrained(trainer_opt['weights']).to_dict())
            config.update({"max_knowledge_length":trainer_opt['max_knowledge_length']})
            config.update({"max_source_length":trainer_opt['max_source_length']})
        elif 't5-' in trainer_opt: config = transformers.T5Config.from_pretrained(trainer_opt['weights'])
        else: raise NotImplementedError('OPTION "{}" is not supported in set_model fuction.'.format(trainer_opt['weigths']))

        ## set model
        if trainer_opt['model'] == 'realm-t5-generator': model = m.m3_t5.generator.M3T5Generate
        elif trainer_opt['model'] == 't5-generator': model = transformers.T5ForConditionalGeneration
        else: raise NotImplementedError('OPTION "{}" is not supported in set_model fuction.'.format(trainer_opt['model']))
        
        if '_rand' in trainer_opt['weights'] :
            self.model = model(config)
        else:
            try:
                # 사전학습 모델 불러오기 
                pretrained_model = transformers.T5ForConditionalGeneration.from_pretrained(trainer_opt["weights"])
                ## predict, train
                if trainer_opt['predict']: self.model = model.from_pretrained(trainer_opt['load_path'], config=config, tokenizer=self.tokenizer, block_records =  self.block_records) 
                else: 
                    self.model = model.from_pretrained(trainer_opt['weights'], config=config, tokenizer=self.tokenizer, block_records =  self.block_records)
                   
                    # 사전학습모델 weight copy
                    pretrained_state = dict()
                    current_model_dict = self.model.state_dict()
                    pretrained_model_dict = pretrained_model.state_dict() 
                    for pre_k, pre_v in pretrained_model_dict.items():
                        for crt_k, crt_v in current_model_dict.items():
                            if pre_k in crt_k: pretrained_state[crt_k] = pre_v
                    # 사전학습 때 없는 parameter : random으로 주기
                    for each_key in list(set(current_model_dict.keys())-set(pretrained_state.keys())):
                        pretrained_state[each_key] = current_model_dict.get(each_key)
                
                    current_model_dict.update(pretrained_state)
                    self.model.load_state_dict(current_model_dict)

            except OSError as e:
                logging.error(e)
                self.model = model(config)
        if torch.cuda.is_available(): self.model.to('cuda')

        # update config     
        merged_dict = {"t5":config.to_dict(), "trainer": trainer_opt}
        config = merged_dict
        with open('config/config.json', 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def set_loss_func(self):
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        if torch.cuda.is_available(): loss_func.to('cuda')

        def cross_entropy_for_generator(logit, target):
            loss = loss_func(logit.view(-1, logit.size(-1)), target.view(-1))
            return loss
        if 'generator' in self.args['trainer']['model'] and self.args['trainer']['loss_func'] == 'cross-entropy': self.loss_func = cross_entropy_for_generator
        else: raise NotImplementedError('No loss function for  {} for {}'.format(self.args['trainer']['loss_func'], \
                self.args['trainer']['model']))

    def set_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def set_acc_func(self, opt='token-accuracy'):

        def token_accuracy(logits, target, target_length=None):
            prediction = torch.argmax(logits, dim=-1)
            acc, count = 0, 0
            for pindex in range(prediction.shape[0]):
                gold = target[pindex,:target_length[pindex]]
                pred = prediction[pindex,:target_length[pindex]]
                if len(gold) > len(pred):
                    pad = [0]*(len(gold)-len(pred))
                    pad = torch.tensor(pad).to(prediction.device)
                    pred = torch.cat((pred,pad),dim=0)
                elif len(pred) > len(gold):
                    pad = [0]*(len(pred)-len(gold))
                    pad = torch.tensor(pad).to(prediction.device)
                    gold = torch.cat((gold,pad),dim=0)
                acc += int(sum(pred == gold))
                count += len(pred)
            return acc/count

        def accuracy(logits, target, target_length=None):
            return torch.mean((torch.argmax(logits, dim=-1) == target).type(torch.FloatTensor)).item()

        opt = opt.lower()
        if opt == 'token-accuracy': self.acc_func = token_accuracy
        elif opt == 'accuracy': self.acc_func = accuracy
        else: raise NotImplementedError('OPTION {} is not supported in set_acc_func.'.format(opt))

    def set_dataloader(self):
        if self.args['trainer']['predict']:
            if self.args['trainer']['test_dataset']:
                self.test_loader = get_dataloader(self.args['trainer']['model'], self.args['trainer']['test_dataset'],
                    self.tokenizer, self.args['trainer']['batch_size'],
                    labels = self.label_list,
                    max_source_length = self.args['trainer']['max_source_length'],
                    max_knowledge_length = self.args['trainer']['max_knowledge_length'],
                    max_target_length = self.args['trainer']['max_target_length'],
                    large_dataset = self.args['trainer']['large_dataset'],
                    num_workers = self.args['trainer']['num_workers'],
                    shuffle = False)
        else:
            if self.args['trainer']['training_dataset']:
                self.train_loader = get_dataloader(self.args['trainer']['model'], self.args['trainer']['training_dataset'],
                        self.tokenizer, self.args['trainer']['batch_size'],
                        labels = self.label_list,
                        max_source_length = self.args['trainer']['max_source_length'],
                        max_knowledge_length = self.args['trainer']['max_knowledge_length'],
                        max_target_length = self.args['trainer']['max_target_length'],
                        large_dataset = self.args['trainer']['large_dataset'],
                        num_workers = self.args['trainer']['num_workers'],
                        shuffle = False)
            if self.args['trainer']['validation_dataset']:
                self.valid_loader = get_dataloader(self.args['trainer']['model'], self.args['trainer']['validation_dataset'],
                        self.tokenizer, self.args['trainer']['batch_size'],
                        labels = self.label_list,
                        max_source_length = self.args['trainer']['max_source_length'],
                        max_knowledge_length = self.args['trainer']['max_knowledge_length'],
                        max_target_length = self.args['trainer']['max_target_length'],
                        large_dataset = self.args['trainer']['large_dataset'],
                        num_workers = self.args['trainer']['num_workers'],
                        shuffle = False)

    def generate(self, source, logits_processor=None):

        if not logits_processor: logits_processor = transformers.LogitsProcessorList()

        logits = self.model.generate(input_ids=source,
                early_stopping = self.args['trainer']['early_stopping'],
                top_k = self.args['trainer']['top_k'],
                num_beams = self.args['trainer']['num_beams'],
                max_length = self.args['trainer']['max_length'],
                min_length = self.args['trainer']['min_length'],
                repetition_penalty = self.args['trainer']['repetition_penalty'],
                length_penalty = self.args['trainer']['length_penalty'],
                temperature = self.args['trainer']['temperature'],
                num_return_sequences = self.args['trainer']['num_return_sequences'],
                #logits_processor = logits_processor, ## transformers>=4.15.0
                )
        return logits

    def generator_decode(self, srce, pred, gold):
        data = {'srce':srce, 'pred':pred, 'gold':gold}
        for key in data:
            data[key] = self.tokenizer.convert_ids_to_tokens(data[key])
            data[key] = ''.join(data[key]).replace('▁',' ').strip()
        return data

    def get_output(self, batch, topk, **kwargs):
        is_train = kwargs.pop('is_train',True)
        verbose = kwargs.pop('verbose',False)

        source = batch['source']
        target = batch['target']
        source_attns = batch['source_attns']
        target_length = batch['target_length']
        kb_idx = batch['kb_idx']

        topk_target = []; topk_target_length =[]
        for bidx, each_target in enumerate(target):
            for kidx in range(topk): 
                topk_target_length.append(target_length[bidx].tolist())

        if torch.cuda.is_available():
            source = source.cuda()
            target = target.cuda()
            source_attns = source_attns.cuda()

        if is_train: 
            output = self.model(input_ids=source, labels=target, kb_answers=kb_idx, is_train=is_train, attention_mask=source_attns)
        else:
            with torch.no_grad(): output = self.model(input_ids=source, labels=target, kb_answers=kb_idx, is_train=is_train, attention_mask=source_attns)

        logits = output.logits
        retriever_loss  = output.retriever_loss
        reader_loss  = output.reader_loss
        if output.loss != None: loss = output.loss
        elif self.loss_func:
            loss = self.loss_func(logits, target)
        else: None

        if self.acc_func: acc = self.acc_func(logits, output.labels, topk_target_length)
        else: None

        return {'logits':output.logits, 'loss':loss, 'acc':acc, 'retriever_loss':retriever_loss, 'reader_loss':reader_loss}

    def run_batch(self, opt, epoch = 0):
        is_train = opt == 'train'

        if is_train: self.model.train()
        else: self.model.eval()

        if opt == 'train':
            if not self.train_loader: self.set_dataloader()
            dataloader = tqdm(self.train_loader)
        elif opt == 'valid':
            if not self.valid_loader: self.set_dataloader()
            dataloader = tqdm(self.valid_loader)
        elif opt == 'test':
            if not self.test_loader: self.set_dataloader()
            dataloader = tqdm(self.test_loader)
        else: raise NotImplementedError('OPTION {} is not supported in run_batch function.'.format(opt))
        losses, acces = 0, 0
        retriever_losses, reader_losses = 0, 0
        for b_index, batch in enumerate(dataloader):
            if is_train: self.optimizer.zero_grad()

            verbose = b_index % self.args['trainer']['logging_steps'] == 0
            output = self.get_output(batch, self.args['trainer']['reader_beam_size'])

            loss = output['loss']
            logits= output['logits']
            retriever_loss = output['retriever_loss']
            reader_loss = output['reader_loss']
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
            
            losses += loss.item()
            loss = losses/(b_index+1)
            
            retriever_losses += retriever_loss.item()
            retriever_loss = retriever_losses/(b_index+1)
            
            reader_losses += reader_loss.item()
            reader_loss = reader_losses/(b_index+1)

            acces += output['acc']
            acc = acces/(b_index+1)

            dataloader.set_description('[{}] Epoch:{}-L{:.3f}_RTL{:.3f}_RDL{:.3f}_A{:.3f}'.format(opt.upper(), epoch, loss, retriever_loss, reader_loss, acc))
            global_step = epoch*len(dataloader)+b_index+1
            extra_info = {'mode':'train','epoch':epoch,'loss':loss,'acc':acc}
            if (b_index+1) % self.args['trainer']['logging_steps'] == 0:
                self.tensorboard.add_scalar(tag='{}_loss'.format(opt), scalar_value=loss, global_step=global_step)
                self.tensorboard.add_scalar(tag='{}_accuracy'.format(opt), scalar_value=acc, global_step=global_step)
            if (b_index+1)%self.args['trainer']['save_every_n_steps'] == 0:
                for p in self.optimizer.param_groups: learning_rate = p['lr'].item()
                self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_batch_{}_loss_{:.4f}_acc_{:.4f}'.format(
                    self.args['trainer']['save_path'], opt, self.args['trainer']['model'], self.args['trainer']['learning_rate'], \
                            self.args['trainer']['patience'], epoch, b_index+1, loss, acc), args=args, extra_info=extra_info)
        if self.args['trainer']['all_save'] or not self.args['trainer']['validation_dataset']:
            self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_loss_{:.4f}_acc_{:.4f}'.format(
                self.args['trainer']['save_path'], opt, self.args['trainer']['model'], self.args['trainer']['learning_rate'], \
                        self.args['trainer']['patience'], epoch, loss, acc), args=args, extra_info=extra_info)

        return {'loss':loss, 'acc':acc, 'retriever_loss':retriever_loss, 'reader_loss':reader_loss}

    def view_sample(self, source, prediction, target, tensorboard=None):
        if tensorboard: pass
        else:
            srce = self.tokenizer.decode(source)
            pred = self.tokenizer.decode(prediction)
            gold = self.tokenizer.decode(target)
    
    def train(self):
        self.set_tensorboard(os.path.join(self.args['trainer']['save_path'], 'tensorboard'))
        self.set_tokenizer(self.args['trainer']['model'], self.args['trainer']['tokenizer_path'])
        self.set_knowledge_base(self.args['trainer']['additional_documents_path'])
        self.set_model(self.args['t5'], self.args['trainer'])
        self.set_knowledge_base_in_model(self.tokenizer, self.model, self.args['trainer']['projected_size'], \
                self.args['trainer']['batch_size'], self.args['trainer']['predict'])
        self.set_loss_func()
        self.set_acc_func(opt=self.args['trainer']['acc_func'])
        if torch.cuda.device_count() > 1: self.set_parallel()
        self.set_optimizer(lr=self.args['trainer']['learning_rate'])
        # self.set_lr_scheduler(optimizer=self.optimizer, num_training_steps=len(self.train_loader))
        best_val_loss, best_val_acc  = 1e+5, -1e+5
        patience = 0
        global_step = 0

        for epoch in range(self.args['trainer']['epoch']):
            sys.stderr.write('\n')
            output = self.run_batch('train', epoch)
            if self.args['trainer']['validation_dataset']:
                output = self.run_batch('valid',epoch)

                val_loss = output['loss']
                val_acc = output['acc']
                val_retriever_loss =  output['retriever_loss']
                val_reader_loss = output['reader_loss']
                
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    extra_info = {'mode':'valid','epoch':epoch,'loss':val_loss,'acc':val_acc}
                    self.save_model(self.model,'{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_valLoss_{:.4f}_valAcc_{:.4f}'.format(self.args['trainer']['save_path'], 'valid', 
                        self.args['trainer']['model'], self.args['trainer']['learning_rate'], self.args['trainer']['patience'], epoch, val_loss, val_acc), args=self.args, extra_info=extra_info)
                    self.save_model(self.model, f"{self.args['trainer']['save_path']}/trained_model", args=self.args, extra_info=extra_info)
                    patience = 0
                else:
                    patience += 1
                    if patience > self.args['trainer']['patience']:
                        logging.info('Ran out of patience.')
                        sys.exit()
    
    @torch.no_grad()
    def predict(self):
        self.set_tokenizer(self.args['trainer']['model'], self.args['trainer']['tokenizer_path'])
        self.set_knowledge_base(self.args['trainer']['additional_documents_path'])
        self.set_model(self.args['t5'], self.args['trainer'])
        self.set_knowledge_base_in_model(self.tokenizer, self.model, self.args['trainer']['projected_size'], \
                self.args['trainer']['batch_size'], self.args['trainer']['predict'])
        self.set_loss_func()
        self.set_acc_func(opt=self.args['trainer']['acc_func'])
        if torch.cuda.device_count() > 1: self.set_parallel()

        if self.args['trainer']['load_path'] == None: ofp = sys.stdout
        else: ofp = open(self.args['trainer']['load_path']+self.args['trainer']['result_path'], 'w')

        self.set_dataloader()
        if self.test_loader == None:
            raise AttributeError('No loaded test file.')
        dataloader = tqdm(self.test_loader)

        if self.args['trainer']['evaluate']: outs = list()
        for b_index, batch in enumerate(dataloader):
            if 'generator' in self.args['trainer']['model']:
                source = batch['source']
                target = batch['target']
                kb_answers = batch['kb_idx']
                target_length = batch['target_length']
                source_attns = batch['source_attns']

                if torch.cuda.is_available():
                    source = source.cuda()
                    target = target.cuda()
                    source_attns = source_attns.cuda()

                prediction, retrieved_ids= self.model.generate(batch['source'].to(self.model.device), \
                        kb_answers=kb_answers, attention_mask=source_attns, labels=target)
            for index in range(len(prediction)):
                srce = batch['source'][index].detach().cpu().tolist()
                gold = batch['target'][index].detach().cpu().tolist()
                pred = prediction[index].detach().cpu().tolist()

                gold_kb = batch['kb_idx'][index].detach().cpu().tolist()
                pred_kb = retrieved_ids[index].detach().cpu().tolist()

                # eos_token remove
                if self.tokenizer.eos_token_id in srce : srce = srce[:srce.index(self.tokenizer.eos_token_id)]
                if self.tokenizer.eos_token_id in gold : gold = gold[:gold.index(self.tokenizer.eos_token_id)]
                if self.tokenizer.eos_token_id in pred : pred = pred[:pred.index(self.tokenizer.eos_token_id)]
                
                if self.args['trainer']['model'] in ['t5-generator', 't5-pgn-generator', 'realm-t5-generator']:
                    out = {'srce':self.tokenizer.decode(srce), 'gold':self.tokenizer.decode(gold), \
                            'pred':self.tokenizer.decode(pred), 'gold_kb':gold_kb, 'pred_kb':pred_kb}
                            #'pred':self.tokenizer.decode(pred), 'gold_kb':gold_kb_text, 'pred_kb':pred_kb_text}
                else: raise NotImplementedError('No predict function for {}.'.format(self.args['trainer']['model']))
                if self.args['trainer']['evaluate']: outs.append(out)
                result = {'data':batch['data'][index],'output':out}
                ofp.write(f"{json.dumps(result, ensure_ascii=False)}\n")
                ofp.flush()

        scores = {}
        if self.args['trainer']['evaluate']:
            pred = [d['pred'] for d in outs]
            gold = [d['gold'] for d in outs]
            pred_kb = [d['pred_kb'] for d in outs]
            gold_kb = [d['gold_kb'] for d in outs]
            if 'generator' in self.args['trainer']['model']:
                bleu = m.evaluate.get_bleu(gold, pred)
                rouge = m.evaluate.get_rouge(gold, pred)
                #cider = m.evaluate.get_cider(gold, pred)
                mrr5 = m.evaluate.get_mrr(gold_kb, pred_kb, topk=5)
                r1 = m.evaluate.get_recall_at_k(gold_kb, pred_kb, topk=1)
                r5 = m.evaluate.get_recall_at_k(gold_kb, pred_kb, topk=5)
                r10 = m.evaluate.get_recall_at_k(gold_kb, pred_kb, topk=10)
                scores = {'bleu':bleu, 'rouge':rouge, 'mrr':mrr5, 'r@1':r1, 'r@5':r5, 'r@10':r10}#, 'cider':cider}
            elif 'classifier' in self.args['trainer']['model']:
                scores = m.evaluate.get_cls(gold, pred)

        merged_dict = {**self.args['t5'], **self.args['trainer']}
        args_dict = dict(merged_dict)
        ofp.write('{}\n'.format(json.dumps({'args':args_dict,'scores':scores}, ensure_ascii=False)))


if __name__ == '__main__':
    logging.info('START {}'.format(datetime.datetime.now()))
    logging.info("\ncommands > python "+" ".join(sys.argv))
    args = ArgumentParser(description='M3_realm Trainer')
    from config.parse_config import ConfigParser
    args.add_argument('-c', '--config', default=None, type=str,help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    trainer = Trainer(config)
    
    if bool(config['trainer']['predict']) : # 예측
        trainer.predict()
    else: # 학습
        trainer.train()
