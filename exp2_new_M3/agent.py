#-*-coding:utf-8-*-
# 2022 AIR Lab.


from asyncore import read
import sys, torch, logging, os
import json
from time import time
import datetime
from os.path import exists
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import shutil

from action import Action, time_this
from dataset import Dataset
#from retriever import M3RetrieverConfig, M3Retriever
from retriever import M3Retriever
#from net import M3NetConfig, M3T5Encoder, M3T5Generator
from net import M3T5Generator
from nets.T5 import M3PreTrainedModel

from transformers import RealmConfig, T5Config
import transformers
transformers.logging.set_verbosity_error()
import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_utils import PreTrainedModel,apply_chunking_to_forward
#from transformers.utils.doc import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import (
        BaseModelOutputWithPastAndCrossAttentions,
        BaseModelOutputWithPoolingAndCrossAttentions,
        MaskedLMOutput,
        ModelOutput,
)
#from huggingface_hub import hf_hub_download
#from transformers.activations import ACT2FN
from action import getBack


from torch.nn.parallel import DistributedDataParallel
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.getLogger().setLevel(logging.INFO)

torch.manual_seed(100)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Agent:
    '''Agent controls training and testing model
    '''

    def __init__(self, config: dict):
        super(Agent, self).__init__()

        self.config = config
        self.log_dir = os.path.join(self.config['log_dir'], self.config['experiment_index'])
        self.model_dir = os.path.join(self.config['model_dir'], self.config['experiment_index'])
        
        # linux, macos 환경에 따라 gpu devie 선택 
        if torch.cuda.is_available() or torch.backends.mps.is_available() :
            self.device = torch.device(self.config["device"])        # mps, cuda
        else:
            self.device = torch.device('cpu')

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.set_environment()

        self.action = Action(self.config)
        self.tokenizer = self.action.get_tokenizer(self.config['tokenizer_path'])
        self.action.set_tensorboard(os.path.join(self.config['save_checkpoint_path'], 'tensorboard'))

        # 데이터 읽기 설정
        self.dataset = Dataset(self.config, self.tokenizer)
        self.block_records = self.dataset.load_external_memory(self.config['external_memory_path'])

        self.set_retriever_environment()
        # 반드시 named parameter로 할 것 
        self.retriever = M3Retriever.from_pretrained(self.config["pretrained_weight_path"], \
                                                                config=self.retriever_config, \
                                                                tokenizer=self.tokenizer, \
                                                                block_records= self.block_records)
        
        #for name, tmp in self.retriever.state_dict.items():
        #    print(name, tmp.shape)
        #print(self.retriever.state_dict())

        self.retriever.to(self.device)

        # 멀티모달을 위한 모델 구조 선택 
        #self.set_net_environment()
        #self.generator = M3T5Generator.from_pretrained(self.config["pretrained_weight_path"], \
                #config=self.m3net_config, \
                #tokenizer=self.tokenizer)
        #self.generator.to(self.device)

        # model type이 classifier / generator / pin-pinter 인지에 따른 선택
        self.label_list = None
        if self.config['net_type'] == 'classifier':
            try:
                with open(self.config['label_path'], 'r') as f: self.label_list = [d.strip() for d in f]
            except:
                raise AttributeError('Classifier model need "label_path".')

        self.action = Action(self.config) # 실제 동작 

        self.tensorboard = self.writer

        self.best_val_loss = 1e+5
        self.best_val_acc = -1e+5
        #self.best_metrics = None # rely on val_metrics to be best
        self.patience = 0

    def set_environment(self):
        """ Set environment, eg. del and create file """
        # remove and create logdir
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)

        # remove and create modeldir 나중에 만들 것
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.makedirs(self.model_dir)

    def set_retriever_environment(self):
        self.retriever_config = self.action.get_pretrained_config(self.config["pretrained_weight_path"])
        self.retriever_config.model_type = 'retriever'
        self.retriever_config.hidden_size = self.config["hidden_size"]
        self.retriever_config.searcher_beam_size = self.config["searcher_beam_size"]
        self.retriever_config.searcher_seq_len = self.config["searcher_seq_len"]
        self.retriever_config.projected_size = self.config['retriever_proj_size']
        self.retriever_config.update_emb = self.config["update_retriever_emb"]
        self.retriever_config.max_knowledge_length = self.config["max_knowledge_length"]
        self.retriever_config.max_source_length = self.config["max_source_length"]
        self.retriever_config.num_block_records = len(self.block_records)
        self.retriever_config.update({'device':self.config["device"]})
        self.retriever_config.update(T5Config.from_pretrained(self.config['pretrained_weight_path']).to_dict())
        #print(self.retriever_config)

    def set_net_environment(self):
        self.m3net_config = self.action.get_pretrained_config(self.config["pretrained_weight_path"])
        self.m3net_config.model_type = 'generator'
        self.m3net_config.hidden_size = self.config["hidden_size"]
        self.m3net_config.span_hidden_size = self.config["span_hidden_size"]
        self.m3net_config.max_span_width = self.config["max_span_width"]
        self.m3net_config.reader_layer_norm_eps = self.config["reader_layer_norm_eps"]
        self.m3net_config.reader_beam_size = self.config["reader_beam_size"]
        self.m3net_config.reader_seq_len = self.config["reader_seq_len"]
        self.m3net_config.d_model = self.config["d_model"]
        self.m3net_config.num_layers = self.config["num_layers"]
        self.m3net_config.relative_attention_num_buckets = self.config["relative_attention_num_buckets"]
        self.m3net_config.relative_attention_max_distance = self.config["relative_attention_max_distance"]
        self.m3net_config.d_kv = self.config["d_kv"]
        self.m3net_config.num_heads = self.config["num_heads"]
        self.m3net_config.dropout_rate = self.config["dropout_rate"]
        self.m3net_config.layer_norm_epsilon = self.config["layer_norm_epsilon"]
        self.m3net_config.is_gated_act = self.config["is_gated_act"]
        self.m3net_config.d_ff = self.config["d_ff"]
        self.m3net_config.dense_act_fn = self.config["dense_act_fn"]
        self.m3net_config.num_decoder_layers = self.config["num_decoder_layers"]
        self.m3net_config.searcher_beam_size = self.config["searcher_beam_size"]
        self.m3net_config.update_emb = self.config["update_generator_emb"]
        self.m3net_config.projected_size = self.config["net_projected_size"]   # 이건 수정이 필요함 
        self.m3net_config.max_length = self.config["max_target_length"]        # 확인 요망         
        self.m3net_config.update({'device':self.config["device"]})
        self.m3net_config.update(T5Config.from_pretrained(self.config['pretrained_weight_path']).to_dict())
        #print(self.m3net_config)


    def run(self):
        """Iterative trainig and evaluate model"""
        # 데이터 읽기
        train_loader, val_loader = self.dataset.get_dataloader()
        # 외부 메모리 인코딩 
        self.action.set_external_memory_in_model(self.tokenizer, \
                    self.retriever, \
                    self.config['retriever_proj_size'], \
                    self.config['batch_size'], \
                    self.block_records, \
                    mode=None)


        # 모델

        # momory, 나중에 수정하자.
        #self.knowledge_base = self.block_records

        learning_rate = self.config["learning_rate"]
        retriever_optimizer = self.action.get_optimizer(self.retriever, learning_rate)
        #generator_optimizer = self.action.get_optimizer(self.generator, learning_rate)
        retriever_loss, generator_loss = self.action.get_loss_fn()


        self.fit(train_loader, val_loader, retriever_loss, generator_loss, retriever_optimizer)
        #self.fit(train_loader, val_loader, retriever_loss, generator_loss, retriever_optimizer, generator_optimizer)

    
    #def fit(self, train_loader, val_loader, retriever_loss, generator_loss, retriever_optimizer, generator_optimizer): 
    def fit(self, train_loader, val_loader, retriever_loss, generator_loss, retriever_optimizer) :
        """학습이 시작되는 곳"""
        n_epochs = self.config["n_epochs"]
        exp = self.config["experiment_index"]
        eval_frequency = self.config["eval_frequency"] # 평가 주기  
        #val_loss, val_metric = self.eval_epoch(val_loader, self.model, loss_fn, threshold)

        #print(">>> Eval Loss At Val: {:.8f}".format(val_loss), flush=True)
        #print("-" * 50, flush=True)
        #self.plot_epoch(val_metric, val_loss, 0, self.writer, is_train=False)


        for epoch in range(1, n_epochs+1):
            #print("\n>> Exp: {} -> Train at Epoch {}/{}".format(exp, epoch,n_epochs), flush=True)

            #train_loss, train_metrics = self.run_batch(tqdm(train_loader), retriever_loss, generator_loss, retriever_optimizer, generator_optimizer, epoch, is_train=True)
            train_loss, train_metrics = self.run_batch(tqdm(train_loader), retriever_loss, generator_loss, retriever_optimizer, epoch, is_train=True)
            #print(">>> Train Loss:{:.8f}".format(train_loss), flush=True)

            #if epoch % eval_frequency == 0:
            # eval_frequency 분리하자 

            #print(">> Exp:{} -> Eval at Epoch: {}/{}".format(exp, epoch, n_epochs), flush=True)
            #val_loss, val_metrics = self.run_batch(tqdm(val_loader), retriever_loss, generator_loss, retriever_optimizer, generator_optimizer, epoch, is_train=False)
            fp_result = open(os.path.join(self.log_dir, "%s_result.txt" % epoch), "w") # 230406 ojy
            val_loss, val_metrics = self.run_batch(tqdm(val_loader), retriever_loss, generator_loss, retriever_optimizer, epoch, is_train=False, eval_fp=fp_result)
            fp_result.close()
            #print(">>> Eval Loss At Val:{:.4f}".format(val_loss), flush=True)
            #self.plot_epoch(val_metrics, val_loss, epoch, self.writer, is_train=False)

            if self.best_val_loss > val_loss:
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics
                extra_info = {'mode':'valid','epoch':epoch,'loss':val_loss,'acc':val_metrics}
                self.action.save_model(self.retriever, \
                        '{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_trainLoss_{:.4f}_trainAcc_{:.4f}'.format(self.model_dir, \
                                'Best', 'm3_retriever', \
                                self.config['learning_rate'], \
                                self.config['patience'], \
                                epoch, train_loss, train_metrics['acc']), \
                        tokenizer=self.tokenizer, optimizer=retriever_optimizer, config=self.config, block_records=self.block_records, extra_info=extra_info)

                extra_info = {'mode':'valid','epoch':epoch,'loss':val_loss,'acc':val_metrics}
#                self.action.save_model(self.generator, \
        #                        '{}/{}_{}_lr_{}_pat_{}_epoch_{:07d}_trainLoss_{:.4f}_trainAcc_{:.4f}'.format(self.model_dir, \
        #                                'Best', 'm3_generator', \
        #                                self.config['learning_rate'], \
        #                                self.config['patience'], \
        #                                epoch, train_loss, train_metrics['acc']), \
        #                        tokenizer=self.tokenizer, optimizer=generator_optimizer, config=self.config, extra_info=extra_info)
                
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.config['patience']:
                    logging.info('Ran out of patience.')
                    sys.exit()


    @time_this
    #def run_batch(self, data_loader, retriever_loss, generator_loss, retriever_optimizer, generator_optimizer, epoch, is_train):
    def run_batch(self, data_loader, retriever_loss, generator_loss, retriever_optimizer, epoch, is_train, eval_fp=None):
        if is_train: 
            self.retriever.train()
            #self.generator.train()
            opt = 'train'
        else: 
            self.retriever.eval()
            #self.generator.eval()
            opt = 'valid'

        losses,acces,retriever_losses,generator_losses = 0,0,0,0
        eval_frequency = self.config["eval_frequency"] # 평가 주기

        for b_index, batch in enumerate(data_loader):
            if is_train: 
                retriever_optimizer.zero_grad()
                #generator_optimizer.zero_grad()

            #print("\n--", b_index, len(batch), batch) 
            #print("\n--", batch['source'].dtype) 
            #print("\n--", batch['source_attns'].dtype) 
            #print("\n--", batch['target'].dtype) 
            #print("\n--", batch['target_length'].dtype) 
            #exit()

            #verbose = b_index%self.config['logging_steps']==0
            acc_, retriever_outputs, generator_outputs = self.get_output(b_index, batch, is_train)

            #loss, retriever_loss, generator_loss = self.eval_model(retriever_outputs, None, generator_outputs, batch['target'])
            loss, retriever_loss, generator_loss = self.eval_model(retriever_outputs, None, None, batch['target'])

            if is_train:
                loss.backward(retain_graph=True)
                #torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10.0) # adam일 경우에는 불필요 
                retriever_optimizer.step()
                #generator_optimizer.step()
            else :
                # 230410 ojy - predict result writing 
                top_n_list = torch.topk(retriever_outputs.retrieved_scores, 15).indices.detach().cpu().numpy()
                for eND, eNT in zip(batch['data'], top_n_list) :
                    eval_fp.write("%s\t%s\t%s\t%s\t%s\n" % (eND['source'], eND['target'], eND['kb_idx'], " ".join(np.char.mod('%d', eNT)), (eND['kb_idx'] in eNT)))
            
            losses += loss.item()
            loss = losses/(b_index+1)
        
            retriever_losses += retriever_loss.item()
            retriever_loss = retriever_losses/(b_index+1)

            #generator_losses += generator_loss.item()
            #generator_loss = generator_losses/(b_index+1)

            #acces += acc_
            acces += np.mean(list(acc_.values())) # 230309 ojy - 모든 top k 성능 평균을 더함
            #acces += acc_[1] # 221213 ojy - top1만 추가
            acc = acces/(b_index+1)

            data_loader.set_description('[{}] Epoch:{}-L{:.3f}_RTL{:.3f}_ACC{:.3f}'.format(opt.upper(), \
                    epoch, loss, retriever_loss, acc))
            #data_loader.set_description('[{}] Epoch:{}-L{:.3f}_RTL{:.3f}_GNL{:.3f}_ACC{:.3f}'.format(opt.upper(), \
                    #epoch, loss, retriever_loss, generator_loss, acc))
            global_step = epoch*len(data_loader)+b_index+1
            metrics = dict()
            metrics['acc'] = acc

            extra_info = {'mode':'train','epoch':epoch,'loss':loss,'acc':metrics}
        
            if is_train and global_step % eval_frequency == 0:                            # 일정 주기마나 업데이트 
                self.action.set_external_memory_in_model(self.tokenizer, \
                            self.retriever, \
                            self.config['retriever_proj_size'], \
                            self.config['batch_size'], \
                            self.block_records, \
                            mode=None)
            
        return loss, metrics

    def get_output(self, b_index, batch, is_train):
        #is_train = kwargs.pop('is_train',True)
        #verbose = kwargs.pop('verbose',False)

        source = batch['source']
        target = batch['target']
        source_attns = torch.tensor(batch['source_attns'])
        target_length = batch['target_length']

        source = source.to(torch.int32 if self.config['device'] == 'mps' else torch.int64)        # 추가 
        source_attns = source_attns.to(torch.int32 if self.config['device']=='mps' else torch.int64)        # 추가 
        target = target.to(torch.int32 if self.config['device']=='mps' else torch.int64)        # 추가 
        source = source.to(self.device)
        source_attns = source_attns.to(self.device)
        target = target.to(self.device)

        #print('\n\n@@@@@1', source.dtype)
        #print('\n\n@@@@@2', source_attns.dtype)
        #print('\n\n@@@@@3', target.dtype)


        # 초기화, 검색한 상위 k개 만큼 답도 만듦  
        target = target.repeat(self.config["searcher_beam_size"],1)
        target = target.to(dtype=torch.int32 if self.config['device']=='mps' else torch.int64)        # 추가 
        #print('\n\n@@@@@4', target.dtype)
        #-------------------------------------------------

        if is_train: 
            retriever_outputs = self.retriever(input_ids=source, b_index=b_index, kb_answers=None, attention_mask=source_attns)
            #generator_outputs = self.generator(input_ids=source, retriever_outputs=retriever_outputs, labels=target, batch_size=self.config['batch_size'], opt=is_train)
        else:
            with torch.no_grad(): 
                retriever_outputs = self.retriever(input_ids=source, b_index=b_index, kb_answers=None, attention_mask=source_attns)
                #generator_outputs = self.generator(input_ids=source, retriever_outputs=retriever_outputs, labels=target, batch_size=self.config['batch_size'], opt=is_train)

        # 
#        logits    =    generator_outputs.logits
#        target    =    generator_outputs.labels
#        target_length = target_length.repeat(self.config["searcher_beam_size"], 1) # [[..],[..],...,[..]]
#        target_length = torch.flatten(target_length) # [......]

        acc_func = self.action.get_acc_func(opt='acc_k')
        acc = acc_func(batch['target'], retriever_outputs.retrieved_scores, topk=[1, 3, 5, 10, 15])
#        return acc, retriever_outputs, generator_outputs
        return acc, retriever_outputs, None


    def eval_model(self, retriever_outputs, retriever_labels, generator_outputs, generator_labels):
        ''' 
        모델을 평가하여 loss를 돌려준다. 
        '''
        retriever_loss_fn, generator_loss_fn = self.action.get_loss_fn()    # 함수 지정 
        from torch.nn import functional as F

        # 221214 ojy retriever loss 구현
        retrieved_logits2 = retriever_outputs.retrieved_logits.squeeze(-1)
        know_vec = torch.zeros(self.config['batch_size'], self.retriever_config.num_block_records) # batch_size, knowledge size
        gen_label_mask = torch.zeros(self.config['batch_size'], self.retriever_config.num_block_records) # batch_size, knowledge size
        for eRI1, eRB in enumerate(retriever_outputs.retrieved_block_ids.tolist()) :
            gen_label_mask[eRI1, generator_labels[eRI1]] = 1
            for eRI2, eRK in enumerate(eRB) : know_vec[eRI1, eRK] = retrieved_logits2[eRI1, eRI2]
        retriever_loss = retriever_loss_fn(know_vec, gen_label_mask)

#        print(retriever_outputs.retrieved_block_ids)
#        print(generator_labels)
#        print(retriever_loss)
#        exit()

        return retriever_loss.mean(), retriever_loss.mean(), retriever_loss.mean()


        lm_logits = generator_outputs.logits        # 결과물의 head logits, [batch_size,seq_len,voc_size] -> #[batch_size, voc_size, seq_len]
        lm_logits = lm_logits.to(self.device)
        generator_labels = generator_labels.to(self.device)
        generator_labels = generator_labels.to(torch.int32 if self.device=='mps' else torch.int64)        # 추가 
        # [batch_size, searcher_beam_size]
        retriever_relevance_score = retriever_outputs.retrieved_logits.squeeze(-1)
        #print('---', lm_logits.shape, lm_logits, flush=True)
        #print('---', generator_labels.shape, generator_labels, flush=True)
        #print('### relevance socre', retriever_relevance_score.shape, retriever_relevance_score)

        # =================================================
        # retriever loss 
        retriever_loss = retriever_loss_fn(retriever_relevance_score)
        #print("+++", retriever_relevance_score.shape, retriever_relevance_score)
        #print("\n+++retriever loss: {}".format(retriever_loss), flush=True)

        # =================================================
        # generator loss : top-k를 위한 특별한 방법
        #print('generator_labels.size -----', generator_labels.size(0))
        #print('searcher_beam_size -----', self.config["searcher_beam_size"])
        generator_labels = generator_labels.repeat(self.config["searcher_beam_size"], 1)
        #-------------------------------------------------

        #print("---0---", lm_logits.shape, generator_labels.shape)
        #print("---0---", lm_logits, generator_labels)
        # [batch_size*top-k, voc_size, seq_len] ::: [batch_size*top-k, seq_len]
        generator_loss    = generator_loss_fn(lm_logits, generator_labels)
        #print("----1-----", generator_loss.shape, generator_loss)
        generator_loss    = generator_loss.mean(dim=1)    # 하나의 입력에 대한 값으로 변환, batch size * top-k 만큼 
        #print("-----2----", generator_loss.shape, generator_loss)

        # batch_size * top-k => [top-k, batch_size] => (transpose) => [batch_size, top-k]
        generator_loss = generator_loss.reshape(retriever_relevance_score.size(1), retriever_relevance_score.size(0))
        generator_loss = generator_loss.transpose_(0,1)
        #print("\n generator_loss(before) ---{}---{} ---".format(generator_loss.shape, generator_loss), flush=True)
        
        # -------------------------------------    
        # 수진이 다중 추론 QA 를 여기에 넣으면 될 듯 
        if self.config["searcher_beam_size"]>1:
            # top-k 중에 1등과 나머지의 상대적 차이로 계
            new_generator_loss=torch.empty(generator_loss.size(0),generator_loss.size(1))
            new_generator_loss = new_generator_loss.to(self.device)
            for n,tmp in enumerate(generator_loss):
                maxx=max(tmp)
                #print(n, tmp, maxx)
                new_generator_loss[n]=(generator_loss[n]-maxx)*-1

            generator_loss=new_generator_loss.to(self.device)
            #print("\n generator_loss(after) ---{}---{} ---".format(generator_loss.shape, generator_loss), flush=True)
            # --------------------------------------

        #print("_----3---- relevance score ", retriever_relevance_score.shape, retriever_relevance_score)
        #generator_correct = F.softmax(retriever_relevance_score,dim=-1).clone().detach().to(self.device)
        generator_correct = F.softmax(retriever_relevance_score,dim=-1).to(self.device)
        #print("_----3/1----", generator_correct)
        any_generator_correct = torch.any(generator_correct) # return True if the element of the list is not 0 or false
        any_generator_correct.type(torch.float32 if self.config['device']=='mps' else torch.float64)
        #print("_----3/2----", any_generator_correct.shape, any_generator_correct.dtype, any_generator_correct, flush=True)

        generator_loss = self.action.marginal_log_loss(generator_loss, generator_correct)
        #print("_----4----", generator_loss)
        generator_loss *= any_generator_correct.type(torch.float32 if self.config['device']=='mps' else torch.float64)
        #print("_----5----", generator_loss.dtype, generator_loss)

        #print("\n generator_loss ------{} --{}/{}-".format(generator_loss, torch.sum(generator_loss), generator_loss.size()), flush=True)

        generator_loss=torch.mean(generator_loss)
        
        #print("+++generator loss: {}".format(generator_loss), flush=True)


        # =================================================
        loss = generator_loss
        #loss = retriever_loss
        #loss = (retriever_loss + generator_loss).mean()

        return loss, retriever_loss, generator_loss



    @time_this
    def plot_epoch(self, metrics, loss, epoch, writer, is_train):
        ''' Plot epoch '''
        self.action.plot_loss(loss, epoch, writer, is_train)
        self.action.plot_metrics(metrics, epoch, writer, is_train)
        
