#-*-coding:utf-8-*-
# Created by Microsoft Corporation
# Licensed under the MIT license.

# Modified by Adaptive Intelligence Research Lab(https://air.changwon.ac.kr/)., 2020. 01. ~

import os
import json
import logging
import torch
import sys
import re
import numpy as np
from torch.utils import data
from time import time
import pandas as pd

def cleaning(sentence):

    sent = sentence.strip()

    #sent = re.sub('\[[^\]]*\]','',sent) ## 대괄호 제거
    #sent = re.sub('\([^\)]*\)','',sent) ## 소괄호 제거
    #sent = re.sub('[^ㅏ-ㅣㄱ-ㅎ가-힣0-9a-zA-Z\.%, ]',' ', sent) ## 특수문자 모두 제거
    sent = re.sub('  *',' ',sent).strip() ## 다중 공백 제거

    return sent

def convert_sentence_to_input(inputs, tokenizer, max_len, direction='left', special_token=False):
    inputs = tokenizer.tokenize(inputs)
    if special_token: inputs = inputs + [tokenizer.eos_token] ## for bert
    dif = abs(max_len - len(inputs))
    if direction == 'left':
        if len(inputs) < max_len: inputs += [tokenizer.pad_token] * dif
        elif max_len < len(inputs): inputs = inputs[dif:]
    else:
        if len(inputs) < max_len: inputs += [tokenizer.pad_token] * dif
        elif max_len < len(inputs): inputs = inputs[:max_len]
    inputs = tokenizer.convert_tokens_to_ids(inputs)
    inputs = tokenizer.decode(inputs)
    return inputs.strip()

class Dataset(data.Dataset):
    def __init__(self, file_path, tokenizer,
                 max_source_length=None,
                 max_knowledge_length=None,
                 max_target_length=None,
                 large_dataset=False):

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.large_dataset = large_dataset

        if max_source_length is None: self.max_source_length = 10e+10
        else: self.max_source_length = max_source_length
        if max_target_length is None: self.max_target_length = 10e+10
        else: self.max_target_length = max_target_length
        if max_knowledge_length is None: self.max_knowledge_length = 10e+10
        else: self.max_knowledge_length = max_knowledge_length

        if self.large_dataset:
            self.set_file_list()
            data_max_source, data_max_target = self.load_file(self.file_list.pop(), self.max_source_length, self.max_target_length)
            self.data = list()
        else:
            assert not os.path.isdir(self.file_path)
            data_max_source, data_max_target = self.load_file(self.file_path, self.max_source_length, self.max_target_length)

            self.data_size = len(self.data)
            print('Data size = {}'.format(self.data_size), flush=True)
        if max_source_length is None: self.max_source_length = data_max_source
        if max_target_length is None: self.max_target_length = data_max_target

        logging.info('Total batch : {}'.format(self.data_size))
        logging.info('Max Source Length : {}'.format(self.max_source_length))
        logging.info('Max Knowledge Length : {}'.format(self.max_knowledge_length))
        logging.info('Max Target Length : {}'.format(self.max_target_length))

    def set_file_list(self):
        if os.path.isdir(self.file_path):
            file_list = sum([[os.path.join(d[0],f) for f in d[-1]] for d in list(os.walk(self.file_path))],[])
            file_list = sorted(file_list)
        else:
            file_list = [self.file_path]
        self.file_list = file_list
        self.data_size = self.set_data_size(file_list)
        print('Data size = {}'.format(self.data_size), flush=True)

    def set_data_size(self, file_list):
        data_size = 0
        for filename in self.file_list:
            if os.path.splitext(filename)[-1] in ['.json']:
                with open(filename, 'r') as f: data = json.load(f)
                data_size += len(data)
            else: ## .txt, .tsv, .jsonl
                with open(filename, 'r') as f:
                    for line in f: data_size += 1
        return data_size

    def load_file(self, filename, max_source_length, max_target_length):
        logging.info('Load {} '.format(filename))

        source_lengths = list()
        target_lengths = list()

        with open(filename, 'r') as ifp:
            key = os.path.splitext(filename)[-1]
            if key in ['.jsonl']:
                data = [json.loads(d) for d in ifp]
            elif key in ['.json']:
                data = json.load(ifp)
            elif key in ['.tsv','.txt']:
                data = [d.strip().split('\t') for d in ifp]
                data = [{'source':d[0], 'target':d[1]} for d in data]
            else:
                raise KeyError('No rule for {} filetype.'.format(key))

        self.data = list()
        for index, item in enumerate(data):
            sys.stderr.write('\r{}/{}'.format(index, len(data)))
            # XX external_kb 추가
            source = item['source']
            target = item['target']
            kb_idx = item['kb_idx']

            if type(source) == list: source = source[0]
            if type(target) == list: target = target[0]

            source = cleaning(source.lower())
            target = cleaning(target.lower())
            
            source_length = len(self.tokenizer.encode(source))
            target_length = len(self.tokenizer.encode(target))

            if source_length > max_source_length:
                # false 면 앞에서부터 잘라냄 
                source= convert_sentence_to_input(source, self.tokenizer, max_source_length, special_token=False)
                source_length = len(self.tokenizer.encode(source))

            if target_length > max_target_length: 
                target= convert_sentence_to_input(target, self.tokenizer, max_target_length, special_token=False)
                target_length = len(self.tokenizer.encode(target))

            source_lengths.append(source_length)
            target_lengths.append(target_length)
            self.data.append({
                'source':source,
                'target':target,
                'source_length':source_length,
                'target_length':target_length,
                'kb_idx' : kb_idx,
                'data':item,
                })
        sys.stderr.write('\n'); sys.stderr.flush()
        
        source_df = pd.DataFrame(source_lengths)
        print(source_df.describe())
        target_df = pd.DataFrame(target_lengths)
        print(target_df.describe())
        return max(source_lengths+[0]), max(target_lengths+[0])


    def __getitem__(self, index):
        if self.large_dataset:
            if len(self.data) == 0:
                if len(self.file_list) == 0: self.set_file_list()
                filename = self.file_list.pop()
                self.load_file(filename)
            return self.data.pop(0)
        else:
            return self.data[index]

    def __len__(self):
        ## for setting total_batch, 10%: removed data if source_length < 5
        #return int(1000000 * len(self.file_list) * 0.9)
        return self.data_size

def generator_collate_fn(data, tokenizer, max_source_length, max_target_length):

    sources = []
    source_attns =[]
    targets = []
    target_lengths = []
    origin = []
    kb_idxs = []

    for items in data:
        # source
        source = items['source'].strip()
        source_input = tokenizer(source, max_length=max_source_length, padding='max_length', truncation=True)

        tokenized_source=source_input['input_ids']
        source_attn=source_input['attention_mask']

        # target
        target = items['target']
        tokenized_target = tokenizer.encode(target, max_length=max_target_length, padding='max_length', truncation=True)
        target_length = tokenized_target.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in tokenized_target else len(tokenized_target)

        # kb_idx 
        kb_idx = items['kb_idx']

        origin.append(items['data'])

        sources.append(tokenized_source)
        source_attns.append(source_attn)
        targets.append(tokenized_target)
        target_lengths.append(target_length)
        kb_idxs.append(kb_idx)

    source_tensor = torch.tensor(sources)
    target_tensor = torch.tensor(targets)
    target_lengths = torch.tensor(target_lengths)
    kb_tensor = torch.tensor(kb_idxs)
    source_attns_tensor = torch.tensor(source_attns)    
    result = {
            'source':source_tensor,
            'source_attns':source_attns_tensor,
            'target':target_tensor,
            'target_length':target_lengths,
            'kb_idx': kb_tensor,
            'data':origin,
            }
    return result

def get_dataloader(opt, file_path, tokenizer, batch_size,
                   labels=None,
                   max_source_length=None,
                   max_knowledge_length=None,
                   max_target_length=None,
                   large_dataset=False,
                   num_workers=0,
                   shuffle=True):
    logging.info('Reading from {}.'.format(file_path))

    dataset = Dataset(file_path, tokenizer,
                      max_source_length=max_source_length,
                      max_knowledge_length=max_knowledge_length,
                      max_target_length=max_target_length,
                      large_dataset=large_dataset)

    if max_source_length is None: max_source_length = dataset.max_source_length
    if max_target_length is None: max_target_length = dataset.max_target_length

    if 'generator' in opt:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=lambda data: generator_collate_fn(data, tokenizer, max_source_length, max_target_length))
    elif 'classifier' in opt:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=lambda data: classifier_collate_fn(data, tokenizer, max_source_length, max_target_length, labels = labels))
    else:
        raise NotImplementedError('OPTION {} is not supported.'.format(opt))
    return data_loader

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='M3 Trainer')
    # training
    # hyper params
    parser.add_argument('-b', '--batch', default=16, type=int)
    # dataset
    parser.add_argument('-td', '--dataset', required=True, type=str)
    parser.add_argument('-ms', '--max_source_length', default=512, type=int)
    parser.add_argument('-mt', '--max_target_length', default=None, type=int)
    # etc
    parser.add_argument('-tk','--tokenizer_path', default='tokenizer/tokenizer_30000_addUmjeol/spiece.model', help='path of pretrained tokenizer model file')
    args = parser.parse_args()

    return args


if __name__=='__main__':
    from transformers import M3Tokenizer

    args = parse_args()

    tokenizer = M3Tokenizer.from_pretrained(args.tokenizer_path)

    train_dataloader = get_dataloader(args.dataset, tokenizer, args.batch,
                                      max_source_length = args.max_source_length,
                                      max_target_length = args.max_target_length,
                                      shuffle = False)
    count = 0
    while(1):
        for di, d in enumerate(train_dataloader):
            start = time()
            sys.stderr.write('\rCHECK UTILS: epoch {}\tstep {}\ttime {:.5f}'.format(count,di,time()-start))
            count += 1
