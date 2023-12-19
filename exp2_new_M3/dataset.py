import os
import json
import logging
import torch
import sys
import re
import numpy as np
from torch.utils import data
from time import time
from tqdm import tqdm
import pandas as pd
from utils.jit_cleaning import jit_cleaning
from utils.truncate_input import truncate_input # TODO 나중에는 파일 하나로 묶어서 사용할 수 있다.
#from action import Action


class Dataset(data.Dataset):
	def __init__(self, config, tokenizer):

		self.config = config
		#action = Action(config)

		self.data_dir = config['data_dir']
		self.dataset = config['dataset']
		self.tokenizer = tokenizer
		self.datatype = self.config["datatype"] # text, image, video, ...

		self.max_source_length = self.config["max_source_length"]
		self.max_target_length = self.config["max_target_length"]
		self.max_knowledge_length = self.config["max_knowledge_length"]

		# 멀티모달을 위한 코드 
		if self.datatype == 'text':
			self.filepath = os.path.join(self.data_dir, self.dataset)
			self.train_iterator = None
			self.val_iterator = None
			self.test_iterator = None
			self.train_dataset, self.valid_dataset = self.load_data()
			self.train_loader, self.valid_loader = self.text_dataloader()
		elif self.datatype == 'image': # add code for image
			pass
		else:
			pass

		logging.info('Total batch : {}'.format(len(self.train_loader)))
		logging.info('Max Source Length : {}'.format(self.max_source_length))
		logging.info('Max Knowledge Length : {}'.format(self.max_knowledge_length))
		logging.info('Max Target Length : {}'.format(self.max_target_length))

	def load_data(self):
		''' Load specific dataset '''
		if 'jit' in self.dataset:
			train_dataset, valid_dataset = self._load_jit()
		elif 'dstc' in self.dataset:
			train_dataset, valid_dataset = self._load_dstc()
		elif self.dataset == 'paraphase':
			pass
		else:
			print(">> ERROR: No dataset available")
			sys.exit(-1)

		return train_dataset, valid_dataset

	def _load_jit(self):
		train_dataset = self.__load_file('train')
		valid_dataset = self.__load_file('valid')
		#test_dataset  = self.__load_file('test')

		return train_dataset, valid_dataset

	def _load_dstc(self):
		train_dataset = self.__load_file('train')
		valid_dataset = self.__load_file('valid')
		#test_dataset  = self.__load_file('test')

		return train_dataset, valid_dataset

	def __load_file(self, role):

		sources = list()
		source_lengths = list()
		targets = list()
		target_lengths = list()
		file_base = os.path.splitext(self.filepath)[0]
		file_ext = os.path.splitext(self.filepath)[-1]
		filepath = file_base + '_' + role + file_ext
		logging.info('Load {} '.format(filepath))

		with open(filepath, 'rt') as ifp:
			if file_ext in ['.jsonl']:
				data = [json.loads(d) for d in ifp]
			elif file_ext in ['.json']:
				data = json.load(ifp)
			elif file_ext in ['.tsv','.txt']:
				data = [d.strip().split('\t') for d in ifp]
				data = [{'source':d[0], 'target':d[1]} for d in data]
			else:
				raise KeyError('No rule for {} filetype.'.format(key))

		__data__ = list()
		for index, item in enumerate(data):
			sys.stderr.write('\r{}/{}'.format(index, len(data)))
			start_time = time()

			source = item['source']
			#target = item['target']
			target = item['kb_idx']

			# 데이터 파일이 여러 개의 리스트로 되어 있는 경우 
			if type(source) == list: source = source[0]
			if type(target) == list: target = target[0]

			#source = jit_cleaning(source)
			#target = jit_cleaning(target)

			source_length = len(self.tokenizer.encode(source))
			if type(target) != int : target_length = len(self.tokenizer.encode(target))
			else : target_length = 1
			#target_length = len(self.tokenizer.encode(target))
	
			if source_length > self.max_source_length: # 230309 ojy 오류 수정
				source= truncate_input(source, self.tokenizer, self.max_source_length, special_token=False)
				source_length = len(self.tokenizer.encode(source))

			if (type(target) != int) and (target_length > self.max_target_length):
				target= truncate_input(target, self.tokenizer, self.max_target_length, special_token=False)
				target_length = len(self.tokenizer.encode(target))
			
			source_lengths.append(source_length)
			target_lengths.append(target_length)
			__data__.append({
				'source':source,
				'target':target,
				'source_length':source_length,
				'target_length':target_length,
				'data':item,
				})

		sys.stderr.write('\n'); sys.stderr.flush()

		#source_df = pd.DataFrame(source_lengths)
		#print(source_df.describe())
		#target_df = pd.DataFrame(target_lengths)
		#print(target_df.describe())

		return __data__


	def get_dataloader(self):
		''' return dataloader for agent '''
		return self.train_loader, self.valid_loader


	def text_dataloader(self):
		''' make dataloader '''
		logging.info('Reading from {}.'.format(self.filepath))

		if self.config['net_type'] == 'generator' and self.config['datatype'] == 'text':
			self.train_loader = torch.utils.data.DataLoader(self.train_dataset,\
					batch_size=self.config['batch_size'],\
					shuffle=False,\
					num_workers=self.config['num_workers'],\
					collate_fn=lambda data: self.generator_collate_fn(data))
			self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset,\
					batch_size=self.config['batch_size'],\
					shuffle=False,\
					num_workers=self.config['num_workers'],\
					collate_fn=lambda data: self.generator_collate_fn(data))
		else:
			raise NotImplementedError('OPTION {} is not supported.'.format(self.config['net_type']))

		return self.train_loader, self.valid_loader

	def image_loader(slef):
		pass


	def load_external_memory(self, path=None):
		'''
		메모리를 읽어들이는 함수
		'''
		if not bool(path): 
			if exists('./retreiver/enwiki_block_record.npy') is False:
				os.system("scp -r odin.nlp.wo.tc:/mnt/odin3/share/Projects/M3/enwiki_documents/block_records.npy ./data/enwiki_block_record.npy")
			block_records_path = "./retreiver/enwiki_block_record.npy"
			self.block_records = np.load(block_records_path, allow_pickle=True)

		elif 'jit' in self.dataset:
			key = os.path.splitext(path)[-1]
			if key in ['.npy']: self.block_records = np.load(path, allow_pickle=True)
			elif key in ['.tsv','.txt']:
				with open(path, 'r') as ifp: data = [bytes(d.strip(),'utf-8') for d in ifp]
				self.block_records = np.array(data, dtype=object)
			else: raise KeyError('No rule for {} filetype.'.format(key))

		elif 'dstc' in self.dataset:
			key = os.path.splitext(path)[-1]
			if key in ['.npy']: self.block_records = np.load(path, allow_pickle=True)
			elif key in ['.tsv','.txt']:
				with open(path, 'r') as ifp: data = [bytes(d.strip().split("\t")[-1],'utf-8') for d in ifp] # 221212 ojy
				#with open(path, 'r') as ifp: data = [bytes(d.strip(),'utf-8') for d in ifp]
				data = np.append(data, [bytes("<extra_id_97>",'utf-8')]) # 221214 ojy - None 추가
				self.block_records = np.array(data, dtype=object)
			else: raise KeyError('No rule for {} filetype.'.format(key))

		else: raise KeyError('External memory is not defined')
		
		return self.block_records

	def generator_collate_fn(self, data):

		sources = []
		targets = []
		source_attns=[]
		target_lengths = []
		origin = []

		for items in data:
			# source
			source = items['source']
			#tokenized_source = tokenizer.encode(source, max_length=max_source_length, padding='max_length', truncation=True)
			source_input = self.tokenizer(source, max_length=self.max_source_length, padding='max_length', truncation=True)
			tokenized_source=source_input['input_ids']
			source_attn=source_input['attention_mask']

			# target
			target = items['target']
			if type(target) != int : # 221213 ojy
				tokenized_target = self.tokenizer.encode(target, max_length=self.max_target_length, padding='max_length', truncation=True)
				target_length = tokenized_target.index(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id in tokenized_target else len(tokenized_target)
			else : 
				tokenized_target = target
				target_length = 1
			#tokenized_target = self.tokenizer.encode(target, max_length=self.max_target_length, padding='max_length', truncation=True)
			#target_length = tokenized_target.index(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id in tokenized_target else len(tokenized_target)

			origin.append(items['data'])

			source_attns.append(source_attn)
			sources.append(tokenized_source)
			targets.append(tokenized_target)
			target_lengths.append(target_length)

		source_tensor = torch.tensor(sources)
		target_tensor = torch.tensor(targets)
		target_lengths = torch.tensor(target_lengths)

		source_tensor = source_tensor.to(torch.int32 if self.config['device']=='mps' else torch.int64)
		target_tensor = target_tensor.to(torch.int32 if self.config['device']=='mps' else torch.int64)
		target_lengths = target_lengths.to(torch.int32 if self.config['device']=='mps' else torch.int64)
		
		result = {
				'source':source_tensor,
				'source_attns':source_attns,
				'target':target_tensor,
				'target_length':target_lengths,
				'data':origin,
				}
		
		return result


	# Dataset을 위한 필수 함수들  
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



