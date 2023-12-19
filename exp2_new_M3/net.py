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
import os, logging
import torch, math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
from packaging import version
import sys
from nets.T5 import *
#from transformers import RealmConfig, T5Config
import copy
#xxx
#torch.set_printoptions(threshold=10000)



@dataclass
class M3T5GenerateModelOutput(ModelOutput):
	retriever_relevance_score: torch.FloatTensor = None
	logits: torch.FloatTensor = None
	last_hidden_state: torch.FloatTensor = None
	past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
	retrieved_block_ids : torch.Tensor = None
	retrieved_blocks:torch.Tensor=None
	labels:torch.FloatTensor=None



class M3T5Generator(M3PreTrainedModel):
	def __init__(self, config, tokenizer=None):
		super().__init__(config)
		self.config					= config
		self.tokenizer			= tokenizer
		self.m3_t5_encoder	= transformers.T5ForConditionalGeneration(self.config).get_encoder()
		self.m3_t5_decoder	= transformers.T5ForConditionalGeneration(self.config).get_decoder()
		self.lm_head				= transformers.T5ForConditionalGeneration(self.config).lm_head

	def forward(self,
		input_ids=None,															# 
		labels=None,																#
		retriever_outputs=None,											#
		#kb_answers = None,
		decoder_input_ids=None,
		decoder_inputs_embeds=None,
		encoder_outputs=None,
		past_key_values=None,
		attention_mask=None,
		decoder_attention_mask=None,
		head_mask=None,
		decoder_head_mask=None,
		inputs_embeds=None,
		use_cache=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		pad_id=0,
		#loss_func=torch.nn.CrossEntropyLoss(ignore_index=-100,reduce=False),
		#b_index=None,
		batch_size=None,														#
		mode=None,
		block_records=None,
		opt=None,																		#
		**kwargs
		):

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		concat_inputs = retriever_outputs.concat_inputs.to(self.config.device)
		concat_inputs.input_ids = concat_inputs.input_ids.to(torch.int32 if self.config.device=='mps' else torch.int64)
		concat_inputs.attention_mask = concat_inputs.attention_mask.to(torch.int32 if self.config.device=='mps' else torch.int64)
		#print('concat_inputs.input_ids : ', concat_inputs.input_ids.dtype)
		#print('concat_inputs.attention_mask : ', concat_inputs.attention_mask.dtype)

		# t5_encoder 	
		encoder_outputs = self.m3_t5_encoder(
			input_ids=concat_inputs.input_ids, # [0 : self.config.reader_beam_size],
			attention_mask=concat_inputs.attention_mask, #[0 : self.config.reader_beam_size],
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		# 학습 시에 teacher forcing
		if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = self.m3_t5_decoder._shift_right(labels)

		# [batch_size,seq_len,hs]
		hidden_states = encoder_outputs.last_hidden_state 
		#print(hidden_states.shape, hidden_states, hidden_states.dtype, flush=True)

		decoder_outputs = self.m3_t5_decoder(
			input_ids							=	decoder_input_ids,
			#attention_mask				=	decoder_attention_mask,
			#inputs_embeds					=	decoder_inputs_embeds,
			#past_key_values				=	past_key_values,
			encoder_hidden_states	=	hidden_states,
			#encoder_attention_mask=	attention_mask,
			#head_mask							=	decoder_head_mask,
			#use_cache							=	use_cache,
			#output_attentions			=	output_attentions,
			#output_hidden_states	=	output_hidden_states,
			#return_dict						=	return_dict,
		)
		#print("@@", decoder_outputs[0].shape, decoder_outputs, flush=True)
		#[batch_size, seq_len, hs]
		sequence_output = decoder_outputs[0]		# lhs: last hidden states
		#print("##", sequence_output.shape, sequence_output, flush=True)
		sequence_output = sequence_output * (self.config.d_model ** -0.5)			# ????

		#[batch_size,seq_len,voc_size]
		# [0, 1, 2]
		lm_logits = self.lm_head(sequence_output)
		lm_logits = lm_logits.permute(0,2,1)
		#print("\n\n!!!", labels.size(), labels.dtype, flush=True)
		#print("@@@", lm_logits.size(), lm_logits.dtype, flush=True)

		
		return M3T5GenerateModelOutput(
			#retriever_relevance_score = encoder_outputs.retriever_logits, 
			logits=lm_logits,
			#past_key_values=decoder_outputs.past_key_values,
			#decoder_hidden_states=decoder_outputs.hidden_states,
			#decoder_attentions=decoder_outputs.attentions,
			#cross_attentions=decoder_outputs.cross_attentions,
			#encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			#encoder_hidden_states=encoder_outputs.hidden_states,
			#encoder_attentions=encoder_outputs.attentions,
			#retrieved_blocks=encoder_outputs.retrieved_blocks,
			labels=labels,
		)
			
		

	@torch.no_grad()
	def generate(
		self,
		input_ids=None,
		max_length=None,
		min_length=None,
		do_sample=None,
		early_stopping=None,
		num_beams=None,
		temperature=None,
		top_k=None,
		top_p=None,
		repetition_penalty=None,
		bad_words_ids=None,
		bos_token_id=None,
		pad_token_id=0,
		eos_token_id=1,
		length_penalty=None,
		no_repeat_ngram_size=None,
		encoder_no_repeat_ngram_size=None,
		num_return_sequences=None,
		max_time=None,
		decoder_start_token_id=None,
		use_cache=None,
		num_beam_groups=None,
		diversity_penalty=None,
		prefix_allowed_tokens_fn=None,
		output_attentions=None,
		output_hidden_states=None,
		output_scores=None,
		return_dict_in_generate=None,
		forced_bos_token_id=None,
		forced_eos_token_id=None,
		remove_invalid_values=None,
		b_index=None,
		batch_size=None,
		mode=None,
		attention_mask=None,
		**model_kwargs,
		):

		if max_length==None: max_length = self.config.max_length
		decoder_input_ids = torch.tensor([[pad_token_id]]*input_ids.shape[0]).to(self.device)

		encoder_outputs = None
		for i in range(max_length):
			model_output = self.forward(input_ids=input_ids, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs,b_index=b_index,batch_size=batch_size,mode=mode,attention_mask=attention_mask)

			logits = model_output.logits.detach()
			predict = torch.argmax(logits, dim=-1)[:,-1:]

			decoder_input_ids = torch.cat((decoder_input_ids, predict),1)
		ret=dict()
		ret['predict']= decoder_input_ids
		ret['picked']=model_output.retrieved_blocks
		predict = decoder_input_ids
		#return predict
		return ret


