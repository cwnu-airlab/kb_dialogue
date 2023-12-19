'''
T5와 interface
'''
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
#xxx
#torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=10000)


class RealmConfig(PretrainedConfig):
	
	model_type = "realm"

	def __init__(
		self,
		config,
		**kwargs
	):
		super().__init__(pad_token_id=config["pad_token_id"], \
				bos_token_id=config["bos_token_id"], \
				eos_token_id=config["eos_token_id"], **kwargs)

		# Common config
		self.vocab_size = config["vocab_size"]
		self.max_position_embeddings = config["max_position_embeddings"]
		self.hidden_size = config["hidden_size"]
		#self.retriever_proj_size = config["retriever_proj_size"]
		self.num_hidden_layers = config["num_hidden_layers"]
		self.num_attention_heads = config["num_attention_heads"]
		self.num_candidates = config["num_candidates"]
		self.intermediate_size = config["intermediate_size"]
		self.hidden_act = config["hidden_act"]
		self.hidden_dropout_prob = config["hidden_dropout_prob"]
		self.attention_probs_dropout_prob = config["attention_probs_dropout_prob"]
		self.initializer_range = config["initializer_range"]
		self.type_vocab_size = config["type_vocab_size"]
		self.layer_norm_eps = config["layer_norm_eps"]
		self.update_emb = None # retriever, generator에서 각각 결정 

		# Reader config
		#self.span_hidden_size = None
		#self.max_span_width = None
		#self.reader_layer_norm_eps = None
		#self.reader_beam_size = None
		#self.reader_seq_len = None

		# Retrieval config
		#self.searcher_beam_size = config["searcher_beam_size"]
		#self.searcher_seq_len = config["searcher_seq_len"]
		#self.projected_size = config['projected_size']
		#self.num_block_records = None
		#self.max_knowledge_length = config["max_knowledge_length"]
		#self.max_source_length = config["max_source_length"]


	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

class M3ScorerProjection(nn.Module):
	''' Score projection for retriever '''
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.projected_size)
		self.LayerNorm = nn.LayerNorm(config.projected_size, eps=config.layer_norm_eps)

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.LayerNorm(hidden_states)
		return hidden_states

class M3PreTrainedModel(PreTrainedModel):
	"""
	An abstract class to handle weights initialization and a simple interface 
	for downloading and loading pretrained models.
	"""

	def _init_weights(self, module):
		"""Initialize the weights"""
		if isinstance(module, nn.Linear):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def _flatten_inputs(self, *inputs):
		"""Flatten inputs' shape to (-1, input_shape[-1])"""
		flattened_inputs = []
		for tensor in inputs:
			if tensor is None:
				flattened_inputs.append(None)
			else:
				input_shape = tensor.shape
				if len(input_shape) > 2:
					tensor = tensor.view((-1, input_shape[-1]))
				flattened_inputs.append(tensor)
		return flattened_inputs



