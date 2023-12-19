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

'''
입력을 최대 길이 만큼 자름
direction: 왼쪽에서부터 자를 것인지 오른쪽에서부터 자를 것인지 결정 
'''
def truncate_input(inputs, tokenizer, max_len, direction='left', special_token=False):
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

