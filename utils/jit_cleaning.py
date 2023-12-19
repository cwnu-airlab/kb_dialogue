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

def jit_cleaning(sentence):

	sent = sentence.strip()

	#sent = re.sub('\[[^\]]*\]','',sent) ## 대괄호 제거
	#sent = re.sub('\([^\)]*\)','',sent) ## 소괄호 제거
	#sent = re.sub('[^ㅏ-ㅣㄱ-ㅎ가-힣0-9a-zA-Z\.%, ]',' ', sent) ## 특수문자 모두 제거
	sent = re.sub('  *',' ',sent).strip() ## 다중 공백 제거

	return sent

