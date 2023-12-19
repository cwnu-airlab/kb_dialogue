import pandas as pd


import transformers
import sys

text=open(sys.argv[1],'r')

tokenizer = transformers.T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-base')
total = []

for tmp in text:
	#je=tmp.split('\t')[2]
	#ko=tmp.split('\t')[1]
	#tmp=tmp.split('\t')[0]
	tmp=tmp.rstrip('\n')
	enc=tokenizer.encode(tmp)
	'''
	a=2

	if a in tokenizer.encode(je):
		continue

	if len(enc)>4 and len(enc)<10:
		print(tmp,end='\t')
		print(ko,end='\t')
		print(je,end='')
	'''
	total.append(len(enc))

	
s=pd.Series(total)
print(s.describe())




print(s.quantile([.8,.9,.95,.99]))
