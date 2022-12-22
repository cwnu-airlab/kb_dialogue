
def get_cls(pred, gold):
	import sklearn.metrics as met

	targets = sorted(list(set(gold)))
	macro_precision = met.precision_score(gold, pred, average = 'macro', zero_division=0)
	macro_recall = met.recall_score(gold, pred, average = 'macro', zero_division=0)
	macro_f1 = met.f1_score(gold, pred, average = 'macro', zero_division=0)
	micro_f1 = met.f1_score(gold, pred, average = 'micro', zero_division=0)
	weighted_f1 = met.f1_score(gold, pred, average = 'weighted', zero_division=0)
	accuracy = met.accuracy_score(gold, pred)

	precisions = met.precision_score(gold, pred, average = None, labels = targets, zero_division=0)
	recalls = met.recall_score(gold, pred, average = None, labels = targets, zero_division=0)
	f_measures = met.f1_score(gold, pred, average = None, labels = targets, zero_division=0)

	print("class\tprecision\trecall\tf1-score")
	for i, target in enumerate(targets):
		print("%s\t%0.3f\t%0.3f\t%0.3f"%(target,precisions[i],recalls[i],f_measures[i]))
	print()
	print("%s\t%0.3f\t%0.3f\t%0.3f"%("MACRO",macro_precision,macro_recall,macro_f1))
	print("Accuracy\t%0.3f"%accuracy)
	print()
	print("%s\t%0.3f\t%0.3f\t%0.3f"%("MEAN",sum(precisions)/len(precisions),sum(recalls)/len(recalls),sum(f_measures)/len(f_measures)))
	print()

	result = {'accuracy':accuracy, 'macr-f1':macro_f1, 'micro-f1':micro_f1, 'weighted_f1':weighted_f1}
	return result

def get_rouge(gold, pred):
	from nltk import eval
	null_list = [' ','',[''],[],None]

	result = {1:list(), 2:list(), 3:list(), 'l':list()}
	for i in range(len(gold)):
		p = pred[i]
		g = gold[i]
		for key in result:
			try:
				if p in null_list or g in null_list: raise ZeroDivisionError
				if 'l' == key:
					result[key].append(eval.rouge_l([g],p))
				else:
					result[key].append(eval.rouge_n([g],p,key))
			except ZeroDivisionError as e:
				result[key].append(0.0)

	result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
	for key in result:
		print('rouge-{}: {}'.format(key, result[key]), flush=True)
	return result

def get_bleu(gold, pred):
	from nltk import eval
	null_list = [' ','',[''],[],None]

	result = {1:list(), 2:list(), 3:list(), 4:list()}
	for i in range(len(gold)):
		p = pred[i]
		g = gold[i]
		for key in result:
			try:
				if p in null_list or g in null_list: raise ZeroDivisionError
				result[key].append(eval.bleu_n([g],[p],key))
			except ZeroDivisionError as e:
				result[key].append(0.0)

	result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
	result['a'] = sum([result[key] for key in result])/len(result)
	for key in result:
		print('BLEU-{}: {}'.format(key, result[key]), flush=True)
	return result

def get_cider(gold, pred):
	from nltk import eval
	null_list = [' ','',[''],[],None]

	result = list()
	for i in range(len(gold)):
		p = pred[i]
		g = gold[i]
		try:
			if p in null_list or g in null_list: raise ZeroDivisionError
			result.append(float(eval.cider([g],[p])[0]))
		except ZeroDivisionError as e:
			result.append(0.0)
	result = sum(result)/len(result) if len(result)>0 else 0
	result = {'cider':result}
	print('CIDER: {}'.format(result['cider']), flush=True)
	return result
	
def _match(gold_knowledge, pred_knowledge):
	result = []
	for pred in pred_knowledge:
		matched = False
		if gold_knowledge == pred : matched =True
		result.append(matched)
	return result

def get_mrr(gold_knowledge, pred_knowledge, topk=5):
	relevance = _match(gold_knowledge, pred_knowledge)[:topk]
	if True in relevance:
		idx = relevance.index(True)
		result = 1.0/(idx+1)
	else: result = 0.0
	return result 

def get_recall_at_k(gold_knowledge, pred_knowledge, topk=5):
	relevance = _match(gold_knowledge, pred_knowledge)[:topk]
	if True in relevance: result = 1.0
	else: result = 0.0
	
	return result

def knoweldge_select(gold_knowledge, pred_knowledge):
	total_mrr, total_r1, total_r5 =0,0,0
	for gidx, gold in enumerate(gold_knowledge):
		gold, pred = gold, pred_knowledge[gidx]
		total_mrr += get_mrr(gold, pred, topk=5)
		total_r1 += get_recall_at_k(gold, pred, topk=1)
		total_r5 += get_recall_at_k(gold, pred, topk=5)

	print (f"MRR@5 : {total_mrr/len(gold_knowledge)}")	
	print (f"R@1 : {total_r1/len(gold_knowledge)}")	
	print (f"R@5 : {total_r5/len(gold_knowledge)}\n")	

if __name__=='__main__':
	import os
	import json
	import sys, tqdm
	
	filename = sys.argv[1]
	ext = os.path.splitext(filename)[-1]
	task = sys.argv[2]
	if ext == '.txt':
		with open(filename, 'r') as f:
			data = [d.strip() for d in f.readlines()]
		columns = data[0]
		
		pred = [d.replace('PRED:','').strip() for d in data if 'PRED:' in d]
		gold = [d.replace('GOLD:','').strip() for d in data if 'GOLD:' in d]
		srce = [d.replace('SRCE:','').strip() for d in data if 'SRCE:' in d]
	elif ext == '.jsonl':
		with open(filename, 'r') as f:
			data = [json.loads(d) for d in f]
		pred = [d['output']['pred'] for d in data]
		gold = [d['output']['gold'] for d in data]
		srce = [d['output']['srce'] for d in data]
		gold_kb = [d['output']['gold_kb'] for d in data]
		pred_kb = [d['output']['pred_kb'] for d in data]

	assert len(pred) == len(gold) and len(gold) == len(srce) 
	print('data size: {}\n'.format(len(gold)))

	if 'gener' in task:
		knoweldge_select(gold_kb, pred_kb)
		print()
		get_rouge(gold, pred)
		print()
		get_bleu(gold, pred)

	elif 'class' in task:
		get_cls(gold, pred)

