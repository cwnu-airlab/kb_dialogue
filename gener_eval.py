import os, re
import json
import sys
from nltk.metrics import StringMetric
from nltk.translate.meteor_score import meteor_score
import nltk

nltk.download('wordnet')
RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def recall_at_k(label_idx, hyp_knowledge, k=5):
    relevance = match(label_idx, hyp_knowledge)[:k]

    if True in relevance:
        result = 1.0
    else:
        result = 0.0

    return result


def reciprocal_rank(label_idx, hyp_knowledge, k=5):
    relevance = match(label_idx, hyp_knowledge)[:k]

    if True in relevance:
        idx = relevance.index(True)
        result = 1.0/(idx+1)
    else:
        result = 0.0

    return result


def match(label_idx, pred_idx):
    result = []
    for pred in pred_idx:
        matched = False
        if int(pred) == int(label_idx):
            matched = True
        result.append(matched)
    return result

def compute(score_sum):
    if detection_tp + detection_fp > 0.0:
        score_p = score_sum/(detection_tp + detection_fp)
    else:
        score_p = 0.0

    if detection_tp + detection_fn > 0.0:
        score_r = score_sum/(detection_tp + detection_fn)
    else:
        score_r = 0.0

    if score_p + score_r > 0.0:
        score_f = 2*score_p*score_r/(score_p+score_r)
    else:
        score_f = 0.0

    return (score_p, score_r, score_f)

def normalize_text(text):
        result = text.lower()
        result = RE_PUNC.sub(' ', result)
        result = RE_ART.sub(' ', result)
        result = ' '.join(result.split())

        return result

def get_rouge(gold, pred):
    null_list = [' ','',[''],[],None]

    result = {1:list(), 2:list(), 3:list(), 'l':list()}
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        for key in result:
            try:
                if p in null_list or g in null_list: raise ZeroDivisionError
                if 'l' == key:
                    result[key].append(StringMetric().rouge_l([g],p))
                else:
                    result[key].append(StringMetric().rouge_n([g],p,key))
            except ZeroDivisionError as e:
                result[key].append(0.0)

    result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
    #for key in result:
    #    print('rouge-{}: {}'.format(key, result[key]), flush=True)
    return result

def get_bleu(gold, pred):
    null_list = [' ','',[''],[],None]

    result = {1:list(), 2:list(), 3:list(), 4:list()}
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        for key in result:
            try:
                if p in null_list or g in null_list: raise ZeroDivisionError
                result[key].append(StringMetric().bleu_n([g],[p],key))
            except ZeroDivisionError as e:
                result[key].append(0.0)

    result = {key: sum(result[key])/len(result[key]) if len(result[key])>0 else 0 for key in result}
    result['a'] = sum([result[key] for key in result])/len(result)
    #for key in result:
    #    print('BLEU-{}: {}'.format(key, result[key]), flush=True)
    return result

def get_cider(gold, pred):
    null_list = [' ','',[''],[],None]

    result = list()
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        try:
            if p in null_list or g in null_list: raise ZeroDivisionError
            #print(StringMetric().cider([g],[p]))
            #result.append(float(StringMetric().cider([g],[p])[0])) #TODO: 수정함
            result.append(float(StringMetric().cider([g],[p])))
        except ZeroDivisionError as e:
            result.append(0.0)
    result = sum(result)/len(result) if len(result)>0 else 0
    result = {'cider':result}
    #print('CIDER: {}'.format(result['cider']), flush=True)
    return result

def get_meteor(gold, pred):
    #from nltk import StringMetric()
    null_list = [' ','',[''],[],None]

    result = list()
    for i in range(len(gold)):
        p = pred[i]
        g = gold[i]
        try:
            if p in null_list or g in null_list: raise ZeroDivisionError
            #result.append(float(StringMetric().meteor([g],p)))
            result.append(float(meteor_score([g],p)))
        except (IndexError, ZeroDivisionError) as e:
            result.append(0.0)
    result = sum(result)/len(result) if len(result)>0 else 0
    result = {'meteor':result}
    #print('METEOR: {}'.format(result['meteor']), flush=True)
    return result


#file_path = sys.argv[1]
file_path = 'checkpoint/dstc9_v29_recent/trained_model/prediction_test.jsonl'
none_idx = 12039
with open(file_path, 'r') as f:
    prediction_results = [json.loads(line) for line in f]

print("-----"*10)
print(f"{file_path}", end='\n\n')
print("detection 성능")
detection_tp, detection_fp, detection_fn, detection_tn = 0, 0, 0, 0
none_idx = 12039 if "test" in file_path else 2900
if "fit_knowledge" in file_path:
    none_idx = 2900

for result_dict in prediction_results:
    label_idx = result_dict["label_idx"]
    pred_idx = result_dict["pred_idx"]

    if label_idx == none_idx:
        if none_idx != pred_idx[0]:
            detection_fp += 1
        else:
            detection_tn += 1
    else:
        if none_idx != pred_idx[0]:
            detection_tp += 1
        else:
            detection_fn += 1

score_p, score_r, score_f = compute(detection_tp)

print(f"tp : {detection_tp}, fp : {detection_fp}, fn : {detection_fn}, tn : {detection_tn}")
print(f"score_p : {score_p}\nscore_r : {score_r}\nscore_f : {score_f}\n")

print("selection 성능")

recall1, recall5, recall15, mrr5 = 0, 0, 0, 0
for result_dict in prediction_results:
    label_idx = result_dict["label_idx"]
    pred_idx = result_dict["pred_idx"]

    if label_idx != none_idx and pred_idx[0] != none_idx:
        recall1 += recall_at_k(label_idx, pred_idx, k=1)
        recall5 += recall_at_k(label_idx, pred_idx, k=5)
        mrr5 += reciprocal_rank(label_idx, pred_idx, k=5)

recall1_p, recall1_r, recall1_f = compute(recall1)
recall5_p, recall5_r, recall5_f = compute(recall5)
mrr5_p, mrr5_r, mrr5_f = compute(mrr5)

print(f"r@1 : {recall1_f}\nr@5 : {recall5_f}\nmrr@5 : {mrr5_f}")

print("-----"*10)

prediction_results = [result for result in prediction_results \
    if result['label_idx'] != none_idx and result['pred_idx'][0] != none_idx]

pred = [normalize_text(d['pred_response']) for d in data]
gold = [normalize_text(d['label_response']) for d in data]

assert len(pred) == len(gold)
print('data size: {}\n'.format(len(gold)))

bleu_result = get_bleu(gold, pred)
rouge_result = get_rouge(gold, pred)
cider_result = get_cider(gold, pred)
meteor_result = get_meteor(gold, pred)

for key in bleu_result:
    print('BLEU-{}: {}'.format(key, compute(bleu_result[key])[2]), flush=True)

for key in rouge_result:
    print('rouge-{}: {}'.format(key, compute(rouge_result[key])[2]), flush=True)

print('CIDER: {}'.format(compute(cider_result['cider'])[2]), flush=True)
print('METEOR: {}'.format(compute(cider_result['meteor'])[2]), flush=True)