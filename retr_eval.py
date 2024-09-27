import pdb, json
from M3.data_module import KnowledgeReader

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


def print_passage(idx):
    print(passages[idx])


file_path = 'checkpoint/dstc9_v29_recent/trained_model/prediction_test.jsonl'
with open(file_path, 'r') as fp:
    prediction_results = [json.loads(line) for line in fp]

knowledge_reader = KnowledgeReader("data/data_eval")
passages = list(knowledge_reader.passages.values())

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




