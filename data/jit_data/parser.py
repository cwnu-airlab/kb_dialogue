import json, os

## python makingDataset.py -i train.jsonl -o m3_train.jsonl -k hotpotQA_knowledge.txt
## python makingDataset.py -i valid.jsonl -o m3_valid.jsonl -k hotpotQA_knowledge.txt
## python makingDataset.py -i test.jsonl -o m3_test.jsonl -k hotpotQA_knowledge.txt




import sys


total=open(sys.argv[1],'r')

f=open(sys.argv[2],'w')

tot=list()
for tmp in total:
	tmp=tmp.rstrip('\n').split('\t')
	
	k=tmp[0]
	ko=tmp[1]
	#je=tmp[2]

	result = {'keywords':k, 'korean':ko}

	tot.append(result)

	
f.write(json.dumps(tot,ensure_ascii=False,indent=4))


