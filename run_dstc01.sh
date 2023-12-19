#!/bin/bash
# Program:
#   Run main.py with specific arguments in root path

set -e # Exit immediately if a pipeline returns non-zeros signal
set -x # Print a trace of simple command

#export PYTHONPATH='src'
log_dir="saved/logdirs/"
#cd src

# non-directory portion of the name of the shell scirpts
file_name=`basename $0`
# ##-> Deletes longest match of (*_) from front of $file_name.
experiment_index=${file_name##*_}
# %%-> Deletes longest match of $s (.*) from back of $experiment_index.
experiment_index=${experiment_index%%.*}
log_file=$log_dir$experiment_index/log.txt

python main.py \
    --experiment_index=$experiment_index \
    --device=cuda \
    	--device_num=0 \
	--data_dir=data/dstc_data/\
		--dataset=dstc.json \
		--datatype=text \
    --model_type=generator \
		--tokenizer_path=t5-small \
	--max_source_length=130 \
	--max_target_length=3 \
	--max_knowledge_length=50 \
    --n_epochs=30 \
    --num_workers=4 \
    --eval_frequency=100 \
    --seed=False \
		--batch_size=16 \
		--searcher_beam_size=15 \
		--update_retriever_emb=True \
		--update_generator_emb=True \
		--external_memory_path=data/dstc_data/knowledge.txt \
		--pretrained_weight_path=t5-small \
    2>&1 | tee $log_file
