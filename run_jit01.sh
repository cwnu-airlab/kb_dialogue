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
log_file=$log_dir$experiment_index.log.txt

python3 main.py \
	--experiment_index=$experiment_index \
	--device=cuda \
	--data_dir=data/jit_data/run/\
	--dataset=jit_real.json \
	--datatype=text \
	--model_type=generator \
	--tokenizer_path=KETI-AIR/ke-t5-small\
	--n_epochs=30 \
	--num_workers=0 \
	--eval_frequency=100 \
	--seed=False \
	--batch_size=64\
	--searcher_beam_size=5 \
	--update_retriever_emb=True \
	--update_generator_emb=True \
	--external_memory_path=data/jit_data/kb/valid_kb.txt \
	--pretrained_weight_path=valid_t5-generator_lr_0.0001_pat_7_epoch_0000051_valLoss_0.1840_valAcc_0.9731\
    2>&1 | tee $log_file
