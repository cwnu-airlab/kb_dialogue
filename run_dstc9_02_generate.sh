# train
# python run.py \
torchrun --nproc_per_node=4 run.py \
	--version=2 \
	--gpu_id=0 \
	--model_type="generator" \
	--mode="train" \
	--gradient_accumulation_steps=1 \
	--max_grad_norm=1.0 \
	--model_all_save=False \
	--history_max_utterances=-1 \
	--history_max_tokens=128 \
	--data_language="eg" \
	--retriever_prediction_result_path="checkpoint/dstc9_v1_recent/trained_model/prediction_test.jsonl" \
	--per_gpu_batch_size=64 \
	--predict_batch_size=64 \
	--shuffle=True \
	--learning_rate=1e-05 \
	--pretrained_model_path="openai-community/gpt2" \
	#--pretrained_model_path="skt/kogpt2-base-v2" \
	# --pretrained_model_path="skt/ko-gpt-trinity-1.2B-v0.5" \

## predict
#python run.py \
#	--version=2 \
#	--gpu_id=0 \
#	--model_type="generator" \
#	--mode="predict" \
#	--gradient_accumulation_steps=1 \
#	--max_grad_norm=1.0 \
#	--model_all_save=False \
#	--history_max_utterances=-1 \
#	--history_max_tokens=128 \
#	--data_language="eg" \
#	--retriever_prediction_result_path="checkpoint/dstc9_v1_recent/trained_model/prediction_test.jsonl" \
#	--per_gpu_batch_size=64 \
#	--predict_batch_size=64 \
#	--shuffle=True \
#	--learning_rate=1e-05 \
#	--predict_model_path="checkpoint/dstc9_v2_generate/trained_model" \
