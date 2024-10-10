# english
#python run.py \
torchrun --nproc_per_node=4 run.py \
	--version=0 \
	--gpu_id=0 \
	--model_type="ict" \
	--mode="train" \
	--gradient_accumulation_steps=1 \
	--max_grad_norm=1.0 \
	--model_all_save=True \
	--history_max_utterances=-1 \
	--history_max_tokens=128 \
	--data_language="eg" \
	--per_gpu_batch_size=128 \
	--predict_batch_size=128 \
	--shuffle=True \
	--vector_similarity='inner_product' \
	--use_proj_layer=False \
	--pretrained_model_path="google-bert/bert-base-uncased" \
	--learning_rate=1e-05 \

## korean
## python run.py \
#torchrun --nproc_per_node=4 run.py \
#	--version=0 \
#	--gpu_id=0 \
#	--model_type="ict" \
#	--mode="train" \
#	--gradient_accumulation_steps=1 \
#	--max_grad_norm=1.0 \
#	--model_all_save=True \
#	--history_max_utterances=-1 \
#	--history_max_tokens=128 \
#	--data_language="ko" \
#	--per_gpu_batch_size=128 \
#	--predict_batch_size=128 \
#	--shuffle=True \
#	--vector_similarity='inner_product' \
#	--use_proj_layer=False \
#	--pretrained_model_path="klue/bert-base" \
#	--learning_rate=1e-05 \
