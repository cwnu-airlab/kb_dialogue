# train
torchrun --nproc_per_node=4 run.py \
	--version=1 \
	--gpu_id=0 \
	--model_type="retriever" \
	--mode="train" \
	--refresh_frequency=1 \
	--gradient_accumulation_steps=1 \
	--max_grad_norm=1.0 \
	--model_all_save=False \
	--input_type='recent' \
	--history_max_utterances=2 \
	--history_max_tokens=128 \
	--data_language="eg" \
	--per_gpu_batch_size=64 \
	--predict_batch_size=128 \
	--shuffle=True \
	--num_candidates=10 \
	--vector_similarity='inner_product' \
	--use_proj_layer=False \
	--use_passage_body=True \
	--learning_rate=1e-05 \
	--pretrained_model_path="checkpoint/dstc9_v0_ict/trained_model" \
	#--pretrained_model_path="google-bert/bert-base-uncased" \


# # predict
# python run.py \
# 	--version=1 \
# 	--gpu_id=0 \
# 	--model_type="retriever" \
# 	--mode="predict" \
# 	--refresh_frequency=1 \
# 	--gradient_accumulation_steps=1 \
# 	--max_grad_norm=1.0 \
# 	--model_all_save=False \
# 	--input_type='recent' \
# 	--history_max_utterances=2 \
# 	--history_max_tokens=128 \
# 	--data_language="eg" \
# 	--per_gpu_batch_size=64 \
# 	--predict_batch_size=128 \
# 	--shuffle=True \
# 	--num_candidates=10 \
# 	--vector_similarity='inner_product' \
# 	--use_proj_layer=False \
# 	--use_passage_body=True \
# 	--learning_rate=1e-05 \
# 	--predict_model_path="checkpoint/dstc9_v1_recent/trained_model" \
