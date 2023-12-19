"""Config"""

import argparse

from action import str2bool
# bool type 이 먹지 않는다. 


class Config:
	"""Config
	"""
	def __init__(self):
		"""
		parser: to read all config
		config: save config in pairs like key:value
		"""
		super(Config, self).__init__()
		self.parser = argparse.ArgumentParser(description='M3')
		self.config = dict()
		self.args = None
		self._load_necessary()

	def _load_necessary(self):
		"""load necessary part in config"""
		self._add_default_setting()
		self._add_special_setting()
		self.args = self.parser.parse_args()
		self._load_default_setting()
		self._load_special_setting()

	def _load_default_setting(self):
		"""Load default setting from Parser"""
		self.config['experiment_index'] = self.args.experiment_index
		self.config['is_save_model'] = self.args.is_save_model
		#self.config['cuda'] = self.args.cuda
		self.config['device'] = self.args.device
		self.config['device_num'] = self.args.device_num
		self.config["num_workers"] = self.args.num_workers

		self.config["data_dir"] = self.args.data_dir
		self.config['dataset'] = self.args.dataset
		self.config['max_source_length'] = self.args.max_source_length
		self.config['max_target_length'] = self.args.max_target_length
		self.config['max_knowledge_length'] = self.args.max_knowledge_length
		#self.config['max_sen_len'] = self.args.max_sen_len
		self.config['dropout_keep'] = self.args.dropout_keep
		self.config['datatype'] = self.args.datatype
		self.config['net_type'] = self.args.net_type

		self.config["learning_rate"] = self.args.learning_rate
		self.config["patience"] = self.args.patience
		self.config['n_epochs'] = self.args.n_epochs
		self.config['batch_size'] = self.args.batch_size

		self.config['seed'] = self.args.seed

		self.config["eval_frequency"] = self.args.eval_frequency
		self.config['log_dir'] = self.args.log_dir
		self.config['model_dir'] = self.args.model_dir
		self.config['external_memory_path'] = self.args.external_memory_path
		self.config['save_checkpoint_path'] = self.args.save_checkpoint_path
		self.config['logging_steps'] = self.args.logging_steps
		self.config['save_every_n_steps'] = self.args.save_every_n_steps

	def _load_special_setting(self):
		#self.config["model_type"] = self.args.model_type # retriever, net에서 결정함 
		self.config["vocab_size"] = self.args.vocab_size # 30522,
		self.config["retriever_proj_size"] = self.args.retriever_proj_size # 128,
		self.config["num_hidden_layers"] = self.args.num_hidden_layers # 12,
		self.config["num_attention_heads"] = self.args.num_attention_heads # 12,
		self.config["num_candidates"] = self.args.num_candidates # 8,
		self.config["intermediate_size"] = self.args.intermediate_size # 3072,
		self.config["hidden_act"] = self.args.hidden_act # "gelu_new",
		self.config["hidden_dropout_prob"] = self.args.hidden_dropout_prob # 0.1,
		self.config["attention_probs_dropout_prob"] = self.args.attention_probs_dropout_prob # 0.1,
		self.config["max_position_embeddings"] = self.args.max_position_embeddings # 512,
		self.config["type_vocab_size"] = self.args.type_vocab_size # 2,
		self.config["initializer_range"] = self.args.initializer_range # 0.02,
		self.config["layer_norm_eps"] = self.args.layer_norm_eps # 1e-12,
		self.config["pad_token_id"] = self.args.pad_token_id # 1,
		self.config["bos_token_id"] = self.args.bos_token_id # 0,
		self.config["eos_token_id"] = self.args.eos_token_id # 2,
		self.config["update_retriever_emb"] = self.args.update_retriever_emb # False
		self.config["dropout_rate"] = self.args.dropout_rate # 
		self.config["d_model"] = self.args.d_model # 512 
		self.config["num_layers"] = self.args.num_layers # 8 
		self.config["relative_attention_num_buckets"] = self.args.relative_attention_num_buckets # 32 
		self.config["relative_attention_max_distance"] = self.args.relative_attention_max_distance # 128 
		self.config["d_kv"] = self.args.d_kv # 64 
		self.config["num_heads"] = self.args.num_heads # 6 
		self.config["layer_norm_epsilon"] = self.args.layer_norm_epsilon # 1e-06 
		self.config["is_gated_act"] = self.args.is_gated_act # true
		self.config["d_ff"] = self.args.d_ff # 1024
		self.config["dense_act_fn"] = self.args.dense_act_fn # gelu_new
		self.config["num_decoder_layers"] = self.args.num_decoder_layers # 8
		self.config["update_generator_emb"] = self.args.update_generator_emb # False
		#
		self.config["tokenizer_path"] = self.args.tokenizer_path # 
		self.config["net_projected_size"] = self.args.net_projected_size # 
		self.config["hidden_size"] = self.args.hidden_size # 
		self.config["pretrained_weight_path"] = self.args.pretrained_weight_path # 
		# net config
		self.config["span_hidden_size"] = self.args.span_hidden_size # 256,
		self.config["max_span_width"] = self.args.max_span_width # 10,
		self.config["reader_layer_norm_eps"] = self.args.reader_layer_norm_eps # 1e-3,
		self.config["reader_beam_size"] = self.args.reader_beam_size # 5,
		self.config["reader_seq_len"] = self.args.reader_seq_len # 320,  # 288 + 32
		# retriever config
		self.config["num_block_records"] = self.args.num_block_records # 13353718,   
		self.config["searcher_beam_size"] = self.args.searcher_beam_size # 5000,
		self.config["searcher_seq_len"] = self.args.searcher_seq_len # 64,


	def _add_default_setting(self):
		# need defined each time
		self.parser.add_argument('--experiment_index', default="None", type=str,
								 help="001, 002, ...")
		self.parser.add_argument("--is_save_model", default=True, type=str2bool,
								 help="is save model")
		self.parser.add_argument('--device', default="mps", type=str,
								 help="gpu visible device [mps, cuda] on multiple environment")
		self.parser.add_argument('--device_num', default='0', type=str,
								 help="visible device number")
		self.parser.add_argument("--num_workers", default=0, type=int,
								 help="num_workers of dataloader")

		self.parser.add_argument("--data_dir", default="data", type=str,
								 help="data directory, where to store datasets")
		self.parser.add_argument('--dataset', default="jit", type=str,
								 help="file name")
		self.parser.add_argument('--net_type', default="generator", type=str,
								 help="generator / classifier / QA")
		self.parser.add_argument('--max_source_length', default=32, type=int,
								 help="max_source_length")
		self.parser.add_argument('--max_target_length', default=32, type=int,
								 help="max_target_length")
		self.parser.add_argument('--max_knowledge_length', default=32, type=int,
								 help="max_knowledge_length")
		#self.parser.add_argument('--max_sen_len', default="30", type=int,
		#						 help="30")
		#self.parser.add_argument('--output_size', default="2", type=int,
		#						 help="200")
		self.parser.add_argument('--dropout_keep', default="0.8", type=float,
								 help="0.8")
		self.parser.add_argument('--datatype', default="text", type=str,
								 help="image, text")

		self.parser.add_argument("--learning_rate", default=1e-3, type=float,
								 help="learning rate")
		self.parser.add_argument("--patience", default=1000, type=int,
								 help="patience")
		self.parser.add_argument("--batch_size", default=1000, type=int,
								 help="batch size of each epoch")
		self.parser.add_argument("--n_epochs", default=20, type=int,
								 help="n epochs to train")

		self.parser.add_argument('--seed', default=False, type=str2bool,
								 help="Random seed for pytorch and Numpy")

		self.parser.add_argument("--eval_frequency", default=10, type=int,
								 help="Eval train and test frequency")
		self.parser.add_argument('--log_dir', default="saved/logdirs/",
								 type=str, help='store tensorboard files, \
								 None means not to store')
		self.parser.add_argument('--model_dir', default="saved/models/",
								 type=str, help='store models')
		self.parser.add_argument('--external_memory_path', default="data/jit_data/kb/valid_kb.txt",
								 type=str, help='external memory path')
		self.parser.add_argument('--save_checkpoint_path', default="checkpoint",
								 type=str, help='save checkpoint path')
		self.parser.add_argument('--logging_steps', default=100,
								 type=int, help='logging_steps')
		self.parser.add_argument('--save_every_n_steps', default=20000,
								 type=int, help='save_every_n_steps')

	def _add_special_setting(self):
		self.parser.add_argument("--model_type", default='retriever', type=str,
								 help="retriever, generator")
		self.parser.add_argument("--vocab_size", default=30522, type=int,
								 help="vocabulary size")
		self.parser.add_argument("--retriever_proj_size", default=512, type=int,
								 help="")
		self.parser.add_argument("--num_hidden_layers", default=12, type=int,
								 help="numbers of hidden layers")
		self.parser.add_argument("--num_attention_heads", default=12, type=int,
								 help="num_attention_heads")
		self.parser.add_argument("--num_candidates", default=8, type=int,
								 help="num_candidates")
		self.parser.add_argument("--intermediate_size", default=3072, type=int,
								 help="intermediate_size")
		self.parser.add_argument("--hidden_act", default="gelu_new", type=str,
								 help="gelu_new")
		self.parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
								 help="hidden_dropout_prob")
		self.parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
								 help="attention_probs_dropout_prob")
		self.parser.add_argument("--max_position_embeddings", default=512, type=int,
								 help="max_position_embeddings")
		self.parser.add_argument("--type_vocab_size", default=2, type=int,
								 help="type_vocab_size")
		self.parser.add_argument("--initializer_range", default=0.02, type=float,
								 help="initializer_range")
		self.parser.add_argument("--layer_norm_eps", default=1e-12, type=float,
								 help="layer_norm_eps")
		self.parser.add_argument("--pad_token_id", default=1, type=int,
								 help="pad_token_id")
		self.parser.add_argument("--bos_token_id", default=0, type=int,
								 help="bos_token_id")
		self.parser.add_argument("--eos_token_id", default=2, type=int,
								 help="eos_token_id")
		self.parser.add_argument("--update_retriever_emb", default=False, type=str2bool,
								 help="update_retriever_emb")
		self.parser.add_argument("--dropout_rate", default=0.0, type=float,
								 help="dropout_rate")
		self.parser.add_argument("--d_model", default=512, type=int,
								 help="d_model")
		self.parser.add_argument("--num_layers", default=8, type=int,
								 help="num_layers")
		self.parser.add_argument("--relative_attention_num_buckets", default=32, type=int,
								 help="relative_attention_num_buckets")
		self.parser.add_argument("--relative_attention_max_distance", default=128, type=int,
								 help="relative_attention_max_distance")
		self.parser.add_argument("--d_kv", default=64, type=int,
								 help="d_kv")
		self.parser.add_argument("--num_heads", default=6, type=int,
								 help="num_heads")
		self.parser.add_argument("--layer_norm_epsilon", default=1e-06, type=float,
								 help="layer_norm_epsilon")
		self.parser.add_argument("--is_gated_act", default="true", type=str,
								 help="is_gated_act")
		self.parser.add_argument("--d_ff", default=1024, type=int,
								 help="d_ff")
		self.parser.add_argument("--dense_act_fn", default="gelu_new", type=str,
								 help="dense_act_fn")
		self.parser.add_argument("--num_decoder_layers", default=8, type=int,
								 help="num_decoder_layers")
		self.parser.add_argument("--update_generator_emb", default=False, type=str2bool,
								 help="update_generator_emb")

		self.parser.add_argument("--tokenizer_path", default="KETI-AIR/ke-t5-small", type=str,
								 help="KETI-AIR/ke-t5-small")
		self.parser.add_argument("--net_projected_size", default=512, type=int,
								 help="net projected_size")
		self.parser.add_argument("--hidden_size", default=512, type=int,
								 help="")
		self.parser.add_argument("--pretrained_weight_path", default="KETI-AIR/ke-t5-small", type=str,
								 help="pretrained weight path")
		self.parser.add_argument("--span_hidden_size", default=256, type=int,
								 help="span_hidden_size")
		self.parser.add_argument("--max_span_width", default=10, type=int,
								 help="max_span_width")
		self.parser.add_argument("--reader_layer_norm_eps", default=1e-3, type=float,
								 help="reader_layer_norm_eps")
		self.parser.add_argument("--reader_beam_size", default=5, type=int,
								 help="reader_beam_size")
		self.parser.add_argument("--reader_seq_len", default=320, type=int,
								 help="reader_seq_len")
		self.parser.add_argument("--num_block_records", default=13353718, type=int,
								 help="num_block_records")
		self.parser.add_argument("--searcher_beam_size", default=2, type=int,
								 help="searcher_beam_size")
		self.parser.add_argument("--searcher_seq_len", default=64, type=int,
								 help="searcher_seq_len")
		#self.parser.add_argument("--track_running_stats", default=True,
		#						 type=str2bool, help="track_running_stats for\
		#						 batch normalization in network")

	def print_config(self):
		"""print config
		"""
		print('=' * 20, 'basic setting start', '=' * 20)
		for arg in self.config:
			print('{:20}: {}'.format(arg, self.config[arg]))
		print('=' * 20, 'basic setting end', '=' * 20)

	def get_config(self):
		"""return config"""
		self.print_config()
		return self.config
