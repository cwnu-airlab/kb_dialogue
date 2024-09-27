import os
import sys
import random
import logging
import signal
import shutil
import traceback
from argparse import (ArgumentParser, Namespace)

import torch
import transformers
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel

from M3.agent import Agent
#from M3.data_module import ICTDataset, SelectionDataset, GenerationDataset, KnowledgeReader, SPECIAL_TOKENS
from M3.upgrade_data_module import ICTDataset, SelectionDataset, GenerationDataset, KnowledgeReader, SPECIAL_TOKENS
from M3.model import M3ICT, M3Retriever


logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    raise argparse.ArgumentTypeError(f'input value : {v}\nBoolean value required.')


def str2lower(v):
    return v.lower()


def set_seed(seed: int, deterministic: bool=False):
    if not (isinstance(seed, int) and seed >= 0):
        raise argparse.ArgumentTypeError("Seed requires a positive integer value.")
    if not isinstance(deterministic, bool):
        raise TypeError("Determinstic requires a boolean value")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = ArgumentParser()
    # Main parameters
    parser.add_argument("--task_name", type=str, default="dstc9")
    parser.add_argument("--version", type=int)
    parser.add_argument("--gpu_id", type=int, default=0, help="학습하거나 추론할 때 사용할 GPU를 지정(torchrun인 경우 해당 변수는 무시)")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--model_type", type=str2lower, choices=['ict', 'retriever', 'generator'])
    parser.add_argument("--mode", type=str2lower, choices=['train', 'predict'])
    # Agent parameters
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5, help="학습 조기 종료 patience")
    parser.add_argument("--refresh_frequency", type=int, default=1, help="KB 임베딩 업데이트 주기(REALM 논문 참고)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Gradient Clipping")
    parser.add_argument('--model_all_save', type=str2bool, default=False, help="모든 모델 저장 여부")
    # Dataset parameters
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--input_type", type=str, default="history")
    parser.add_argument("--history_max_utterances", type=int, default=2, help="대화 내역 최대 발화 수(input_type이 recent 혹은 entire인 경우에만 사용)")
    parser.add_argument("--history_max_tokens", type=int, default=128, help="대화 내역 최대 토큰 길이(input_type이 history인 경우만 사용)")
    parser.add_argument("--data_language", type=str2lower, default="eg", choices=['eg', 'ko'])
    parser.add_argument("--retriever_prediction_result_path", type=str, default="", help="Retriever가 검색한 top k 문서를 사용하여 Generator를 추론하기 위한 Retriever 추론 결과 파일 경로")
    # DataLoader parameters
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="학습 batch size")
    parser.add_argument("--predict_batch_size", type=int, default=128, help="추론 batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle", type=str2bool, default=True)
    # Model parameters
    parser.add_argument("--num_candidates", type=int, default=10, help="학습 시 사용할 후보 문서 개수")
    parser.add_argument("--vector_similarity", type=str2lower, default="inner_product", choices=['inner_product', 'cosine'])
    parser.add_argument("--use_proj_layer", type=str2bool, default=False, help="임베딩 차원을 줄이는 layer 사용 여부")
    parser.add_argument("--proj_size", type=int, default=128)
    parser.add_argument("--use_passage_body", type=str2bool, default=True, help="외부 지식 passage의 body 사용 여부")
    parser.add_argument("--pretrained_model_path", type=str, help="학습 시 사용할 사전학습된 모델")
    parser.add_argument("--predict_model_path", type=str, default='', help="추론할 모델 path")
    # Generation strategy parameters
    # https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/text_generation#transformers.GenerationConfig
    parser.add_argument("--max_new_tokens", type=int, default=48, help="The maximum numbers of tokens to generate")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="The value used to modulate the next token probabilities")
    parser.add_argument("--top_k", type=int, default=0, help="The number of highest probability vocabulary tokens to keep for top-k-filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation")
    # Optimizer parameters
    parser.add_argument("--learning_rate", type=float, default=1e-05)

    # Create arguments
    args = parser.parse_args()
    if args.model_type == "ict":
        args.input_type = "ict"
    if args.model_type == "generator":
        args.input_type = "generate"
    args.checkpoint = f"checkpoint/{args.task_name}_v{args.version}_{args.input_type}/"
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = (args.local_rank != -1)

    # Set logging config
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )

    # Verify argumets
    if not(isinstance(args.gpu_id, int) and args.gpu_id in range(torch.cuda.device_count())):
        raise argparse.ArgumentTypeError(f"Invalid gpu_id")
    if args.mode == "predict" and args.local_rank != -1:
        raise Exception("predict mode only support single GPU or CPU")
    if args.data_language == "ko":
        raise Exception('데이터 셋 수정 필요')

    # Set CUDA, GPU & distributed training
    if not args.distributed:
        device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    else:
        device = f"cuda:{args.local_rank}"
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.device = device

    # Check checkpoint directory
    # local_rank in [-1, 0] => 메인 프로세스만 동작
    if args.mode == "train":
        if args.local_rank in [-1, 0]:
            if os.path.exists(args.checkpoint) and bool(os.listdir(args.checkpoint)):
                logger.warn(f"\"{args.checkpoint}\" already exists. Do you want overwrite? (y/n)")
                answer = str2bool(input(">> "))
                if not answer:
                    os.kill(os.getpid(), signal.SIGTERM)
                else:
                    shutil.rmtree(args.checkpoint)
                    os.makedirs(args.checkpoint)
            else:
                os.makedirs(args.checkpoint, exist_ok=True)
    else:
        if args.model_type == "generator" and \
                (not bool(args.retriever_prediction_result_path) or not os.path.exists(args.retriever_prediction_result_path)):
            raise FileNotFoundError(f"Generator를 추론하기 위해서는 Retriever의 추론 결과가 필요합니다.")
        if "checkpoint" in args.predict_model_path and \
                str(args.version) in args.predict_model_path and args.input_type in args.predict_model_path:
            if not os.path.exists(args.predict_model_path):
                raise FileNotFoundError(f"Model file does not exist.")
        else:
            predict_model_path = os.path.join(args.checkpoint, "trained_model")
            if os.path.exists(predict_model_path):
                args.predict_model_path = predict_model_path
            else:
                raise FileNotFoundError(f"Model file does not exist.")

    # 메인 프로세스가 입력을 받는동안 다른 프로세스들은 대기
    if args.distributed:
        torch.distributed.barrier()

    logger.info('COMMAND: python '+' '.join(sys.argv))
    logger.info(f"Arguments : {args}")

    # Set seed
    set_seed(args.seed)

    # Load tokenizer
    model_path = args.pretrained_model_path if args.mode == "train" else args.predict_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'right' if args.model_type != "generator" else 'left'
    tokenizer.truncation_side = 'left'
    if args.mode == "train":
        tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # Load Dataset
    fit_knowledge_reader = KnowledgeReader(os.path.join(args.data_dir, 'data_fit'))
    eval_knowledge_reader = KnowledgeReader(os.path.join(args.data_dir, 'data_eval'))

    if args.model_type == "ict":
        dataset = ICTDataset
    elif args.model_type == "retriever":
        dataset = SelectionDataset
    else:
        dataset = GenerationDataset
    train_dataset = dataset(
        args,
        os.path.join(args.data_dir, 'data_fit'),
        'train',
        tokenizer,
        fit_knowledge_reader
    )
    valid_dataset = dataset(
        args,
        os.path.join(args.data_dir, 'data_fit'),
        'valid',
        tokenizer,
        fit_knowledge_reader,
    )
    test_dataset = dataset(
        args,
        os.path.join(args.data_dir, 'data_eval'),
        'test',
        tokenizer,
        eval_knowledge_reader,
    )

    # Set dataloader
    train_dataloader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset, shuffle=args.shuffle) \
            if args.distributed else None,
        shuffle=None if args.distributed else args.shuffle,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=DistributedSampler(valid_dataset, shuffle=args.shuffle) \
            if args.distributed else None,
        shuffle=None if args.distributed else args.shuffle,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        collate_fn=valid_dataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.predict_batch_size,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn
    )

    # Load model
    config = AutoConfig.from_pretrained(model_path)
    if args.model_type == "ict":
        addition_config = {
            "vector_similarity": args.vector_similarity,
            "use_proj_layer": args.use_proj_layer,
            "proj_size": args.proj_size,
            "device": args.device
        }
    elif args.model_type == "retriever":
        addition_config = {
            "num_candidates": args.num_candidates,
            "vector_similarity": args.vector_similarity,
            "use_proj_layer": args.use_proj_layer,
            "proj_size": args.proj_size,
            "use_passage_body": args.use_passage_body,
            "device": args.device
        }
    else:
        addition_config = dict()
    config.update(addition_config)

    fit_passages = list(fit_knowledge_reader.passages.values())
    eval_passages = list(eval_knowledge_reader.passages.values())

    if args.model_type == "ict":
        model = M3ICT.from_pretrained(
            model_path,
            config=config
        )
    elif args.model_type == "retriever":
        model = M3Retriever.from_pretrained(
            model_path,
            config=config,
            fit_passages=fit_passages,
            eval_passages=eval_passages,
            tokenizer=tokenizer,
        )
    else:
        model = GPT2LMHeadModel.from_pretrained(
            model_path,
            config=config
        )
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    # Load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate) \
        if args.mode == "train" else None

    agent = Agent(
        args,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        tokenizer,
        model,
        optimizer
    )
    if args.mode == "train":
        agent.fit()
    else:
        agent.predict()


if __name__ == "__main__":
    TOTAL_DEVICE_CNT = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(TOTAL_DEVICE_CNT)])
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    try:
        main()
    except Exception as e:
        local_rank = os.environ.get("LOCAL_RANK", -1)
        traceback_msg = traceback.format_exc()
        logger.error(f"\n{'='*40}\nError: {str(e)} 발생\nProcess-[{local_rank}] 종료\nTraceback: {traceback_msg}{'='*40}")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            