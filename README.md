# 개요

# 디렉토리 구조

~~~
.
├── checkpoint
├── M3
     ├── agent.py
     ├── data_module.py
     ├── model.py
     └── upgrade_data_module.py
├── data
     ├── data_eval
     ├── test
          ├── labels.json
          └── logs.json
     └── knowledge.json
     ├── data_fit
          ├── train
                ├── labels.json
                └── logs.json
          ├── valid
                ├── labels.json
                └── logs.json
          └── knowledge.json
     ├── hotel_db.json
     ├── restaurant_db.json
     ├── taxi_db.json
     └── san_francisco_db.json
├── gener_eval.py
├── retr_eval.py
├── run.py
├── run_dstc9_00_ict.sh
├── run_dstc9_01_history.sh
└── run_dstc9_02_generate.sh
~~~

- `checkpoint` : 학습 중에 모델이 저장되는 폴더
- `M3` : 학습 및 추론 코드가 존재하는 폴더
	- `agent.py` : 학습 및 추론 수행 코드
	- `data_module.py` : 데이터 셋 코드
	- `model.py` : 모델 코드
	- `upgrade_data_module.py` : 기존 데이터 셋에서 multiWOZ 2.1과 DSTC 9 eval 폴더에 존재하는 데이터베이스의 지역 정보를 활용하여 개체명을 추출을 개선한 코드
- `data` : 데이터가 저장되어 있는 폴더
	- `data_eval` : 평가 시 사용하는 데이터가 저장되어 있는 폴더
		- `test` : test 데이터가 저장되어 있는 폴더
			- `labels.json` : test 정답 데이터
			- `logs.json` : test 대화 내역 데이터
		- `knowledge.json` : 평가 시 사용하는 외부 지식 데이터
	- `data_fit` : 학습 시 사용하는 데이터가 저장되어 있는 폴더
		- `train` : train 데이터가 저장되어 있는 폴더
			- `labels.json` : train 정답 데이터
			- `logs.json` : train 대화 내역 데이터
		- `valid` : valid 데이터가 저장되어 있는 폴더
			- `labels.json` : valid 정답 데이터
			- `logs.json` : valid 대화 내역 데이터
		- `knowledge.json` : 학습 시 사용하는 외부 지식 데이터
	- `hotel_db.json` : MultiWOZ 2.1에 있는 hotel 엔티티의 정보가 담긴 데이터베이스
	- `restaurant_db.json` : MultiWOZ 2.1에 있는 restaurant 엔티티의 정보가 담긴 데이터베이스
	- `taxi_db.json` : MultiWOZ 2.1에 있는 taxi 엔티티의 정보가 담긴 데이터베이스
	- `san_francisco_db.json` : DSTC 9에 있는 샌프란시스코 엔티티의 정보가 담긴 데이터베이스
- `gener_eval.py` : Generator 결과를 평가하는 코드
- `retr_eval.py` : Retriever 결과를 평가하는 코드
- `run.py` : 실험 파라미터를 설정한 후 agent를 호출하여 실험을 진행시키는 코드
- `run_dstc9_00_ict.sh` : ict 사전 학습을 위한 baseline 쉘 파일
- `run_dstc9_01_recent.sh` : 최근 도메인 + 최근 개체명 + 시스템의 마지막 발화 + 사용자의 마지막 발화를 Retriever의 입력으로 사용하는 recent 학습을 위한 baseline 쉘 파일
- `run_dstc9_02_generate.sh` : Generator 학습을 위한 baseline 쉘 파일

# 주요 실험 파라미터

- `version` : 실험 version
- `gpu_id` : 학습 및 추론 시 사용할 GPU 지정
- `model_type` : 사용할 모델 종류(ict, retriever, generator 중 선택)
- `mode` : 실행 모드(train, predict 중 선택)
- `refresh_frequency` : 외부 지식 임베딩 업데이트 주기
- `gradient_accumulation_steps` : 기울기 누적 단계
- `max_grad_norm` : 그래디언트 클리핑에 사용되는 기울기 최대 norm 값
- `model_all_save` : 모든 중간 모델 저장 여부
- `input_type` : Retriever 입력 형식 종류(history, entire, recent 중 선택)
	- history : 일정 토큰 이내의 대화 내역을 입력
	- entire : 전처리 모듈이 추출한 모든 도메인과 모든 개체명 그리고 시스템의 마지막 발화와 사용자의 마지막 발화를 입력
	- recent : 전처리 모듈에서 추출된 도메인과 개체명 중 마지막으로 추출된 도메인과 개체명만을 시스템의 마지막 발화와 사용자의 마지막 발화와 함께 입력
- `history_max_utterances` : 입력으로 사용할 대화 내역의 발화 갯수
- `history_max_tokens` : 입력으로 사용할 대화 내역의 토큰 수
- `retriever_prediction_result_path` : generator 추론 시 사용되는 retriever 결과 값이 저장된 파일 경로
- `per_gpu_batch_size` : 학습 시에 gpu 당 batch 크기
- `predict_batch_size` : 추론 시에 batch 크기
- `num_workers` : dataloader를 위한 병렬 프로세스 수
- `shuffle` : 데이터 shuffle 여부
- `num_candidates` : 사용자의 질의와 유사한 후보 문서의 수
- `vector_similarity` : 벡터 유사도 계산 방법(cosine 유사도, 내적 중 선택)
- `use_proj_layer` : 프로젝션 레이어 사용 여부
- `proj_size` : 프로젝션 레이어의 크기
- `use_passage_body` : 외부 지식 문서 임베딩 생성 시 문서의 본문(body) 사용 여부
- `pretrained_model_path` : 사전 학습된 모델 경로
- `predict_model_path` : 추론 할 모델 경로
- `max_new_tokens` : 새로 생성할 수 있는 최대 토큰 수
- `num_beams` : 빔 search에서 사용되는 빔 수
- `temperature` : 샘플링 시 사용할 온도 값
- `top_k` : 상위 k개의 후보 단어 중에 sampling
- `top_p` : 누적 확률 p안의 후보 단어 중에 sampling
- `learning_rate` : 학습률

# 학습

1. 실험 파라미터 값들을 파라미터로 전달받아 run 파일을 실행하는 쉘 파일 생성
2. `sh run_dstc9_{version}_ict.sh` 명령어로 쉘 파일 실행

※ multi gpu로 학습할 시 쉘 파일에 `python run.py` 대신 `torchrun --nproc_per_node={use_gpu_cnt} run.py`를 사용

# 추론

- mode를 predict로 설정한 후 쉘 파일을 실행할 시 자동으로 valid loss가 가장 적은 모델을 추론 진행
- 추론할 모델을 직접 선택하고 싶을 경우 mode를 predict로 설정하고, `predict_model_path`를 모델이 저장된 디렉토리로 직접 지정 후 실행

※ 추론 시에는 multi gpu 사용 X

# 평가

- Retriever 실험 결과 파일 경로를 retr_eval.py에 설정하고 실행
- Generator 실험 결과 파일 경로를 gener_eval.py에 설정하고 실행
