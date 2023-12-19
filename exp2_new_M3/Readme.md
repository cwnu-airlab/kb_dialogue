
# 개요
M3는 외부 메모리를 사용하는 멀티모달 처리 시스템이다. 본 문서는 M3의 기본 개념과 구조, 각 파일에 대한 설명, 사용법, 그리고 응용에 대해서 설명한다.

# 디렉토리 구조

~~~
.
├── checkpoint
├── data
├── nets
├── saved
├── utils
├── valid_t5-generator_lr_0.0001_pat_7_epoch_0000051_valLoss_0.1840_valAcc_0.9731
├── action.py
├── agent.py
├── config.py
├── dataset.py
├── main.py
├── net.py
├── requirement.txt
├── retriever.py
└── run_jit01.sh
~~~
M3의 폴더 구조는 위와 같다. 각각에 대한 설명은 다음과 같다.

- checkpoint: 학습 중간에 chekpoint를 저장하는 폴더
- data: 학습하는 데이터를 저장하는 폴더. 여러 개의 데이터에 대한 실험을 진행하는 경우에 이 아래에 다시 폴더를 만들어서 진행한다.
- nets: M3의 하부 구조를 이루는 사전학습 모델(pretrained model)과의 인터페이스를 위해 만든 파일이 있는 폴더
- saved: 미세조정된 모델, 로그 정보를 저장하는 폴더. 여기에는 'models'와 'logdirs'가 있다.
- utils: action 클래스에서 사용하는 개별 함수들이 구현된 파일이 있는 폴더
- valid_t5... : 사전학습된 모델이 저장된 공간. 우리는 KET5를 사용하고 있다.

다음은 M3를 구성하는 파일들에 대한 설명이다.

- main.py : 프로그램의 시작이 되면 부분. 
- config.py: 프로그램의 모든 설정이 저장된 클래스. 실제 실험에는 shell 파일을 이용해 조정이 가능하다.
- agent.py: 프로그램의 핵심 구조가 있는 부분. 
- dataset.py: 학습데이터와 외부 지식을 읽는 부분.
- retriever.py: 외부지식을 검색하는 부분
- net.py:  결과를 생성하는 부분. 응용에 따라서 생성(generation), 선택(QA), 분류(classification) 등의 구조를 가질 수 있다. 이것은 'net_type'이라는 변수를 통해서 결정한다. 


# 실행
실험을 위해서 실행을 하려면 'run_xxx01.sh'을 수정한다. 이 파일 이름에서 'xxx'는 학습할 데이터를 사용하도록 설계되었다. 그 다음의 번호는 실험 번호이다. 첫번째 실험 '01', 두번째 실험 '02' 이러한 방법으로 사용하면 된다. 예를들어 'run_jit01.sh'이면 'jit' 데이터를 사용하는 첫번째 실험이라는 말이 된다.

shell 파일의 내용은 다음과 같다.
~~~shell
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
    --device=mps \
	--data_dir=data/jit_data/run/\
	--dataset=jit_real.json \
	--datatype=text \
    --model_type=generator \
	--tokenizer_path=KETI-AIR/ke-t5-small\
    --n_epochs=30 \
    --num_workers=0 \
    --eval_frequency=100 \
    --seed=-1 \
	--batch_size=64\
	--searcher_beam_size=5 \
	--update_retriever_emb=True \
	--update_generator_emb=True \
	--external_memory_path=data/jit_data/kb/valid_kb.txt \
	--pretrained_weight_path=valid_t5-generator_lr_0.0001_pat_7_epoch_0000051_valLoss_0.1840_valAcc_0.9731\
    2>&1 | tee $log_file
~~~
__각 항목에 대한 설명들__

실험을 진행한다면 다음과 같은 순서를 따라서 진행하면 된다. 

1. linux 환경인지 macos 환경인지 결정한다.
2. 학습할 데이터를 결정한다. 예를 들어 jit_real.json이라고 하자. 입력 파일 형식에 대해서는 아래에서 설명한다.
3. dataset.py의 load_data(), \_load_jit() 를 수정한다. 
4. action.get_loss_fn()에서 입력 파일에 따라 손실함수를 수정한다. 
5. shell 파일에서 실행환경, 각 폴더와 파일, 파라미터를 수정한다. 

이제 다음과 같이 실행한다. 

~~~
$> bash ./run_jit01.sh
~~~


# 학습 데이터
학습 데이터는 다음과 같은 형식을 지원한다.  또한 데이터는 'text', 'image', 'video'가 가능한다. 이 형식은 'data_type' 변수를 사용하여 결정한다. 

> - json
> - jsonl
> - txt
> - csv
> - tsv



# 외부 지식
검색기에서 사용하는 외부 지식이다. 텍스트 형식을 가지고 있으며 dataset.py에서 읽어들인다. 


