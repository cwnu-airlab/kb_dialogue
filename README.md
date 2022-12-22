# 외부 지식을 이용한 대화 모델
외부 지식을 이용한 대화 모델

### 01. 외부 지식 설정

기존의 영어 위키피디아 문서로 구성된 외부지식을 사용하거나 사용자가 진행할 task에 맞춰서 진행하기 위해서 `/config/config.json` 에 명시된 파라미터 내 `"additional_documents_path":false`이면 share에 저장되어 있는 영어 위키피디아 문서를 복사해와서 `'data'` 폴더 내 저장한다. "additional_documents_path"에 사용자가 진행할 task에 대한 문서를 `.npy` 파일로 불러오도록 한다.

실제로 `"additional_documents_path"` 의 확장자가  `.npy` 파일이라면 해당 파일을 불러와서 사용하고, 확장자가 `.tsv` 또는 `.txt`라면 해당 데이터를 각 라인을 외부 문서로 읽어서 외부지식으로 사용한다.

### 02. 학습

`/config/config.json` 에 명시된 파라미터를 입력으로 하여 학습을 진행한다. config.json에는 크게 2가지로 나눌 수 있으며, "t5"와 "trainer"로 나눌 수 있다.

* t5 : 기존 realmConfig와 t5 config를 merge하여 생성된 config
* trainer : 실제 학습 및 예측에 사용되는 하이퍼파라미터에 대한 config

이 때 config를 변경해야한다. 학습 시`"predict" : false`, ` "evalutate : false"` 으로 설정해야 한다. t5의 모델 선택에 따라 `projected_size` 또한 동일하게 맞춰주어야 한다. 예를 들어 `t5-small`이면  `projected_sizes`는 512이여야 한다.

#### 실행 예시

```python
python run.py -c config/config.json -d 0
```

* -c : config 파일 path, default=None
* -d : gpu device 번호

#### 학습 데이터 형식 예제

1. json

   ```json
   [
     {
     "source": "This is input sentence.",
     "target": "This is output sentence.",
     "kb_idx": 4995
     },
     {
     "source": "This is input sentence.",
     "target": "This is output sentence.",
     "kb_idx": 4995
     }
    ]
   ```

2. jsonl

   ```json
   {"source": "This is input sentence.", "target": "This is output sentence.", "kb_idx": 4995}
   {"source": "This is input sentence.", "target": "This is output sentence.", "kb_idx": 4995}
   ```

3. txt, tsv : `[SOURCE]\t[TARGET]\t[KB_INDEX] `형식으로 구성되어 있어야 함

   ```tex
   This is input sentence.	This is output sentence.	4995
   This is input sentence.	This is output sentence.	4995
   ```

### 03. 예측

이 때 config를 변경해야한다. 학습 시`"predict" :true ` 으로 설정하고 실행한다.  `"evalutate : true"`로 설정하면 정답 문장과 예측 문장을 이용하여 nltko 라이브러리의 measure들을 이용하여 성능까지 측정한다.

#### 실행 예시

```python
python run.py -c config/config.json -d 0
```

* -c : config 파일 path, default=None
* -d : gpu device 번호

#### 예측 결과 예시

```json
{
	"data" : {
		"source" : "This is sentence",
		"target" : "This is sentence",
    "kb_idx" : 4995
	},
	"output" : {
		"srce" : "This is sentence",
		"gold" : "This is sentence",
		"pred" : "This is sentence",
    "gold_kb" : 4995,
    "pred_kb" : [3343, 3452, 4112,...] 
	}
}
```

## Acknowledgement
본 연구는 정부(과학기술정보통신부)의 재원으로 지원을 받아 수행된 연구입니다.   
(정보통신기획평가원, 2022-0-00320, 상황인지 및 사용자 이해를 통한 인공지능 기반 1:1 복합대화 기술 개발)
