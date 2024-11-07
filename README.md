
# 🏆 다국어 영수증 OCR

## 🥇 팀 구성원

### 박재우, 이상진, 유희석, 정지훈, 천유동, 임용섭


## 프로젝트 소개
카메라로 영수증을 인식할 경우 자동으로 영수증 내용이 입력되는 어플리케이션이 있습니다. 이처럼 OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR은 글자 검출 (Text detection), 글자 인식 (Text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다.

본 대회에서는 `글자 검출`만을 수행합니다. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작합니다.

이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작하고 학습 데이터 추정을 통한 Data-Centric 다국어 영수증 속 글자 검출을 진행합니다.

본 대회는 Data-Centric AI의 관점에서 모델 활용을 경쟁하는 대회입니다. 이에 따라 제공되는 베이스라인 코드 중 모델 관련 부분을 변경하는 것이 금지되어 있습니다.

평가지표는 `DetEval`을 사용합니다. `DetEval`은 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가 방법 입니다.

최종 리더보드에는 recall, precision과 F1-score 가 기록되고, recall과 precision 의 조화평균인 `F1 score` 를 기준으로 랭킹이 산정됩니다.
<br />

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2024.10.28(월) ~ 2024.11.7(목)

<br />

## 🥈 프로젝트 결과
### Public
- **** / 24
- F1 Score : **0.9200**
### Private
- **** / 24
- F1 Score : **0.**

<br />

## 🥉 데이터셋 구조
```
data/
    ├─ chinese_receipt/
    │  ├─ ufo/
    │  │    └─ train.json
    │  │    └─ test.jsons
    │  │    └─ sample_submission.csv
    │  └─ images/
    │        └─ train/
    │             └─ images
    │        └─ test/
    │             └─ images
    ├─ japanese_receipt/
    ├─ thai_receipt/
    └─ vietnamese_receipt/
 
```
이 코드는 `부스트캠프 AI Tech`에서 제공하는 데이터셋으로 다음과 같은 구성을 따릅니다. 
- 전체 이미지 개수 : 520장
- 분류 언어(4개) : 중국어, 일본어, 태국어, 베트남어
- 전체 데이터 중 학습데이터 400장, 평가데이터 120장으로 사용
- 제출 형식 : UFO(Upstage Format for OCR) format csv 파일
- **저작권 문제 없는 외부 데이터셋 사용 가능**
<br />

## 🥉 프로젝트 구조
```
project/
│   README.md
│   requirements.txt
│   start_ngrok.py
│
├───tools
│       cleansing_labels.ipynb
│       coco2yolo.py
│       csv_bbox_visualize.ipynb
│       ensemble.ipynb
│       json_bbox_visualize.ipynb
│       json_coco2pascal.ipynb
│
└───yolo
    │   yolo_inference.ipynb
    │   yolo_train.ipynb
    │
    └───cfg
            coco-trash.yaml
code
└─  model.py
└─  loss.py
└─  train.py
└─  inference.py
└─  dataset.py
└─  detect.py
└─  deteval.py
└─  east_dataset.py
└─  requirements.txt

```
### 1) Services
- `configs/a_custom/`: MMDetection 모델의 학습과 추론에 필요한 설정 파일들을 포함하고 있습니다.
- `tools/fold_train.py`: Stratified Group K-Fold 교차 검증을 통한 학습을 제공합니다.

### 2) Streamlit_viz
- `data/`: 데이터셋 로딩 및 증강 관련 파일들을 포함하고 있습니다.
- `models/`: 모델 로드 및 저장 관련 파일들을 포함하고 있습니다.
- `process/`: 이미지 전처리 기능을 제공합니다.
- `train/`: 모델 학습 및 평가에 필요한 파일들을 포함하고 있습니다.

### 3) tools
- `cloba2datu.ipynb`: cloba 데이터셋을 datumaro 형식으로 변환합니다.
- `datu2ufo.ipynb`: datumaro 형식의 데이터셋을 UFO 형식으로 변환합니다.
- `ufo2datu.ipynb`: UFO 형식의 데이터셋을 datumaro 형식으로 변환합니다.
- `easyocr_pseudo.ipynb`: Easyocr 라이브러리를 활용해서 pseudo-labeling을 진행합니다.
- `img_hash.ipynb`: 이미지 hash값을 구해 중복된 데이터를 검출합니다.
- `inference_visualize.ipynb`: inference한 결과를 test이미지에 시각화합니다.
- `server-status.py`: 현재 서버상태를 불러옵니다.

<br />

## ⚙️ 설치

### Dependencies
이 모델은 Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

```bash
pip install -r requirements.txt
```

- lanms==1.0.2
- opencv-python==4.10.0.84
- shapely==2.0.5
- albumentations==1.4.12
- torch==2.1.0
- tqdm==4.66.5
- albucore==0.0.13
- annotated-types==0.7.0
- contourpy==1.1.1
- cycler==0.12.1
- eval_type_backport==0.2.0
- filelock==3.15.4
- fonttools==4.53.1
- fsspec==2024.6.1
- imageio==2.35.0
- importlib_resources==6.4.2
- Jinja2==3.1.4
- kiwisolver==1.4.5
- lazy_loader==0.4
- MarkupSafe==2.1.5
- matplotlib==3.7.5
- mpmath==1.3.0
- networkx==3.1
- numpy==1.24.4
- nvidia-cublas-cu12==12.1.3.1
- nvidia-cuda-cupti-cu12==12.1.105
- nvidia-cuda-nvrtc-cu12==12.1.105
- nvidia-cuda-runtime-cu12==12.1.105
- nvidia-cudnn-cu12==8.9.2.26
- nvidia-cufft-cu12==11.0.2.54
- nvidia-curand-cu12==10.3.2.106
- nvidia-cusolver-cu12==11.4.5.107
- nvidia-cusparse-cu12==12.1.0.106
- nvidia-nccl-cu12==2.18.1
- nvidia-nvjitlink-cu12==12.6.20
- nvidia-nvtx-cu12==12.1.105
- packaging==24.1
- pillow==10.4.0
- pydantic==2.8.2
- pydantic_core==2.20.1
- pyparsing==3.1.2
- python-dateutil==2.9.0.post0
- PyWavelets==1.4.1
- PyYAML==6.0.2
- scikit-image==0.21.0
- scipy==1.10.1
- six==1.16.0
- sympy==1.13.2
- tifffile==2023.7.10
- tomli==2.0.1
- triton==2.1.0
- typing_extensions==4.12.2


<br />

## 🚀 빠른 시작
### Train
#### MMDetection

```python
# fold train
python tools/fold_train.py {config_path}

# train
python tools/train.py {config_path}
```

#### Torchvision
```python
python main.py
```
##### Torchvision Parser
기본 설정
- `--annotations_path` : train.json path
- `--data_dir` : Dataset directory
- `--model_name` : 학습 진행할 모델 이름 ( 기본값: Faster RCNN )
- `--device` : `cuda` or `cup` ( 기본값 : cuda )
- `--base_dir` : result path

학습 설정
- `--num_epochs` : 학습할 에폭 수 (기본값 : 1)
- `--batch_size` : 배치 크기 결정 ( 기본값 : 32 )
- `--n_split` : fold split 수량 ( 기본값 : 5 )
- `--training_mode` : `standard` or `fold` (필수)

옵티마이저 설정
- `--optimizer` : `SGD` or `AdamW` ( 기본값 : SGD )
- `--learning_rate` : 학습률 설정 ( 기본값 : 0.001)
- `--momentum` : SGD Momentum 값 설정 ( 기본값 0.9 )
- `--weight_decay` : 옵티마이저 weight decay 설정 ( 기본값 : 0.0009 )

스케쥴러 설정 (CosineAnnealing)
- `--scheduler_t_max` : 코사인 어널링 t max 설정 ( 기본값 : 40)
- `--scheduler_eta_min` : 코사인 어널링 eta min 설정 ( 기본값 : 0)

### Test
#### MMDetection
```python
python tools/test.py {config_path} {pth_file_path}
```

##### MMDetection Parser
- `--tta` : Test Time Augmentation 활성화
<br />

## 🏅 Wrap-Up Report   
### [ Wrap-Up Report 👑](https://github.com/boostcampaitech7/level2-objectdetection-cv-08/blob/main/WRAP_UP/CV08_level2_%EB%9E%A9%EC%97%85%EB%A6%AC%ED%8F%AC%ED%8A%B8.pdf)
