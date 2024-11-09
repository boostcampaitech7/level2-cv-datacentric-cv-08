
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
- 4 / 24
- F1 Score : **0.9200**
### Private
- 5 / 24
- F1 Score : **0.9073**

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
    ├─  README.md
    ├─  requirements.txt
    ├─  setup.sh
    ├─  custom_train.py
    ├─  find_hyper_pram.py
    │
    ├── services
    │     ├─  kakao.py
    │     ├─  refresh_kakao_token.py
    │     ├─  sheet_kakao_key_update.py
    │     ├─  sheet_pull_kakao_key.py
    │     ├─  slack.py
    │     └─  spreadsheet.py
    │
    └─── tools
          ├─  cloba2datu.ipynb
          ├─  datu2ufo.ipynb
          ├─  ufo2datu.ipynb
          ├─  easyocr_pseudo.ipynb
          ├─  img_hash.ipynb
          ├─  inference_visualize.ipynb
          ├─  server-status.py
          ├─  data_duplication_check.py
          └─  streamlit_viz.py  
    
```
### 1) Services
- `kakao.py`: 카카오톡 액세스 토큰을 불러와 카카오톡 메시지로 학습현황을 전송합니다.
- `refresh_kakao_token.py`: 카카오톡 액세스 토큰을 새로 발급합니다.
- `sheet_kakao_key_update.py`: 구글 스프레드 시트에 카카오톡 액세스 토큰을 업데이트합니다.
- `sheet_pull_kakao_key.py`: 구글 스프레드 시트에서 액세스 토큰을 가져와 로컬 토큰을 업데이트합니다.
- `slack.py`: 학습현황을 슬랙 메시지로 보냅니다.
- `spreadsheet.py`: 서버의 학습 현황을 구글 스프레드 시트에 업데이트합니다.
  
### 2) tools
- `cloba2datu.ipynb`: cloba 데이터셋을 datumaro 형식으로 변환합니다.
- `datu2ufo.ipynb`: datumaro 형식의 데이터셋을 UFO 형식으로 변환합니다.
- `ufo2datu.ipynb`: UFO 형식의 데이터셋을 datumaro 형식으로 변환합니다.
- `easyocr_pseudo.ipynb`: Easyocr 라이브러리를 활용해서 pseudo-labeling을 진행합니다.
- `img_hash.ipynb`: 이미지 hash값을 구합니다.
- `inference_visualize.ipynb`: inference한 결과를 test이미지에 시각화합니다.
- `server-status.py`: 현재 서버상태를 불러옵니다.
- `data_duplication_check.py`: hash값으로 이미지가 겹치는지 확인하고 제거합니다.
- `streamlit_viz.py`: dataset의 annotation을 streamlit으로 시각화합니다.

<br />

## ⚙️ 설치

### Dependencies
이 모델은 Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

```bash
pip install -r requirements.txt
```
<details>
<summary>requirements 접기/펼치기</summary>

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
</details>
<br />

## 🚀 빠른 시작
### Train
```python
python custom_train.py 
```
### Train Parser
기본 설정
- `--data_dir` : Dataset directory
- `--model_dir` : Model directory (기본값 : EAST Model)
- `--device` : `cuda` or `cpu` ( 기본값 : cuda )

학습 설정
- `--num_workers` : 학습할 프로세스 수 (기본값 : 8)
- `--image_size` : 학습할 이미지 크기 (기본값 : 2048)
- `--input_size` : 학습할 입력 이미지 크 (기본값 : 1024)
- `--batch_size` : 배치 크기 결정 ( 기본값 : 8)
- `--learning_rate` : 학습률 설정 ( 기본값 : 0.001)
- `--max_epochs` : 학습할 에폭 수 (기본값 : 150)
- `--save_interval` : 가중치를 저장할 epoch 간격 (기본값 : 5)

## 🏅 Wrap-Up Report   
### [ Wrap-Up Report 👑]
