# 🎵 AI_Music — 강의_AI_Pair 스타일 (1강~8강)

`Part III. LLM & Agent/music_ai`의 Phase0(D1~D4)·Phase1(D5~D8)을 코드는 그대로 유지한 채,
`Part II. Transformer/1. DL_Fundamentals/강의_AI_Pair` 스타일(목표·계약·비유·실행 전 예측·AI Pair 4단계)로
재구성한 버전입니다. (예전 이름 `music_ai_v2`에서 `AI_Music`으로 폴더명이 바뀌었습니다.)

## 빠른 시작

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

jupyter notebook "강의_AI_Pair/1강_내노래를_숫자로보기(AI_Pair).ipynb"
```

노트북 8개는 `강의_AI_Pair/` 서브폴더 안에 있고, `Data/`·모델 파일·`app*.py`는 그 위(`AI_Music/`) 루트에 있습니다 — 노트북을 어느 위치에서 열어도(`강의_AI_Pair/`든 `AI_Music/`든) 각 노트북이 cwd와 상위 폴더를 자동으로 탐색해 `Data/`를 찾습니다.

## 커리큘럼 (1강~8강)

모든 파일은 `강의_AI_Pair/` 안에 있습니다.

| 강 | 파일 | 핵심 |
|---|---|---|
| 1강 | `1강_내노래를_숫자로보기(AI_Pair).ipynb` | 오디오 로드, 파형, RMS 에너지 |
| 2강 | `2강_요즘음악_에너지템포비교(AI_Pair).ipynb` | BPM, spectral centroid, zero crossing rate |
| 3강 | `3강_플레이리스트_클러스터링(AI_Pair).ipynb` | KMeans + PCA 음악 지도 |
| 4강 | `4강_소리를이미지로_멜스펙트로그램(AI_Pair).ipynb` | 멜스펙트로그램 → CNN 입력 직관 |
| 5강 | `5강_멜스펙트로그램_장르분류(AI_Pair).ipynb` | librosa 피처 추출 + RandomForest |
| 6강 | `6강_Streamlit_장르예측앱(AI_Pair).ipynb` | st.file_uploader → RF 예측 → 멜스펙트로그램 UI |
| 7강 | `7강_Streamlit고급_ResNet도입(AI_Pair).ipynb` | 신뢰도, 멀티파일 비교, ResNet conv1 시각화 |
| 8강 | `8강_ResNet18_스펙트로그램분류(AI_Pair).ipynb` ★★ | ResNet-18 fine-tuning (GPU 트랙, RUN_HEAVY 게이트) |

각 노트북은 강의_AI_Pair 스타일의 12개 공통 요소(목표·이전 강 연결·계약·실행 전 예측·표 기반 코드 설명·확인해보기·AI Agent 연결·불확실성 참고·Solo/Review/Debug/Prompt Card·세션 요약)를 포함합니다.

## 폴더 구조

```
AI_Music/
├── 강의_AI_Pair/
│   └── 1강~8강 (AI_Pair).ipynb      ← 강의 노트북 8개
├── app.py, app_v2.py, app_preview.py ← 6·7강(및 5강 예고)이 생성/참조하는 Streamlit 앱
├── model_rf.joblib, label_encoder.joblib, scaler.joblib, resnet18_gtzan.pth
├── d8_fallback_data/                ← GPU 없는 환경용 8강 폴백 데이터
├── Data/
│   ├── generated_loops/            ← 1~2강용 합성 루프
│   ├── latest_music_local/         ← 1강 데모 mp3/wav
│   └── Music_genres/               ← GTZAN 원본 (WAV 883개 + PNG 937개 + CSV 2개)
├── requirements.txt, packages.txt
└── _archive/                        ← 강사용 내부 자료(작업계획서, 일회성 편집 스크립트) — 수강생 배포 불필요
```

## 참고

- MIDI·K-pop RAG 데이터셋, Phase2(LangChain)·Phase3(LangGraph 배포) 노트북은 이 버전의 범위 밖입니다 — 원본 `music_ai/` 참고.
- 6강·8강 코드는 실습 중 이 폴더에 `requirements.txt`를 자체 생성하는 셀을 포함합니다. 노트북을 실행한 뒤에는 프로젝트 셋업용 정본(이 파일 상단 내용)으로 되돌려야 합니다.
- GTZAN: 학술 비상업 목적 사용 (Tzanetakis & Cook, 2002).
- `resnet18_gtzan.pth`는 `RUN_HEAVY=True`로 실제 GPU 학습을 돌려야 진짜 학습된 가중치로 갱신됩니다 — 파일이 이미 있으면 dry-run이 덮어쓰지 않습니다(8강 [셀 15] 참고).
