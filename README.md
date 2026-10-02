# 임상텍스트 LLM. 의료 특화 모델이 노트를 읽는가

MIMIC-IV-Note 퇴원요약문에서 퇴원 약물 목록을 JSON으로 추출하고, 의료 특화
LLM이 기반 모델보다 실제로 나은지를 같은 계열 짝 비교로 측정함.

## 핵심 질문

모델이 이 노트를 읽고 답하는지, 흔한 처방을 기억해 두었다가 내놓는지를 가름. 노트를 지우거나,
남의 환자 노트로 바꿔치거나, 문장 순서를 흩어도 점수가 유지되면 그 점수는 사전 지식에서 나온 것으로 봄.
이 구분을 사전 등록된 기준으로 검증함.

## 요약

1. 외워서 답하지 않음. 노트를 빼면 16모델 중 11개가 micro-F1 0.000 이고 나머지도 0.06\~0.08 임. 흔한 약 10개만 찍어도
   0.220 이 나오는데 그러지 않음.
2. 의료 특화 모델은 대체로 기반 모델보다 낫지 않음. 사전등록한 세로 짝 2쌍은 모두 기반이 이겼음 (MedGemma −0.207,
   Meditron3 −0.115). 뒤에 더한 세로 짝 5쌍은 차이 없음 3, 기반 우세 1 (Apertus MeditronFO −0.069, 여덟 번 비교를 보정하면
   차이 없음), 의료 우세 1 (OLMo-2 32B MeditronFO +0.045, 두 모델 모두 문맥 4,096 안에 드는 노트 133건에서만 잰 값).
3. 같은 절차도 기반에 따라 갈림. 같은 MeditronFO 절차를 세 기반에 걸었더니 Gemma 3 27B 는 차이 없음, Apertus 8B 는
   나빠짐(−0.069, 보정하면 차이 없음), OLMo-2 32B 는 좋아짐(+0.045, 133건). 진 짝은 지시를 지키지 못함.
   MedGemma 환각률 29.8%, JSON 준수 0%.
4. 고른다면 32B 를 올릴 수 있을 때 Baichuan-M2(0.951)나 Qwen2.5 Instruct(0.945). 둘의 차이는 없음. 8B bf16 이면
   Llama 3.1 Instruct(0.917).
5. 드문 약에서는 이 평가로 판단할 수 없음. 흔한 약(등장 1000회 이상)은 상위 다섯 모델이 93\~97% 를 맞힘. 등장 10회 미만인
   약은 표본에 12건뿐이고 상위 다섯 모델이 그중 3\~4건(25\~33%)을 맞혔는데, 3/12 의 정확 95% 구간이 5\~57% 임. 놓치면
   위험한 쪽이 드문 약이라, 쓰기 전에 드문 약을 따로 모아 재야 함.

판정 기준은 모델을 돌리기 전에 [docs/prereg.md](docs/prereg.md) 에 고정함. 사전등록한 것은 4모델, 3쌍이고,
나머지 12모델, 5쌍은 1차 결과를 본 뒤 더한 탐색임. 프롬프트는 변형 3종 가운데 모델마다 가장 유리한 것을 씀.
설계 메모 전체는 [docs/report.md](docs/report.md) 1절.

## 설계

### 모델 16개, 짝 8쌍

같은 계열 안에서 의료 사전학습 유무만 다르게 짝지음(세로 짝). 크기, 계열이 함께 다르면 차이의 원인을 말할 수 없기 때문임.

| 쌍 | 의료 특화 | 기반 모델 | 정밀도 | 구분 |
|---|---|---|---|---|
| 쌍1 | MedGemma 1.5 4B | Gemma 3 4B | bf16 | 사전등록, 세로(Google 의료 특화) |
| 쌍2 | Meditron3 8B | Llama 3.1 8B Instruct | bf16 | 사전등록, 세로(EPFL 의료 특화) |
| 쌍3 | MedGemma 1.5 4B 대 Meditron3 8B | | | 사전등록, 가로(실무 권고, 크기와 계열이 섞임) |
| 쌍4 | Baichuan-M2 32B | Qwen2.5 32B Instruct | 4비트 | 추가, 세로 |
| 쌍5 | HuatuoGPT-o1 8B | Llama 3.1 8B Instruct | bf16 | 추가, 세로 |
| 쌍6 | Gemma 3 27B MeditronFO | Gemma 3 27B Instruct | 4비트 | 추가, 세로 |
| 쌍7 | Apertus 8B MeditronFO | Apertus 8B Instruct | bf16 | 추가, 세로 |
| 쌍8 | OLMo-2 32B MeditronFO | OLMo-2 32B Instruct | 4비트 | 추가, 세로 |

짝 없는 세 모델(Lingshu 7B, II-Medical 8B, MedReason 8B)은 기반을 특정할 수 없어 순위표에만 올림.
정밀도는 짝 안에서 같게 맞춤. 한쪽만 양자화하면 그 손해가 의료 특화 효과에 섞이기 때문임(27B, 32B 는 양쪽 모두 4비트).
사고를 쓰는 모델과 문맥이 짧은 모델의 처리 규칙은 판정 전에 고정함(트러블슈팅 9, 12).
짝을 이렇게 둔 이유는 [docs/report.md](docs/report.md) 6절.

### 감사 조건

| 조건 | 입력 | 드러내는 것 |
|---|---|---|
| `normal` | 노트 전체 | 기준 성능 |
| `no_section` | 퇴원약 절 제거 | 답이 본문에 없어도 맞히는가 |
| `empty` | 인적사항만 | 노트 없이 내놓는 사전 지식 |
| `other_patient` | 다른 환자 노트 | 주어진 노트를 실제로 읽는가 |
| `shuffled` | 문장 순서 셔플 | 구조를 읽는가, 이름만 긁는가 |
| `frequency_baseline` | 모델 없음 | 실격 기준선 |

### 정답과 채점

정답은 노트의 `Discharge Medications:` 절을 규칙으로 파싱해 만듦(`src/notes.py`). `prescriptions` 테이블은
입원 중 투약 전체라 정답으로 쓰지 않음. 노트 33만 건 중 앞 20,000건은 절 파싱률 98.1%, 항목 추출 94.0%,
약물명의 약국 어휘 일치 96.6%, 시드를 고정한 무작위 25,000건은 98.0%, 94.1%, 96.6% 다(`scripts/phase0_audit.py`).
파서와 채점기의 규칙 전체는 [docs/report.md](docs/report.md) 3절.

## 데이터

MIMIC-IV-Note + MIMIC-IV v3.1 (`hosp` 모듈). 크리덴셜 접근이 필요하며 데이터와
파생 산출물은 저장소에 포함되지 않음(PhysioNet DUA).

```
<저장소가 있는 곳>/
├── <폴더>/clinical-llm-memorization-audit/     <- 이 저장소
└── 데이터/mimic/data/raw/{mimic-iv-note/note/, mimiciv/}
```

경로는 저장소 기준 상대경로로 정함(`src/datapaths.py`). 데이터를 다른 곳에
두었다면 환경변수로 덮어씀.

```bash
export MIMIC_DATA_ROOT=/my/path/mimic/data
```

추론은 전부 로컬에서 함. MIMIC 노트를 외부 LLM API로 보내면 DUA 위반임
(DUA는 제3자와의 데이터 공유를 금지하며 API 전송이 여기 포함됨).


## 실행

추론은 RTX 4090(24GB) 한 장에서 돌림.

```bash
conda env create -f environment.yml && conda activate cltext
sh scripts/install_hooks.sh && sh scripts/verify_hooks.sh   # 데이터 유출 차단 훅(clone 마다 1회)
# torch 는 CUDA 휠이라 별도 인덱스에서 받음
pip install torch==2.14.0+cu132 --index-url https://download.pytorch.org/whl/cu132

# 평가 대상 LLM 16개를 HuggingFace 캐시로 받는다 (521GB, 485GiB 실측).
# Llama 는 원본 .pth(16GB) 를 빼고 safetensors 만 받음.
python scripts/download_llm_models.py

python scripts/phase0_audit.py
python scripts/run_phase1_sample.py
python scripts/run_phase2_conditions.py

# 모델별로: 프롬프트 선택용 추론 -> 선택 -> 본 평가 추론
# 배치는 4비트 27B, 32B 와 II-Medical 이 1, 나머지 8B 가 2, 4B 가 8 이었음(응답 행의 batch_size).
# 배치가 다르면 몇 건의 출력이 갈리므로 같은 값으로 돌림(트러블슈팅 14).
MODELS="medgemma-4b gemma3-4b baichuan-m2-32b qwen25-32b gemma3-27b-meditronfo gemma3-27b \
        apertus-8b-meditronfo apertus-8b olmo2-32b-meditronfo olmo2-32b \
        meditron3-8b huatuo-o1-8b llama31-8b lingshu-7b ii-medical-8b medreason-8b"
batch_of() {
    case $1 in
        medgemma-4b|gemma3-4b) echo 8 ;;
        apertus-8b|apertus-8b-meditronfo|meditron3-8b|huatuo-o1-8b|llama31-8b|lingshu-7b|medreason-8b) echo 2 ;;
        *) echo 1 ;;
    esac
}
for m in $MODELS; do
    python scripts/run_phase3_infer.py --model $m --stage select --batch "$(batch_of $m)"
done
python scripts/run_phase3_select_prompt.py
for m in $MODELS; do
    python scripts/run_phase3_infer.py --model $m --stage final --batch "$(batch_of $m)"
done
# 쌍8 은 기반 토크나이저로 한 번 더 (트러블슈팅 12)
python scripts/check_tokenizer_sensitivity.py --model olmo2-32b-meditronfo --stage select --batch 1
python scripts/check_tokenizer_sensitivity.py --model olmo2-32b-meditronfo --stage final --batch 1

python scripts/run_phase4_evaluate.py
python scripts/render_results_tables.py --write     # README 표
python scripts/render_report_tables.py --write      # docs/report.md 의 표
```

GPU 없이 배관만 점검하려면 추론 명령에 `--stub` 을 붙임.

결과를 낸 모델 커밋과 본 평가 밖 보조 스크립트는 [docs/report.md](docs/report.md) 재현 방법 절.

## 결과

`outputs/phase4_results.json`. 모델당 600건(main 정상 200 + 감사 4조건 x 100) 로컬 추론이고 주 지표는
main 200건의 micro-F1 임. 아래 표는 `scripts/render_results_tables.py` 가 채점 결과 파일에서 만듦.
감사 조건, 드문 약, 출력 형식 표와 해석은 [docs/report.md](docs/report.md) 7\~9절.

<!-- 결과표 시작 -->
빈도 베이스라인(가장 흔한 약 10개 고정 출력) micro-F1 0.220. 채점된 16개 모델 모두 크게 넘겨 실격 없음.

### 순위표. 정상 노트 200건 micro-F1

| # | 모델 | 특화 | 짝 | 정밀도 | 프롬프트 | micro-F1 [95% CI] | 환각률 | 누락률 | JSON 준수 | 노트 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| 1 | Baichuan-M2 32B | 의료 | 쌍4 | 4bit | grounded | 0.951 [0.939, 0.962] | 5.2% | 4.5% | 100% | 200 |
| 2 | Qwen2.5 32B Instruct | 기반 | 쌍4 | 4bit | grounded | 0.945 [0.931, 0.958] | 5.2% | 5.8% | 100% | 200 |
| 3 | Gemma 3 27B Instruct | 기반 | 쌍6 | 4bit | minimal | 0.923 [0.896, 0.945] | 8.2% | 7.2% | 80% | 200 |
| 4 | Gemma 3 27B MeditronFO | 의료 | 쌍6 | 4bit | grounded | 0.923 [0.904, 0.940] | 7.9% | 7.6% | 100% | 200 |
| 5 | Llama 3.1 8B Instruct | 기반 | 쌍2, 쌍5 | bf16 | schema | 0.917 [0.892, 0.937] | 7.8% | 8.8% | 100% | 200 |
| 6 | HuatuoGPT-o1 8B | 의료 | 쌍5 | bf16 | schema | 0.914 [0.895, 0.931] | 6.9% | 10.3% | 100% | 200 |
| 7 | OLMo-2 32B MeditronFO | 의료 | 쌍8 | 4bit | schema | 0.912 [0.887, 0.935] | 7.7% | 9.8% | 97% | 133 |
| 8 | Lingshu 7B | 의료 | 없음 | bf16 | schema | 0.899 [0.874, 0.920] | 8.5% | 11.7% | 99% | 200 |
| 9 | Gemma 3 4B | 기반 | 쌍1 | bf16 | schema | 0.867 [0.842, 0.891] | 13.8% | 12.7% | 1% | 200 |
| 10 | OLMo-2 32B Instruct | 기반 | 쌍8 | 4bit | schema | 0.852 [0.824, 0.880] | 9.9% | 19.3% | 99% | 148 |
| 11 | II-Medical 8B | 의료 | 없음 | bf16 | minimal | 0.848 [0.813, 0.880] | 9.0% | 20.6% | 0% | 200 |
| 12 | Apertus 8B Instruct | 기반 | 쌍7 | bf16 | minimal | 0.812 [0.765, 0.858] | 16.0% | 21.5% | 100% | 200 |
| 13 | MedReason 8B | 의료 | 없음 | bf16 | schema | 0.803 [0.772, 0.832] | 14.6% | 24.3% | 74% | 200 |
| 14 | Meditron3 8B | 의료 | 쌍2 | bf16 | minimal | 0.802 [0.748, 0.848] | 20.0% | 19.6% | 100% | 200 |
| 15 | Apertus 8B MeditronFO | 의료 | 쌍7 | bf16 | schema | 0.742 [0.694, 0.792] | 20.1% | 30.7% | 8% | 200 |
| 16 | MedGemma 1.5 4B | 의료 | 쌍1 | bf16 | minimal | 0.660 [0.593, 0.726] | 29.8% | 37.7% | 0% | 200 |

정밀도가 다른 모델끼리는 가로로 비교하지 않음. 4비트 27B, 32B 는 짝 안에서만 읽음. OLMo-2 두 모델은 문맥 4,096 을 넘는 노트를 채점에서 뺐으므로 노트 수가 200 보다 적음. 짝 없는 모델(Lingshu, II-Medical, MedReason)은 기반을 특정할 수 없어 순위표에만 올림.

### 짝 비교 8쌍. 페어드 부트스트랩 95% CI

| 쌍 | 비교 (의료 − 기반) | 차이 | 95% CI | 판정 | 노트 |
|---|---|---:|---|---|---:|
| 1 (세로) | MedGemma 1.5 4B − Gemma 3 4B | −0.207 | [−0.270, −0.142] | Gemma 3 4B 우세 | 200 |
| 2 (세로) | Meditron3 8B − Llama 3.1 8B Instruct | −0.115 | [−0.171, −0.059] | Llama 3.1 8B Instruct 우세 | 200 |
| 3 (가로) | MedGemma 1.5 4B − Meditron3 8B | −0.142 | [−0.210, −0.069] | Meditron3 8B 우세 | 200 |
| 4 (세로) | Baichuan-M2 32B − Qwen2.5 32B Instruct | +0.006 | [−0.002, +0.016] | 차이 없음 | 200 |
| 5 (세로) | HuatuoGPT-o1 8B − Llama 3.1 8B Instruct | −0.003 | [−0.024, +0.020] | 차이 없음 | 200 |
| 6 (세로) | Gemma 3 27B MeditronFO − Gemma 3 27B Instruct | −0.001 | [−0.025, +0.029] | 차이 없음 | 200 |
| 7 (세로) | Apertus 8B MeditronFO − Apertus 8B Instruct | −0.069 | [−0.127, −0.011] | Apertus 8B Instruct 우세 | 200 |
| 8 (세로) | OLMo-2 32B MeditronFO − OLMo-2 32B Instruct | +0.045 | [+0.024, +0.069] | OLMo-2 32B MeditronFO 우세 | 133 |
<!-- 결과표 끝 -->

## 보고 지침

TRIPOD-LLM(Nature Medicine 2025) 항목별 대응표는 [docs/TRIPOD_LLM.md](docs/TRIPOD_LLM.md).

## 트러블슈팅

실측으로 잡은 결함 18건의 증상, 원인, 수정, 회귀 테스트는 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
에 있음. 번호는 문서들이 가리키는 번호 그대로임.

1. 정답 출처가 애초에 틀림. `prescriptions` 는 퇴원약이 아님
2. 비식별화가 괄호를 반만 지워 용량이 이름에 딸려옴
3. 번호 목록에 복용약이 아닌 것이 섞여 있음
4. 제형을 떼면 다른 약이 망가짐
5. 전 구간 무작위 표본에서만 나온 것 3건
6. 감사 기준이 오탐을 냈고, 고쳤다고 적은 수정이 코드에 없었음\
   6-b. 정답 파서가 실제 약을 의료용품으로 버리고 있었음
7. DUA. 집계 파일도 그냥 올리면 안 됨
8. 감사 조건의 분모가 다른 노트였음
9. 사고하는 모델은 답을 내기 전에 생성 상한에 걸림
10. 사고 안의 초안을 최종 답으로 읽을 수 있었음
11. 읽지 못한 배열을 "JSON 준수, 약 0개"로 세고 있었음
12. OLMo-2 MeditronFO 는 배포된 토크나이저가 기반 모델과 다름
13. MedGemma 는 사고 스위치가 없는데 숨은 표시로 사고를 씀
14. 배치 크기가 섞인 파일 하나가 프롬프트 선택을 뒤집음
15. 실행이 중간에 끊김. 이어받은 파일은 지우고 처음부터 다시 만듦
16. 채점기가 정답 쪽 중복만 그대로 두어 맞힐 수 없는 누락을 만들고 있었음
17. 별칭 규칙이 정답에만 걸리고 예측에는 걸리지 않았음
