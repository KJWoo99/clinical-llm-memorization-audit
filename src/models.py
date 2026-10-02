"""평가 대상 모델 정의와 로컬 로더.

전부 로컬에서 돌림. MIMIC 노트를 외부 API 로 보내면 PhysioNet DUA 위반임.
가중치는 HuggingFace 캐시에 받아 둔 것을 씀.

정밀도는 짝 안에서 같게 둠. 양자화 손해가 모델마다 달라, 하나로 고정하면 결과가
의료 특화 때문인지 양자화 때문인지 갈리지 않음. 들어가는 것은 bf16, 27B 는 bf16 이
약 54GB 라 24GB 에 안 들어가므로 짝의 양쪽을 4비트로 맞춤. 4비트 27B 와 bf16 8B 는
가로로 비교하지 않음.

세로 짝은 같은 기반에서 의료 사전학습만 갈라진 것이어야 함. OpenBioLLM 은 Llama 3
기반이라 Llama 3.1 과 짝지을 수 없음.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    label: str
    family: str        # 짝 비교의 계열: 같은 family 끼리가 세로 비교임
    medical: bool      # 의료 특화 여부
    approx_params_b: float
    precision: str = "bf16"   # "bf16" | "4bit"
    base_of_record: str = ""  # 이 모델이 어디서 갈라졌는지. 짝의 근거
    # 채팅 템플릿에 넘길 인자. 사고(thinking) 스위치가 있는 모델은 여기서 끔.
    # 이 과제는 512토큰 안에 답을 내는 추출 과제이고 짝의 상대는 바로 답함.
    # Baichuan-M2 는 <think> 로 사고를 쓰다 512토큰에서 잘려 답을 못 냄:
    # 그 상태로 채점하면 사고 조각에서 약 3개를 긁어낸 점수가 됨.
    template_kwargs: tuple[tuple[str, object], ...] = ()
    # 생성 상한. 기본 512. 사고가 학습에 박혀 스위치가 없는 모델(HuatuoGPT-o1,
    # MedReason)은 답 앞에 사고를 쓰므로 상한을 늘려야 답이 나옴.
    # 응답 행마다 실제 상한을 기록하므로 어느 모델이 얼마를 받았는지 남음.
    max_new_tokens: int = 512
    # 모델이 학습한 최대 문맥 길이(config.json 의 max_position_embeddings).
    # 입력 + 생성 상한이 이것을 넘는 건은 모델이 학습하지 않은 위치를 읽으므로
    # 채점에서 뺌(`exceeds_context`). OLMo-2 만 4096 이고 나머지는 이 과제의
    # 입력(최대 약 7,800토큰)보다 넉넉함.
    context_window: int = 131072


MODELS: dict[str, ModelSpec] = {
    # ══ 세로 짝: 같은 기반에서 의료 사전학습만 갈라진 것 ═════════════
    #
    # 규모 3단계(4B, 8B, 32B)를 덮음. 8B 두 개는 같은 기반 위 다른 팀이라,
    # 한 팀이 잘못 만든 것인지 방법 자체의 문제인지를 가름.

    # ── 쌍A: 4B. 2026-01 공개된 의료 특화 모델 ──────────────────────────
    #    구글이 명시한 용도가 "비정형 의료 문서에서 구조화 데이터 추출"로,
    #    이 프로젝트의 과제와 정확히 같음.
    "medgemma-4b": ModelSpec(
        key="medgemma-4b", hf_id="google/medgemma-1.5-4b-it",
        label="MedGemma 1.5 4B (의료)", family="gemma-4b", medical=True,
        approx_params_b=4.0, precision="bf16", base_of_record="google/gemma-3-4b-pt",
    ),
    "gemma3-4b": ModelSpec(
        key="gemma3-4b", hf_id="google/gemma-3-4b-it",
        label="Gemma 3 4B (기반)", family="gemma-4b", medical=False,
        approx_params_b=4.0, precision="bf16", base_of_record="google/gemma-3-4b-pt",
    ),

    # ── 쌍B: 32B. HealthBench 오픈소스 1위 ───────────────────────────
    #    제작자가 GPTQ-Int4 판도 배포하지만 원본을 받아 NF4 로 올림. 짝인
    #    Qwen2.5-32B 가 NF4 라, 제작자 판을 쓰면 양자화 방식 차이가 섞임.
    "baichuan-m2-32b": ModelSpec(
        key="baichuan-m2-32b", hf_id="baichuan-inc/Baichuan-M2-32B",
        label="Baichuan-M2 32B (의료)", family="qwen25-32b", medical=True,
        approx_params_b=32.8, precision="4bit", base_of_record="Qwen/Qwen2.5-32B",
        template_kwargs=(("thinking_mode", "off"),),
    ),
    "qwen25-32b": ModelSpec(
        key="qwen25-32b", hf_id="Qwen/Qwen2.5-32B-Instruct",
        label="Qwen2.5 32B Instruct (기반)", family="qwen25-32b", medical=False,
        approx_params_b=32.8, precision="4bit", base_of_record="Qwen/Qwen2.5-32B",
        context_window=32768,
    ),

    # ── MeditronFO 세 짝: 같은 의료화 절차를 서로 다른 기반에 적용한 것 ──
    # EPFL 이 2026-05 에 공개. 같은 파이프라인, 기반만 다름(Gemma 3 27B, Apertus 8B,
    # OLMo-2 32B). 짝 하나로는 팀의 솜씨와 의료 사전학습 효과를 못 가르는데, 같은
    # 절차를 세 기반에 걸면 갈림. 27B, 32B 는 bf16 54~64GB 라 양쪽 다 4비트.
    "gemma3-27b-meditronfo": ModelSpec(
        key="gemma3-27b-meditronfo", hf_id="EPFLiGHT/Gemma-3-27B-MeditronFO",
        label="Gemma 3 27B MeditronFO (의료, EPFL)", family="gemma-27b", medical=True,
        approx_params_b=27.0, precision="4bit", base_of_record="google/gemma-3-27b-it",
    ),
    "gemma3-27b": ModelSpec(
        key="gemma3-27b", hf_id="google/gemma-3-27b-it",
        label="Gemma 3 27B Instruct (기반)", family="gemma-27b", medical=False,
        approx_params_b=27.0, precision="4bit", base_of_record="google/gemma-3-27b-it",
    ),
    "apertus-8b-meditronfo": ModelSpec(
        key="apertus-8b-meditronfo", hf_id="EPFLiGHT/Apertus-8B-MeditronFO",
        label="Apertus 8B MeditronFO (의료, EPFL)", family="apertus-8b", medical=True,
        approx_params_b=8.0, precision="bf16",
        base_of_record="swiss-ai/Apertus-8B-Instruct-2509",
        template_kwargs=(("enable_thinking", False),),
        context_window=65536,
    ),
    "apertus-8b": ModelSpec(
        key="apertus-8b", hf_id="swiss-ai/Apertus-8B-Instruct-2509",
        label="Apertus 8B Instruct (기반)", family="apertus-8b", medical=False,
        approx_params_b=8.0, precision="bf16",
        base_of_record="swiss-ai/Apertus-8B-Instruct-2509",
        template_kwargs=(("enable_thinking", False),),
        context_window=65536,
    ),
    "olmo2-32b-meditronfo": ModelSpec(
        key="olmo2-32b-meditronfo", hf_id="EPFLiGHT/OLMo-2-32B-MeditronFO",
        label="OLMo-2 32B MeditronFO (의료, EPFL)", family="olmo2-32b", medical=True,
        approx_params_b=32.0, precision="4bit",
        base_of_record="allenai/OLMo-2-0325-32B-Instruct",
        context_window=4096,
    ),
    "olmo2-32b": ModelSpec(
        key="olmo2-32b", hf_id="allenai/OLMo-2-0325-32B-Instruct",
        label="OLMo-2 32B Instruct (기반)", family="olmo2-32b", medical=False,
        approx_params_b=32.0, precision="4bit",
        base_of_record="allenai/OLMo-2-0325-32B-Instruct",
        context_window=4096,
    ),

    # ── 쌍C, D: 8B. 같은 기반에 두 팀이 각각 의료 적응을 함 ──────────
    "meditron3-8b": ModelSpec(
        key="meditron3-8b", hf_id="EPFLiGHT/Meditron3-8B",
        label="Meditron3 8B (의료, EPFL)", family="llama31-8b", medical=True,
        approx_params_b=8.0, precision="bf16",
        base_of_record="meta-llama/Llama-3.1-8B-Instruct",
    ),
    "huatuo-o1-8b": ModelSpec(
        key="huatuo-o1-8b", hf_id="FreedomIntelligence/HuatuoGPT-o1-8B",
        label="HuatuoGPT-o1 8B (의료, Huatuo)", family="llama31-8b", medical=True,
        approx_params_b=8.0, precision="bf16",
        base_of_record="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=1536,
    ),
    "llama31-8b": ModelSpec(
        key="llama31-8b", hf_id="meta-llama/Llama-3.1-8B-Instruct",
        label="Llama 3.1 8B Instruct (기반)", family="llama31-8b", medical=False,
        approx_params_b=8.0, precision="bf16",
        base_of_record="meta-llama/Meta-Llama-3.1-8B",
    ),

    # ══ 짝 없는 의료 모델: 기반을 짝지을 수 없어 순위표(실무 권고, 쌍3)에만 올림 ══
    "lingshu-7b": ModelSpec(
        key="lingshu-7b", hf_id="lingshu-medical-mllm/Lingshu-7B",
        label="Lingshu 7B (의료, 알리바바)", family="_unpaired", medical=True,
        approx_params_b=7.0, precision="bf16", base_of_record="(카드 미선언)",
        context_window=128000,
    ),
    "ii-medical-8b": ModelSpec(
        key="ii-medical-8b", hf_id="Intelligent-Internet/II-Medical-8B-1706",
        label="II-Medical 8B (의료, 추론특화)", family="_unpaired", medical=True,
        approx_params_b=8.0, precision="bf16", base_of_record="(카드 미선언)",
        context_window=40960,
        # Qwen3 기반 추론 모델인데 사고 스위치가 없음. 탐침 12건 전부 <think> 로 시작,
        # 끝낸 10건은 653~2,794토큰, 2건은 greedy 반복으로 8,192 까지. 처음 준 4096 에
        # 선택 단계 300건 중 62건이 닿았고(반복 58, 결론 직전 4, 끝낸 최장 4,011)
        # prereg 규칙대로 8192 로 올려 그 행만 다시 생성함.
        max_new_tokens=8192,
    ),
    "medreason-8b": ModelSpec(
        key="medreason-8b", hf_id="UCSC-VLAA/MedReason-8B",
        label="MedReason 8B (의료, UCSC)", family="_unpaired", medical=True,
        approx_params_b=8.0, precision="bf16", base_of_record="(카드 미선언)",
        max_new_tokens=1536,
    ),
}

# 세로 짝 비교에서 제외함. 기반을 짝지을 수 없어 인과를 말할 수 없음.
REFERENCE_ONLY = ("lingshu-7b", "ii-medical-8b", "medreason-8b")

# ── 사전등록된 짝 (docs/prereg.md 3절) ──────────────────────────────
#
# 결과를 보기 전에 고정한 것임. 여기서 빼지 않음. 사전등록의 요점이
# 결과를 본 뒤 분석을 고르지 않는 것인데, 불편한 비교를 나중에 지우면 그 원칙을
# 어기게 됨. 쌍3 은 교란돼 있다는 사실까지 등록 시점에 함께 적어둠.
PAIRS_PREREGISTERED = (
    ("medgemma-4b", "gemma3-4b", "쌍1 세로: Google 의료 특화 효과"),
    ("meditron3-8b", "llama31-8b", "쌍2 세로: EPFL 의료 특화 효과"),
    ("medgemma-4b", "meditron3-8b", "쌍3 가로: 실무 권고 (크기, 계열 교란)"),
)

# ── 나중에 추가한 짝 (탐색적) ───────────────────────────────────────
#
# 1차 결과를 본 뒤에 더한 것임. 사전등록된 것과 같은 지위로 취급하면 안 됨.
# 확증이 아니라 탐색이며, 리포트에서 그렇게 구분해 적음.
PAIRS_EXPLORATORY = (
    ("baichuan-m2-32b", "qwen25-32b", "쌍4 세로: Baichuan-M2 (32B). 현행 최고 성능"),
    ("huatuo-o1-8b", "llama31-8b", "쌍5 세로: HuatuoGPT-o1 (8B). 쌍2 와 같은 기반"),
    ("gemma3-27b-meditronfo", "gemma3-27b", "쌍6 세로: MeditronFO on Gemma 3 27B"),
    ("apertus-8b-meditronfo", "apertus-8b", "쌍7 세로: MeditronFO on Apertus 8B"),
    ("olmo2-32b-meditronfo", "olmo2-32b", "쌍8 세로: MeditronFO on OLMo-2 32B"),
)

# 쌍6~8 은 같은 의료화 절차를 서로 다른 기반에 적용한 것임. 세 짝의 차이가
# 같은 방향이면 효과는 절차에서 오고, 기반마다 갈리면 기반에서 옴. 짝 하나로는
# 구분되지 않던 것이 셋을 나란히 놓으면 갈림.
PAIRS_SAME_PROCEDURE = (
    ("gemma3-27b-meditronfo", "gemma3-27b"),
    ("apertus-8b-meditronfo", "apertus-8b"),
    ("olmo2-32b-meditronfo", "olmo2-32b"),
)

PAIRS = PAIRS_PREREGISTERED + PAIRS_EXPLORATORY


# 결과를 낸 가중치의 HuggingFace 커밋. 내려받기 스크립트가 이 판을 고정해 받고, 로더도 이 판을 찾음.
# 값은 서버 캐시의 스냅샷 이름(모델마다 하나, refs/main 과 같음)임.
REVISIONS: dict[str, str] = {
    "google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
    "baichuan-inc/Baichuan-M2-32B": "360f684d3cd527a4ef986c54071015eb0e6def58",
    "Qwen/Qwen2.5-32B-Instruct": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    "EPFLiGHT/Gemma-3-27B-MeditronFO": "b1154b288a1dea45c79fb24c9a882ecbdb7215bc",
    "google/gemma-3-27b-it": "005ad3404e59d6023443cb575daa05336842228a",
    "EPFLiGHT/Apertus-8B-MeditronFO": "ef2b141da7ccc347c2a13b2518370ba6a8a2b745",
    "swiss-ai/Apertus-8B-Instruct-2509": "b946d40447b2b597999b9c86d44bee0b452c919f",
    "EPFLiGHT/OLMo-2-32B-MeditronFO": "a651dba27da26f1459f714b3b24476fa7c17b165",
    "allenai/OLMo-2-0325-32B-Instruct": "b96024342a77a69aa0dda815c3454a671f477463",
    "EPFLiGHT/Meditron3-8B": "783c241b18b84692689e0336170b345e5732e48e",
    "FreedomIntelligence/HuatuoGPT-o1-8B": "afc8b260e5b3dee9233863cf2de3080f3094442a",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "lingshu-medical-mllm/Lingshu-7B": "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9",
    "Intelligent-Internet/II-Medical-8B-1706": "a364a7cb987287fad5fefd512da3042e464d74f2",
    "UCSC-VLAA/MedReason-8B": "2730640542b2629e26bdc31938da7ee2a85ebd03",
}

def local_kwargs(hf_id: str) -> dict:
    """캐시에서만, 결과를 낸 판으로 불러오는 인자.

    커밋 해시로 내려받으면 huggingface_hub 가 `refs/main` 을 만들지 않음. 그래서 판을 주지 않고
    `local_files_only=True` 로만 부르면 main 을 찾다가 실패함. 판을 함께 넘겨, 새 환경에서 README 순서대로
    받아도 로딩이 멈추지 않게 함.
    """
    return {"local_files_only": True, "revision": REVISIONS[hf_id]}


def load_model(spec: ModelSpec):
    """`spec.precision` 대로 올림.

    torch/transformers 는 여기서만 import 함: 평가, 집계 스크립트는 GPU 없이도
    돌아가야 하기 때문임.
    """
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
    if spec.precision == "4bit":
        # 우리가 직접 양자화함.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif spec.precision == "prequantized":
        # 저장소가 이미 양자화된 가중치를 담고 있음(예: GPTQ-Int4).
        # 여기에 BitsAndBytes 를 또 걸면 안 됨. 설정은 config.json 에 들어 있음.
        #
        # 제작자가 배포한 양자화를 쓰는 이유: 32B 를 24GB 에 올리는 건 빠듯해서
        # 어떤 계층을 어떻게 줄였는지가 실제로 결과를 가름. 직접 양자화하면
        # 그 선택이 우리 것이 되고, 모델 카드가 보고한 성능과 달라질 수 있음.
        pass
    elif spec.precision != "bf16":
        raise ValueError(
            f"모르는 정밀도: {spec.precision!r} (bf16 | 4bit | prequantized)")

    # 캐시에 있는 것만 씀. 네트워크로 나가지 않음.
    #
    # 이유가 둘임.
    #
    # 하나. 게이티드 저장소는 토큰 없이는 401 을 냄. 가중치는 이미 받아 뒀는데
    # transformers 가 `chat_template.jinja` 처럼 캐시에 없는 파일을 새로 받으려다
    # 통째로 실패함. 토큰을 서버에 저장해두는 선택지는 쓰지 않음.
    #
    # 둘. 이 저장소는 MIMIC 노트를 다루므로 추론 중에 바깥으로 나가는 연결이
    # 아예 없어야 함. 그래서 로컬 전용을 코드로 못박아 둠.
    #
    # 캐시에 없는 모델을 부르면 여기서 바로 실패함. 그것이 맞는 동작임:
    # 알리지 않고 내려받아 쓰면 "무엇으로 쟀는가"가 실행할 때마다 달라질 수 있음.
    local = local_kwargs(spec.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, **local)
    # Lingshu 7B 는 Qwen2.5-VL 구조라 AutoModelForCausalLM 이 받지 않음. 글만 넣으므로 영상 모델 클래스로 올려도 같은 언어 모델로
    # 동작함. CausalLM 이 받는 구조는 지금까지와 똑같이 올림.
    config = AutoConfig.from_pretrained(spec.hf_id, **local)
    model_cls = AutoModelForCausalLM
    if type(config) not in AutoModelForCausalLM._model_mapping:
        model_cls = AutoModelForImageTextToText
    model = model_cls.from_pretrained(spec.hf_id, **kwargs, **local)
    model.eval()
    # KV 캐시를 켬. OLMo-2 는 config 에 use_cache=False 가 박혀 있어 토큰마다
    # 프롬프트 전체를 다시 계산함: 같은 크기의 Qwen 이 19 tok/s 일 때 1 tok/s.
    # 기본값을 믿지 않고 여기서 명시함.
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    return tokenizer, model


def exceeds_context(row: dict, spec: ModelSpec) -> bool:
    """이 응답 행이 모델의 문맥 길이를 넘는 조건에서 만들어졌는가.

    넘친 건은 오류 없이 점수가 나빠짐. 짝의 두 모델이 서로 다른 건수에서 넘치면
    (OLMo-2 는 토크나이저가 달라 226 대 153 이었음) 의료 특화 효과와 문맥 초과가
    섞이므로, 채점과 프롬프트 선택 모두에서 이 건을 뺌. 행에 기록된 생성 상한을
    쓰고, 없으면 모델 설정값을 씀.
    """
    budget = row.get("max_new_tokens", spec.max_new_tokens)
    return int(row["n_input_tokens"]) + int(budget) > spec.context_window
