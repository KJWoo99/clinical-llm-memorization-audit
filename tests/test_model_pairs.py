"""짝 정의가 비교의 전제를 지키는지 봄.

세로 비교는 "같은 기반에서 의료 사전학습만 갈라진 두 모델"이라는 전제 위에서만
성립함. 전제가 깨진 짝은 오류를 내지 않음. 그냥 틀린 결론을 냄:
의료 특화 효과라고 적은 차이가 사실은 기반 버전 차이거나 정밀도 차이가 됨.

예: OpenBioLLM 은 Llama 3 기반이라 Llama 3.1 과 짝지으면 거기서 나온 차이는 의료 특화가 아니라
3 과 3.1 의 차이임. 그래서 그 짝은 두지 않음.

여기서 보는 것은 셋임.

1. 짝에 적힌 키가 실제로 정의돼 있는가 (오타는 당장은 문제없다가 나중에 KeyError 로 터짐)
2. 세로 짝의 두 모델이 같은 계열인가
3. 세로 짝의 두 모델이 같은 정밀도인가: 한쪽만 4비트면 그 손해가 결론에 섞임
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import (  # noqa: E402
    MODELS,
    PAIRS,
    PAIRS_PREREGISTERED,
    PAIRS_SAME_PROCEDURE,
    REFERENCE_ONLY,
)

VERTICAL = [(a, b, why) for a, b, why in PAIRS if "세로" in why]


@pytest.mark.parametrize("a,b,why", PAIRS)
def test_pair_members_are_defined(a, b, why):
    for key in (a, b):
        assert key in MODELS, f"{why}: '{key}' 가 MODELS 에 없다"


@pytest.mark.parametrize("a,b,why", VERTICAL)
def test_vertical_pairs_share_family(a, b, why):
    """세로 짝은 같은 계열이어야 함. 계열이 다르면 가로 비교임."""
    fa, fb = MODELS[a].family, MODELS[b].family
    assert fa == fb, f"{why}: 계열이 다르다 ({a}={fa}, {b}={fb})"


@pytest.mark.parametrize("a,b,why", VERTICAL)
def test_vertical_pairs_share_precision(a, b, why):
    """세로 짝은 정밀도가 같아야 함.

    한쪽만 4비트로 재면 그 차이가 의료 특화 효과에 섞임. 무엇이 원인인지
    말할 수 없는 비교가 됨. 제작자가 배포한 사전양자화 가중치는 4비트와
    같은 자리로 봄: 둘 다 4비트 급이라는 뜻임.
    """
    def bucket(p: str) -> str:
        return "4bit" if p in ("4bit", "prequantized") else p

    pa, pb = bucket(MODELS[a].precision), bucket(MODELS[b].precision)
    assert pa == pb, (
        f"{why}: 정밀도가 다르다 ({a}={MODELS[a].precision}, {b}={MODELS[b].precision})")


@pytest.mark.parametrize("a,b,why", VERTICAL)
def test_vertical_pairs_contrast_medical_flag(a, b, why):
    """세로 짝은 한쪽만 의료 특화여야 함. 둘 다 의료면 무엇과 비교하는지가 없음."""
    assert MODELS[a].medical != MODELS[b].medical, (
        f"{why}: medical 표시가 둘 다 {MODELS[a].medical} 다")


def test_preregistered_pairs_are_still_present():
    """사전등록된 짝은 빼지 않음.

    결과를 본 뒤 불편한 비교를 지우면 사전등록의 의미가 없어짐.
    """
    for a, b, why in PAIRS_PREREGISTERED:
        assert (a, b, why) in PAIRS, f"사전등록 짝이 PAIRS 에서 빠졌다: {why}"


def test_unpaired_models_are_not_used_in_pairs():
    """짝 없는 모델은 짝 비교에 끼면 안 됨. 기반을 모르므로 인과를 말할 수 없음."""
    used = {k for a, b, _ in PAIRS for k in (a, b)}
    overlap = sorted(set(REFERENCE_ONLY) & used)
    assert not overlap, f"참고용 모델이 짝에 들어가 있다: {overlap}"


def test_same_procedure_group_shares_one_procedure():
    """같은 절차를 서로 다른 기반에 적용한 묶음이어야 함.

    이 묶음의 요점은 기반이 서로 달라야 한다는 것임. 같은 기반이 둘 있으면
    "절차의 효과"와 "그 기반에서의 효과"가 다시 섞임.
    """
    families = [MODELS[a].family for a, _ in PAIRS_SAME_PROCEDURE]
    assert len(set(families)) == len(families), f"기반이 겹친다: {families}"
    for a, b in PAIRS_SAME_PROCEDURE:
        assert MODELS[a].medical and not MODELS[b].medical
        assert "MeditronFO" in MODELS[a].hf_id, (
            f"{a} 는 같은 절차 묶음에 있는데 MeditronFO 가 아니다")


def test_reasoning_models_get_a_direct_answer_path():
    """사고(thinking)를 쓰는 모델은 끄거나, 못 끄면 답이 나올 만큼의 상한을 받음.

    Baichuan-M2 처럼 <think> 로 사고를 쓰다 512토큰에서 잘리면 답을 못 내고, 파서가
    사고 조각에서 약 3개씩을 긁어냄(다른 모델은 9~10개).
    스위치가 있는 모델은 끄고(짝의 상대가 바로 답하므로 조건을 맞추는 쪽),
    사고가 학습에 박혀 스위치가 없는 모델은 상한을 늘림. 어느 쪽이든 응답
    행에 그 조건이 기록됨(run_phase3_infer.py).
    """
    kw = {k: dict(v.template_kwargs) for k, v in MODELS.items()}
    assert kw["baichuan-m2-32b"] == {"thinking_mode": "off"}
    assert kw["apertus-8b"] == {"enable_thinking": False}
    assert kw["apertus-8b-meditronfo"] == {"enable_thinking": False}
    for key in ("huatuo-o1-8b", "medreason-8b", "ii-medical-8b"):
        assert MODELS[key].max_new_tokens > 512, f"{key} 는 사고를 끌 수 없어 상한이 더 커야 한다"
    for key, spec in MODELS.items():
        assert spec.max_new_tokens >= 512, key


def test_context_window_exclusion_rule():
    """문맥 길이를 넘은 조건의 응답은 채점, 선택에서 빠짐.

    OLMo-2 32B 두 모델은 문맥이 4096 이라, 긴 퇴원기록에서 입력+생성 상한이
    한도를 넘음(900건 중 226건, 153건). 넘친 건은 오류 없이
    점수가 나빠지고, 짝의 두 모델이 서로 다른 건수에서 넘치므로 그대로 채점하면
    의료 특화 효과와 문맥 초과가 섞임.
    """
    from models import exceeds_context
    olmo = MODELS["olmo2-32b"]
    assert olmo.context_window == 4096
    assert MODELS["olmo2-32b-meditronfo"].context_window == 4096
    assert not exceeds_context({"n_input_tokens": 3584, "max_new_tokens": 512}, olmo)
    assert exceeds_context({"n_input_tokens": 3585, "max_new_tokens": 512}, olmo)
    # 행에 상한이 없으면 모델 설정값을 씀
    assert exceeds_context({"n_input_tokens": 3600}, olmo)
    # 문맥이 넉넉한 모델은 이 과제의 최대 입력(약 7,800토큰)에서도 넘지 않음
    assert not exceeds_context({"n_input_tokens": 7819, "max_new_tokens": 512}, MODELS["apertus-8b"])
    for key, spec in MODELS.items():
        assert spec.context_window >= 4096, key
