"""프롬프트 변형과 감사 조건 생성.

프롬프트를 여러 개 두는 이유

의료 특화 LLM을 기반 모델과 비교한 선행 연구(arXiv 2411.08870)가 지적한
약점 중 하나가 "모델마다 잘 먹히는 프롬프트가 다른데 하나로 고정해 비교했다"
는 것임. 한 프롬프트만 쓰면 그 프롬프트에 우연히 맞는 모델이 이김.
여기서는 변형 3종을 두고 모델마다 개발 표본에서 가장 좋은 것을 고른 뒤
그 선택으로 본 평가를 돌림. 어떤 변형이 뽑혔는지는 리포트에 함께 실음.

감사 조건

정상 조건의 점수만으로는 모델이 노트를 읽었는지 알 수 없음. 노트에서 답을
지우거나(no_section), 아예 비우거나(empty), 다른 환자 것으로 바꿔치거나
(other_patient), 문장 순서를 흩어(shuffled) 점수가 어떻게 변하는지를 봐야
함. 정상과 감사 조건의 점수 차이가 곧 "노트를 읽은 정도"다.
"""
from __future__ import annotations

import random
import re

# 노트 첫머리의 인적사항 블록. empty 조건은 여기까지만 남김.
_HEADER_END = re.compile(r"^\s*(Service|Allergies|Chief Complaint)\s*:", re.MULTILINE)

# 사전등록한 세 변형. 모델별 선택은 이 셋 안에서만 함.
PROMPT_VARIANTS = ("minimal", "schema", "grounded")

# 나중에 더한 변형. 절제실험에서만 쓰고 사전등록된 선택에는 넣지 않음.
#
# 사전등록의 요점은 결과를 본 뒤 유리한 설정을 고르지 않는 것임. 그래서
# 선택 대상(PROMPT_VARIANTS)은 그대로 두고, 새 변형은 별도 목록에 둔 뒤
# 리포트에서 "탐색적"으로 구분해 적음. 여기서 좋은 것이 나오면 그것은
# 결론이 아니라 다음 사전등록의 후보임.
PROMPT_VARIANTS_EXPLORATORY = ("fewshot", "terse")

ALL_PROMPT_VARIANTS = PROMPT_VARIANTS + PROMPT_VARIANTS_EXPLORATORY

_INSTRUCTIONS = {
    # (a) 최소 지시: 군더더기 없이 과제만 말함
    "minimal": (
        "List the discharge medications from this discharge summary.\n"
        'Answer with a JSON array of medication names only, like ["drug a", "drug b"].\n'
    ),
    # (b) 스키마와 예시를 줌: 형식 준수율이 오르는 대신 예시에 끌려갈 수 있음
    "schema": (
        "Extract the discharge medication list from this discharge summary.\n\n"
        "Output format: a JSON array of strings. Each string is one medication name "
        "without dose, route, or frequency.\n"
        'Example output: ["Furosemide", "Aspirin", "Metoprolol Tartrate"]\n\n'
        "Output only the JSON array. No explanation, no markdown fences.\n"
    ),
    # (c) 근거 제약을 명시: 노트에 없는 약을 지어내지 말라고 못박음
    "grounded": (
        "Extract the discharge medication list from this discharge summary.\n\n"
        "Rules:\n"
        "- Include only medications that appear in the note itself.\n"
        "- Do not add medications that are commonly prescribed but absent from this note.\n"
        "- If the note contains no discharge medications, output an empty array [].\n"
        "- Give the medication name only, without dose, route, or frequency.\n\n"
        "Output only a JSON array of strings. No explanation, no markdown fences.\n"
    ),
    # (d) 예시 두 건을 실제 형태로 보여줌: 탐색적
    #     schema 의 예시는 약 이름 세 개짜리 한 줄이라 "형식"만 알려줌.
    #     여기서는 짧은 노트 조각과 그에 대한 답을 함께 보여줌. 형식이 아니라
    #     무엇을 뽑아야 하는지를 보여주려는 것임. 퇴원약 목록과 입원 중
    #     투약을 섞어 쓰는 실수가 이것으로 줄어드는지 봄.
    "fewshot": (
        "Extract the discharge medication list from this discharge summary.\n\n"
        "Example 1\n"
        "Note: ... Discharge Medications: 1. Aspirin 81 mg PO daily "
        "2. Metoprolol Tartrate 25 mg PO BID ...\n"
        'Answer: ["Aspirin", "Metoprolol Tartrate"]\n\n'
        "Example 2\n"
        "Note: ... Medications on Admission: Lisinopril 10 mg ... "
        "Discharge Medications: None. Patient expired. ...\n"
        "Answer: []\n\n"
        "Give the medication name only, without dose, route, or frequency.\n"
        "Output only a JSON array of strings. No explanation, no markdown fences.\n"
    ),
    # (e) 한 문장: 탐색적
    #     minimal 도 두 문장임. 지시를 더 줄이면 형식 준수가 떨어지는지,
    #     아니면 지시문이 애초에 별 영향이 없는지 가름.
    "terse": (
        'Discharge medications as a JSON array of names:\n'
    ),
}


def build_prompt(note_text: str, variant: str) -> str:
    if variant not in _INSTRUCTIONS:
        msg = f"알 수 없는 프롬프트 변형: {variant!r} (가능: {ALL_PROMPT_VARIANTS})"
        raise ValueError(msg)
    return f"{_INSTRUCTIONS[variant]}\n--- DISCHARGE SUMMARY ---\n{note_text}\n--- END ---\n"


def header_only(text: str) -> str:
    """인적사항 블록만 남김: 임상 내용이 없으면 모델은 사전 지식만으로 답함."""
    m = _HEADER_END.search(text)
    return text[: m.start()] if m else text[:400]


def shuffle_sentences(text: str, *, seed: int) -> str:
    """문장 단위로 순서를 흩음.

    노트의 구조(어느 절에 무엇이 있는지)를 깨뜨려도 점수가 유지되면, 모델이
    구조를 읽은 것이 아니라 약 이름을 긁어모은 것에 가까움.
    """
    sentences = re.split(r"(?<=[.\n])\s+", text)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    return " ".join(sentences)
