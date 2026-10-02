"""모델 출력에서 약물 목록을 꺼내는 파서.

모델은 지시해도 순수 JSON 만 내놓지 않음. 펜스로 감싸거나, 앞에 문장을 붙이거나,
배열 대신 객체를 주거나, 번호 목록으로 답함. 전부 실패로 치면 형식 준수와 추출
성능이 섞이므로 두 값을 나눠 냄.
  - `strict_json`: 지시대로 순수 JSON 배열을 냈는가 (형식 준수율)
  - `medications`: 형식이 어긋나도 건져낸 약물 목록 (성능)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)
_NUMBERED = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])\s*(.+)$", re.MULTILINE)

# 배열 대신 객체를 줄 때 흔히 쓰는 키
_LIST_KEYS = ("medications", "discharge_medications", "drugs", "medication_list", "items")


@dataclass
class ParsedOutput:
    medications: list[str] = field(default_factory=list)
    strict_json: bool = False
    recovered_by: str = "none"  # 어떤 경로로 건졌는지: 리포트에 분포를 실음


def _coerce_list(obj: object) -> list[str] | None:
    if isinstance(obj, list):
        out = []
        for x in obj:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                # [{"name": "Aspirin", "dose": "81 mg"}] 형태
                for key in ("name", "medication", "drug"):
                    if isinstance(x.get(key), str):
                        out.append(x[key])
                        break
        # 원소가 있는데 하나도 못 읽었으면 성공이 아님. [["약", "용량"]] 이나
        # [{"약이름": "용량"}] 을 빈 목록으로 돌려주면 "지시대로 JSON 을 냈고 약은
        # 0개"로 채점돼, 형식 실패가 형식 준수로 잘못 집계됨(트러블슈팅 11).
        if obj and not out:
            return None
        return out
    if isinstance(obj, dict):
        for key in _LIST_KEYS:
            if key in obj:
                return _coerce_list(obj[key])
    return None


def _unreadable_list(obj: object) -> bool:
    """JSON 으로는 읽혔지만 약 목록 자리의 배열을 해석하지 못한 경우."""
    if isinstance(obj, dict):
        obj = next((obj[k] for k in _LIST_KEYS if k in obj), None)
    return isinstance(obj, list) and bool(obj)


# 우리가 템플릿으로 넣은 빈 사고 블록의 닫힘 표시. 사고를 끄면 템플릿이
# '<think>' 와 빈 줄을 넣고 모델이 '</think>' 로 닫은 뒤 답을 시작함. 이
# 꼬리표는 모델이 지시를 안 지킨 것이 아니라 우리 스위치의 흔적이므로 형식
# 판정 전에 걷어냄. 모델이 스스로 사고를 쓴 경우(<think>…</think>)는
# 그대로 둠: 그것은 지시 이행의 일부임.
_EMPTY_THINK_TAIL = re.compile(r"^\s*</think>\s*")

# 모델이 스스로 쓴 사고의 시작과 끝. HuatuoGPT-o1 은 "## Thinking" 뒤에
# "## Final Response", MedReason 은 "## Final Answer" 로 답을 따로 쓰고, Qwen3
# 계열(II-Medical)은 <think>…</think> 를 씀. 사고 모델을 더하면 그 모델의
# 실제 응답에서 끝 표시를 확인하고 여기에 넣어야 함.
#
# MedGemma 1.5 는 템플릿에 사고 스위치가 없는데도 응답 일부를 "<unused94>thought" 로
# 시작해 사고를 쓰고 "<unused95>" 뒤에 답을 씀(Gemma 의 예약 토큰. 최종 600건 중 240건.
# 정상 노트보다 섞인, 비운 노트에서 잦음). 이 표시를 모르면 사고 안의 초안 목록을 답으로 읽음
# (트러블슈팅 13). 기반 Gemma 3 4B 는 이 표시를 한 번도 쓰지 않음.
_REASONING_START = re.compile(r"^(?:<think>|#{1,3}\s*Thinking\b|<unused94>thought\b)", re.IGNORECASE)
_REASONING_END = re.compile(r"</think>|^#{1,3}\s*Final (?:Response|Answer)\s*$|<unused95>",
                            re.IGNORECASE | re.MULTILINE)


def _answer_part(text: str) -> tuple[str | None, bool]:
    """사고를 쓴 응답이면 사고 뒤의 답만 돌려줌. (답, 사고가 있었는가)

    사고 안에도 초안 목록이 들어 있음("so far: [...]"). 본문 배열 탐색은 앞에서부터
    찾으므로 사고를 남겨 두면 최종 답이 아니라 초안을 채점함. 사고 도중에 상한에
    걸려 끝 표시가 없으면 답을 내지 않은 것이라 None 임: 사고 조각에서 약을
    긁어내면 답하지 않은 모델이 점수를 받음.
    """
    if not _REASONING_START.match(text):
        return text, False
    ends = list(_REASONING_END.finditer(text))
    if not ends:
        return None, True
    return text[ends[-1].end():].strip(), True


def parse_model_output(raw: str) -> ParsedOutput:
    text = _EMPTY_THINK_TAIL.sub("", raw or "", count=1).strip()
    if not text:
        return ParsedOutput()
    answer, reasoned = _answer_part(text)
    if answer is None:
        return ParsedOutput(recovered_by="failed")
    parsed = _parse_answer(answer)
    if reasoned and parsed.recovered_by == "strict":
        # 답 부분은 순수 JSON 이지만 응답 전체가 JSON 만은 아님. 지시 준수로 세지 않음.
        return ParsedOutput(medications=parsed.medications, recovered_by="after_reasoning")
    return parsed


def _parse_answer(text: str) -> ParsedOutput:
    if not text:
        return ParsedOutput()

    # 1) 지시대로 순수 JSON
    try:
        obj = json.loads(text)
        items = _coerce_list(obj)
        if items is not None:
            return ParsedOutput(medications=items, strict_json=True, recovered_by="strict")
        if _unreadable_list(obj):
            # 읽을 수 없는 배열임. 아래 3) 로 넘기면 [["A", "10 mg"], ["B", "5 mg"]]
            # 에서 두 번째 쌍을 약 목록으로 오인함.
            return ParsedOutput(recovered_by="failed")
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) 마크다운 펜스 안의 JSON
    fence = _FENCE.search(text)
    if fence:
        try:
            obj = json.loads(fence.group(1).strip())
            items = _coerce_list(obj)
            if items is not None:
                return ParsedOutput(medications=items, recovered_by="fence")
            if _unreadable_list(obj):
                return ParsedOutput(recovered_by="failed")
        except (json.JSONDecodeError, ValueError):
            pass

    # 3) 본문 어딘가에 있는 배열
    for m in _ARRAY.finditer(text):
        try:
            items = _coerce_list(json.loads(m.group(0)))
            if items is not None:
                return ParsedOutput(medications=items, recovered_by="embedded_array")
        except (json.JSONDecodeError, ValueError):
            continue

    # 4) 번호, 불릿 목록으로 답한 경우
    bullets = [b.strip().strip('",') for b in _NUMBERED.findall(text)]
    bullets = [b for b in bullets if b]
    if bullets:
        return ParsedOutput(medications=bullets, recovered_by="bullets")

    return ParsedOutput(recovered_by="failed")
