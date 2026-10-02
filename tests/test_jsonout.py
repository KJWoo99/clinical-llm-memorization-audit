"""모델 출력 파서 검증.

여기서 쓰는 형태는 전부 LLM 이 실제로 자주 내놓는 것들임. 형식이 어긋난
출력을 무조건 실패로 처리하면 '형식 준수율'과 '추출 성능'이 뒤섞이므로,
strict_json 플래그와 건져낸 목록을 따로 확인함.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jsonout import parse_model_output


class TestStrict:
    def test_pure_json_array(self):
        r = parse_model_output('["Furosemide", "Aspirin"]')
        assert r.medications == ["Furosemide", "Aspirin"]
        assert r.strict_json is True
        assert r.recovered_by == "strict"

    def test_empty_array_is_valid_not_a_failure(self):
        r = parse_model_output("[]")
        assert r.medications == []
        assert r.strict_json is True


class TestRecovery:
    def test_markdown_fence(self):
        r = parse_model_output('```json\n["Furosemide", "Aspirin"]\n```')
        assert r.medications == ["Furosemide", "Aspirin"]
        assert r.strict_json is False
        assert r.recovered_by == "fence"

    def test_prose_before_array(self):
        r = parse_model_output('Here are the discharge medications:\n["Furosemide"]')
        assert r.medications == ["Furosemide"]
        assert r.strict_json is False

    def test_object_with_medications_key(self):
        r = parse_model_output('{"medications": ["Furosemide", "Aspirin"]}')
        assert r.medications == ["Furosemide", "Aspirin"]
        assert r.strict_json is True

    def test_array_of_objects_with_name_field(self):
        r = parse_model_output('[{"name": "Furosemide", "dose": "40 mg"}]')
        assert r.medications == ["Furosemide"]

    def test_unreadable_array_is_failure_not_empty_success(self):
        # 원소가 있는데 하나도 못 읽은 배열을 "JSON 준수, 약 0개"로 세면 안 됨.
        # 뒤의 배열 탐색으로 넘겨도 안 됨: 두 번째 쌍 ["B", "5 mg"] 을 약으로 읽음.
        for raw in ('[["Allopurinol", "100 mg Tablet"], ["Aspirin", "325 mg Tablet"]]',
                    '[{"valsartan": "40 mg"}, {"omeprazole": "20 mg"}]',
                    '```json\n[["Aspirin", "81 mg"], ["Senna", "8.6 mg"]]\n```',
                    '{"medications": [["Aspirin", "81 mg"], ["Senna", "8.6 mg"]]}'):
            r = parse_model_output(raw)
            assert r.recovered_by == "failed", raw
            assert r.medications == [] and r.strict_json is False

    def test_empty_array_is_still_a_valid_answer(self):
        r = parse_model_output("[]")
        assert r.medications == [] and r.recovered_by == "strict"

    def test_numbered_list_fallback(self):
        r = parse_model_output("1. Furosemide\n2. Aspirin\n3. Senna")
        assert r.medications == ["Furosemide", "Aspirin", "Senna"]
        assert r.recovered_by == "bullets"

    def test_bullet_list_fallback(self):
        r = parse_model_output("- Furosemide\n- Aspirin")
        assert r.medications == ["Furosemide", "Aspirin"]


class TestFailures:
    def test_empty_output(self):
        r = parse_model_output("")
        assert r.medications == []
        assert r.strict_json is False

    def test_unparseable_prose(self):
        r = parse_model_output("I cannot determine the medications from this note.")
        assert r.medications == []
        assert r.recovered_by == "failed"

    def test_none_input_does_not_crash(self):
        r = parse_model_output(None)
        assert r.medications == []


class TestOrdering:
    def test_strict_path_preferred_over_bullets(self):
        """번호 목록과 JSON 이 함께 있으면 JSON 을 택해야 함."""
        r = parse_model_output('Here:\n1. ignore me\n["Furosemide"]')
        assert r.medications == ["Furosemide"]


def test_empty_think_tail_from_template_is_not_a_format_violation():
    """템플릿이 넣은 빈 사고 블록의 닫힘 표시는 형식 판정에서 뺌.

    Baichuan-M2 의 사고를 끄면 응답이 '</think>' 와 빈 줄로 시작함. 그 꼬리표는
    우리 스위치의 흔적이지 모델이 지시를 어긴 것이 아님. 걷어내지 않으면 순수
    JSON 을 낸 모델이 형식 위반으로 잡힘.
    """
    p = parse_model_output('</think>' + chr(10) * 2 + '["Acetaminophen", "Albuterol"]')
    assert p.strict_json and p.recovered_by == "strict"
    assert p.medications == ["Acetaminophen", "Albuterol"]
    # 모델이 스스로 쓴 사고 블록은 그대로 둠
    q = parse_model_output('<think>' + chr(10) + 'some reasoning' + chr(10) + '</think>' + chr(10) + '["A"]')
    assert not q.strict_json


def test_reasoning_draft_is_not_scored_as_the_answer():
    """사고 안의 초안 목록이 아니라 사고 뒤의 최종 답을 채점함."""
    nl = chr(10)
    huatuo = ("## Thinking" + nl * 2 + 'so far: ["Draft", "Wrong"]' + nl * 2
              + "## Final Response" + nl * 2 + 'Here is the list:' + nl * 2 + '["Aspirin", "Senna"]')
    r = parse_model_output(huatuo)
    assert r.medications == ["Aspirin", "Senna"] and r.recovered_by == "embedded_array"
    medreason = ("## Thinking" + nl * 2 + "### Conclusion:" + nl + 'draft ["Wrong"]' + nl * 2
                 + "## Final Answer" + nl * 2 + '["Aspirin", "Senna"]')
    m = parse_model_output(medreason)
    assert m.medications == ["Aspirin", "Senna"] and m.recovered_by == "after_reasoning"
    q = parse_model_output("<think>" + nl + '["Draft"]' + nl + "</think>" + nl + '["Aspirin"]')
    assert q.medications == ["Aspirin"] and q.recovered_by == "after_reasoning"
    assert not q.strict_json


def test_reasoning_cut_off_before_answer_is_a_failure():
    """상한에 걸려 사고가 끝나지 않았으면 사고 조각에서 약을 긁어내지 않음."""
    nl = chr(10)
    for raw in ("<think>" + nl + 'maybe ["Aspirin", "Senna"] and then',
                "## Thinking" + nl + "1. Aspirin" + nl + "2. Senna",
                # MedGemma 1.5 의 숨은 사고 표시. 잘리면 불릿 초안을 읽지 않음.
                "<unused94>thought" + nl + "Medications so far:" + nl + "*   Aspirin" + nl + "*   Senna"):
        r = parse_model_output(raw)
        assert r.recovered_by == "failed" and r.medications == []


def test_medgemma_hidden_thought_reads_answer_after_unused95():
    """<unused94>thought … <unused95> 뒤의 답만 읽음. 사고 안의 펜스 초안은 무시함."""
    nl = chr(10)
    raw = ("<unused94>thought" + nl + "draft:" + nl + "```json" + nl + '["Wrong"]' + nl + "```" + nl
           + "<unused95>```json" + nl + '["Aspirin", "Senna"]' + nl + "```")
    r = parse_model_output(raw)
    assert r.medications == ["Aspirin", "Senna"] and r.recovered_by == "fence"
    plain = "<unused94>thought" + nl + "ok" + nl + "<unused95>" + '["Aspirin"]'
    p = parse_model_output(plain)
    assert p.medications == ["Aspirin"] and p.recovered_by == "after_reasoning" and not p.strict_json
