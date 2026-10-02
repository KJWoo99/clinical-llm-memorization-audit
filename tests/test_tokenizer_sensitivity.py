"""쌍8 토크나이저 민감도 비교의 짝짓기와 판정 검증.

비교는 두 토크나이저 모두에서 문맥 안에 드는 행만 같은 행끼리 짝지어야 함.
기반 토크나이저는 입력이 짧아 문맥 초과가 덜 나므로, 각자 행을 쓰면 표본이
달라져 토큰화 효과와 노트 길이 효과가 섞임.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_tokenizer_sensitivity import compare  # noqa: E402

from models import MODELS  # noqa: E402

SPEC = MODELS["olmo2-32b-meditronfo"]
GOLD = {f"n{i}": (["aspirin", "metoprolol"], []) for i in range(30)}


def _row(note: str, variant: str, meds: list[str], n_in: int = 1000) -> dict:
    return {"note_id": note, "condition": "normal", "variant": variant,
            "response": json.dumps(meds), "n_input_tokens": n_in, "max_new_tokens": 512}


def _rows(meds_by_variant: dict[str, list[str]], n_in: int = 1000) -> list[dict]:
    return [_row(n, v, m, n_in) for n in GOLD for v, m in meds_by_variant.items()]


def test_identical_responses_do_not_trigger() -> None:
    rows = _rows({"minimal": ["aspirin"], "schema": ["aspirin", "metoprolol"]})
    res = compare(rows, [dict(r) for r in rows], SPEC, GOLD, {})
    assert res["triggers_followup"] is False
    for x in res["by_variant"].values():
        assert x["diff_base_minus_shipped"] == 0
        assert x["identical_responses"] == x["n_rows"] == 30


def test_large_consistent_gain_triggers() -> None:
    shipped = _rows({"schema": ["aspirin"]})
    base = _rows({"schema": ["aspirin", "metoprolol"]})
    res = compare(shipped, base, SPEC, GOLD, {})
    x = res["by_variant"]["schema"]
    assert x["diff_base_minus_shipped"] > 0.02 and x["ci_low"] > 0
    assert res["triggers_followup"] is True


def test_changed_selection_triggers_even_when_small() -> None:
    # 배포는 minimal 이 1등, 기반은 schema 가 1등: 선택이 바뀌면 후속 조치 대상임
    shipped = _rows({"minimal": ["aspirin", "metoprolol"], "schema": ["aspirin"]})
    base = _rows({"minimal": ["aspirin"], "schema": ["aspirin", "metoprolol"]})
    res = compare(shipped, base, SPEC, GOLD, {})
    assert res["selected_variant_shipped"] == "minimal"
    assert res["selected_variant_base_tokenizer"] == "schema"
    assert res["triggers_followup"] is True


def test_final_pair8_is_compared_on_the_same_notes_both_ways() -> None:
    # 배포 토크나이저에서만 문맥을 넘는 노트 5개는 두 방식 모두에서 빠져야 함.
    # 한쪽만 빼면 두 짝 차이가 서로 다른 표본 위에서 계산됨.
    from check_tokenizer_sensitivity import compare_final

    def fin(meds, n_in_first5, key):
        return [{**_row(n, "schema", meds, n_in_first5 if int(n[1:]) < 5 else 1000),
                 "split": "main", "model": key} for n in GOLD]

    meta = {(n, "normal"): {"gold": g, "neutral": ne} for n, (g, ne) in GOLD.items()}
    shipped = fin(["aspirin"], 3800, "olmo2-32b-meditronfo")
    base = fin(["aspirin", "metoprolol"], 3500, "olmo2-32b-meditronfo")
    partner = fin(["aspirin"], 3500, "olmo2-32b")
    res = compare_final(shipped, base, partner, SPEC, MODELS["olmo2-32b"], meta, {})
    assert res["n_notes_common"] == 25
    assert res["n_notes_within_context"] == {"shipped": 25, "base_tokenizer": 30, "partner": 30}
    assert res["pair8_shipped_minus_partner"]["diff"] == 0
    assert res["pair8_base_tokenizer_minus_partner"]["diff"] > 0
    assert res["base_tokenizer_minus_shipped"]["significant"]


def test_only_rows_within_context_on_both_sides_are_paired() -> None:
    # 배포 토크나이저에서만 문맥을 넘는 노트 10개는 짝 비교에서 빠져야 함
    shipped = [_row(n, "schema", ["aspirin"], 3800 if int(n[1:]) < 10 else 1000) for n in GOLD]
    base = [_row(n, "schema", ["aspirin"], 3500 if int(n[1:]) < 10 else 900) for n in GOLD]
    res = compare(shipped, base, SPEC, GOLD, {})
    assert res["n_rows_common"] == 30
    assert res["n_excluded_context_shipped"] == 10
    assert res["n_excluded_context_base_tokenizer"] == 0
    assert res["by_variant"]["schema"]["n_rows"] == 20
