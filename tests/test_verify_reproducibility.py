"""재현 검증의 배치 묶음 복원과 표본 선택 검증.

배치가 달라지면 greedy 출력도 달라질 수 있으므로, 다시 만들 때는 원래 묶음을 그대로
써야 함. 묶음을 잘못 복원하면 손상이 없는데도 불일치로 보임.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_reproducibility import batch_groups, pick_groups, run_settings  # noqa: E402

from models import MODELS  # noqa: E402

FINAL_COMPOSITION = {"normal": 200, "no_section": 100, "empty": 100,
                     "other_patient": 100, "shuffled": 100}


def _rows(batch: int) -> list[dict]:
    """추론 스크립트가 쓰는 순서 그대로: 정상 200행 뒤 감사 조건 4종이 노트마다 번갈아 옴.

    배치마다 len(batch) 를 남기고 마지막은 자투리임.
    """
    flat = [{"note_id": f"n{i}", "condition": "normal", "variant": "minimal"} for i in range(200)]
    audit = [c for c in FINAL_COMPOSITION if c != "normal"]
    flat += [{"note_id": f"a{i}", "condition": c, "variant": "minimal"}
             for i in range(100) for c in audit]
    for i in range(0, len(flat), batch):
        chunk = flat[i:i + batch]
        for r in chunk:
            r["batch_size"] = len(chunk)
    return flat


def test_groups_follow_written_batches():
    groups = batch_groups(_rows(8))
    # 600 = 8 x 75, 자투리 없음
    assert [len(g) for g in groups] == [8] * 75


def test_trailing_partial_batch_is_its_own_group():
    rows = _rows(8)[:300]           # 300 = 8 x 37 + 4
    for r in rows[-4:]:
        r["batch_size"] = 4
    groups = batch_groups(rows)
    assert [len(g) for g in groups[-2:]] == [8, 4]
    assert sum(len(g) for g in groups) == 300


def test_resume_boundary_splits_broken_group():
    rows = _rows(2)
    # 이어받기가 배치 2 짝의 한가운데서 배치 1 로 바뀐 경우
    rows[5]["batch_size"] = 1
    groups = batch_groups(rows)
    assert sum(len(g) for g in groups) == len(rows)
    assert all(len({r["batch_size"] for r in g}) == 1 for g in groups)


def test_pick_covers_interleaved_conditions():
    # 간격 뽑기였다면 4주기에 걸려 한 조건만 나왔던 배치임.
    picked = pick_groups(batch_groups(_rows(1)), sample=12)
    conds = Counter(r["condition"] for g in picked for r in g)
    assert set(conds) == set(FINAL_COMPOSITION)


def test_pick_covers_interleaved_variants():
    rows = [{"note_id": f"n{i}", "condition": "normal", "variant": v, "batch_size": 1}
            for i in range(100) for v in ("minimal", "schema", "grounded")]
    picked = pick_groups(batch_groups(rows), sample=12)
    assert Counter(g[0]["variant"] for g in picked) == {"minimal": 4, "schema": 4, "grounded": 4}


def test_big_batches_cover_every_condition():
    picked = pick_groups(batch_groups(_rows(8)), sample=12)
    assert len(picked) >= 5
    conds = {r["condition"] for g in picked for r in g}
    assert conds == set(FINAL_COMPOSITION)


def test_old_rows_without_settings_use_current_spec():
    # 먼저 만든 행은 기록 칸이 없음. 그 모델들은 기본값이라 현재 설정과 같음.
    spec = MODELS["qwen25-32b"]
    s = run_settings({"note_id": "n", "condition": "normal", "variant": "grounded"}, spec)
    assert s.max_new_tokens == 512 and s.template_kwargs == ()


def test_recorded_settings_win_over_current_spec():
    spec = MODELS["ii-medical-8b"]            # 지금 8192
    s = run_settings({"max_new_tokens": 4096, "template_kwargs": {}}, spec)
    assert s.max_new_tokens == 4096 and s.template_kwargs == ()


def test_recorded_template_kwargs_are_kept():
    spec = MODELS["baichuan-m2-32b"]
    s = run_settings({"max_new_tokens": 512, "template_kwargs": {"thinking_mode": "off"}}, spec)
    assert dict(s.template_kwargs) == {"thinking_mode": "off"}


def test_pin_template_date_changes_strftime_now():
    import pytest
    jinja_utils = pytest.importorskip("transformers.utils.chat_template_utils")
    from verify_reproducibility import pin_template_date

    original = jinja_utils.datetime
    try:
        pin_template_date("2026-09-14")
        tpl = jinja_utils._compile_jinja_template("{{ strftime_now('%Y-%m-%d') }}")
        assert tpl.render() == "2026-09-14"
    finally:
        jinja_utils.datetime = original
        jinja_utils._compile_jinja_template.cache_clear()


def test_picked_groups_stay_in_file_order():
    groups = batch_groups(_rows(1))
    picked = pick_groups(groups, sample=12)
    idx = [groups.index(g) for g in picked]
    assert idx == sorted(idx)
