"""절제 일지의 종료 판단이 규칙대로 세어지는지 봄.

규칙: 의무 5회 뒤 부모 대비 개선이 아닌 회차가 3회 연속이면 멈춤.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location(
        "render_experiment_log", ROOT / "scripts" / "render_experiment_log.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _r(rid, parent, f1, lo, hi):
    return {"id": rid, "parent": parent,
            "result": {"micro_f1": f1, "ci_low": lo, "ci_high": hi}}


def test_only_separated_higher_interval_counts_as_improvement(rl):
    recs = [_r(1, 0, 0.88, 0.85, 0.91), _r(6, 1, 0.89, 0.86, 0.92),   # 겹침
            _r(7, 1, 0.95, 0.93, 0.97),                                # 갈리고 높음
            _r(8, 7, 0.60, 0.50, 0.70), _r(9, 7, 0.94, 0.92, 0.96),
            _r(10, 7, 0.96, 0.94, 0.98)]
    rows = rl.stop_trace(recs)
    assert [(x["id"], x["streak"]) for x in rows] == [(6, 1), (7, 0), (8, 1), (9, 2), (10, 3)]


def test_mandatory_runs_are_not_counted(rl):
    recs = [_r(1, 0, 0.88, 0.85, 0.91), _r(2, 1, 0.40, 0.30, 0.50), _r(5, 2, 0.10, 0.05, 0.15)]
    assert rl.stop_trace(recs) == []


ABL = ROOT / "outputs" / "ablation"


@pytest.mark.skipif(not (ABL / "08_minimal-medgemma.json").exists(), reason="절제 기록 없음")
def test_real_records_stop_at_8(rl):
    recs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in ABL.glob("*.json")),
                  key=lambda r: r["id"])
    rows = rl.stop_trace(recs)
    assert next(x["id"] for x in rows if x["streak"] >= 3) == 8
