"""짝 비교의 노트 정렬 검증.

`paired_bootstrap_diff` 는 `a_scores[i]` 와 `b_scores[i]` 가 같은 노트라는
전제 위에서만 성립함. 이 전제가 깨지면 에러 없이 틀린 신뢰구간이 나오므로,
호출 경로에서 정렬이 실제로 보장되는지 확인함.

추론은 JSONL 에 순차 append 되고 중단 시 이어받는 구조라, 모델마다 응답 순서가
다를 수 있음. 따라서 "쓰인 순서"에
의존하면 안 되고 note_id 로 맞춰야 함.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import NoteScore, paired_bootstrap_diff


def _score(f1: float) -> NoteScore:
    """f1 값을 직접 지정한 최소 NoteScore. 정렬 검증에만 씀.

    NoteScore 자체에는 note_id 가 없음: 노트와의 대응은 호출자가 리스트 순서로
    유지함. 그래서 정렬이 중요하고, 이 파일이 그것을 검증함.
    """
    return NoteScore(
        n_gold=1, n_pred=1,
        true_positive=1, false_positive=0, false_negative=0,
        neutral_hits=0, precision=f1, recall=f1, f1=f1,
        matched_gold_freqs=[], missed_gold_freqs=[],
    )


def _mean_f1(scores: list[NoteScore]) -> float:
    return float(np.mean([s.f1 for s in scores])) if scores else 0.0


def test_paired_bootstrap_uses_same_indices_for_both():
    """같은 표본 인덱스를 두 모델에 적용해야 노트 난이도가 상쇄됨.

    a 가 b 보다 항상 정확히 0.1 높으면, 짝지은 차이는 노트를 어떻게 뽑든
    항상 0.1 이어야 하고 신뢰구간 폭이 0 이어야 함. 인덱스를 따로 뽑으면
    구간이 0 보다 넓게 벌어짐.
    """
    a = [_score(v) for v in (0.2, 0.5, 0.9, 0.4, 0.7)]
    b = [_score(v - 0.1) for v in (0.2, 0.5, 0.9, 0.4, 0.7)]

    point, lo, hi = paired_bootstrap_diff(a, b, _mean_f1, n_boot=200)
    assert point == pytest.approx(0.1, abs=1e-9)
    assert hi - lo == pytest.approx(0.0, abs=1e-9), "짝이 맞으면 구간 폭이 0 이어야 한다"


def test_paired_bootstrap_rejects_length_mismatch():
    a = [_score(0.5)] * 3
    b = [_score(0.4)] * 2
    with pytest.raises(ValueError, match="노트 수가 다르다"):
        paired_bootstrap_diff(a, b, _mean_f1, n_boot=10)


def test_misaligned_pairing_gives_different_answer():
    """정렬이 어긋나면 실제로 다른 답이 나옴: 방어가 필요한 이유의 증거.

    같은 점수 집합이라도 순서를 섞어 짝지으면 점추정은 같아도(평균 차이라서)
    구간은 넓어짐. 정렬 오류는 오류 메시지 없이 신뢰구간을 틀리게 만듦.
    """
    vals_a = [0.2, 0.5, 0.9, 0.4, 0.7]
    a = [_score(v) for v in vals_a]
    b_aligned = [_score(v - 0.1) for v in vals_a]
    b_shuffled = [_score(v - 0.1) for v in [0.9, 0.2, 0.4, 0.7, 0.5]]

    _, lo1, hi1 = paired_bootstrap_diff(a, b_aligned, _mean_f1, n_boot=300)
    _, lo2, hi2 = paired_bootstrap_diff(a, b_shuffled, _mean_f1, n_boot=300)

    assert hi1 - lo1 == pytest.approx(0.0, abs=1e-9)
    assert hi2 - lo2 > 0.05, "어긋난 짝은 구간이 넓어져야 한다(= 잘못된 결론 위험)"


def test_note_id_sorting_makes_order_independent():
    """응답이 어떤 순서로 쓰였든 note_id 로 정렬하면 같은 짝이 나옴.

    run_phase4_evaluate.py 가 쓰는 방식(dict[note_id] -> sorted)을 그대로 재현함.
    """
    # 모델 A 는 순서대로, 모델 B 는 뒤섞인 순서로 응답이 쌓였다고 가정
    rows_a = [("n1", 0.3), ("n2", 0.6), ("n3", 0.9)]
    rows_b = [("n3", 0.8), ("n1", 0.2), ("n2", 0.5)]

    ord_a = {nid: _score(v) for nid, v in rows_a}
    ord_b = {nid: _score(v) for nid, v in rows_b}
    ids_a, ids_b = sorted(ord_a), sorted(ord_b)
    assert ids_a == ids_b

    sa = [ord_a[k] for k in ids_a]
    sb = [ord_b[k] for k in ids_b]
    # n1:0.3-0.2, n2:0.6-0.5, n3:0.9-0.8 -> 전부 0.1 차이
    point, lo, hi = paired_bootstrap_diff(sa, sb, _mean_f1, n_boot=200)
    assert point == pytest.approx(0.1, abs=1e-9)
    assert hi - lo == pytest.approx(0.0, abs=1e-9)


def test_common_subset_recovery_when_note_sets_differ():
    """노트 집합이 다를 때 공통 노트만 남기는 복구 로직(평가 스크립트와 동일).

    길이가 같아도 집합이 다르면(A={n1,n2,n3}, B={n1,n2,n4}) 길이 검사는 통과함.
    공통 노트로 좁혀야 짝이 맞음.
    """
    ord_a = {"n1": _score(0.3), "n2": _score(0.6), "n3": _score(0.9)}
    ord_b = {"n1": _score(0.2), "n2": _score(0.5), "n4": _score(0.1)}
    ids_a, ids_b = sorted(ord_a), sorted(ord_b)

    assert len(ids_a) == len(ids_b)   # 길이 검사만으로는 못 잡음
    assert ids_a != ids_b

    common = sorted(set(ids_a) & set(ids_b))
    assert common == ["n1", "n2"]

    pos_a = {n: i for i, n in enumerate(ids_a)}
    pos_b = {n: i for i, n in enumerate(ids_b)}
    sa = [[ord_a[k] for k in ids_a][pos_a[n]] for n in common]
    sb = [[ord_b[k] for k in ids_b][pos_b[n]] for n in common]

    point, lo, hi = paired_bootstrap_diff(sa, sb, _mean_f1, n_boot=200)
    assert point == pytest.approx(0.1, abs=1e-9)
    assert hi - lo == pytest.approx(0.0, abs=1e-9)
