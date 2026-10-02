"""채점 규칙 검증: 여기가 틀리면 모든 모델 비교가 틀어짐."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from evaluate import (
    bootstrap_ci,
    frequency_baseline,
    hallucination_rate,
    micro_f1,
    omission_rate,
    score_note,
)


def _s(gold, predicted, neutral=(), freq=None):
    return score_note(
        gold=list(gold), predicted=list(predicted),
        neutral=list(neutral), drug_frequency=freq,
    )


class TestBasicScoring:
    def test_perfect_match(self):
        r = _s(["Furosemide", "Aspirin"], ["Furosemide", "Aspirin"])
        assert (r.true_positive, r.false_positive, r.false_negative) == (2, 0, 0)
        assert r.f1 == 1.0

    def test_hallucination_counts_as_false_positive(self):
        r = _s(["Furosemide"], ["Furosemide", "Metformin"])
        assert r.true_positive == 1
        assert r.false_positive == 1

    def test_omission_counts_as_false_negative(self):
        r = _s(["Furosemide", "Aspirin"], ["Furosemide"])
        assert r.false_negative == 1
        assert r.recall == 0.5

    def test_empty_prediction_scores_zero(self):
        r = _s(["Furosemide"], [])
        assert r.f1 == 0.0
        assert r.false_negative == 1

    def test_case_and_space_differences_still_match(self):
        r = _s(["Docusate Sodium"], ["  docusate   sodium "])
        assert r.true_positive == 1


class TestAliases:
    def test_brand_name_prediction_is_accepted(self):
        r = _s(["Emtricitabine-Tenofovir (Truvada)"], ["Truvada"])
        assert r.true_positive == 1
        assert r.false_positive == 0

    def test_generic_name_prediction_is_accepted(self):
        r = _s(["Emtricitabine-Tenofovir (Truvada)"], ["Emtricitabine-Tenofovir"])
        assert r.true_positive == 1

    def test_dosage_form_variant_is_accepted(self):
        r = _s(["Multivitamin Tablet"], ["Multivitamin"])
        assert r.true_positive == 1


class TestNeutralItems:
    """보류약, 의료용품은 정답도 오답도 아님."""

    def test_predicting_a_neutral_item_is_not_a_hallucination(self):
        r = _s(["Furosemide"], ["Furosemide", "Trimethoprim"], neutral=["Trimethoprim"])
        assert r.false_positive == 0
        assert r.neutral_hits == 1
        assert r.f1 == 1.0

    def test_omitting_a_neutral_item_is_not_penalised(self):
        r = _s(["Furosemide"], ["Furosemide"], neutral=["Trimethoprim"])
        assert r.false_negative == 0
        assert r.f1 == 1.0


class TestDuplicateHandling:
    def test_same_drug_twice_counts_once(self):
        """성분명과 제품명으로 두 번 써도 정답 하나를 두 번 맞힌 게 되면 안 됨."""
        r = _s(["Furosemide", "Aspirin"], ["Furosemide", "furosemide", "Aspirin"])
        assert r.true_positive == 2
        assert r.false_positive == 0

    def test_one_gold_is_consumed_by_only_one_prediction(self):
        r = _s(["Emtricitabine-Tenofovir (Truvada)"],
               ["Truvada", "Emtricitabine-Tenofovir"])
        assert r.true_positive == 1
        # 두 번째 예측은 이미 소비된 정답을 다시 맞힐 수 없어 환각이 됨
        assert r.false_positive == 1


class TestAggregates:
    def test_micro_f1_weights_by_drug_count(self):
        big = _s(["a", "b", "c", "d"], ["a", "b", "c", "d"])
        small = _s(["x"], ["wrong"])
        p, r, f = micro_f1([big, small])
        assert p == pytest.approx(4 / 5)
        assert r == pytest.approx(4 / 5)

    def test_hallucination_rate(self):
        s = [_s(["a"], ["a", "z"]), _s(["b"], ["b"])]
        # 예측 4개(a,z,b) 중 환각 1개 -> 1/3
        assert hallucination_rate(s) == pytest.approx(1 / 3)

    def test_omission_rate(self):
        s = [_s(["a", "b"], ["a"]), _s(["c"], ["c"])]
        assert omission_rate(s) == pytest.approx(1 / 3)

    def test_bootstrap_ci_brackets_point_estimate(self):
        s = [_s(["a"], ["a"]) for _ in range(20)] + [_s(["b"], ["z"]) for _ in range(20)]
        point = micro_f1(s)[2]
        lo, hi = bootstrap_ci(s, lambda xs: micro_f1(xs)[2], n_boot=300)
        assert lo <= point <= hi

    def test_bootstrap_ci_is_deterministic(self):
        s = [_s(["a"], ["a"]), _s(["b"], ["z"])]
        f = lambda xs: micro_f1(xs)[2]  # noqa: E731
        assert bootstrap_ci(s, f, n_boot=200) == bootstrap_ci(s, f, n_boot=200)


class TestFrequencyBaseline:
    def test_returns_most_common_drugs_in_order(self):
        freq = {"aspirin": 100, "senna": 50, "rare": 1}
        assert frequency_baseline(freq, k=2) == ["aspirin", "senna"]

    def test_baseline_can_be_scored_like_a_model(self):
        freq = {"aspirin": 100, "senna": 50}
        pred = frequency_baseline(freq, k=2)
        r = _s(["Aspirin", "Furosemide"], pred)
        assert r.true_positive == 1   # aspirin 은 맞고
        assert r.false_positive == 1  # senna 는 이 노트에 없으니 환각


class TestRarityTracking:
    def test_matched_and_missed_frequencies_are_recorded(self):
        freq = {"aspirin": 5000, "rarezolid": 2}
        r = _s(["Aspirin", "Rarezolid"], ["Aspirin"], freq=freq)
        assert r.matched_gold_freqs == [5000]
        assert r.missed_gold_freqs == [2]


class TestPairedBootstrap:
    def test_identical_models_have_zero_difference(self):
        from evaluate import paired_bootstrap_diff
        s = [_s(["a"], ["a"]), _s(["b"], ["z"])]
        point, lo, hi = paired_bootstrap_diff(s, list(s), lambda xs: micro_f1(xs)[2], n_boot=300)
        assert point == 0.0
        assert lo == 0.0 and hi == 0.0

    def test_better_model_gives_positive_difference(self):
        from evaluate import paired_bootstrap_diff
        good = [_s(["a"], ["a"]) for _ in range(30)]
        bad = [_s(["a"], ["z"]) for _ in range(30)]
        point, lo, hi = paired_bootstrap_diff(good, bad, lambda xs: micro_f1(xs)[2], n_boot=300)
        assert point > 0
        assert lo > 0   # 구간이 0 을 넘지 않아야 '실재하는 차이'로 판정됨

    def test_mismatched_lengths_raise(self):
        from evaluate import paired_bootstrap_diff
        with pytest.raises(ValueError, match="노트 수가 다르다"):
            paired_bootstrap_diff([_s(["a"], ["a"])], [], lambda _: 0.0)


class TestRarityBuckets:
    def test_common_and_rare_are_bucketed_separately(self):
        from evaluate import recall_by_rarity
        freq = {"common": 5000, "rare": 3}
        s = [_s(["Common", "Rare"], ["Common"], freq=freq)]
        out = recall_by_rarity(s)
        assert out["10회 미만"] == (0, 1)    # 드문 약은 놓침
        assert out["1000회 이상"] == (1, 1)  # 흔한 약은 맞힘

    def test_bucket_edges_are_lower_inclusive(self):
        """구간은 [아래, 위) 이고 이름이 그 경계를 그대로 말함."""
        from evaluate import recall_by_rarity
        freq = {"a": 10, "b": 1000, "c": 9, "d": 999}
        s = [_s(["A", "B", "C", "D"], ["A", "B", "C", "D"], freq=freq)]
        out = recall_by_rarity(s)
        assert list(out) == ["10회 미만", "10~99회", "100~999회", "1000회 이상"]
        assert out["10회 미만"] == (1, 1)     # 9회
        assert out["10~99회"] == (1, 1)      # 10회
        assert out["100~999회"] == (1, 1)    # 999회
        assert out["1000회 이상"] == (1, 1)  # 1000회

def test_duplicate_gold_is_counted_once():
    """같은 약이 정답에 두 번 있어도 한 번으로 셈.

    프롬프트는 약 이름만 요구하고 채점기는 예측의 중복을 합침. 정답만 중복을
    남기면 그 항목은 누구도 맞힐 수 없는 누락이 됨(실측 60건, 정답의 2.2%).
    """
    s = score_note(gold=["Dexamethasone", "Dexamethasone", "Aspirin"],
                   predicted=["Dexamethasone", "Aspirin"], neutral=[])
    assert s.n_gold == 2
    assert s.true_positive == 2
    assert s.false_negative == 0
    assert s.recall == 1.0


def test_duplicate_gold_merges_by_alias_too():
    """표기가 달라도 별칭이 겹치면 같은 약으로 봄(제형이 붙은 표기 등)."""
    s = score_note(gold=["Multivitamin Tablet", "Multivitamin"],
                   predicted=["Multivitamin"], neutral=[])
    assert s.n_gold == 1
    assert s.false_negative == 0


def test_two_different_drugs_are_not_merged():
    """합치기가 서로 다른 약까지 묶으면 안 됨."""
    s = score_note(gold=["Aspirin", "Warfarin"], predicted=["Aspirin"], neutral=[])
    assert s.n_gold == 2
    assert s.false_negative == 1


def test_rarity_follows_the_same_gold_list():
    """희귀도 집계가 맞힌 정답과 같은 목록을 봄.

    별칭이 비는 이름(비식별 표시만 남은 항목 등)은 채점 목록에서 빠짐. 희귀도가
    원래 이름 목록을 따로 훑으면 그만큼 인덱스가 밀려 엉뚱한 약의 빈도가 "맞힌 것"
    으로 들어감. 여기서는 맞힌 약(빈도 5)의 빈도만 matched 에 있어야 함.
    """
    s = score_note(gold=["___", "Rifaximin"], predicted=["Rifaximin"], neutral=[],
                   drug_frequency={"rifaximin": 5, "": 9999})
    assert s.n_gold == 1
    assert s.matched_gold_freqs == [5]
    assert s.missed_gold_freqs == []


# --- 예측 쪽에도 정답과 같은 별칭 규칙(트러블슈팅 17) ------------------------

def test_prediction_with_dosage_form_matches_gold_without_it():
    """정답 `Multivitamin` 에 예측 `Multivitamin Tablet` 은 맞힌 것임(환각+누락이 아님)."""
    s = score_note(gold=["Multivitamin"], predicted=["Multivitamin Tablet"], neutral=[])
    assert (s.true_positive, s.false_positive, s.false_negative) == (1, 0, 0)


def test_prediction_with_parenthesized_generic_matches_gold_generic():
    """예측 `Lipitor (Atorvastatin)` 는 괄호 안 성분명으로도 대조함."""
    s = score_note(gold=["Atorvastatin"], predicted=["Lipitor (Atorvastatin)"], neutral=[])
    assert (s.true_positive, s.false_positive, s.false_negative) == (1, 0, 0)


def test_alias_duplicate_prediction_is_counted_once_not_as_hallucination():
    """같은 약을 제품명과 성분명(괄호)으로 두 번 쓰면 한 번으로 셈. 두 번째를 환각으로 세면
    노트에 있는 약을 환각이라 부르게 됨. 정답 쪽 merge 와 같은 규칙임."""
    s = score_note(gold=["Emtricitabine-Tenofovir (Truvada)"],
                   predicted=["Truvada", "Emtricitabine-Tenofovir (Truvada)"], neutral=[])
    assert (s.true_positive, s.false_positive, s.n_pred) == (1, 0, 1)
