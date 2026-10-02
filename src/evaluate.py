"""채점 규칙.

중립 항목을 왜 따로 두는가

노트의 번호 목록에는 보류된 약(HELD-)이나 의료용품처럼 "퇴원 복용약"이라
단정하기 어려운 항목이 섞여 있음. 이걸 정답에 넣으면 올바르게 제외한 모델이
누락으로 깎이고, 오답으로 두면 노트를 그대로 읽은 모델이 환각으로 깎임.
어느 쪽도 공정하지 않으므로 정답에서 빼되 환각으로도 세지 않음.

별칭으로 맞히는 것을 왜 인정하는가

같은 약을 성분명으로도 제품명으로도 씀('Emtricitabine-Tenofovir' = 'Truvada').
표기 차이로 깎으면 약물 지식이 아니라 표기 습관을 재게 됨.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datapaths import SEED
from notes import drug_aliases, normalize_drug


@dataclass
class NoteScore:
    n_gold: int
    n_pred: int
    true_positive: int
    false_positive: int   # 환각: 노트에 없는 약
    false_negative: int   # 누락
    neutral_hits: int     # 중립 항목을 맞힌 것: 상도 벌도 없음
    precision: float
    recall: float
    f1: float
    matched_gold_freqs: list[int]
    missed_gold_freqs: list[int]


def _alias_entries(names: list[str], *, merge: bool = False) -> list[tuple[str, set[str]]]:
    """이름 목록을 (이름, 별칭 집합) 목록으로. `merge` 면 같은 약을 한 번만 남김.

    정답 쪽에 같은 약이 두 번 들어오는 노트가 있음(용량만 다른 감량 요법 등).
    예측은 같은 약을 두 번 써도 한 번으로 세고, 프롬프트도 "약 이름만" 을
    요구하므로, 정답의 두 번째 항목은 어떤 모델도 맞힐 수 없는 누락이 됨
    (표본 300건에서 60개, 정답의 2.2%). 정답도 예측과 같은 규칙으로 합침.

    이름을 함께 돌려주는 이유는 희귀도 집계 때문임. 맞힌 정답은 이 목록의
    인덱스로 들고 있는데, 희귀도가 원래 이름 목록을 따로 훑으면 별칭이 비어
    빠진 항목만큼 인덱스가 밀림.
    """
    out: list[tuple[str, set[str]]] = []
    for n in names:
        al = drug_aliases(n)
        if not al:
            continue
        if merge and any(al & prev for _, prev in out):
            continue
        out.append((n, al))
    return out


def score_note(
    *,
    gold: list[str],
    predicted: list[str],
    neutral: list[str],
    drug_frequency: dict[str, int] | None = None,
) -> NoteScore:
    """노트 하나를 채점함.

    정답 하나는 예측 하나에만 대응시킴. 모델이 같은 약을 성분명과 제품명으로
    두 번 쓰면 정답 하나를 두 번 맞힌 것으로 세면 안 되기 때문임.
    """
    gold_entries = _alias_entries(gold, merge=True)
    gold_aliases = [al for _, al in gold_entries]
    neutral_aliases = [al for _, al in _alias_entries(neutral)]
    # 예측에도 정답과 같은 규칙을 적용함: 별칭 집합(괄호 안 이름, 제형 뗀 이름)으로 대조하고,
    # 별칭이 겹치는 예측은 같은 약으로 보고 한 번만 셈. 정규화 문자열 하나로만 대조하면
    # `Multivitamin Tablet`, `Lipitor (Atorvastatin)` 같은 표기가 환각+누락으로 깎이고,
    # 같은 약을 성분명, 제품명으로 두 번 쓰면 두 번째가 환각으로 세어짐(트러블슈팅 17).
    pred_entries = _alias_entries(predicted, merge=True)

    used_gold: set[int] = set()
    tp = fp = neutral_hits = 0

    for _, pal in pred_entries:
        hit = next((i for i, al in enumerate(gold_aliases) if i not in used_gold and pal & al), None)
        if hit is not None:
            used_gold.add(hit)
            tp += 1
            continue
        if any(pal & al for al in neutral_aliases):
            neutral_hits += 1
            continue
        fp += 1

    fn = len(gold_aliases) - len(used_gold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    freq = drug_frequency or {}
    matched_freqs, missed_freqs = [], []
    for i, (name, _) in enumerate(gold_entries):
        f = freq.get(normalize_drug(name), 0)
        (matched_freqs if i in used_gold else missed_freqs).append(f)

    return NoteScore(
        n_gold=len(gold_aliases),
        n_pred=len(pred_entries),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        neutral_hits=neutral_hits,
        precision=precision,
        recall=recall,
        f1=f1,
        matched_gold_freqs=matched_freqs,
        missed_gold_freqs=missed_freqs,
    )


def micro_f1(scores: list[NoteScore]) -> tuple[float, float, float]:
    """노트 전체를 한 덩어리로 합쳐 계산함. 약이 많은 노트가 더 큰 비중을 가짐."""
    tp = sum(s.true_positive for s in scores)
    fp = sum(s.false_positive for s in scores)
    fn = sum(s.false_negative for s in scores)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def hallucination_rate(scores: list[NoteScore]) -> float:
    """예측한 약 중 노트에 없던 약의 비율."""
    total_pred = sum(s.true_positive + s.false_positive for s in scores)
    if not total_pred:
        return 0.0
    return sum(s.false_positive for s in scores) / total_pred


def omission_rate(scores: list[NoteScore]) -> float:
    total_gold = sum(s.n_gold for s in scores)
    if not total_gold:
        return 0.0
    return sum(s.false_negative for s in scores) / total_gold


def bootstrap_ci(
    scores: list[NoteScore],
    statistic,
    *,
    n_boot: int = 2000,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """노트 단위 부트스트랩 신뢰구간.

    표본이 300건이라 점추정만 보고 모델 순위를 말하면 안 됨. 노트를 복원
    추출해 통계량 분포를 만들고 그 구간을 함께 보고함.
    """
    if not scores:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(statistic([scores[i] for i in idx]))
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def frequency_baseline(drug_frequency: dict[str, int], k: int = 10) -> list[str]:
    """가장 흔한 약 k개를 노트와 무관하게 고정 출력하는 베이스라인.

    이걸 못 넘는 모델은 노트를 전혀 읽지 못한 것이므로 사전등록상 실격임.
    """
    return [name for name, _ in sorted(drug_frequency.items(), key=lambda kv: -kv[1])[:k]]


def paired_bootstrap_diff(
    a_scores: list[NoteScore],
    b_scores: list[NoteScore],
    statistic,
    *,
    n_boot: int = 2000,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """같은 노트에서 두 모델의 통계량 차이(a - b)와 그 신뢰구간.

    모델마다 따로 부트스트랩해 구간을 겹쳐 보면 안 됨. 같은 노트를 함께
    뽑아 차이를 재야 노트 난이도라는 공통 변동이 상쇄되고, 두 모델 사이의
    차이만 남음.

    반환: (점추정 차이, 하한, 상한)
    """
    if len(a_scores) != len(b_scores):
        msg = f"두 모델의 노트 수가 다르다: {len(a_scores)} vs {len(b_scores)}"
        raise ValueError(msg)
    if not a_scores:
        return (0.0, 0.0, 0.0)

    point = statistic(a_scores) - statistic(b_scores)
    rng = np.random.default_rng(seed)
    n = len(a_scores)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(
            statistic([a_scores[i] for i in idx]) - statistic([b_scores[i] for i in idx])
        )
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point, lo, hi


def recall_by_rarity(
    scores: list[NoteScore], *, quantile_edges: tuple[int, ...] = (10, 100, 1000)
) -> dict[str, tuple[int, int]]:
    """약물 희귀도 구간별 (맞힌 수, 전체 수).

    흔한 약만 맞히고 드문 약에서 재현율이 떨어지면 노트를 읽은 것이 아니라 사전
    지식을 뱉은 것에 가까움: 그 신호를 여기서 잡음.
    """
    # 구간은 [아래, 위) 다. 이름도 그대로 적음: 10회 미만, 10~99회, 100~999회, 1000회 이상.
    buckets = {}
    labels = []
    prev = 0
    for e in quantile_edges:
        name = f"{e}회 미만" if prev == 0 else f"{prev}~{e - 1}회"
        labels.append((name, prev, e))
        prev = e
    labels.append((f"{prev}회 이상", prev, float("inf")))
    for name, _, _ in labels:
        buckets[name] = [0, 0]

    for s in scores:
        for f in s.matched_gold_freqs:
            for name, lo, hi in labels:
                if lo <= f < hi:
                    buckets[name][0] += 1
                    buckets[name][1] += 1
                    break
        for f in s.missed_gold_freqs:
            for name, lo, hi in labels:
                if lo <= f < hi:
                    buckets[name][1] += 1
                    break
    return {k: (v[0], v[1]) for k, v in buckets.items()}
