"""Phase 4: 채점과 사전등록 기준 판정.

docs/prereg.md 7, 8절에 등록한 지표와 판정 기준을 그대로 적용함.
모델을 돌리지 않으므로 GPU 없이 실행됨.

    python scripts/run_phase4_evaluate.py [--stub]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl

from datapaths import SEED
from evaluate import (
    bootstrap_ci,
    frequency_baseline,
    hallucination_rate,
    micro_f1,
    omission_rate,
    paired_bootstrap_diff,
    recall_by_rarity,
    score_note,
)
from jsonout import parse_model_output
from models import MODELS, PAIRS, exceeds_context

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
RESPONSE_DIR = OUT_DIR / "responses"
SELECTED_PATH = OUT_DIR / "phase3_selected_prompts.json"
REPORT_PATH = OUT_DIR / "phase4_results.json"

# docs/prereg.md 8절 판정 경계
MEMORISATION_STRONG = 0.50
MEMORISATION_PRESENT = 0.25
STRUCTURE_INDEPENDENT = 0.90
BASELINE_K = 10
# 보정 구간의 끝은 부트스트랩 양 끝 여섯 건 남짓으로 정해져 시드에 따라 움직임. 그 폭을 시드 20개로 측정함.
BONFERRONI_SEEDS = 20


def _f1(scores) -> float:
    return micro_f1(scores)[2]


def audit_normal_scores(key, suffix, meta, freq):
    """감사 분할 정상 조건의 채점 결과: 감사 조건 비율의 분모로 씀.

    감사 조건(no_section, empty, other_patient, shuffled)은 감사 분할 100건에
    걸려 있는데, 분모로 main 분할 200건의 정상 점수를 쓰면 "조건의 효과"와
    "노트 집합의 차이"가 섞임. 실측하니 두 집합의 정상 F1 은 모델에 따라
    최대 0.09 벌어짐. 같은 노트의 정상 점수로 나눠야 함.

    그 응답은 프롬프트 선택 단계(--stage select)에 이미 있음. 디코딩이 greedy
    고정이라 같은 변형으로 다시 돌려도 같은 출력이 나오므로 재추론하지 않고
    고른 변형의 것만 꺼내 씀.

    다만 이 분모는 변형 선택에 쓰인 바로 그 점수라 선택 편향이 있음(고른
    변형이 그 100건에서 가장 높았으므로 분모가 부풀려져 비율이 작게 나옴).
    그 방향과 크기를 결과에 함께 남김.
    """
    if not SELECTED_PATH.exists():
        return None
    selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    if key not in selected:
        return None
    variant = selected[key]["variant"]
    path = RESPONSE_DIR / f"{key}_select{suffix}.jsonl"
    if not path.exists():
        return None
    by_variant: dict[str, list] = {}
    for r in load_responses(path):
        if not suffix and exceeds_context(r, MODELS[key]):
            continue
        m = meta.get((r["note_id"], r["condition"]))
        if m is None:
            continue
        parsed = parse_model_output(r["response"])
        s_ = score_note(gold=list(m["gold"]), predicted=parsed.medications,
                        neutral=list(m["neutral"]), drug_frequency=freq)
        by_variant.setdefault(r.get("variant", variant), []).append(s_)
    if variant not in by_variant:
        return None
    return {
        "variant": variant,
        "scores": by_variant[variant],
        "f1_by_variant": {v: _f1(sc) for v, sc in sorted(by_variant.items())},
    }


def load_responses(path: Path) -> list[dict]:
    """응답 JSONL 을 읽음. 깨진 줄은 세어서 알리고 건너뜀.

    추론은 몇 시간짜리라 중단으로 마지막 줄이 잘려 있을 수 있음. 한 줄
    때문에 채점 전체가 죽으면 그 시간을 통째로 날림. 다만 말없이 넘기면
    표본이 줄어든 걸 모르므로 몇 줄을 버렸는지 반드시 출력함.
    """
    if not path.exists():
        return []
    rows, broken = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            broken += 1
    if broken:
        print(f"  *** {path.name}: 깨진 줄 {broken}개를 건너뛰었다 "
              f"(중단으로 잘린 기록으로 보인다). 재실행하면 그만큼 다시 채운다.")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()
    suffix = "_stub" if args.stub else ""

    conditions = pl.read_parquet(OUT_DIR / "phase2_conditions.parquet")
    meta = {
        (r["note_id"], r["condition"]): r
        for r in conditions.iter_rows(named=True)
    }
    freq = json.loads((OUT_DIR / "phase1_drug_frequency.json").read_text(encoding="utf-8"))

    if args.stub:
        print("*** 스텁 모드: 가짜 응답을 채점한다. 이 수치는 리포트에 쓸 수 없다. ***\n")

    # --- 빈도 베이스라인 (모델 없이 흔한 약 k개 고정 출력) ---
    baseline_pred = frequency_baseline(freq, k=BASELINE_K)
    main_notes = conditions.filter(
        (pl.col("split") == "main") & (pl.col("condition") == "normal")
    )
    baseline_scores = [
        score_note(gold=list(r["gold"]), predicted=baseline_pred,
                   neutral=list(r["neutral"]), drug_frequency=freq)
        for r in main_notes.iter_rows(named=True)
    ]
    baseline_f1 = _f1(baseline_scores)
    print(f"빈도 베이스라인 (흔한 약 {BASELINE_K}개 고정): micro-F1 {baseline_f1:.4f}")
    print(f"  출력 목록: {baseline_pred}\n")

    results: dict[str, dict] = {}
    main_scores_by_model: dict[str, list] = {}
    # 짝 비교는 같은 노트끼리 맞물려야 성립함. 길이만 같고 노트 집합이 다르면
    # (한 모델이 특정 노트에서만 실패한 경우 등) 오류 없이 엉뚱한 짝이 만들어지므로
    # note_id 목록을 함께 들고 다니며 대조함.
    main_note_ids_by_model: dict[str, list[str]] = {}

    for key in sorted(MODELS):
        rows = load_responses(RESPONSE_DIR / f"{key}_final{suffix}.jsonl")
        if not rows:
            print(f"{MODELS[key].label}: 응답 없음: 건너뜀")
            continue
        # 문맥 길이를 넘은 조건에서 만든 응답은 뺌(models.exceeds_context).
        # 짝 비교는 아래에서 공통 노트로 좁히므로 두 모델이 모두 문맥 안인 노트만 남음.
        n_all = len(rows)
        rows = [r for r in rows if suffix or not exceeds_context(r, MODELS[key])]
        n_ctx_excluded = n_all - len(rows)
        if n_ctx_excluded:
            print(f"  [문맥 초과 제외] {MODELS[key].label}: {n_ctx_excluded}/{n_all}건 "
                  f"(문맥 {MODELS[key].context_window})")

        by_condition: dict[str, list] = {}
        strict_by_condition: dict[str, list[int]] = {}
        recovery_paths: dict[str, int] = {}
        ordered_main: dict[str, object] = {}

        for r in rows:
            m = meta.get((r["note_id"], r["condition"]))
            if m is None:
                continue
            parsed = parse_model_output(r["response"])
            s = score_note(gold=list(m["gold"]), predicted=parsed.medications,
                           neutral=list(m["neutral"]), drug_frequency=freq)
            by_condition.setdefault(r["condition"], []).append(s)
            strict_by_condition.setdefault(r["condition"], []).append(int(parsed.strict_json))
            recovery_paths[parsed.recovered_by] = recovery_paths.get(parsed.recovered_by, 0) + 1
            if r["condition"] == "normal" and r["split"] == "main":
                ordered_main[r["note_id"]] = s

        normal = by_condition.get("normal", [])
        if not normal:
            print(f"{MODELS[key].label}: 정상 조건 응답이 없다: 건너뜀")
            continue
        ordered_ids = sorted(ordered_main)
        main_scores_by_model[key] = [ordered_main[k] for k in ordered_ids]
        main_note_ids_by_model[key] = ordered_ids

        p, r_, f = micro_f1(normal)
        lo, hi = bootstrap_ci(normal, _f1)
        strict = sum(strict_by_condition["normal"]) / len(strict_by_condition["normal"])

        print(f"\n=== {MODELS[key].label} ===")
        print(f"  정상 micro-F1 {f:.4f}  [95% CI {lo:.4f}~{hi:.4f}]  "
              f"정밀도 {p:.4f} 재현율 {r_:.4f}")
        print(f"  환각률 {hallucination_rate(normal)*100:.1f}%  "
              f"누락률 {omission_rate(normal)*100:.1f}%  "
              f"JSON 준수 {strict*100:.1f}%")

        disqualified = f <= baseline_f1
        if disqualified:
            print(f"  *** 실격: 빈도 베이스라인({baseline_f1:.4f}) 이하")

        # 감사 조건의 분모: 같은 노트(감사 분할)의 정상 점수를 씀.
        audit_norm = audit_normal_scores(key, suffix, meta, freq)
        if audit_norm is None:
            denom = f
            denom_source = "main_normal(대체)"
            print("  [주의] 감사 분할 정상 응답이 없어 main 분할 정상으로 나눈다 "
                  ": 노트 집합이 달라 비율이 조건 효과와 섞인다.")
        else:
            denom = _f1(audit_norm["scores"])
            denom_source = "audit_normal"

        cond_f1 = {}
        for cond, scores in sorted(by_condition.items()):
            cf = _f1(scores)
            cond_f1[cond] = cf
            if cond != "normal":
                print(f"  {cond:<14} micro-F1 {cf:.4f}  (감사 정상 대비 {cf/denom*100:5.1f}%)"
                      if denom else f"  {cond:<14} micro-F1 {cf:.4f}")

        # 암기 판정은 `empty` 만으로 함 (docs/prereg.md 6, 8절).
        # `other_patient` 는 순수하지 않음: 실측 결과 정답 약물의 16.5%가 남의
        # 노트에도 그대로 존재함(흔한 약일수록 심함). 그 조건을 판정에 넣으면
        # "우연한 겹침"을 "암기"로 오판함. 참고값으로만 함께 보고함.
        empty_ratio = cond_f1.get("empty", 0.0) / denom if denom else 0.0
        other_ratio = cond_f1.get("other_patient", 0.0) / denom if denom else 0.0
        if empty_ratio >= MEMORISATION_STRONG:
            memo = "암기 강함"
        elif empty_ratio >= MEMORISATION_PRESENT:
            memo = "암기 있음"
        else:
            memo = "암기 경미"
        print(f"  암기 판정: {memo} (빈 노트로도 정상 대비 {empty_ratio*100:.1f}%)")
        print(f"    참고: 남의 노트: 정상 대비 {other_ratio*100:.1f}% "
              f"(정답의 16.5%가 남의 노트에도 존재하므로 판정에는 쓰지 않는다)")

        shuffled_ratio = cond_f1.get("shuffled", 0.0) / denom if denom else 0.0
        structure = "구조 무관" if shuffled_ratio >= STRUCTURE_INDEPENDENT else "구조 의존"
        print(f"  구조 판정: {structure} (문장을 흩어도 정상 대비 {shuffled_ratio*100:.1f}%)")

        rarity = recall_by_rarity(normal)
        print("  희귀도별 재현율:")
        for bucket, (hit, tot) in rarity.items():
            if tot:
                print(f"    {bucket:<12} {hit}/{tot} ({hit/tot*100:5.1f}%)")

        results[key] = {
            "label": MODELS[key].label,
            "medical": MODELS[key].medical,
            "family": MODELS[key].family,
            "normal": {"precision": p, "recall": r_, "micro_f1": f, "ci": [lo, hi]},
            "hallucination_rate": hallucination_rate(normal),
            "omission_rate": omission_rate(normal),
            "json_compliance": strict,
            "condition_micro_f1": cond_f1,
            "memorisation_verdict": memo,
            # 판정 근거를 명시적으로 남김: 어떤 조건의 어떤 비율로 판정했는지
            # 나중에 되짚을 수 있어야 함.
            "memorisation_basis": {
                "condition": "empty",
                "ratio_vs_normal": empty_ratio,
                "other_patient_ratio_vs_normal": other_ratio,
                "note": "other_patient 는 정답의 16.5%가 남의 노트에도 존재해 판정에서 제외",
                # 분모를 어디서 가져왔는지 남김. main 분할로 나누면 조건 효과와
                # 노트 집합 차이가 섞이므로 감사 분할 정상을 씀.
                "denominator_source": denom_source,
                "denominator_f1": denom,
                "main_normal_f1": f,
                "denominator_selection_bias": (
                    None if audit_norm is None else {
                        "variant": audit_norm["variant"],
                        "f1_by_variant": audit_norm["f1_by_variant"],
                        "note": "이 분모는 변형 선택에 쓰인 점수라 위로 편향돼 "
                                "비율이 실제보다 작게 나온다",
                    }
                ),
            },
            "structure_verdict": structure,
            "recall_by_rarity": {k: list(v) for k, v in rarity.items()},
            "recovery_paths": recovery_paths,
            "disqualified_vs_baseline": disqualified,
            "context_window": MODELS[key].context_window,
            "n_excluded_context": n_ctx_excluded,
            "n_responses": n_all,
        }

    # --- 짝 비교 ---
    print("\n" + "=" * 60)
    print("짝 비교: 페어드 부트스트랩 95% CI 가 0 을 걸치면 '차이 없음'")
    print("=" * 60)
    pair_out = []
    for a, b, desc in PAIRS:
        if a not in main_scores_by_model or b not in main_scores_by_model:
            print(f"  {desc}: 응답이 모자라 비교 불가")
            continue
        sa, sb = main_scores_by_model[a], main_scores_by_model[b]
        ids_a, ids_b = main_note_ids_by_model[a], main_note_ids_by_model[b]
        if ids_a != ids_b:
            # 길이가 같아도 노트 집합이 다르면 짝이 어긋남: 길이 검사만으로는
            # 못 잡음. 공통 노트만 남겨 짝을 다시 맞춤.
            common = sorted(set(ids_a) & set(ids_b))
            if not common:
                print(f"  {desc}: 공통 노트가 없어 비교 불가")
                continue
            print(f"  {desc}: 노트 집합이 달라 공통 {len(common)}건으로 좁힘 "
                  f"({a} {len(ids_a)}건 / {b} {len(ids_b)}건)")
            pos_a = {nid: i for i, nid in enumerate(ids_a)}
            pos_b = {nid: i for i, nid in enumerate(ids_b)}
            sa = [sa[pos_a[nid]] for nid in common]
            sb = [sb[pos_b[nid]] for nid in common]
        point, lo, hi = paired_bootstrap_diff(sa, sb, _f1)
        # 같은 기준으로 여덟 번 재면 우연히 0 을 벗어나는 짝이 생김. 판정은
        # 사전등록대로 보정 없는 95% 구간으로 하되, 보정하면 어떻게 되는지를
        # 함께 남김. 문서가 이 값을 인용하므로 산출물에 있어야 낡지 않음.
        _, blo, bhi = paired_bootstrap_diff(sa, sb, _f1, alpha=0.05 / len(PAIRS))
        ends = np.array([paired_bootstrap_diff(sa, sb, _f1, alpha=0.05 / len(PAIRS), seed=SEED + k)[1:]
                         for k in range(1, BONFERRONI_SEEDS + 1)])
        real = not (lo <= 0 <= hi)
        verdict = ("차이 있음" if real else "차이 없음")
        winner = MODELS[a].label if point > 0 else MODELS[b].label
        print(f"\n  {desc}")
        print(f"    {MODELS[a].label} - {MODELS[b].label} = {point:+.4f} "
              f"[95% CI {lo:+.4f}~{hi:+.4f}] -> {verdict}")
        print(f"      보정(alpha 0.05/{len(PAIRS)}) [{blo:+.4f}~{bhi:+.4f}] -> "
              f"{'차이 있음' if not (blo <= 0 <= bhi) else '차이 없음'}")
        if real:
            print(f"    우세: {winner}")
        pair_out.append({
            "a": a, "b": b, "description": desc,
            # 몇 개 노트로 비교했는지 남김: 노트 집합이 어긋나 공통으로 좁힌
            # 경우 이 값이 각 모델의 노트 수보다 작아지므로, 나중에 리포트를
            # 읽을 때 표본이 줄었다는 사실을 놓치지 않음.
            "n_notes": len(sa),
            "diff": point, "ci": [lo, hi],
            # Bonferroni: alpha 0.05/8. 보정해도 유의한지 따로 읽음.
            "ci_bonferroni": [blo, bhi],
            "ci_bonferroni_seed_spread": {"n_seeds": BONFERRONI_SEEDS,
                                           "lower": [float(ends[:, 0].min()), float(ends[:, 0].max())],
                                           "upper": [float(ends[:, 1].min()), float(ends[:, 1].max())]},
            "significant": real,
            "significant_bonferroni": not (blo <= 0 <= bhi),
            "winner": winner if real else None,
        })

    REPORT_PATH.write_text(json.dumps({
        "stub": args.stub,
        "baseline": {"k": BASELINE_K, "micro_f1": baseline_f1, "drugs": baseline_pred},
        "models": results,
        "pairs": pair_out,
        "prereg_thresholds": {
            "memorisation_strong": MEMORISATION_STRONG,
            "memorisation_present": MEMORISATION_PRESENT,
            "structure_independent": STRUCTURE_INDEPENDENT,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"\n저장: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
