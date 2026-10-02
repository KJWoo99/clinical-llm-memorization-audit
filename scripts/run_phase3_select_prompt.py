"""Phase 3: 모델별 프롬프트 변형 선택.

--stage select 로 받아둔 audit 분할 응답에서, 모델마다 micro-F1 이 가장 높은
변형을 고름. 이 선택은 audit 분할로만 하고 main 분할은 건드리지 않음:
같은 데이터로 고르고 평가하면 그 선택만큼 성능이 부풀려짐.

    python scripts/run_phase3_select_prompt.py [--stub]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from evaluate import micro_f1, score_note
from jsonout import parse_model_output
from models import MODELS, exceeds_context

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
RESPONSE_DIR = OUT_DIR / "responses"
SELECTED_PATH = OUT_DIR / "phase3_selected_prompts.json"


def load_responses(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _note_order(rows: list[dict], variant: str) -> list[str]:
    """해당 변형의 점수가 쌓인 순서와 같은 순서의 note_id 목록.

    `per_variant[variant]` 는 rows 를 순회하며 append 한 결과이므로, 같은 필터를
    같은 순서로 적용하면 인덱스가 일대일로 맞음. 공통 노트로 좁힐 때 이 대응이
    필요함: 순서를 가정하지 않으려면 애초에 (note_id, score) 쌍으로 들고
    다녀야 하지만, 기존 구조를 최소 변경하기 위해 여기서 재현함.
    """
    return [r["note_id"] for r in rows if r["variant"] == variant]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()
    suffix = "_stub" if args.stub else ""

    conditions = pl.read_parquet(OUT_DIR / "phase2_conditions.parquet")
    gold_by_note = {
        r["note_id"]: (r["gold"], r["neutral"])
        for r in conditions.filter(pl.col("condition") == "normal").iter_rows(named=True)
    }
    freq = json.loads((OUT_DIR / "phase1_drug_frequency.json").read_text(encoding="utf-8"))

    selected: dict[str, dict] = {}
    for key in sorted(MODELS):
        path = RESPONSE_DIR / f"{key}_select{suffix}.jsonl"
        rows = load_responses(path)
        if not rows:
            print(f"  {key}: 응답 없음: 건너뜀 ({path.name})")
            continue
        # 문맥 길이를 넘은 조건의 응답은 선택에 쓰지 않음. 넘친 건이 변형마다
        # 조금씩 다르게 섞이면 "어느 프롬프트가 나은가"에 길이 효과가 끼어듦.
        # 변형 사이는 아래에서 공통 노트로 좁히므로 세 변형 모두 문맥 안인 노트만 남음.
        n_all = len(rows)
        if not suffix:
            rows = [r for r in rows if not exceeds_context(r, MODELS[key])]
        if len(rows) < n_all:
            print(f"  {key}: 문맥 초과 {n_all - len(rows)}/{n_all}건 제외 (문맥 {MODELS[key].context_window})")

        per_variant: dict[str, list] = {}
        strict_counts: dict[str, list[int]] = {}
        notes_by_variant: dict[str, set[str]] = {}
        for r in rows:
            if r["note_id"] not in gold_by_note:
                # 정답을 못 찾으면 빈 정답으로 채점돼 F1 이 0 이 됨:
                # 경고 없이 그 변형을 불리하게 만들므로 즉시 멈춤.
                msg = (f"{key}: note_id {r['note_id']!r} 의 정답이 없다. "
                       "phase2_conditions.parquet 이 응답과 다른 표본이다.")
                raise KeyError(msg)
            gold, neutral = gold_by_note[r["note_id"]]
            parsed = parse_model_output(r["response"])
            per_variant.setdefault(r["variant"], []).append(
                score_note(gold=list(gold), predicted=parsed.medications,
                           neutral=list(neutral), drug_frequency=freq)
            )
            strict_counts.setdefault(r["variant"], []).append(int(parsed.strict_json))
            notes_by_variant.setdefault(r["variant"], set()).add(r["note_id"])

        # 변형끼리 같은 노트 집합 위에서 비교해야 공정함. 추론이 중간에 끊겨
        # 한 변형만 덜 채워지면, 그 변형이 우연히 쉬운/어려운 노트만 갖게 되어
        # 선택이 왜곡됨. 공통 노트로 좁혀 비교함.
        sets = list(notes_by_variant.values())
        common = set.intersection(*sets) if sets else set()
        if any(s != common for s in sets):
            sizes = {v: len(s) for v, s in notes_by_variant.items()}
            print(f"  [경고] 변형별 노트 집합이 다르다 {sizes}: 공통 {len(common)}건으로 좁혀 비교한다")
            for variant in per_variant:
                keep = [
                    s for s, nid in zip(per_variant[variant],
                                        _note_order(rows, variant), strict=True)
                    if nid in common
                ]
                per_variant[variant] = keep
            strict_counts = {
                v: [c for c, nid in zip(strict_counts[v], _note_order(rows, v), strict=True)
                    if nid in common]
                for v in strict_counts
            }

        print(f"\n{MODELS[key].label}")
        best, best_f1 = None, -1.0
        # 변형별 점수를 파일에도 남김. 리포트의 선택 표를 로그가 아니라 여기서 만듦.
        by_variant: dict[str, dict] = {}
        for variant, scores in sorted(per_variant.items()):
            if not scores:
                print(f"  {variant:<10} 채점할 노트가 없어 건너뜀")
                continue
            _, _, f1 = micro_f1(scores)
            strict = sum(strict_counts[variant]) / len(strict_counts[variant])
            by_variant[variant] = {"micro_f1": f1, "json_compliance": strict, "n_notes": len(scores)}
            print(f"  {variant:<10} micro-F1 {f1:.4f}   JSON 준수 {strict*100:5.1f}%  "
                  f"(노트 {len(scores)}건)")
            # 동점이면 정렬 순서상 먼저 온 변형이 유지됨(grounded < minimal < schema).
            # 결정적이어야 재현이 되므로 무작위 선택은 쓰지 않음.
            if f1 > best_f1:
                best, best_f1 = variant, f1
        if best is None:
            print(f"  {key}: 비교할 변형이 없다: 건너뜀")
            continue
        selected[key] = {
            "variant": best,
            "audit_micro_f1": best_f1,
            "n_notes_compared": len(per_variant[best]),
            "by_variant": by_variant,
        }
        print(f"  -> 선택: {best} (F1 {best_f1:.4f}, 노트 {len(per_variant[best])}건)")

    if not selected:
        print("\n선택할 수 있는 모델이 없다. 먼저 --stage select 추론을 돌릴 것.")
        return 1

    # 스텁은 실제 선택 파일을 건드리지 않음. 같은 경로에 쓰면 본 추론(--stage final)이
    # 가짜 응답으로 고른 변형을 읽음.
    out = SELECTED_PATH.with_name(SELECTED_PATH.stem + "_stub.json") if args.stub else SELECTED_PATH
    out.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
