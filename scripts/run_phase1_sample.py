"""Phase 1: 평가 표본 구성과 약물 희귀도 산출.

전체 33만 건을 다 돌릴 수 없음(로컬 추론으로 노트 하나에 수 초~수십 초).
그래서 표본을 쓰되, 표본이 전체를 대표하도록 두 축으로 층화함.

  - 노트 길이 3분위: 긴 노트에서 성능이 떨어지는지 보려면 길이가 고루 섞여야 함
  - 약물 개수 3분위: 약이 20개인 환자와 3개인 환자는 과제 난이도가 다름

같은 환자의 노트가 여러 번 뽑히면 표본이 한 사람에게 쏠리므로 환자당 1건만 씀.

약물 희귀도는 전체 코퍼스에서 셈. 표본에서 세면 표본에 몇 번 나왔는지가
되어 버려서, "이 약이 임상에서 얼마나 흔한가"라는 원래 뜻과 달라짐.

    python scripts/run_phase1_sample.py [--n-main 200] [--n-audit 100]
    python scripts/run_phase1_sample.py --frequency-only   # 표본은 두고 약물 빈도만 지금 파서로 다시 셈
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from datapaths import NOTE, SEED
from notes import gold_medications, neutral_items, normalize_drug

ROOT = Path(__file__).resolve().parents[1]
NOTE_DIR = NOTE
OUT_DIR = ROOT / "outputs"

# 길이 상한. 잘라내는 대신 표본에서 제외해 "잘린 노트를 평가한" 상황을 피함.
# 상한을 두는 이유는 메모리가 아니라 비교 가능성임: 모델마다 컨텍스트 창이
# 다른데 어떤 모델에서만 잘리면 그 모델이 불리해져 짝 비교가 성립하지 않음.
# 24000자는 이 코퍼스에서 상위 몇 %만 걸러내는 값임.
MAX_NOTE_CHARS = 24000
MIN_MEDICATIONS = 1


def build_corpus_frequency(notes: pl.DataFrame) -> Counter:
    """전체 코퍼스에서 약물별 등장 노트 수를 셈."""
    freq: Counter = Counter()
    for text in notes["text"]:
        g = gold_medications(text)
        if not g:
            continue
        freq.update({normalize_drug(n) for n in g if normalize_drug(n)})
    return freq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-main", type=int, default=200, help="주 조건 표본 크기")
    ap.add_argument("--n-audit", type=int, default=100, help="감사 조건 표본 크기")
    ap.add_argument("--force", action="store_true",
                    help="기존 표본과 달라도 덮어쓴다 (추론 결과가 무효가 된다)")
    ap.add_argument("--frequency-only", action="store_true",
                    help="표본은 건드리지 않고 약물 빈도(희귀도)만 지금 파서로 다시 센다")
    args = ap.parse_args()

    print("전체 퇴원요약문 로딩...")
    notes = (
        pl.scan_csv(NOTE_DIR / "discharge.csv.gz", infer_schema_length=0)
        .select("note_id", "subject_id", "hadm_id", "text")
        .collect(engine="streaming")
    )
    print(f"  {len(notes):,}건, 환자 {notes['subject_id'].n_unique():,}명")

    print("\n전체 코퍼스 약물 빈도 계산...")
    freq = build_corpus_frequency(notes)
    print(f"  고유 약물 {len(freq):,}종, 총 등장 {sum(freq.values()):,}회")

    if args.frequency_only:
        # 희귀도는 정답과 같은 파서로 세야 함. 파서를 고친 뒤 빈도 파일만 옛 값으로 남으면, 옛 파서가 버리던
        # 약(glucose gel 등)이 빈도 0 으로 읽혀 '10회 미만' 구간에 들어감(트러블슈팅 5). 표본의 정답이 지금
        # 파서와 같은지 먼저 확인함.
        sample = pl.read_parquet(OUT_DIR / "phase1_sample.parquet")
        text_by_id = dict(zip(notes["note_id"], notes["text"], strict=True))
        changed = sum((gold_medications(text_by_id[r["note_id"]]) or []) != list(r["gold"])
                      for r in sample.select("note_id", "gold").to_dicts())
        if changed:
            print(f"*** 표본 정답 {changed}건이 지금 파서와 다르다. 정답부터 다시 만들 것(트러블슈팅 6-b 절차).")
            return 1
        (OUT_DIR / "phase1_drug_frequency.json").write_text(
            json.dumps(dict(freq), ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        print(f"표본 정답은 지금 파서와 같다. 빈도만 저장: {OUT_DIR / 'phase1_drug_frequency.json'} ({len(freq):,}종)")
        return 0

    print("\n표본 후보 선별...")
    rows = []
    for note_id, subject_id, hadm_id, text in zip(
        notes["note_id"], notes["subject_id"], notes["hadm_id"], notes["text"], strict=True
    ):
        if len(text) > MAX_NOTE_CHARS:
            continue
        gold = gold_medications(text)
        if gold is None or len(gold) < MIN_MEDICATIONS:
            continue
        norm = [normalize_drug(g) for g in gold]
        norm = [n for n in norm if n]
        if not norm:
            continue
        rows.append({
            "note_id": note_id,
            "subject_id": int(subject_id),
            "hadm_id": int(hadm_id),
            "n_chars": len(text),
            "n_gold": len(norm),
            # 이 노트에서 가장 드문 약이 얼마나 드문가: 롱테일 층화에 씀
            "rarest_drug_freq": min(freq[n] for n in norm),
        })
    cand = pl.DataFrame(rows)
    print(f"  후보 {len(cand):,}건 (환자 {cand['subject_id'].n_unique():,}명)")

    # 환자당 1건만 남김
    cand = cand.sort(["subject_id", "note_id"]).unique(subset=["subject_id"], keep="first")
    print(f"  환자당 1건으로 정리: {len(cand):,}건")

    cand = cand.with_columns(
        pl.col("n_chars").qcut(3, labels=["짧음", "보통", "긺"]).alias("length_stratum"),
        pl.col("n_gold").qcut(3, labels=["적음", "보통", "많음"]).alias("count_stratum"),
    ).with_columns(
        (pl.col("length_stratum").cast(pl.Utf8) + "/" + pl.col("count_stratum").cast(pl.Utf8))
        .alias("stratum")
    )

    total_needed = args.n_main + args.n_audit
    n_strata = cand["stratum"].n_unique()
    per_stratum = total_needed // n_strata + 1
    print(f"\n층 {n_strata}개, 층당 {per_stratum}건씩 뽑아 총 {total_needed}건 구성")

    picked = (
        cand.with_columns(
            pl.int_range(pl.len()).shuffle(seed=SEED).over("stratum").alias("_r")
        )
        .filter(pl.col("_r") < per_stratum)
        .drop("_r")
        .sample(fraction=1.0, shuffle=True, seed=SEED)
    )
    if len(picked) < total_needed:
        print(f"  *** 층화로 {len(picked)}건밖에 못 뽑았다. 요청 {total_needed}건")
        return 1
    picked = picked.head(total_needed)

    main_set = picked.head(args.n_main).with_columns(pl.lit("main").alias("split"))
    audit_set = picked.tail(args.n_audit).with_columns(pl.lit("audit").alias("split"))
    sample = pl.concat([main_set, audit_set])

    # 본문과 정답을 붙임
    sample = sample.join(notes.select("note_id", "text"), on="note_id", how="left")
    golds, neutrals = [], []
    for text in sample["text"]:
        golds.append(gold_medications(text) or [])
        neutrals.append(neutral_items(text))
    sample = sample.with_columns(
        pl.Series("gold", golds),
        pl.Series("neutral", neutrals),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 이미 표본이 있는데 구성이 달라지면 덮어쓰지 않음.
    #
    # 정답 파서를 고치면 노트별 `n_gold` 가 바뀌고, 그러면 `qcut` 3분위 경계에
    # 걸린 노트들이 층을 옮김. 층이 바뀌면 층 내부 셔플이 통째로 재배열돼
    # 시드가 같아도 표본이 크게 달라짐(실측: 300건 중 229건 교체).
    #
    # 이미 그 표본으로 추론을 돌렸다면 응답 수천 건이 한 번에 무효가 되고,
    # 무엇보다 결과를 본 뒤 표본을 다시 뽑는 것은 사전등록의 취지에 어긋남.
    # 정답만 바뀐 경우라면 표본은 그대로 두고 `gold` 컬럼만 재계산해야 함.
    existing_path = OUT_DIR / "phase1_sample.parquet"
    if existing_path.exists() and not args.force:
        existing = pl.read_parquet(existing_path)
        if existing["note_id"].to_list() != sample["note_id"].to_list():
            n_diff = len(set(existing["note_id"]) - set(sample["note_id"]))
            print("\n*** 기존 표본과 구성이 다르다: 덮어쓰지 않고 중단한다.")
            print(f"    기존 {existing.height}건 중 {n_diff}건이 다른 노트로 바뀐다.")
            print("    정답 파서를 고쳤다면 표본을 다시 뽑는 대신 gold 컬럼만")
            print("    재계산할 것(README.md 6-b 절에 절차가 있다).")
            print("    정말 새로 뽑아야 한다면 --force 를 준다.")
            return 1

    sample.write_parquet(existing_path)
    (OUT_DIR / "phase1_drug_frequency.json").write_text(
        json.dumps(dict(freq), ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    print(f"\n저장: {OUT_DIR / 'phase1_sample.parquet'} ({len(sample)}건)")
    print(f"      {OUT_DIR / 'phase1_drug_frequency.json'} ({len(freq):,}종)")
    print("\n표본 구성:")
    print(sample.group_by("split", "stratum").len().sort("split", "stratum"))
    print(f"\n  노트 길이: 중앙 {sample['n_chars'].median():,.0f}자, "
          f"최소 {sample['n_chars'].min():,} 최대 {sample['n_chars'].max():,}")
    print(f"  정답 약물 수: 중앙 {sample['n_gold'].median():.0f}, "
          f"최소 {sample['n_gold'].min()} 최대 {sample['n_gold'].max()}")
    print(f"  환자 중복: {len(sample) - sample['subject_id'].n_unique()}건 (0 이어야 정상)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
