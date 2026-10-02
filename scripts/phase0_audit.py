"""Phase 0: 데이터 전수조사와 정답 파서 실측 검증.

합성 예시로 짠 테스트가 통과하는 것과, 33만 건의 실제 노트에서 파서가
제대로 도는 것은 다른 문제임. 이 스크립트는 실제 노트에 파서를 돌려
(1) 얼마나 파싱되는지 (2) 뽑힌 약물명이 실제 약국 어휘에 있는지를 확인함.

약물명이 prescriptions 어휘에 없다면 둘 중 하나임: 파서가 용량이나 지시문을
약물명으로 잘못 잡았거나, 그 병원에서 쓰지 않는 표기이거나. 전자는 고쳐야
할 결함이므로 미매칭 표기를 전부 눈으로 확인할 수 있게 출력함.

    python scripts/phase0_audit.py                      # 앞 20,000건. 결과: outputs/phase0_audit_first20000.json
    python scripts/phase0_audit.py --random 25000       # 전 구간 무작위 25,000건(시드 고정). outputs/phase0_audit_random25000.json

결과 JSON 에는 비율과 개수만 남김(약물명, 노트 내용은 남기지 않음). 문서의 파싱률
98.1%, 94.0%, 96.6% 와 무작위 표본 검증은 이 파일에서 옴.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl

from datapaths import MIMICIV, NOTE, SEED
from notes import (
    drug_aliases,
    extract_section,
    gold_medications,
    normalize_drug,
    parse_items,
)

NOTE_DIR = NOTE
MIMIC_DIR = MIMICIV
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

REQUIRED_FILES = (
    NOTE_DIR / "discharge.csv.gz",
    NOTE_DIR / "radiology.csv.gz",
    MIMIC_DIR / "hosp/prescriptions.csv.gz",
    MIMIC_DIR / "hosp/admissions.csv.gz",
    MIMIC_DIR / "hosp/patients.csv.gz",
)

# 약물명 자리에 들어오면 파서가 어긋난 것이 확실한 표기: 용법, 조제 지시문임.
HARD_DEFECT_TOKENS = ("sig", "disp", "refill", "by mouth", "puff", "times a day")

# 제형 표기는 결함이 아님. 용량이 없는 약은 원래 'Multivitamin Tablet' 처럼
# 적히고, drug_aliases 가 제형을 뗀 별칭을 함께 만들어 채점에서 인정함.
# 다만 제형을 떼도 약국 어휘에 없으면 확인이 필요하므로 따로 셈.
DOSAGE_FORM_TOKENS = ("tablet", "capsule", "solution", "ointment", "powder", "packet")


def check_files() -> bool:
    print("=== 1. 필요 파일 존재 확인 ===")
    ok = True
    for p in REQUIRED_FILES:
        exists = p.is_file()
        print(f"  {'있음' if exists else '없음 ***'}  {p.name}")
        ok &= exists
    return ok


def load_prescription_vocabulary() -> set[str]:
    drugs = (
        pl.scan_csv(MIMIC_DIR / "hosp/prescriptions.csv.gz", infer_schema_length=0)
        .select("drug")
        .unique()
        .collect(engine="streaming")
    )
    vocab = set()
    for d in drugs["drug"].drop_nulls().to_list():
        norm = normalize_drug(d)
        if norm:
            vocab.add(norm)
    return vocab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-notes", type=int, default=20000, help="앞에서부터 읽을 노트 수")
    ap.add_argument("--random", type=int, default=0, help="전 구간에서 무작위로 뽑을 노트 수(주면 --n-notes 대신 쓴다)")
    ap.add_argument("--seed", type=int, default=SEED, help="무작위 표본의 시드")
    args = ap.parse_args()

    ok = check_files()
    if not ok:
        print("\n필요 파일이 없다. 이후 검사는 의미가 없으므로 중단한다.")
        return 1

    total_rows = (
        pl.scan_csv(NOTE_DIR / "discharge.csv.gz", infer_schema_length=0)
        .select(pl.len())
        .collect()
        .item()
    )
    scan = pl.scan_csv(NOTE_DIR / "discharge.csv.gz", infer_schema_length=0).select(
        "note_id", "subject_id", "hadm_id", "text")
    if args.random:
        # 앞부분만 보면 파일 뒤쪽의 표기를 놓침(트러블슈팅 5).
        mode = f"random{args.random}"
        print(f"\n=== 2. 퇴원요약문 로딩 (전 구간 무작위 {args.random:,}건, 시드 {args.seed}) ===")
        rows = np.sort(np.random.default_rng(args.seed).choice(total_rows, size=args.random, replace=False))
        notes = scan.with_row_index("row").filter(pl.col("row").is_in(rows.tolist())).drop("row").collect()
    else:
        mode = f"first{args.n_notes}"
        print(f"\n=== 2. 퇴원요약문 로딩 (앞 {args.n_notes:,}건) ===")
        notes = scan.head(args.n_notes).collect()
    if not len(notes):
        print("  *** 노트를 한 건도 읽지 못했다.")
        return 1
    print(f"  전체 노트 {total_rows:,}건 중 {len(notes):,}건 조사")
    print(f"  고유 환자 {notes['subject_id'].n_unique():,}명")

    print("\n=== 3. 퇴원 약물 절 파싱률 ===")
    n_section = n_items = 0
    counts: list[int] = []
    all_names: list[str] = []
    empty_section: list[str] = []
    for note_id, text in zip(notes["note_id"], notes["text"], strict=True):
        sec = extract_section(text)
        if sec is None:
            continue
        n_section += 1
        items = parse_items(sec)
        if not items:
            empty_section.append(note_id)
            continue
        n_items += 1
        names = gold_medications(text) or []
        counts.append(len(names))
        all_names.extend(names)

    print(f"  절이 있는 노트          {n_section:>7,} ({n_section/len(notes)*100:5.1f}%)")
    print(f"  약물 항목까지 뽑힌 노트 {n_items:>7,} ({n_items/len(notes)*100:5.1f}%)")
    print(f"  절은 있으나 항목이 0개  {len(empty_section):>7,}")
    if counts:
        s = pl.Series(counts)
        print(f"  노트당 약물 수: 중앙 {s.median():.0f}, 평균 {s.mean():.1f}, "
              f"최소 {s.min()}, 최대 {s.max()}")

    print("\n=== 4. 뽑힌 약물명이 약국 어휘에 있는가 ===")
    vocab = load_prescription_vocabulary()
    print(f"  prescriptions 고유 약물명 {len(vocab):,}종")

    name_counts = Counter(normalize_drug(n) for n in all_names)
    name_counts.pop("", None)
    matched = {n: c for n, c in name_counts.items() if n in vocab}
    unmatched = {n: c for n, c in name_counts.items() if n not in vocab}
    tot = sum(name_counts.values())
    tot_matched = sum(matched.values())
    print(f"  추출 약물 언급 {tot:,}건 / 고유 {len(name_counts):,}종")
    if not tot:
        print("  *** 추출된 약물이 하나도 없다. --n-notes 가 너무 작거나 파서가 죽었다.")
        return 1
    print(f"  어휘 일치: 언급 기준 {tot_matched/tot*100:.1f}%, "
          f"고유 기준 {len(matched)/len(name_counts)*100:.1f}%")

    print("\n=== 5. 파서 결함 신호 ===")
    # 단어 경계가 없으면 부분문자열로 걸림('rosiglitazone maleate', 'tasigna',
    # 'methadone (dispersible tablet)' 등 약국 어휘 5종이 결함으로 오판됨). \b 로 묶음.
    defect_re = re.compile(r"\b(?:" + "|".join(HARD_DEFECT_TOKENS) + r")\b")
    bad = [n for n in name_counts if defect_re.search(n)]
    print(f"  용법, 조제 지시문이 약물명에 섞인 것: {len(bad)}종 (0 이어야 정상)")
    for n in sorted(bad, key=lambda x: -name_counts[x])[:15]:
        print(f"    {name_counts[n]:>6,}회  {n!r}")
    ok &= len(bad) == 0

    too_long = [n for n in name_counts if len(n.split()) > 6]
    print(f"  7단어 이상으로 지나치게 긴 약물명: {len(too_long)}종 (0 이어야 정상)")
    for n in sorted(too_long, key=lambda x: -name_counts[x])[:10]:
        print(f"    {name_counts[n]:>6,}회  {n!r}")
    ok &= len(too_long) == 0

    form_named = [n for n in name_counts if any(t in n.split() for t in DOSAGE_FORM_TOKENS)]
    unresolved = [n for n in form_named if not (drug_aliases(n) & vocab)]
    print(f"  제형이 붙은 약물명 {len(form_named)}종: 별칭으로도 어휘에 못 닿는 것 "
          f"{len(unresolved)}종")
    for n in sorted(unresolved, key=lambda x: -name_counts[x])[:10]:
        print(f"    {name_counts[n]:>6,}회  {n!r}")

    print("\n  어휘 미매칭 상위 20종 (파서 결함인지 눈으로 확인할 것):")
    for n in sorted(unmatched, key=lambda x: -unmatched[x])[:20]:
        print(f"    {unmatched[n]:>6,}회  {n!r}")

    print("\n=== 6. 약물 흔한 정도 (롱테일 확인) ===")
    ranked = name_counts.most_common()
    top10 = ranked[:10]
    print("  가장 흔한 10종 (빈도 베이스라인 후보):")
    for n, c in top10:
        print(f"    {c:>7,}  {n}")
    singleton = sum(1 for _, c in ranked if c == 1)
    print(f"  단 1회만 등장하는 약물: {singleton:,}종 "
          f"({singleton/len(ranked)*100:.1f}%: 롱테일 감사에 쓸 수 있다)")

    summary = {"mode": mode, "seed": args.seed if args.random else None, "total_notes": total_rows,
               "n_notes": len(notes), "n_patients": notes["subject_id"].n_unique(),
               "section_rate": n_section / len(notes), "item_rate": n_items / len(notes),
               "section_without_items": len(empty_section),
               "mentions": tot, "unique_names": len(name_counts),
               "vocab_match_mentions": tot_matched / tot, "vocab_match_unique": len(matched) / len(name_counts),
               "defect_names": len(bad), "too_long_names": len(too_long), "form_named": len(form_named),
               "form_named_unresolved": len(unresolved), "singleton_names": singleton, "passed": bool(ok)}
    out = OUT_DIR / f"phase0_audit_{mode}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n결과 저장: {out}")

    print("\n" + "=" * 50)
    if ok:
        print("전수조사 통과. 정답 파서가 실제 노트에서 정상 동작한다.")
        return 0
    print("결함이 있다. 위 *** 표시를 해결할 것.")
    return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
