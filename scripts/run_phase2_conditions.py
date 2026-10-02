"""Phase 2: 감사 조건별 입력 텍스트 생성.

docs/prereg.md 6절에 등록한 조건을 그대로 만듦. 조건마다 노트 본문을
바꿔 넣되 정답은 언제나 원래 노트의 것을 유지함. 남의 노트를 주고
"이 환자의 퇴원약"을 물었을 때 원래 환자의 약을 맞히면 그건 노트를 읽은
것이 아니라는 뜻이므로, 정답을 바꿔치면 그 검사가 성립하지 않음.

    python scripts/run_phase2_conditions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from datapaths import SEED
from notes import remove_section
from prompts import header_only, shuffle_sentences

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
SAMPLE_PATH = OUT_DIR / "phase1_sample.parquet"
OUT_PATH = OUT_DIR / "phase2_conditions.parquet"

AUDIT_CONDITIONS = ("no_section", "empty", "other_patient", "shuffled")


def build() -> pl.DataFrame:
    sample = pl.read_parquet(SAMPLE_PATH)
    rows = []

    # 정상 조건은 두 분할 모두에 만듦.
    #   audit 분할 = 프롬프트 선택용, main 분할 = 최종 평가용
    for r in sample.iter_rows(named=True):
        rows.append({
            "note_id": r["note_id"], "split": r["split"], "condition": "normal",
            "input_text": r["text"], "gold": r["gold"], "neutral": r["neutral"],
            "n_chars": r["n_chars"], "length_stratum": r["length_stratum"],
        })

    audit = sample.filter(pl.col("split") == "audit")
    n = len(audit)
    audit_rows = list(audit.iter_rows(named=True))

    for i, r in enumerate(audit_rows):
        base = {
            "note_id": r["note_id"], "split": "audit",
            "gold": r["gold"], "neutral": r["neutral"],
            "n_chars": r["n_chars"], "length_stratum": r["length_stratum"],
        }
        # 퇴원약 절만 들어냄: 답이 본문에 없을 때도 맞히는지
        rows.append({**base, "condition": "no_section",
                     "input_text": remove_section(r["text"])})
        # 인적사항만: 임상 내용 없이 내놓는 사전 지식
        rows.append({**base, "condition": "empty",
                     "input_text": header_only(r["text"])})
        # 다른 환자의 노트. 정답은 원래 환자 것 그대로 둠.
        # 순환 이동이라 자기 자신과 짝지어지지 않음.
        other = audit_rows[(i + 1) % n]
        rows.append({**base, "condition": "other_patient",
                     "input_text": other["text"],
                     "donor_note_id": other["note_id"]})
        # 문장 순서만 흩음: 내용은 그대로, 구조만 깨뜨림
        rows.append({**base, "condition": "shuffled",
                     "input_text": shuffle_sentences(r["text"], seed=SEED + i)})

    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> int:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"저장: {OUT_PATH} ({len(df):,}행)")
    print("\n조건별 건수:")
    print(df.group_by("split", "condition").len().sort("split", "condition"))

    print("\n무결성 확인:")
    norm = df.filter(pl.col("condition") == "normal")
    empty = df.filter(pl.col("condition") == "empty")
    nosec = df.filter(pl.col("condition") == "no_section")
    other = df.filter(pl.col("condition") == "other_patient")

    print(f"  empty 조건 평균 길이 {empty['input_text'].str.len_chars().mean():,.0f}자 "
          f"(normal {norm['input_text'].str.len_chars().mean():,.0f}자보다 훨씬 짧아야 한다)")

    leaked = sum(
        1 for r in nosec.iter_rows(named=True)
        if "Discharge Medications" in r["input_text"]
    )
    print(f"  no_section 에 퇴원약 절이 남은 건수: {leaked} (0 이어야 정상)")

    self_paired = sum(
        1 for r in other.iter_rows(named=True) if r["note_id"] == r["donor_note_id"]
    )
    print(f"  other_patient 가 자기 자신과 짝지어진 건수: {self_paired} (0 이어야 정상)")

    shuffled = df.filter(pl.col("condition") == "shuffled")
    same = sum(
        1 for a, b in zip(
            shuffled.sort("note_id")["input_text"],
            norm.filter(pl.col("split") == "audit").sort("note_id")["input_text"],
            strict=True,
        ) if a == b
    )
    print(f"  shuffled 가 원문과 동일한 건수: {same} (0 이어야 정상)")
    return 0


if __name__ == "__main__":
    import argparse

    # 인자는 없음. 읽어 두어야 --help 가 설명만 찍고 끝남.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
