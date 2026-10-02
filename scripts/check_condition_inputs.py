"""감사 조건 입력(phase2_conditions.parquet)이 사전등록 6절 사양대로 만들어졌는지 셈. 노트 원문은 찍지 않고 비율만 남김.

`other_patient`(남의 노트) 조건을 판정에서 빼는 근거가 "정답 약물의 16.5% 가 남의 노트에도 있다" 임. 이 수치와
사전등록 6절 표의 나머지 값(정답이 하나라도 남은 노트 61%, 중앙값 12.5%, 0% 인 노트 39%, 25% 이상 33%)을
저장소의 별칭 규칙(`notes.drug_aliases`, README 채점 규칙 2번)으로 다시 셈.

잔존의 정의(사전등록 6절): audit 100건에서 정답 항목의 별칭 가운데 하나라도, 입력을 소문자로 바꾸고 a-z, 0-9, -, /, +
밖의 문자를 공백으로 바꾼 글에 부분 문자열로 들어 있으면 남은 것으로 셈. 단어 경계를 요구한 값도 함께 남김.

    python scripts/check_condition_inputs.py     # 결과: outputs/condition_inputs_check.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from notes import drug_aliases  # noqa: E402

CONDITIONS_PATH = ROOT / "outputs" / "phase2_conditions.parquet"
OUT = ROOT / "outputs" / "condition_inputs_check.json"
HEADER = re.compile(r"discharge\s+medications\s*:", re.I)


def normalize(text: str) -> str:
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\-/+ ]", " ", text.lower())) + " "


def residue(gold: list[str], text: str, *, word_boundary: bool) -> float:
    nt = normalize(text)
    hit = sum(1 for g in gold
              if any((f" {a} " if word_boundary else a) in nt for a in drug_aliases(g)))
    return hit / len(gold) if gold else 0.0


def summarize(rs: list[float]) -> dict:
    return {"n": len(rs), "any_pct": round(100 * sum(x > 0 for x in rs) / len(rs), 1),
            "mean_pct": round(100 * statistics.mean(rs), 1), "median_pct": round(100 * statistics.median(rs), 1),
            "zero_pct": round(100 * sum(x == 0 for x in rs) / len(rs), 1),
            "ge25_pct": round(100 * sum(x >= 0.25 for x in rs) / len(rs), 1)}


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df = pl.read_parquet(CONDITIONS_PATH).filter(pl.col("split") == "audit")
    normal = {r["note_id"]: r["input_text"] for r in df.filter(pl.col("condition") == "normal").to_dicts()}
    out: dict = {}
    for cond in ("empty", "other_patient"):
        rows = df.filter(pl.col("condition") == cond).to_dicts()
        for wb in (False, True):
            key = f"{cond}_{'word_boundary' if wb else 'substring'}"
            out[key] = summarize([residue(list(r["gold"]), r["input_text"], word_boundary=wb) for r in rows])
    sh = df.filter(pl.col("condition") == "shuffled").to_dicts()
    out["shuffled"] = {
        "n": len(sh),
        "same_token_multiset": sum(sorted(r["input_text"].split()) == sorted(normal[r["note_id"]].split()) for r in sh),
        "identical_to_normal": sum(r["input_text"] == normal[r["note_id"]] for r in sh)}
    ns = df.filter(pl.col("condition") == "no_section").to_dicts()
    out["no_section"] = {"n": len(ns), "header_left": sum(bool(HEADER.search(r["input_text"])) for r in ns),
                         "normal_header_present": sum(bool(HEADER.search(normal[r["note_id"]])) for r in ns)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False))
    print(f"결과 저장: {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
