"""절제실험 한 회차: 프롬프트, 디코딩 설정을 하나씩 바꿔 audit 분할에서 측정함.

    python scripts/run_ablation.py --id 1 --name minimal \\
        --variant minimal --hypothesis "..." --change "없음(기준선)"

실행마다 `outputs/ablation/<id>_<name>.json` 에 남김. 기록 항목은
가설, 바꾼 것, 결과, 진단, 부모 회차임.

## 왜 모델 하나에서만 하는가

절제실험은 "무엇을 바꾸면 달라지는가"를 보는 것이고, 모델 16개마다 돌리면
16배가 걸림. 그래서 기반 모델 하나(gemma3-4b)에서 설정 축을 훑고, 모델별
선택은 사전등록대로 따로 함(`run_phase3_select_prompt.py`). 둘은 목적이
다름. 절제는 축을 이해하려는 것이고, 선택은 모델마다 공정한 조건을 주려는
것임.

## 왜 audit 분할인가

main 분할은 최종 평가용임. 설정을 고르는 데 쓰면 그 선택만큼 점수가
부풀려짐. audit 분할의 normal 조건 100건만 씀.

## 판정선

여기서는 시드를 바꿔도 결과가 같음: greedy 라 확률적 요소가 없음. 대신
노트 100건을 뽑은 표본 자체에서 오는 폭이 있으므로, 부트스트랩 신뢰구간을
함께 기록하고 구간이 겹치면 "차이 없음"으로 읽음.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "outputs"
ABL_DIR = OUT_DIR / "ablation"
CONDITIONS_PATH = OUT_DIR / "phase2_conditions.parquet"


def _diagnose(scores, parse_fail: int, n: int, truncated: int) -> str:
    """무엇이 문제였는지 한 줄로. 점수만 보면 원인을 모름."""
    if parse_fail > n * 0.1:
        return f"출력 형식 실패가 잦다({parse_fail}/{n}): 점수가 낮은 이유가 추출이 아니라 형식이다"
    if truncated > n * 0.05:
        return f"생성 길이 한도에 걸린 건이 있다({truncated}/{n}): max_new_tokens 를 의심할 것"
    from evaluate import hallucination_rate, omission_rate
    h, o = hallucination_rate(scores), omission_rate(scores)
    if h > o * 1.5:
        return f"환각이 누락보다 많다(환각 {h:.1%}, 누락 {o:.1%}): 없는 약을 지어낸다"
    if o > h * 1.5:
        return f"누락이 환각보다 많다(누락 {o:.1%}, 환각 {h:.1%}): 있는 약을 빠뜨린다"
    return f"환각 {h:.1%}, 누락 {o:.1%} 로 한쪽에 치우치지 않는다"


def main() -> int:
    import polars as pl
    import run_phase3_infer as infer

    from evaluate import bootstrap_ci, micro_f1, score_note
    from jsonout import parse_model_output
    from models import MODELS, load_model
    from prompts import ALL_PROMPT_VARIANTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True, help="실험 번호")
    ap.add_argument("--name", required=True, help="짧은 이름 (파일명에 쓰임)")
    ap.add_argument("--hypothesis", default="", help="이번에 무엇을 기대하고 바꾸는가")
    ap.add_argument("--change", default="", help="기준선 대비 무엇을 바꿨는가")
    ap.add_argument("--parent", type=int, default=0, help="어느 실험을 보고 정했는가")
    ap.add_argument("--model", default="gemma3-4b", choices=sorted(MODELS))
    ap.add_argument("--variant", default="schema", choices=sorted(ALL_PROMPT_VARIANTS))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만 (점검용)")
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()

    spec = MODELS[args.model]
    df = pl.read_parquet(CONDITIONS_PATH)
    rows = df.filter((pl.col("split") == "audit") & (pl.col("condition") == "normal"))
    rows = rows.with_columns(pl.lit(args.variant).alias("variant"))
    rows = list(rows.iter_rows(named=True))
    if args.limit:
        rows = rows[: args.limit]

    print("=" * 74)
    print(f"  절제 #{args.id}: {args.name}")
    print("=" * 74)
    print(f"  모델    : {spec.label} ({spec.precision})")
    print(f"  설정    : variant={args.variant} max_new_tokens={args.max_new_tokens} "
          f"batch={args.batch}")
    print(f"  대상    : audit/normal {len(rows)}건")
    if args.change:
        print(f"  바꾼 것 : {args.change}")

    freq_path = OUT_DIR / "phase1_drug_frequency.json"
    drug_freq = json.loads(freq_path.read_text(encoding="utf-8")) if freq_path.exists() else {}

    t0 = time.time()
    if args.stub:
        raw = [infer._stub_response(r) for r in rows]
    else:
        tokenizer, model = load_model(spec)
        # 한도를 회차마다 바꿀 수 있어야 함. 생성 상한은 ModelSpec 이 들고
        # 있으므로 이 실행에서만 바뀐 사본을 만들어 넘김.
        spec = dataclasses.replace(spec, max_new_tokens=args.max_new_tokens)
        raw = []
        for batch in infer._batches(rows, args.batch):
            raw += infer._generate(tokenizer, model, batch, spec)[0]
            if len(raw) % (args.batch * 5) < args.batch:
                el = time.time() - t0
                print(f"    [{len(raw)}/{len(rows)}] {el / 60:.1f}분, "
                      f"남은 예상 {el / len(raw) * (len(rows) - len(raw)) / 60:.1f}분")
    elapsed = time.time() - t0

    # 어떤 경로로 답을 건졌는지 셈. 점수만 보면 "형식을 못 맞춘 것"과
    # "약을 못 찾은 것"이 구분되지 않음.
    recovered: dict[str, int] = {}
    samples: list[dict] = []

    scores, parse_fail, truncated = [], 0, 0
    for row, text in zip(rows, raw, strict=True):
        parsed = parse_model_output(text)
        recovered[parsed.recovered_by] = recovered.get(parsed.recovered_by, 0) + 1
        if len(samples) < 3:
            # 원문 몇 개를 남김. 나중에 "왜 이 점수인가"를 볼 때 필요함.
            # 단 기록 파일에는 넣지 않음. 모델 응답은 노트의 약 처방을 그대로
            # 옮겨 적으므로 MIMIC 원문의 일부이고, 기록은 저장소에 올라감(DUA).
            # 원문은 깃에서 빠지는 outputs/responses/ 아래 따로 둠.
            samples.append({"note_id": row["note_id"],
                            "response": text[:600],
                            "recovered_by": parsed.recovered_by,
                            "n_parsed": len(parsed.medications)})
        # `recovered_by` 가 "failed" 면 네 경로를 모두 시도하고도 못 건진 것,
        # "none" 은 응답이 비어 있던 것임. 둘 다 형식 실패로 셈.
        failed = parsed.recovered_by in ("none", "failed")
        if failed:
            parse_fail += 1
            # 닫는 괄호 없이 끝났으면 한도에 걸려 잘린 것으로 봄.
            if text.strip() and not text.strip().endswith("]"):
                truncated += 1
        scores.append(score_note(
            gold=list(row["gold"]), predicted=parsed.medications,
            neutral=list(row["neutral"]), drug_frequency=drug_freq))

    p, r, f1 = micro_f1(scores)
    lo, hi = bootstrap_ci(scores, lambda s: micro_f1(s)[2], seed=20260911)
    diagnosis = _diagnose(scores, parse_fail, len(rows), truncated)

    print(f"\n[결과] micro-F1 {f1:.4f}  (95% CI {lo:.4f}~{hi:.4f})")
    print(f"       precision {p:.4f}  recall {r:.4f}  형식실패 {parse_fail}/{len(rows)}")
    print("       형식 경로: " + ", ".join(f"{k} {v}" for k, v in
          sorted(recovered.items(), key=lambda kv: -kv[1])))
    print(f"       {diagnosis}")
    print(f"       소요 {elapsed / 60:.1f}분")

    ABL_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "id": args.id, "name": args.name, "parent": args.parent,
        "hypothesis": args.hypothesis, "change": args.change or "없음(기준선)",
        "config": {
            "model": spec.key, "precision": spec.precision, "variant": args.variant,
            "max_new_tokens": args.max_new_tokens, "batch": args.batch,
            "decoding": "greedy", "split": "audit/normal", "n_notes": len(rows),
        },
        "result": {
            "micro_f1": f1, "ci_low": lo, "ci_high": hi,
            "precision": p, "recall": r,
            "parse_failures": parse_fail, "truncated": truncated,
            "strict_json": recovered.get("strict", 0),
            # strict: 지시대로 순수 JSON / fence: 마크다운 펜스 안 / embedded_array:
            # 설명 사이에 낀 배열 / bullets: 번호목록 / failed: 아무것도 못 건짐
            "recovered_by": dict(sorted(recovered.items(), key=lambda kv: -kv[1])),
        },
        # 원문 없이 형식만 남김: 어느 경로로 건졌고 몇 개를 뽑았는가.
        "samples": [{"recovered_by": x["recovered_by"], "n_parsed": x["n_parsed"],
                     "response_chars": len(x["response"])} for x in samples],
        "samples_raw": f"outputs/responses/ablation_samples/{args.id:02d}_{args.name}.json",
        "diagnosis": diagnosis,
        "elapsed_sec": round(elapsed, 1),
        "note": "audit 분할로만 평가. main 분할은 최종 1회만 본다.",
    }
    path = ABL_DIR / f"{args.id:02d}_{args.name}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    raw_path = ROOT / rec["samples_raw"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"\n[기록] {path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
