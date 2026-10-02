"""쌍8 MeditronFO 를 기반 토크나이저로 다시 돌려 토큰화 차이가 점수를 바꾸는지 측정함.

    python scripts/check_tokenizer_sensitivity.py --model olmo2-32b-meditronfo --batch 1
    python scripts/check_tokenizer_sensitivity.py --model olmo2-32b-meditronfo --stage final --batch 1

## 왜 확인해야 하는가

`EPFLiGHT/OLMo-2-32B-MeditronFO` 의 tokenizer.json 에는 기반 모델(OLMo-2 Instruct)의
사전 분할 정규식이 빠져 있음. 어휘와 채팅 템플릿은 같은데 같은 글이 다르게
쪼개져 입력이 약 9% 길어짐. 그대로 짝 비교를 하면 의료 특화 효과와 토큰화
차이가 섞임. 판정 규칙은 docs/prereg.md 부록에 결과를 보기 전에 적음.

## 무엇을 비교하는가

선택 단계와 같은 300건(audit 정상 조건 100건 x 변형 3개)을 가중치는 그대로 두고
토크나이저만 기반 것으로 바꿔 생성함. 배포 토크나이저로 이미 만든 선택 단계
응답과 같은 행끼리 짝짓음. 두 쪽 모두 문맥 안에 드는 행만 씀: 기반
토크나이저는 입력이 짧아 문맥 초과가 덜 나므로, 각자 행을 쓰면 표본이 달라짐.

응답 원문은 깃에서 빠지는 outputs/responses/ 에만 두고, 결과 파일에는 숫자만 남김.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "outputs"
RESPONSE_DIR = OUT_DIR / "responses"

# docs/prereg.md 부록의 판정선
MIN_ABS_DIFF = 0.02


def _key(r: dict) -> tuple[str, str, str]:
    return (r["note_id"], r["condition"], r["variant"])


def compare(shipped: list[dict], base: list[dict], spec, gold_by_note: dict, freq: dict) -> dict:
    """같은 행끼리 짝지어 변형별 F1 과 짝 부트스트랩 차이(기반 - 배포)를 냄."""
    from evaluate import micro_f1, paired_bootstrap_diff, score_note
    from jsonout import parse_model_output
    from models import exceeds_context

    def f1(scores):
        return micro_f1(scores)[2]

    def score(r):
        gold, neutral = gold_by_note[r["note_id"]]
        parsed = parse_model_output(r["response"])
        return score_note(gold=list(gold), predicted=parsed.medications,
                          neutral=list(neutral), drug_frequency=freq), parsed

    by_s = {_key(r): r for r in shipped}
    by_b = {_key(r): r for r in base}
    common = sorted(set(by_s) & set(by_b))
    both_in = [k for k in common
               if not exceeds_context(by_s[k], spec) and not exceeds_context(by_b[k], spec)]

    variants: dict[str, dict] = {}
    for v in sorted({k[2] for k in both_in}):
        ks = [k for k in both_in if k[2] == v]
        s_sc, b_sc, same, s_fail, b_fail = [], [], 0, 0, 0
        for k in ks:
            ss, sp = score(by_s[k])
            bs, bp = score(by_b[k])
            s_sc.append(ss)
            b_sc.append(bs)
            same += by_s[k]["response"].strip() == by_b[k]["response"].strip()
            s_fail += sp.recovered_by in ("none", "failed")
            b_fail += bp.recovered_by in ("none", "failed")
        diff, lo, hi = paired_bootstrap_diff(b_sc, s_sc, f1)
        variants[v] = {
            "n_rows": len(ks),
            "f1_shipped": f1(s_sc), "f1_base_tokenizer": f1(b_sc),
            "diff_base_minus_shipped": diff, "ci_low": lo, "ci_high": hi,
            "identical_responses": same,
            "parse_failures_shipped": s_fail, "parse_failures_base_tokenizer": b_fail,
        }

    # 선택 단계와 같은 방식(각자 문맥 안의 행, 세 변형 공통 노트)으로 고를 변형이 달라지는가
    def pick(rows_by_key):
        per_v: dict[str, dict[str, object]] = {}
        for k, r in rows_by_key.items():
            if not exceeds_context(r, spec):
                per_v.setdefault(k[2], {})[k[0]] = score(r)[0]
        if not per_v:
            return None
        notes = set.intersection(*(set(d) for d in per_v.values()))
        f1s = {v: f1([d[n] for n in sorted(notes)]) for v, d in per_v.items()}
        # 동점이면 정렬 순서상 앞의 변형(선택 스크립트와 같음). max 는 첫 최댓값을 돌려줌.
        return max(sorted(f1s), key=lambda v: f1s[v])

    tok_s = sum(by_s[k]["n_input_tokens"] for k in common)
    tok_b = sum(by_b[k]["n_input_tokens"] for k in common)
    decisive = any(x["ci_low"] > 0 or x["ci_high"] < 0 for x in variants.values()
                   if abs(x["diff_base_minus_shipped"]) >= MIN_ABS_DIFF)
    sel_s, sel_b = pick({k: by_s[k] for k in common}), pick({k: by_b[k] for k in common})
    return {
        "n_rows_common": len(common),
        "n_rows_both_within_context": len(both_in),
        "n_excluded_context_shipped": sum(exceeds_context(by_s[k], spec) for k in common),
        "n_excluded_context_base_tokenizer": sum(exceeds_context(by_b[k], spec) for k in common),
        "input_tokens_ratio_shipped_over_base": tok_s / tok_b if tok_b else None,
        "by_variant": variants,
        "selected_variant_shipped": sel_s,
        "selected_variant_base_tokenizer": sel_b,
        "rule": f"차이 95% 구간이 0 을 포함하지 않고 |차이| >= {MIN_ABS_DIFF}, 또는 선택 변형이 바뀜",
        "triggers_followup": bool(decisive or sel_s != sel_b),
    }


def compare_final(shipped: list[dict], base: list[dict], partner: list[dict],
                  spec, partner_spec, meta: dict, freq: dict) -> dict:
    """쌍8 을 두 방식으로 봄. main 분할 정상 조건, 세 파일 모두 문맥 안인 노트만.

    같은 노트 위에서 (배포 MeditronFO - 기반 모델) 과 (기반 토크나이저 MeditronFO - 기반 모델)
    을 나란히 내야 두 결론이 표본 차이 없이 비교됨.
    """
    from evaluate import micro_f1, paired_bootstrap_diff, score_note
    from jsonout import parse_model_output
    from models import exceeds_context

    def f1(scores):
        return micro_f1(scores)[2]

    def by_note(rows, sp):
        out = {}
        for r in rows:
            if r["split"] != "main" or r["condition"] != "normal" or exceeds_context(r, sp):
                continue
            m = meta[(r["note_id"], r["condition"])]
            out[r["note_id"]] = score_note(gold=list(m["gold"]),
                                           predicted=parse_model_output(r["response"]).medications,
                                           neutral=list(m["neutral"]), drug_frequency=freq)
        return out

    s, b, p = by_note(shipped, spec), by_note(base, spec), by_note(partner, partner_spec)
    common = sorted(set(s) & set(b) & set(p))
    ss, bs, ps = [s[n] for n in common], [b[n] for n in common], [p[n] for n in common]

    def diff(x, y):
        d, lo, hi = paired_bootstrap_diff(x, y, f1)
        return {"diff": d, "ci": [lo, hi], "significant": not (lo <= 0 <= hi)}

    return {
        "split": "main/normal",
        "n_notes_common": len(common),
        "n_notes_within_context": {"shipped": len(s), "base_tokenizer": len(b), "partner": len(p)},
        "f1": {"meditronfo_shipped": f1(ss), "meditronfo_base_tokenizer": f1(bs), "partner": f1(ps)},
        "pair8_shipped_minus_partner": diff(ss, ps),
        "pair8_base_tokenizer_minus_partner": diff(bs, ps),
        "base_tokenizer_minus_shipped": diff(bs, ss),
    }


def _generate_rows(spec, base_id: str, todo: list[dict], out_path: Path, batch_size: int) -> int:
    """가중치는 그대로, 토크나이저만 기반 것으로 바꿔 생성해 JSONL 로 이어 씀."""
    import run_phase3_infer as infer
    from transformers import AutoTokenizer

    from models import load_model, local_kwargs

    _, model = load_model(spec)
    tokenizer = AutoTokenizer.from_pretrained(base_id, **local_kwargs(base_id))
    shipped_tok = AutoTokenizer.from_pretrained(spec.hf_id, **local_kwargs(spec.hf_id))
    # 템플릿까지 달라지면 토크나이저만 바꾼 비교가 아님. 글자 단위로 같은지 먼저 봄.
    msg = [{"role": "user", "content": infer.build_prompt(todo[0]["input_text"], todo[0]["variant"])}]
    a = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    b = shipped_tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    if a != b or len(tokenizer) != len(shipped_tok):
        print("[중단] 채팅 템플릿이나 어휘 크기가 다르다. 토크나이저만 바꾼 비교가 되지 않는다.")
        return 1
    t_start = time.time()
    n = 0
    with out_path.open("a", encoding="utf-8") as f:
        for batch in infer._batches(todo, batch_size):
            texts, n_ins, n_outs = infer._generate(tokenizer, model, batch, spec)
            for row, text, n_in, n_out in zip(batch, texts, n_ins, n_outs, strict=True):
                f.write(json.dumps({
                    "model": spec.key, "note_id": row["note_id"], "split": row["split"],
                    "condition": row["condition"], "variant": row["variant"],
                    "response": text, "n_input_tokens": n_in, "n_output_tokens": n_out,
                    "batch_size": len(batch), "max_new_tokens": spec.max_new_tokens,
                    "template_kwargs": dict(spec.template_kwargs), "tokenizer": base_id,
                }, ensure_ascii=False) + "\n")
            f.flush()
            n += len(batch)
            if n % (batch_size * 25) < batch_size or n == len(todo):
                el = time.time() - t_start
                print(f"  [{n}/{len(todo)}] 경과 {el / 60:.1f}분, "
                      f"남은 예상 {el / n * (len(todo) - n) / 60:.1f}분")
    return 0


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    import polars as pl
    import run_phase3_infer as infer

    from models import MODELS, PAIRS

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2-32b-meditronfo", choices=sorted(MODELS))
    ap.add_argument("--stage", default="select", choices=("select", "final"),
                    help="final 은 prereg 판정선을 넘었을 때만 돌린다")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    spec = MODELS[args.model]
    base_id = spec.base_of_record
    stage = args.stage
    shipped_path = RESPONSE_DIR / f"{spec.key}_{stage}.jsonl"
    out_path = RESPONSE_DIR / f"{spec.key}_{stage}_basetok.jsonl"
    shipped = _load(shipped_path)
    print(f"모델 {spec.label}  단계 {stage}\n  토크나이저 {base_id} (배포본 대신)\n"
          f"  비교 대상 {shipped_path.name} {len(shipped)}건")

    variant = None
    if stage == "final":
        # 배포 토크나이저로 고른 변형을 그대로 씀. 선택 단계 민감도에서 두 토크나이저의
        # 선택이 같음(schema). 달랐다면 이 비교에 변형 차이까지 섞임.
        selected = json.loads(infer.SELECTED_PATH.read_text(encoding="utf-8"))
        variant = selected[spec.key]["variant"]
        print(f"  변형 {variant} (선택 결과)")
    jobs = infer.build_jobs(stage, variant)
    infer._repair_trailing_line(out_path)
    done = infer._done_keys(out_path)
    todo = [r for r in jobs.iter_rows(named=True) if _key(r) not in done]
    print(f"  전체 {len(jobs)}건, 완료 {len(done)}건, 남은 {len(todo)}건")
    if todo and _generate_rows(spec, base_id, todo, out_path, args.batch):
        return 1

    base = _load(out_path)
    conditions = pl.read_parquet(OUT_DIR / "phase2_conditions.parquet")
    freq = json.loads((OUT_DIR / "phase1_drug_frequency.json").read_text(encoding="utf-8"))
    path = OUT_DIR / f"tokenizer_sensitivity_{spec.key}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    if stage == "final":
        partner_key = next(b for a, b, _ in PAIRS if a == spec.key)
        meta = {(r["note_id"], r["condition"]): r for r in conditions.iter_rows(named=True)}
        fin = compare_final(shipped, base, _load(RESPONSE_DIR / f"{partner_key}_final.jsonl"),
                            spec, MODELS[partner_key], meta, freq)
        fin = {"partner": partner_key, "variant": variant, **fin}
        print(f"\n공통 노트 {fin['n_notes_common']}건 (문맥 안: {fin['n_notes_within_context']})")
        print("  micro-F1 " + ", ".join(f"{k} {v:.4f}" for k, v in fin["f1"].items()))
        for k in ("pair8_shipped_minus_partner", "pair8_base_tokenizer_minus_partner",
                  "base_tokenizer_minus_shipped"):
            x = fin[k]
            sig = "  유의" if x["significant"] else ""
            print(f"  {k:<38} {x['diff']:+.4f} ({x['ci'][0]:+.4f}~{x['ci'][1]:+.4f}){sig}")
        old["final"] = fin
        path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        print(f"[기록] {path} (final 항목)")
        return 0

    gold_by_note = {r["note_id"]: (r["gold"], r["neutral"])
                    for r in conditions.filter(pl.col("condition") == "normal").iter_rows(named=True)}
    res = compare(shipped, base, spec, gold_by_note, freq)
    res = {"model": spec.key, "shipped_tokenizer": spec.hf_id, "base_tokenizer": base_id, **res}
    print(f"\n두 토크나이저 모두 문맥 안 {res['n_rows_both_within_context']}/{res['n_rows_common']}건 "
          f"(문맥 초과: 배포 {res['n_excluded_context_shipped']}, 기반 {res['n_excluded_context_base_tokenizer']})")
    print(f"입력 토큰 배포/기반 {res['input_tokens_ratio_shipped_over_base']:.3f}")
    for v, x in res["by_variant"].items():
        print(f"  {v:<9} 배포 {x['f1_shipped']:.4f}  기반 {x['f1_base_tokenizer']:.4f}  "
              f"차이 {x['diff_base_minus_shipped']:+.4f} ({x['ci_low']:+.4f}~{x['ci_high']:+.4f})  "
              f"같은 응답 {x['identical_responses']}/{x['n_rows']}  "
              f"형식실패 {x['parse_failures_shipped']}->{x['parse_failures_base_tokenizer']}")
    print(f"선택 변형: 배포 {res['selected_variant_shipped']}, 기반 {res['selected_variant_base_tokenizer']}")
    print(f"후속 조치 필요: {res['triggers_followup']}")
    if "final" in old:
        res["final"] = old["final"]
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"[기록] {path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
