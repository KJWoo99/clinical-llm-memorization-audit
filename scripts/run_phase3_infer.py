"""Phase 3: 로컬 추론.

두 단계로 나눠 돌림(docs/prereg.md 5절).

  --stage select : audit 분할 정상 조건에 프롬프트 3종을 전부 돌림.
                   모델마다 어느 변형이 나은지 고르기 위한 단계임.
  --stage final  : 고른 변형으로 main 분할(최종 평가)과 감사 조건 전체를 돌림.

`main` 분할은 프롬프트 선택이 끝나기 전에는 열지 않음. 선택과 평가를 같은
데이터로 하면 그 선택만큼 성능이 부풀려짐.

출력은 JSONL 로 한 줄씩 즉시 씀. 노트 하나에 10~20초라 전체가 몇 시간
걸리므로, 중간에 끊겨도 이어서 돌릴 수 있어야 함(이미 있는 줄은 건너뜀).

    python scripts/run_phase3_infer.py --model medgemma-4b --stage select
    python scripts/run_phase3_infer.py --model medgemma-4b --stage final

GPU 없이 파이프라인만 점검하려면 --stub 을 줌(모델을 올리지 않음).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from models import MODELS
from prompts import PROMPT_VARIANTS, build_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
CONDITIONS_PATH = OUT_DIR / "phase2_conditions.parquet"
SELECTED_PATH = OUT_DIR / "phase3_selected_prompts.json"
RESPONSE_DIR = OUT_DIR / "responses"


def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r["note_id"], r["condition"], r["variant"]))
    return keys


def _pad_token_id(tokenizer) -> int:
    """패딩 토큰 id. 없으면 eos 로 대체함.

    `tokenizer.pad_token_id or tokenizer.eos_token_id` 로 쓰면 안 됨:
    Gemma 계열은 `<pad>` 의 id 가 0 이라 falsy 로 판정돼 항상 eos(=1)로
    바뀜. 배치 추론에서는 패딩이 실제로 일어나므로 오류 없이 결과가 틀어짐.
    """
    pad = tokenizer.pad_token_id
    return tokenizer.eos_token_id if pad is None else pad


def _repair_trailing_line(path: Path) -> None:
    """중단으로 잘린 마지막 줄을 지움.

    이어쓰기(append)는 파일 끝에 그대로 붙으므로, 앞선 실행이 줄 중간에서
    죽었으면 다음 기록이 그 조각과 한 줄로 이어붙어 둘 다 깨진 JSON 이
    됨. 잘린 조각을 먼저 잘라내야 새 기록이 온전히 남음.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return
    cut = raw.rfind(b"\n")
    kept = raw[: cut + 1] if cut >= 0 else b""
    path.write_bytes(kept)
    print(f"  [복구] 중단으로 잘린 마지막 줄 {len(raw) - len(kept):,}바이트를 버렸다.")


def _stub_response(row: dict) -> str:
    """GPU 없이 파이프라인을 점검하기 위한 가짜 응답.

    정답의 절반만 맞히고 없는 약을 하나 얹어, 채점기가 정답, 누락, 환각을
    전부 계산하도록 만듦. 성능 수치로 쓰면 안 되며 리포트에 넣지 않음.
    """
    gold = list(row["gold"])
    half = gold[: max(1, len(gold) // 2)]
    return json.dumps([*half, "Stubcillin"], ensure_ascii=False)


def _batches(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _generate(tokenizer, model, rows: list[dict], spec) -> tuple[list[str], list[int], list[int]]:
    """한 배치를 greedy 로 생성함.

    ## 왜 배치로 돌리는가

    한 건씩 돌리면 GPU 가 대부분의 시간을 놀면서 보냄. 생성은 토큰마다
    가중치 전체를 한 번씩 읽는 구조라 메모리 대역폭이 병목인데, 한 건만
    올리면 그 한 번 읽은 가중치로 토큰 하나만 만듦. 여러 건을 같이 올리면
    같은 읽기로 여러 건의 토큰을 만듦.

    모델 16개 x 900건을 한 건씩 돌리면 5일이 넘음. 그래서 배치를 붙임.

    ## 왼쪽 패딩이어야 하는 이유

    디코더 모델은 마지막 토큰 다음을 이어 씀. 오른쪽에 패딩을 붙이면
    이어 쓰기 시작점이 패딩 뒤가 되어, 짧은 입력일수록 패딩을 읽고 답을
    시작함. 오류가 나지 않아 점수만 보고는 알기 어려움.

    패드 토큰 id 를 `pad or eos` 로 고르면 안 되는 이유는 `_pad_token_id` 에
    적어둠: Gemma 는 pad id 가 0 이라 falsy 판정에 걸림.
    """
    import torch

    prompts = [build_prompt(r["input_text"], r["variant"]) for r in rows]
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False,
            # 사고 스위치가 있는 모델은 여기서 끔(models.py 의 template_kwargs).
            **dict(spec.template_kwargs))
        for p in prompts
    ]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        # 템플릿이 이미 BOS 를 넣으므로 여기서 또 붙이지 않음.
        enc = tokenizer(rendered, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
    finally:
        tokenizer.padding_side = prev_side

    pad_id = _pad_token_id(tokenizer)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=spec.max_new_tokens,
                             do_sample=False,      # 재현성을 위해 greedy 고정
                             use_cache=True,       # OLMo-2 는 기본이 False 라 20배 느림
                             pad_token_id=pad_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    n_ins = [int(v) for v in enc["attention_mask"].sum(dim=1)]
    texts, n_outs = [], []
    for row_tokens in gen:
        texts.append(tokenizer.decode(row_tokens, skip_special_tokens=True))
        # 배치의 다른 건이 길면 짧은 건은 뒤가 패딩으로 채워짐. 패딩을 빼고 셈.
        # pad 와 eos 가 같은 토크나이저에서는 종료 토큰 한 개만큼 적게 세어짐.
        n_outs.append(int((row_tokens != pad_id).sum()))
    return texts, n_ins, n_outs


def build_jobs(stage: str, variant_choice: str | None) -> pl.DataFrame:
    df = pl.read_parquet(CONDITIONS_PATH)
    if stage == "select":
        jobs = df.filter((pl.col("split") == "audit") & (pl.col("condition") == "normal"))
        return jobs.join(
            pl.DataFrame({"variant": list(PROMPT_VARIANTS)}), how="cross"
        )
    # final: main 분할 정상 조건 + audit 분할 감사 조건 전체
    jobs = df.filter(
        (pl.col("split") == "main")
        | ((pl.col("split") == "audit") & (pl.col("condition") != "normal"))
    )
    return jobs.with_columns(pl.lit(variant_choice).alias("variant"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--stage", required=True, choices=("select", "final"))
    ap.add_argument("--stub", action="store_true", help="모델을 올리지 않고 가짜 응답으로 배관만 점검")
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만 (점검용)")
    # 큰 모델일수록 배치가 메모리를 더 먹음. 4비트 32B 는 가중치만 20GB 라
    # 남는 자리가 얼마 없으므로 작게 줌.
    ap.add_argument("--batch", type=int, default=8, help="한 번에 생성할 건수")
    args = ap.parse_args()

    spec = MODELS[args.model]

    variant_choice = None
    if args.stage == "final":
        if not SELECTED_PATH.exists():
            print(f"[중단] 프롬프트 선택 결과가 없다: {SELECTED_PATH}")
            print("       먼저 --stage select 를 돌리고 run_phase3_select_prompt.py 를 실행할 것.")
            return 1
        selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
        if spec.key not in selected:
            print(f"[중단] {spec.key} 의 프롬프트가 아직 선택되지 않았다.")
            return 1
        variant_choice = selected[spec.key]["variant"]
        print(f"선택된 프롬프트 변형: {variant_choice}")

    jobs = build_jobs(args.stage, variant_choice)
    if args.limit:
        jobs = jobs.head(args.limit)

    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_stub" if args.stub else ""
    out_path = RESPONSE_DIR / f"{spec.key}_{args.stage}{suffix}.jsonl"
    _repair_trailing_line(out_path)
    done = _done_keys(out_path)
    todo = [r for r in jobs.iter_rows(named=True)
            if (r["note_id"], r["condition"], r["variant"]) not in done]

    print(f"모델 {spec.label}  단계 {args.stage}")
    print(f"  전체 {len(jobs):,}건, 완료 {len(done):,}건, 남은 {len(todo):,}건")
    print(f"  출력 {out_path}")
    if not todo:
        print("  할 일이 없다.")
        return 0

    tokenizer = model = None
    if not args.stub:
        from models import load_model
        print(f"\n모델 로딩 ({spec.precision})...")
        t0 = time.time()
        tokenizer, model = load_model(spec)
        print(f"  {time.time()-t0:.0f}초")

    t_start = time.time()
    done_n = 0
    with out_path.open("a", encoding="utf-8") as f:
        for batch in _batches(todo, args.batch):
            t0 = time.time()
            if args.stub:
                texts = [_stub_response(r) for r in batch]
                n_ins = n_outs = [0] * len(batch)
            else:
                texts, n_ins, n_outs = _generate(tokenizer, model, batch, spec)
            secs = round((time.time() - t0) / len(batch), 2)

            for row, text, n_in, n_out in zip(batch, texts, n_ins, n_outs, strict=True):
                f.write(json.dumps({
                    "model": spec.key,
                    "note_id": row["note_id"],
                    "split": row["split"],
                    "condition": row["condition"],
                    "variant": row["variant"],
                    "response": text,
                    "n_input_tokens": n_in,
                    "n_output_tokens": n_out,
                    # 배치로 돌리면 한 건의 시간을 따로 잴 수 없음. 배치 전체를
                    # 건수로 나눈 값이며, 배치 크기를 함께 남겨 나중에 구분되게 함.
                    "seconds": secs,
                    "batch_size": len(batch),
                    # 모델마다 다를 수 있는 조건을 행에 남김. "왜 이 모델만 길게
                    # 받았나"를 응답 파일만 보고 알 수 있어야 함.
                    "max_new_tokens": spec.max_new_tokens,
                    "template_kwargs": dict(spec.template_kwargs),
                }, ensure_ascii=False) + "\n")
            f.flush()
            done_n += len(batch)

            if done_n % (args.batch * 5) < args.batch or done_n == len(todo):
                elapsed = time.time() - t_start
                eta = elapsed / done_n * (len(todo) - done_n)
                print(f"  [{done_n}/{len(todo)}] 경과 {elapsed/60:.1f}분, "
                      f"남은 예상 {eta/60:.1f}분")

    print(f"\n완료: {out_path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
