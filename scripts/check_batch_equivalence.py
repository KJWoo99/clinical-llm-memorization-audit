"""배치 추론이 한 건씩 돌린 것과 같은 답을 내는지 확인함.

    python scripts/check_batch_equivalence.py --model gemma3-4b --n 8

## 왜 확인해야 하는가

속도 때문에 배치로 바꿨는데 답이 달라지면 그건 다른 실험임. 그리고 이 종류의
어긋남은 오류를 내지 않음. 왼쪽 패딩을 빠뜨리거나 attention mask 를 잘못
주면 짧은 입력일수록 패딩을 읽고 답을 시작하는데, 출력은 여전히 그럴듯한
JSON 이라 눈으로는 구분되지 않음.

greedy 라 이론상 같아야 함. 다만 부동소수 누적 순서가 배치 크기에 따라 달라져
드물게 한 토큰이 갈릴 수 있음. 그래서 "전부 같아야 한다"가 아니라 몇 건이
갈렸는지를 보고함. 몇 건이 갈리는 정도는 정상이고, 절반이 갈리면 패딩을
의심해야 함.

GPU 를 쓰므로 학습이 돌지 않을 때 실행함.
"""
from __future__ import annotations

import argparse
import difflib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHORT_TOKENS = 512
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    import polars as pl
    from run_phase3_infer import _generate

    from models import MODELS, load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-4b", choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=8, help="비교할 노트 수")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    spec = MODELS[args.model]
    df = pl.read_parquet(ROOT / "outputs" / "phase2_conditions.parquet")
    rows = (df.filter((pl.col("split") == "audit") & (pl.col("condition") == "normal"))
              .head(args.n).with_columns(pl.lit("schema").alias("variant")))
    rows = list(rows.iter_rows(named=True))
    print(f"모델 {spec.label}  노트 {len(rows)}건")

    tokenizer, model = load_model(spec)

    t0 = time.time()
    one, one_len = [], []
    for r in rows:
        texts, _, n_outs = _generate(tokenizer, model, [r], spec)
        one += texts
        one_len += n_outs
    t_one = time.time() - t0

    # 본 추론이 실제로 쓰는 배치 크기로 나눠 넣음. n 건을 한꺼번에 넣으면 "배치 8 이 괜찮다"는 확인이
    # 배치 2 로 도는 8B 모델에는 적용되지 않음.
    t0 = time.time()
    many = []
    for i in range(0, len(rows), args.batch):
        many += _generate(tokenizer, model, rows[i:i + args.batch], spec)[0]
    t_many = time.time() - t0

    diff = [i for i, (a, b) in enumerate(zip(one, many, strict=True)) if a.strip() != b.strip()]
    # 원문이 한 토큰 달라도 뽑힌 약 목록이 같으면 채점 결과는 같음. 이 실험에
    # 보는 것은 목록이므로 둘을 따로 셈.
    from jsonout import parse_model_output
    med_diff = [i for i in diff
                if sorted(parse_model_output(one[i]).medications)
                != sorted(parse_model_output(many[i]).medications)]
    print(f"\n한 건씩 {t_one:.1f}초  배치 {args.batch} {t_many:.1f}초  "
          f"({t_one / max(t_many, 1e-9):.1f}배)")
    print(f"원문이 다른 응답 {len(diff)}/{len(rows)}건, "
          f"약 목록이 다른 응답 {len(med_diff)}/{len(rows)}건")
    for i in diff[:3]:
        print(f"\n[{i}] {rows[i]['note_id']}")
        for line in list(difflib.unified_diff(
                one[i].splitlines(), many[i].splitlines(),
                fromfile="batch=1", tofile=f"batch={args.batch}", lineterm=""))[:12]:
            print("   " + line)

    # 패딩 결함 판정은 짧은 응답에서만 함. 사고 모델(II-Medical 등)은 응답이 수천
    # 토큰이라 부동소수 누적 차이만으로도 원문이 거의 항상 어딘가 갈림. 그것까지
    # "절반 넘게 갈림"으로 세면 약 목록이 같아도 실패로 끝남. 패딩을 잘못 넣으면
    # 짧은 응답부터 망가지므로 짧은 응답만 봐도 결함은 잡힘.
    short = [i for i in range(len(rows)) if one_len[i] <= SHORT_TOKENS]
    short_diff = [i for i in diff if i in short]
    if short:
        print(f"짧은 응답({SHORT_TOKENS}토큰 이하) {len(short)}건 중 원문이 다른 것 {len(short_diff)}건")
    if short and len(short_diff) > len(short) // 2:
        print("\n짧은 응답이 절반 넘게 갈렸다. 부동소수 오차로 보기 어렵다: 패딩 방향과 "
              "attention mask 를 먼저 볼 것.")
        return 1
    # 약 목록까지 달라진 건이 8건 중 1건을 넘으면 그 배치 크기로는 채점 결과가
    # 달라진다고 보고 실패로 끝냄. 0 이 아닌 종료 코드로 실패가 드러남.
    if len(med_diff) * 8 > len(rows):
        print(f"\n약 목록이 {len(med_diff)}건 달라졌다. 이 모델은 배치 1 로 돌려야 한다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
