"""저장된 응답을 같은 조건으로 다시 만들어 글자까지 같은지 봄.

## 왜 필요한가

추론 기간에 서버가 여러 번 꺼져 실행이 중간에 끊김(트러블슈팅 15). 끊기기 직전에는 python 이
segfault 를 낸 기록도 있음. 기계가 불안정하면 멈추기 전에 계산이 틀릴 수 있다는 뜻임.

중간에 끊기거나 계산 도중 segfault 로 끝난 시도가 행을 보탠 파일은 이미 지우고 다시
만듦. 남은 파일은 한 번의 성공한 시도로 만들어졌지만, 같은 기계에서 같은 시기에
돎. "성공으로 끝났다"는 것은 프로세스가 죽지 않았다는 뜻일 뿐 계산이 맞았다는
보증은 아님. 그래서 표본을 다시 만들어 대조함.

## 방법

greedy 디코딩이라 같은 입력, 같은 설정, 같은 배치 구성이면 같은 출력이 나와야 함.

배치 구성까지 맞추는 이유: 배치가 달라지면 패딩 길이가 달라지고 bf16 행렬곱의 덧셈
순서가 바뀜. 앞서 배치 등가성 검사에서 배치 1 과 2 사이에
16건 중 0~2건의 약 목록이 달라짐(II-Medical 은 8건이라 배치 1 로만 돌림). 배치 1 로만 다시 돌리면 그 차이가 손상처럼 보임.
그 검사 기록(`outputs/batch_equivalence_*.log`)은 note_id 와 응답 조각이 들어 있어
깃에서 빠짐(.gitignore). 저장소에는 없고 실행한 기계에만 남음.

응답 행에는 `batch_size` 가 실제 배치 길이로 남아 있고, 한 배치의 행은 파일에 연달아
쓰임. 그래서 파일을 앞에서부터 batch_size 만큼씩 끊으면 원래 묶음이 복원됨.
생성 상한과 템플릿 인자도 행에 남은 값을 씀. 그 사이 models.py 가 바뀌었어도
그 행을 만들 때의 설정으로 돌림.

    python scripts/verify_reproducibility.py --model qwen25-32b --stage select final

결과는 outputs/verify/<model>_<stage>.jsonl 에 쌓이고, 끊겨도 이어서 돌림.
불일치가 한 건이라도 있으면 종료 코드 3 을 내 실패로 드러남.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from models import MODELS, local_kwargs

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
RESPONSE_DIR = OUT_DIR / "responses"
VERIFY_DIR = OUT_DIR / "verify"
MISMATCH_EXIT = 3


def batch_groups(rows: list[dict]) -> list[list[dict]]:
    """파일 순서대로 원래 배치 묶음을 복원함.

    배치 하나는 len(batch) 행을 연달아 쓰고 각 행에 batch_size=len(batch) 를 남김.
    묶음 안에서 batch_size 가 어긋나면(이어받기 경계 등) 그 행만 한 건짜리로 떼어냄.
    """
    groups, i = [], 0
    while i < len(rows):
        b = max(1, int(rows[i].get("batch_size") or 1))
        chunk = rows[i:i + b]
        if len(chunk) == b and all(r.get("batch_size") == b for r in chunk):
            groups.append(chunk)
            i += b
        else:
            groups.append([rows[i]])
            i += 1
    return groups


def pick_groups(groups: list[list[dict]], sample: int, min_groups: int = 5) -> list[list[dict]]:
    """(조건, 변형)마다 고르게, 그리고 파일 앞뒤로 퍼진 묶음을 고름.

    크래시는 파일 어디에서든 났으므로 앞에서부터 자르면 안 됨. 그렇다고 일정 간격으로
    뽑아도 안 됨. 최종 파일은 정상 200행 뒤에 감사 조건 4종이 노트마다 번갈아 쓰이고,
    선택 파일은 변형 3종이 번갈아 쓰여서, 간격이 주기와 맞으면 늘 같은 조건만 걸림.
    그래서 묶음 첫 행의 (조건, 변형)으로 층을 나누고 층 안에서 간격을 둠.

    묶음이 크면(배치 8) 표본 건수를 채우기 전에 묶음 수가 모자라므로 최소 묶음 수를 둠.
    """
    if not groups:
        return []
    strata: dict[tuple[str, str], list[list[dict]]] = {}
    for g in groups:
        strata.setdefault((g[0]["condition"], g[0]["variant"]), []).append(g)
    avg = sum(len(g) for g in groups) / len(groups)
    n = min(len(groups), max(min_groups, len(strata), math.ceil(sample / avg)))
    picked = []
    for members in strata.values():
        k = max(1, round(n * len(members) / len(groups)))
        step = len(members) / k
        picked.extend(members[int(i * step)] for i in range(min(k, len(members))))
    # 파일 순서로 되돌림. 로그를 읽을 때 앞뒤가 섞이지 않게.
    order = {id(g): i for i, g in enumerate(groups)}
    return sorted(picked, key=lambda g: order[id(g)])


def _key(r: dict) -> tuple[str, str, str]:
    return (r["note_id"], r["condition"], r["variant"])


def pin_template_date(day: str) -> None:
    """채팅 템플릿의 strftime_now 가 돌려주는 날짜를 고정함.

    Apertus 두 모델의 템플릿은 시스템 프롬프트에 "Current date: <오늘>" 을 넣음
    (16개 중 이 둘뿐, 두 날짜로 렌더링해 확인). 다른 날 다시 만들면 입력이 달라져 손상이 없어도
    불일치가 남. 그 행을 만든 날짜로 맞춤.
    transformers 는 템플릿을 컴파일할 때 모듈의 datetime.now() 를 부르므로 그 이름을 바꿈.
    """
    from datetime import datetime as _dt

    import transformers.utils.chat_template_utils as ctu

    fixed = _dt.strptime(day, "%Y-%m-%d")

    class _Pinned(_dt):
        @classmethod
        def now(cls, _tz=None):
            return fixed

    ctu.datetime = _Pinned
    ctu._compile_jinja_template.cache_clear()


def run_settings(row: dict, spec):
    """그 행을 만들 때의 생성 설정으로 spec 을 맞춤.

    먼저 만든 행(gemma3-4b, gemma3-27b 계열, qwen25-32b 의 select)에는
    max_new_tokens, template_kwargs 기록 칸이 없음. 그 네 모델은 그때도 지금도 기본값
    (512, 인자 없음)이라 기록이 없으면 현재 설정을 씀. 기록이 있으면 기록을 따름:
    II-Medical 처럼 중간에 상한을 바꾼 모델은 행마다 값이 다를 수 있음.
    """
    cap = int(row["max_new_tokens"]) if "max_new_tokens" in row else spec.max_new_tokens
    kwargs = row["template_kwargs"] if "template_kwargs" in row else dict(spec.template_kwargs)
    return replace(spec, max_new_tokens=cap, template_kwargs=tuple(sorted((kwargs or {}).items())))


def verify_stage(stage: str, spec, tokenizer, model, text_of: dict, sample: int,
                 suffix: str = "", dry_run: bool = False) -> tuple[int, int]:
    src = RESPONSE_DIR / f"{spec.key}_{stage}{suffix}.jsonl"
    if not src.exists():
        print(f"  [{stage}] 응답 파일이 없다: {src.name}")
        return 0, 0
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = pick_groups(batch_groups(rows), sample)

    if dry_run:
        # 모델 없이 실제 파일로 생성 직전까지 감. GPU 로 돌리기 전에 데이터 때문에 나는
        # 오류(기록 칸 없음, 조건 표에 없는 노트)를 잡으려는 것임.
        missing = [r for g in picked for r in g if (r["note_id"], r["condition"]) not in text_of]
        settings = Counter((run_settings(g[0], spec).max_new_tokens,
                            run_settings(g[0], spec).template_kwargs) for g in picked)
        print(f"  [{stage}{suffix}] 모의: {len(rows)}행, 묶음 {len(picked)}개"
              f"({sum(len(g) for g in picked)}행), 조건 표에 없는 행 {len(missing)}, 설정 {dict(settings)}")
        return (1, 0) if missing else (0, 0)

    from run_phase3_infer import _generate

    rec_path = VERIFY_DIR / f"{spec.key}_{stage}{suffix}.jsonl"
    seen = set()
    if rec_path.exists():
        seen = {_key(json.loads(line)) for line in rec_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [g for g in picked if not all(_key(r) in seen for r in g)]
    print(f"  [{stage}] {len(rows)}행, 묶음 {len(picked)}개({sum(len(g) for g in picked)}행) 검사, "
          f"남은 묶음 {len(todo)}개")

    with rec_path.open("a", encoding="utf-8") as f:
        for gi, group in enumerate(todo, 1):
            first = group[0]
            run_spec = run_settings(first, spec)
            jobs = [{"input_text": text_of[(r["note_id"], r["condition"])], "variant": r["variant"]}
                    for r in group]
            t0 = time.time()
            texts, n_ins, _ = _generate(tokenizer, model, jobs, run_spec)
            n_same = 0
            for r, again, n_in in zip(group, texts, n_ins, strict=True):
                same = again == r["response"]
                n_same += same
                f.write(json.dumps({
                    "note_id": r["note_id"], "condition": r["condition"], "variant": r["variant"],
                    "batch_size": len(group), "same": same,
                    # 불일치가 나면 원인부터 가름. 입력 토큰 수가 다르면 그 사이 프롬프트,
                    # 템플릿 처리가 바뀐 것이고(손상 아님), 같은데 출력만 다르면 계산 쪽임.
                    "n_input_stored": r.get("n_input_tokens"), "n_input_again": n_in,
                    "max_new_tokens_used": run_spec.max_new_tokens,
                    "len_stored": len(r["response"]), "len_again": len(again),
                    # 다를 때만 원문을 남김. 같으면 파일만 부풀림.
                    "stored": None if same else r["response"],
                    "again": None if same else again,
                }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"    묶음 {gi}/{len(todo)}  {first['condition']:<14} 배치 {len(group)}  "
                  f"일치 {n_same}/{len(group)}  ({time.time()-t0:.0f}초)")

    recs = [json.loads(line) for line in rec_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    same = sum(1 for d in recs if d["same"])
    print(f"  [{stage}] 누적 {len(recs)}건 중 일치 {same}건, 불일치 {len(recs) - same}건")
    return len(recs), same


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--stage", required=True, nargs="+", choices=("select", "final"))
    ap.add_argument("--sample", type=int, default=12, help="파일마다 다시 만들 대략의 건수")
    # 쌍8 기반 토크나이저 실행(check_tokenizer_sensitivity.py)이 만든 *_basetok 파일을 검사함.
    # 모델 가중치는 같고 토크나이저만 기반 모델 것으로 바꿔 만들었으므로, 행에 남은
    # tokenizer 로 다시 불러 같은 조건을 맞춤.
    ap.add_argument("--basetok", action="store_true", help="기반 토크나이저로 만든 *_basetok 파일")
    ap.add_argument("--dry-run", action="store_true", help="모델을 올리지 않고 표본, 설정, 조건 표 대응만 확인")
    ap.add_argument("--template-date", default=None,
                    help="템플릿에 날짜를 넣는 모델(Apertus)의 원래 실행 날짜, YYYY-MM-DD")
    args = ap.parse_args()
    if args.template_date:
        pin_template_date(args.template_date)

    spec = MODELS[args.model]
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    # 프롬프트 원문은 응답 파일에 없음. 조건 표에서 (note_id, condition) 으로 되찾음.
    cond = pl.read_parquet(OUT_DIR / "phase2_conditions.parquet")
    text_of = {(r["note_id"], r["condition"]): r["input_text"] for r in cond.iter_rows(named=True)}

    sys.path.insert(0, str(ROOT / "scripts"))
    suffix = "_basetok" if args.basetok else ""
    tokenizer = model = None

    if args.dry_run:
        # 날짜를 넣는 템플릿인지, 고정이 먹혔는지 렌더링으로 보임.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.hf_id, **local_kwargs(spec.hf_id))
        shown = tok.apply_chat_template([{"role": "user", "content": "X"}], add_generation_prompt=True,
                                        tokenize=False, **dict(spec.template_kwargs))
        dates = sorted(set(__import__("re").findall(r"20\d\d-\d\d-\d\d", shown)))
        print(f"  모의: 렌더링된 템플릿의 날짜 {dates or '없음'}")
        if args.basetok:
            first = json.loads(next(line for line in (RESPONSE_DIR / f"{spec.key}_{args.stage[0]}{suffix}.jsonl")
                                    .read_text(encoding="utf-8").splitlines() if line.strip()))
            print(f"  모의: 행에 기록된 토크나이저 {first['tokenizer']}")
    else:
        from models import load_model
        print(f"모델 {spec.label}\n모델 로딩 ({spec.precision})...")
        t0 = time.time()
        tokenizer, model = load_model(spec)
        print(f"  {time.time()-t0:.0f}초")
        if args.basetok:
            from transformers import AutoTokenizer
            first = json.loads(next(line for line in (RESPONSE_DIR / f"{spec.key}_{args.stage[0]}{suffix}.jsonl")
                                    .read_text(encoding="utf-8").splitlines() if line.strip()))
            base_id = first["tokenizer"]
            tokenizer = AutoTokenizer.from_pretrained(base_id, **local_kwargs(base_id))
            print(f"  토크나이저를 행에 기록된 {base_id} 로 바꿨다")

    total = same = 0
    for stage in args.stage:
        n, s = verify_stage(stage, spec, tokenizer, model, text_of, args.sample, suffix, args.dry_run)
        total += n
        same += s
    print(f"\n합계 {total}건 중 일치 {same}건")
    return MISMATCH_EXIT if same < total else 0


if __name__ == "__main__":
    raise SystemExit(main())
