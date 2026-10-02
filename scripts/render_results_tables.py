"""채점 결과(outputs/phase4_results.json)에서 README 의 결과 표(순위표, 짝 비교)를 만듦.

손으로 옮겨 적으면 16모델 x 열 여러 개에서 한두 칸은 틀림. 표는 이 스크립트가 만들고,
README 의 `<!-- 결과표 시작 -->` 과 `<!-- 결과표 끝 -->` 사이만 바꿔 넣음. 그 밖의
문장은 사람이 씀.

    python scripts/render_results_tables.py            # 표만 출력
    python scripts/render_results_tables.py --write    # README.md 의 표 구간을 바꿈
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsonout import parse_model_output  # noqa: E402
from models import MODELS, PAIRS_EXPLORATORY, PAIRS_PREREGISTERED, exceeds_context  # noqa: E402

PAIRS = PAIRS_PREREGISTERED + PAIRS_EXPLORATORY

OUT = ROOT / "outputs"
BEGIN, END = "<!-- 결과표 시작 -->", "<!-- 결과표 끝 -->"
BINS = ("10회 미만", "10~99회", "100~999회", "1000회 이상")


def _md_tilde(text: str) -> str:
    """GitHub 는 한 블록 안의 ~ 두 개를 취소선으로 그림. 코드 밖의 ~ 를 \\~ 로 적음."""
    out, fence = [], False
    for ln in text.split("\n"):
        if ln.lstrip().startswith("```"):
            fence = not fence
        elif not fence:
            parts = re.split(r"(`[^`]*`)", ln)
            ln = "".join(p if i % 2 else re.sub(r"(?<!\\)~", r"\\~", p) for i, p in enumerate(parts))
        out.append(ln)
    return "\n".join(out)


def short(label: str) -> str:
    return label.split(" (")[0]


def signed(x: float, nd: int = 3) -> str:
    """본문과 같은 부호 표기. 음수는 하이픈이 아니라 마이너스 기호(−)로 적음."""
    return f"{x:+.{nd}f}".replace("-", "−")


def pair_of(key: str) -> str:
    """세로 짝 번호. 쌍3(가로)은 양쪽 모델이 세로 짝을 따로 가지므로 건너뜀.
    같은 기반을 두 짝이 나눠 쓰는 Llama 는 둘 다 적음."""
    hits = [f"쌍{i}" for i, (a, b, _) in enumerate(PAIRS, 1)
            if key in (a, b) and MODELS[a].medical != MODELS[b].medical]
    return ", ".join(hits) if hits else "없음"


def parse_paths(key: str, spec) -> Counter:
    """최종 600건 중 채점 대상 행의 파싱 경로."""
    p = OUT / "responses" / f"{key}_final.jsonl"
    c: Counter = Counter()
    if not p.exists():
        return c
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if exceeds_context(r, spec):
            continue
        o = parse_model_output(r["response"])
        c["strict" if o.strict_json else o.recovered_by] += 1
    return c


def scored_notes(key: str, spec) -> int:
    """정상 조건에서 채점된 노트 수. 문맥을 넘는 행을 뺀 뒤의 수라 OLMo-2 는 200 보다 적음."""
    p = OUT / "responses" / f"{key}_final.jsonl"
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["condition"] == "normal" and not exceeds_context(r, spec):
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    d = json.loads((OUT / "phase4_results.json").read_text(encoding="utf-8"))
    sel = json.loads((OUT / "phase3_selected_prompts.json").read_text(encoding="utf-8"))
    models = d["models"]
    order = sorted(models, key=lambda k: -models[k]["normal"]["micro_f1"])
    missing = [k for k in MODELS if k not in models]
    lines: list[str] = []
    w = lines.append

    w(f"빈도 베이스라인(가장 흔한 약 10개 고정 출력) micro-F1 {d['baseline']['micro_f1']:.3f}. "
      f"채점된 {len(models)}개 모델 모두 크게 넘겨 실격 없음.")
    if missing:
        w("")
        w(f"> 아직 채점되지 않은 모델: {', '.join(MODELS[k].label for k in missing)}.")
    w("")
    w("### 순위표. 정상 노트 200건 micro-F1")
    w("")
    w("| # | 모델 | 특화 | 짝 | 정밀도 | 프롬프트 | micro-F1 [95% CI] | 환각률 | 누락률 | JSON 준수 | 노트 |")
    w("|---|---|---|---|---|---|---|---:|---:|---:|---:|")
    for i, k in enumerate(order, 1):
        m, s = models[k], MODELS[k]
        n = m["normal"]
        n_notes = scored_notes(k, s)
        w(f"| {i} | {short(s.label)} | {'의료' if s.medical else '기반'} | {pair_of(k)} | {s.precision} "
          f"| {sel[k]['variant']} | {n['micro_f1']:.3f} [{n['ci'][0]:.3f}, {n['ci'][1]:.3f}] "
          f"| {m['hallucination_rate']:.1%} | {m['omission_rate']:.1%} | {m['json_compliance']:.0%} | {n_notes} |")
    w("")
    w("정밀도가 다른 모델끼리는 가로로 비교하지 않음. 4비트 27B, 32B 는 짝 안에서만 읽음. "
      "OLMo-2 두 모델은 문맥 4,096 을 넘는 노트를 채점에서 뺐으므로 노트 수가 200 보다 적음. "
      "짝 없는 모델(Lingshu, II-Medical, MedReason)은 기반을 특정할 수 없어 순위표에만 올림.")
    w("")
    w("### 짝 비교 8쌍. 페어드 부트스트랩 95% CI")
    w("")
    w("| 쌍 | 비교 (의료 − 기반) | 차이 | 95% CI | 판정 | 노트 |")
    w("|---|---|---:|---|---|---:|")
    for i, p in enumerate(d["pairs"], 1):
        a, b = MODELS[p["a"]], MODELS[p["b"]]
        kind = "가로" if a.medical == b.medical else "세로"
        verdict = f"{short(p['winner'])} 우세" if p["significant"] else "차이 없음"
        w(f"| {i} ({kind}) | {short(a.label)} − {short(b.label)} | {signed(p['diff'])} "
          f"| [{signed(p['ci'][0])}, {signed(p['ci'][1])}] | {verdict} | {p['n_notes']} |")
    text = "\n".join(lines)
    if not args.write:
        print(text)
        return 0
    readme = ROOT / "README.md"
    src = readme.read_text(encoding="utf-8")
    if BEGIN not in src or END not in src:
        print(f"README.md 에 {BEGIN} / {END} 표시가 없다.")
        return 1
    head, rest = src.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    # 줄끝은 LF 로 고정함. 플랫폼 기본값에 맡기면 CRLF 가 되어 표 한 줄만 바뀌어도 문서 전체가
    # 바뀐 것으로 나와 대조가 안 됨.
    readme.write_text(head + BEGIN + "\n" + _md_tilde(text) + "\n" + END + tail,
                      encoding="utf-8", newline="\n")
    print(f"README.md 결과표 갱신: 모델 {len(models)}개, 짝 {len(d['pairs'])}쌍")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
