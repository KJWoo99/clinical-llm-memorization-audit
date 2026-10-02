"""docs/report.md 의 표를 산출물에서 만듦. render_results_tables.py 와 짝임.

report 에만 있는 표: 프롬프트 선택(변형별 점수), 감사 조건(감사 정상 분모와 비율, 판정), 희귀도,
출력 형식(파싱 경로). 성능, 짝 표는 README 와 같은 값이므로 render_results_tables 의 함수를 다시 씀.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from render_results_tables import (  # noqa: E402
    BINS,
    _md_tilde,
    parse_paths,
    scored_notes,
    short,
    signed,
)

from models import MODELS  # noqa: E402

OUT = ROOT / "outputs"
MARK = {
    "선택": ("<!-- 선택표 시작 -->", "<!-- 선택표 끝 -->"),
    "성능": ("<!-- 성능표 시작 -->", "<!-- 성능표 끝 -->"),
    "짝": ("<!-- 짝표 시작 -->", "<!-- 짝표 끝 -->"),
    "감사": ("<!-- 감사표 시작 -->", "<!-- 감사표 끝 -->"),
    "희귀": ("<!-- 희귀표 시작 -->", "<!-- 희귀표 끝 -->"),
    "형식": ("<!-- 형식표 시작 -->", "<!-- 형식표 끝 -->"),
}


def tables() -> dict[str, str]:
    d = json.loads((OUT / "phase4_results.json").read_text(encoding="utf-8"))
    sel = json.loads((OUT / "phase3_selected_prompts.json").read_text(encoding="utf-8"))
    models = d["models"]
    order = sorted(models, key=lambda k: -models[k]["normal"]["micro_f1"])
    out: dict[str, list[str]] = {k: [] for k in MARK}

    w = out["선택"].append
    w("| 모델 | grounded | minimal | schema | 선택 | 노트 |")
    w("|---|---:|---:|---:|---|---:|")
    for k in order:
        bv = sel[k].get("by_variant", {})
        cells = []
        for v in ("grounded", "minimal", "schema"):
            if v in bv:
                mark = "**" if v == sel[k]["variant"] else ""
                cells.append(f"{mark}{bv[v]['micro_f1']:.3f}{mark} ({bv[v]['json_compliance']:.0%})")
            else:
                cells.append("")
        w(f"| {short(MODELS[k].label)} | " + " | ".join(cells)
          + f" | {sel[k]['variant']} | {sel[k]['n_notes_compared']} |")

    w = out["성능"].append
    w("| 모델 | 특화 | 정밀도 | precision | recall | micro-F1 [95% CI] | 환각률 | 누락률 | JSON 준수 | 노트 |")
    w("|---|---|---|---:|---:|---|---:|---:|---:|---:|")
    for k in order:
        m, s = models[k], MODELS[k]
        n = m["normal"]
        w(f"| {short(s.label)} | {'의료' if s.medical else '기반'} | {s.precision} | {n['precision']:.3f} "
          f"| {n['recall']:.3f} | {n['micro_f1']:.3f} [{n['ci'][0]:.3f}, {n['ci'][1]:.3f}] "
          f"| {m['hallucination_rate']:.1%} | {m['omission_rate']:.1%} | {m['json_compliance']:.0%} "
          f"| {scored_notes(k, s)} |")

    w = out["짝"].append
    w("| 쌍 | 비교 (의료 − 기반) | 차이 | 95% CI | 판정 | 노트 |")
    w("|---|---|---:|---|---|---:|")
    for i, p in enumerate(d["pairs"], 1):
        a, b = MODELS[p["a"]], MODELS[p["b"]]
        kind = "가로" if a.medical == b.medical else "세로"
        verdict = f"{short(p['winner'])} 우세" if p["significant"] else "차이 없음"
        w(f"| {i} ({kind}) | {short(a.label)} − {short(b.label)} | {signed(p['diff'])} "
          f"| [{signed(p['ci'][0])}, {signed(p['ci'][1])}] | {verdict} | {p['n_notes']} |")

    w = out["감사"].append
    w("| 모델 | main 정상 | 감사 정상 | 절 제거 | 빈 노트 | 남의 노트 | 문장 섞음 | 빈 노트 비율 | 섞음 유지율 | 암기 | 구조 |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for k in order:
        m = models[k]
        c = m["condition_micro_f1"]
        basis = m["memorisation_basis"]
        denom = basis["denominator_f1"]
        w(f"| {short(MODELS[k].label)} | {c['normal']:.3f} | {denom:.3f} | {c['no_section']:.3f} "
          f"| {c['empty']:.3f} | {c['other_patient']:.3f} | {c['shuffled']:.3f} "
          f"| {basis['ratio_vs_normal']:.1%} | {c['shuffled'] / denom:.1%} "
          f"| {m['memorisation_verdict'].replace('암기 ', '')} | {m['structure_verdict'].replace('구조 ', '')} |")

    w = out["희귀"].append
    w("| 모델 | " + " | ".join(BINS) + " |")
    w("|---|" + "---:|" * len(BINS))
    for k in order:
        r = models[k]["recall_by_rarity"]
        w(f"| {short(MODELS[k].label)} | " + " | ".join(
            f"{r[b][0]}/{r[b][1]} ({r[b][0] / r[b][1]:.0%})" for b in BINS) + " |")

    w = out["형식"].append
    w("| 모델 | 순수 JSON | 코드펜스 | 사고 뒤 답 | 그 밖 복구 | 실패 | 합계 |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for k in order:
        c = parse_paths(k, MODELS[k])
        other = sum(v for p, v in c.items() if p not in ("strict", "fence", "after_reasoning", "failed"))
        w(f"| {short(MODELS[k].label)} | {c['strict']} | {c['fence']} | {c['after_reasoning']} "
          f"| {other} | {c['failed']} | {sum(c.values())} |")
    w("")
    # report 에서 이 문단을 손으로만 고치면 다시 그릴 때 옛 문장으로 돌아감.
    # 문장을 고칠 때는 여기를 고치고 --write 로 다시 그림.
    w("순수 JSON 은 응답 전체가 지시대로 배열 하나인 경우임. 사고 뒤 답은 사고를 쓴 뒤 끝 표시 다음의 "
      "답이 순수 JSON 인 경우로, 지시 준수로 세지 않음. 사고 뒤 답을 코드펜스나 본문 배열로 쓴 응답"
      "(16모델 합 96건, 9건)은 이 칸이 아니라 코드펜스, 그 밖 복구 칸에 셈. 어느 칸이든 채점에는 사고 "
      "뒤의 답만 씀. 실패는 상한에 잘린 사고와 읽을 수 없는 배열을 포함함. 순위표의 JSON 준수율은 "
      "정상 노트 200건 기준이고 이 표는 감사 조건을 포함한 채점 대상 전체라 수가 다름. 빈 노트에 "
      "`[]` 만 답한 행은 순수 JSON 으로 셈.")

    return {k: "\n".join(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    t = tables()
    if not args.write:
        for k, v in t.items():
            print(f"### {k}\n{v}\n")
        return 0
    p = ROOT / "docs" / "report.md"
    s = p.read_text(encoding="utf-8")
    for k, (b, e) in MARK.items():
        if b not in s or e not in s:
            print(f"report.md 에 {b} 가 없다")
            return 1
        head, rest = s.split(b, 1)
        _, tail = rest.split(e, 1)
        s = head + b + "\n" + _md_tilde(t[k]) + "\n" + e + tail
    # 줄끝은 LF 로 고정함(플랫폼 기본값에 맡기면 CRLF 가 되어 문서 전체가 바뀐 것으로 보임).
    p.write_text(s, encoding="utf-8", newline="\n")
    print(f"report.md 표 {len(MARK)}개 갱신")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
