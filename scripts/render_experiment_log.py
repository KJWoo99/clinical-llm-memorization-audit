"""절제실험 기록들을 사람이 읽는 실험 일지로 만듦.

    python scripts/render_experiment_log.py

`outputs/ablation/*.json` 을 모아 `docs/EXPERIMENT_LOG.md` 를 씀.

## 판정선을 시드 폭으로 만들지 않는 이유

학습하는 모델이라면 시드를 바꿔 돌린 폭으로 판정선을 만들 수 있음. 여기서는 그렇게
할 수 없음: greedy 디코딩이라 같은 입력에 항상 같은 답이 나오고, 시드를 바꿔도
결과가 그대로임.

대신 노트 100건이라는 표본에서 오는 폭이 있음. 그래서 회차마다 부트스트랩
신뢰구간을 함께 내고, 구간이 겹치면 "차이 없음"으로 읽음. 점추정 두 개를
비교해 순위를 매기는 것보다 보수적임.

## 사전등록과 탐색을 섞지 않음

프롬프트 변형 셋(minimal, schema, grounded)은 결과를 보기 전에 고정함.
나중에 더한 변형(fewshot, terse)은 탐색임. 표에서 구분해 적고, 탐색 쪽이
좋게 나와도 그것은 결론이 아니라 다음 사전등록의 후보임.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ABL_DIR = ROOT / "outputs" / "ablation"
OUT_MD = ROOT / "docs" / "EXPERIMENT_LOG.md"


# 문서에 쓰지 않는 문자들. 이 함수가 찾아서 바꾸는 대상이지만 소스에도 그 문자를 두지 않으려고
# 유니코드 이스케이프로 적음.
_DASH, _DOT, _PM = "\u2014", "\u00b7", "\u00b1"
_ARROWS = {"\u2192": "->", "\u2190": "<-", "\u2191": "^", "\u2193": "v", "\u00d7": "x"}



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

def plain(text: str) -> str:
    """기록에서 가져온 문장을 문서 표기 규칙에 맞춤.

    절제 기록은 산출물이라 손대지 않음. 옛 회차의 진단, 가설 문장에 줄표나
    가운뎃점이 남아 있어도, 문서로 옮길 때는 쉼표와 콜론으로 바꿔 적음.
    """
    text = re.sub(rf"\s*{_DASH}\s*", ": ", text)
    text = re.sub(rf"\s*{_DOT}\s*", ", ", text)
    for a, b in _ARROWS.items():
        text = text.replace(a, b)
    text = re.sub(rf"\s*{_PM}\s*", " +/- ", text)
    return text.replace("( +/- ", "(+/- ")


def _load() -> list[dict]:
    if not ABL_DIR.exists():
        return []
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ABL_DIR.glob("*.json"))]
    return sorted(recs, key=lambda r: r["id"])


def _overlaps(a: dict, b: dict) -> bool:
    """두 회차의 신뢰구간이 겹치는가."""
    ra, rb = a["result"], b["result"]
    return not (ra["ci_high"] < rb["ci_low"] or rb["ci_high"] < ra["ci_low"])


def _verdict(rec: dict, base: dict) -> str:
    if rec["id"] == base["id"]:
        return "기준선"
    d = rec["result"]["micro_f1"] - base["result"]["micro_f1"]
    if _overlaps(rec, base):
        return f"차이 없음 ({d:+.4f}, 구간 겹침)"
    return f"{'개선' if d > 0 else '악화'} ({d:+.4f})"


MANDATORY = 5          # 의무 구간 #1~#5. 연속 미개선은 #6 부터 셈
STOP_AFTER = 3         # 개선 아닌 회차가 이만큼 연속이면 멈춤


def stop_trace(recs: list[dict]) -> list[dict]:
    """#6 부터 부모 대비 판정과 연속 미개선 수."""
    by_id = {r["id"]: r for r in recs}
    rows, streak = [], 0
    for r in recs:
        par = by_id.get(r.get("parent"))
        if r["id"] <= MANDATORY or par is None:
            continue
        verdict = _verdict(r, par)
        streak = 0 if verdict.startswith("개선") else streak + 1
        rows.append({"id": r["id"], "parent": par["id"], "verdict": verdict, "streak": streak})
    return rows


def main() -> int:
    from prompts import PROMPT_VARIANTS

    recs = _load()
    if not recs:
        print(f"기록이 없다: {ABL_DIR}")
        return 1

    base = next((r for r in recs if not r.get("parent")), recs[0])
    lines: list[str] = []
    lines.append("# 실험 일지: 무엇을 바꿨더니 어떻게 됐는가\n")
    lines.append(
        "자동 생성 파일이다. `python scripts/render_experiment_log.py` 로 다시 만든다.\n")
    lines.append(
        "모든 수치는 audit 분할의 정상 조건 100건 기준이다. main 분할은 최종\n"
        "평가용이라 설정을 고르는 데 쓰지 않는다. 같은 데이터로 고르고 평가하면\n"
        "그 선택만큼 점수가 부풀려진다.\n")
    lines.append(
        f"모델은 {base['config']['model']} 하나로 고정했다. 절제는 축을 이해하려는\n"
        "것이고, 모델마다 어느 프롬프트가 맞는지는 사전등록한 선택 절차에서 따로 정한다.\n")
    lines.append(
        "판정은 부트스트랩 95% 신뢰구간이 기준선과 겹치는지로 한다. greedy 라\n"
        "시드를 바꿔도 결과가 같으므로, 학습하는 모델처럼 회차 간 표준편차로\n"
        "판정선을 만들 수 없다.\n")
    lines.append(
        "여덟 회차를 두 번 다시 돌렸다. 처음에는 채점기가 정답 쪽 중복을 합치지 않아\n"
        "누구도 맞힐 수 없는 누락이 생기던 것을 고친 뒤(트러블슈팅 16), 다음에는\n"
        "별칭 규칙을 예측에도 걸도록 고친 뒤다(트러블슈팅 17). 예전 기록에는 점수만 있어\n"
        "재채점이 안 되고 추론부터 다시 해야 했다. 그래서 값의 변화에는\n"
        "채점 수정과 함께 배치 구성, 그 사이 통일한 실행 환경(torch 2.14.0+cu132)의\n"
        "영향이 섞여 있다. 회차 간 판정과 순서는 그대로다.\n")

    lines.append("## 한눈에 보기\n")
    lines.append("| # | 바꾼 것 | micro-F1 | 95% CI | 기준선 대비 | 형식실패 | 등록 |")
    lines.append("|---|---|---:|---|---|---:|---|")
    for r in recs:
        res, cfg = r["result"], r["config"]
        kind = "사전등록" if cfg["variant"] in PROMPT_VARIANTS else "탐색적"
        lines.append(
            f"| {r['id']} | {r['change']} | {res['micro_f1']:.4f} | "
            f"{res['ci_low']:.4f}~{res['ci_high']:.4f} | {_verdict(r, base)} | "
            f"{res['parse_failures']}/{cfg['n_notes']} | {kind} |")
    lines.append("")

    lines.append("## 실험별 경위\n")
    for r in recs:
        res, cfg = r["result"], r["config"]
        lines.append(f"### 실험 #{r['id']}: {r['name']}\n")
        if r.get("parent"):
            lines.append(f"- 부모 회차: 실험 #{r['parent']}")
        if r.get("hypothesis"):
            lines.append(f"- 가설: {plain(r['hypothesis'])}")
        lines.append(f"- 바꾼 것: {plain(r['change'])}")
        lines.append(
            f"- 결과: micro-F1 {res['micro_f1']:.4f} "
            f"(95% CI {res['ci_low']:.4f}~{res['ci_high']:.4f}), {_verdict(r, base)}")
        lines.append(
            f"- 내역: precision {res['precision']:.4f} / recall {res['recall']:.4f}, "
            f"형식 실패 {res['parse_failures']}/{cfg['n_notes']}, "
            f"길이한도 의심 {res['truncated']}건")
        lines.append(f"- 진단: {plain(r['diagnosis'])}")
        lines.append(
            f"- 설정: variant={cfg['variant']} max_new_tokens={cfg['max_new_tokens']} "
            f"batch={cfg['batch']} decoding={cfg['decoding']} "
            f"({cfg['precision']}, {round(r['elapsed_sec'] / 60)}분)")
        lines.append("")

    # ── 형식 준수 ─────────────────────────────────────────────────────
    paths: dict[str, int] = {}
    for r in recs:
        for k, v in (r["result"].get("recovered_by") or {}).items():
            paths[k] = paths.get(k, 0) + v
    if paths:
        total = sum(paths.values())
        strict = paths.get("strict", 0)
        lines.append("## 형식 준수는 점수와 별개다\n")
        lines.append(
            "어떤 경로로 답을 건졌는지 센 것이다. `strict` 는 지시대로 순수 JSON 만 "
            "내놓은 것, `fence` 는 마크다운 펜스로 감싼 것, `bullets` 는 번호목록으로 "
            "답한 것, `failed` 는 네 경로를 다 시도하고도 못 건진 것이다.\n")
        lines.append("| 경로 | 건수 | 비율 |")
        lines.append("|---|---:|---:|")
        for k, v in sorted(paths.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} | {v / total:.1%} |")
        lines.append("")
        lines.append(
            f"전체 {total}건 중 지시대로 순수 JSON 이 나온 것은 {strict}건 "
            f"({strict / total:.1%}) 이다. schema, grounded 프롬프트에는 "
            '"마크다운 펜스를 쓰지 말라"고 명시돼 있는데도 그렇다.\n')
        lines.append(
            "이 말은 위 표의 micro-F1 이 모델의 형식 준수 능력이 아니라 파서의 "
            "복구 능력에 기대고 있다는 뜻이다. 펜스 복구 경로를 빼면 대부분의 "
            "회차가 0점이 된다. 성능을 보고할 때 이 둘을 한 숫자로 섞지 않는다: "
            "과제를 수행했는가와 지시를 지켰는가는 다른 질문이다.\n")

    best = max(recs, key=lambda r: r["result"]["micro_f1"])
    distinguishable = [r for r in recs if r["id"] != base["id"] and not _overlaps(r, base)]
    lines.append("## 여기까지의 결론\n")
    lines.append(
        f"- 가장 높았던 설정: 실험 #{best['id']} ({best['name']}) "
        f"micro-F1 {best['result']['micro_f1']:.4f}")
    lines.append(
        f"- 기준선과 구간이 갈린 회차: {len(distinguishable)}건 "
        f"{[r['name'] for r in distinguishable] or '없음'}")
    if not distinguishable:
        lines.append(
            "- 프롬프트를 바꿔도 이 표본에서는 구분되지 않는다. 점추정 차이는 "
            "있지만 노트 100건의 표본 폭 안이다. 이 경우 프롬프트 선택은 성능을 "
            "올리는 수단이 아니라 모델마다 공정한 조건을 주기 위한 절차로 읽어야 한다.")
    lines.append(f"- 총 {len(recs)}회, 누적 {sum(r['elapsed_sec'] for r in recs) / 3600:.1f}시간")
    lines.append("")

    # ── 종료 판단 ─────────────────────────────────────────────────────
    # 규칙: 의무 5회 뒤, 부모 대비 개선이 아닌 회차가 3회 연속이면 멈춤.
    lines.append("## 종료 판단\n")
    lines.append(
        f"규칙은 의무 {MANDATORY}회 뒤 개선이 아닌 회차가 {STOP_AFTER}회 연속이면 멈추는 것이다. "
        "개선은 부모 회차와 신뢰구간이 갈리고 점추정이 높을 때만 센다.\n")
    lines.append("| # | 부모 | 부모 대비 | 연속 미개선 |")
    lines.append("|---|---|---|---:|")
    trace = stop_trace(recs)
    for x in trace:
        lines.append(f"| {x['id']} | #{x['parent']} | {x['verdict']} | {x['streak']} |")
    lines.append("")
    hit = next((x for x in trace if x["streak"] >= STOP_AFTER), None)
    if hit:
        lines.append(
            f"> #{hit['id']} 에서 {STOP_AFTER}회가 차 멈췄다. 다만 #{MANDATORY + 1}~#{hit['id']} 회차는 "
            "앞 회차 결과를 하나씩 보고 정하지 않고 한꺼번에 계획해 넣었다.\n")
    else:
        lines.append(f"> 아직 {STOP_AFTER}회가 차지 않았다.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    # 줄끝은 LF 로 고정함(플랫폼 기본값에 맡기면 CRLF 가 되어 문서 전체가 바뀐 것으로 보임).
    OUT_MD.write_text(_md_tilde("\n".join(lines)), encoding="utf-8", newline="\n")
    print(f"[저장] {OUT_MD}  (실험 {len(recs)}건)")
    return 0


if __name__ == "__main__":
    import argparse

    # 인자는 없음. 읽어 두어야 --help 가 설명만 찍고 끝남.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
