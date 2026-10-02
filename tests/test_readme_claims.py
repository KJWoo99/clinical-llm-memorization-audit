"""README 와 report 본문 문장 속 수치가 채점 결과 파일과 같은지 검증.

표는 `render_results_tables.py`, `render_report_tables.py` 가 결과 파일에서 다시
그리므로 어긋나면 표 생성기가 잡음. 이 테스트는 사람이 손으로 옮겨 적은 문장을
지킴. 채점기를 고쳐 다시 채점하면 표는 새로 그려도 문장의 숫자는 옛 값으로 남기 쉬움.

결과 파일이 없으면 건너뜀.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "phase4_results.json"
README = ROOT / "README.md"
REPORT = ROOT / "docs" / "report.md"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="채점 결과 파일이 없다")


def _signed(x: float, nd: int = 3) -> str:
    """문서의 부호 표기(음수는 U+2212, 양수는 +)로 적음."""
    s = f"{abs(x):.{nd}f}"
    return ("−" if x < 0 else "+") + s


@pytest.fixture(scope="module")
def res() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    # 문서는 GitHub 에서 취소선이 되지 않게 물결표를 \~ 로 적음. 수치 대조는 ~ 로 되돌려서 함.
    return README.read_text(encoding="utf-8").replace("\\~", "~")


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT.read_text(encoding="utf-8").replace("\\~", "~")


def _pair(res: dict, a: str) -> dict:
    return next(p for p in res["pairs"] if p["a"] == a)


def test_answer_table_pair_differences(res, readme):
    """요약 표의 짝 차이가 결과 파일의 차이와 같음."""
    for label, a in [("MedGemma", "medgemma-4b"), ("Meditron3", "meditron3-8b"),
                     ("Apertus MeditronFO", "apertus-8b-meditronfo"),
                     ("OLMo-2 32B MeditronFO", "olmo2-32b-meditronfo")]:
        want = f"{label} {_signed(_pair(res, a)['diff'])}"
        assert want in readme, want


def test_answer_table_recommended_scores(res, readme):
    m = res["models"]
    for label, key in [("Baichuan-M2", "baichuan-m2-32b"), ("Qwen2.5 Instruct", "qwen25-32b"),
                       ("Llama 3.1 Instruct", "llama31-8b")]:
        want = f"{label}({m[key]['normal']['micro_f1']:.3f})"
        assert want in readme, want


def test_empty_note_claim(res, readme, report):
    """노트를 빼면 11개가 정확히 0 이고 나머지가 0.06~0.08 이라는 문장."""
    empty = [m["condition_micro_f1"]["empty"] for m in res["models"].values()]
    zeros = sum(1 for v in empty if v == 0.0)
    rest = sorted(v for v in empty if v > 0.0)
    assert f"16모델 중 {zeros}개가 micro-F1 0.000" in readme
    assert f"나머지도 {rest[0]:.2f}~{rest[-1]:.2f}" in readme
    assert f"16모델 중 {zeros}개가 정확히 0.000" in report


def test_hallucination_rates_in_prose(res, readme, report):
    m = res["models"]
    pct = {k: f"{v['hallucination_rate'] * 100:.1f}%" for k, v in m.items()}
    assert f"MedGemma 환각률 {pct['medgemma-4b']}" in readme
    assert (f"MedGemma {pct['medgemma-4b']} 대 Gemma 3 4B {pct['gemma3-4b']}, "
            f"Meditron3 {pct['meditron3-8b']} 대") in report


def test_bonferroni_paragraph(res, report):
    """보정 구간 문단(report 6절)의 다섯 짝 값이 결과 파일의 ci_bonferroni 와 같음."""
    text = " ".join(report.split())
    for n, a in [(1, "medgemma-4b"), (2, "meditron3-8b"), (3, "medgemma-4b"),
                 (8, "olmo2-32b-meditronfo"), (7, "apertus-8b-meditronfo")]:
        p = res["pairs"][n - 1]
        assert p["a"] == a
        lo, hi = p["ci_bonferroni"]
        want = f"쌍{n} {_signed(p['diff'], 4)}"
        assert want in text, want
        assert f"[{_signed(lo, 4)}, {_signed(hi, 4)}]" in text, (n, lo, hi)
    sig_only_uncorrected = [i + 1 for i, p in enumerate(res["pairs"])
                            if p["significant"] and not p["significant_bonferroni"]]
    assert sig_only_uncorrected == [7]
    # 보정 구간 끝의 시드 흔들림: 쌍7 위 끝의 범위를 문서가 그대로 인용하고,
    # 스무 시드 모두에서 보정 판정이 한 번 잰 판정과 같아야 함.
    for p in res["pairs"]:
        s = p["ci_bonferroni_seed_spread"]
        assert s["n_seeds"] == 20
        (lo_min, lo_max), (up_min, up_max) = s["lower"], s["upper"]
        if p["significant_bonferroni"]:
            assert up_max < 0 or lo_min > 0
        else:
            assert lo_max <= 0 <= up_min
    up = res["pairs"][6]["ci_bonferroni_seed_spread"]["upper"]
    assert f"{_signed(up[0], 3)}~{_signed(up[1], 3)}" in text


def test_structure_verdict_counts(res, report):
    """섞음 유지율 90% 를 넘은 모델 수와 이름이 문장과 같음."""
    free = [k for k, m in res["models"].items() if m["structure_verdict"] == "구조 무관"]
    assert free == ["baichuan-m2-32b"]
    n_dep = len(res["models"]) - len(free)
    assert f"나머지 {n_dep}모델은" in report
    assert f"{n_dep}모델이 \"구조 의존\"" in report
    ratio = {k: m["condition_micro_f1"]["shuffled"] / m["memorisation_basis"]["denominator_f1"]
             for k, m in res["models"].items()}
    lo, hi = min(ratio.values()), max(ratio.values())
    assert f"정상 대비 {lo * 100:.0f}~{hi * 100:.0f}%" in report
    assert f"Baichuan-M2 하나만 {ratio['baichuan-m2-32b'] * 100:.1f}%로" in report


def test_rare_drug_recall_of_top_five(res, readme):
    """상위 다섯 모델의 등장 10회 미만 재현율 범위."""
    top = sorted(res["models"].values(), key=lambda m: -m["normal"]["micro_f1"])[:5]
    cells = [m["recall_by_rarity"]["10회 미만"] for m in top]
    rates = [hit / n for hit, n in cells]
    lo, hi = min(rates), max(rates)
    hits = [hit for hit, _ in cells]
    assert {n for _, n in cells} == {12}
    # 요약은 건수와 비율을 함께 적음(12건뿐이라 비율만 쓰면 과하게 읽힘).
    assert f"그중 {min(hits)}~{max(hits)}건({lo * 100:.0f}~{hi * 100:.0f}%)을 맞혔는데" in readme


P0 = {m: ROOT / "outputs" / f"phase0_audit_{m}.json" for m in ("first20000", "random25000")}


@pytest.mark.skipif(not all(p.exists() for p in P0.values()), reason="파서 검증 결과 없음")
def test_parser_validation_numbers_come_from_phase0_outputs(readme):
    """파서 검증 수치(98.1%, 94.0%, 96.6% 와 무작위 표본 값)가 phase0_audit 결과 파일과 같음."""
    text = " ".join(readme.split())
    a = json.loads(P0["first20000"].read_text(encoding="utf-8"))
    b = json.loads(P0["random25000"].read_text(encoding="utf-8"))
    assert a["passed"] and b["passed"] and b["seed"] is not None
    assert (f"절 파싱률 {a['section_rate'] * 100:.1f}%, 항목 추출 {a['item_rate'] * 100:.1f}%, "
            f"약물명의 약국 어휘 일치 {a['vocab_match_mentions'] * 100:.1f}%") in text
    assert (f"{b['section_rate'] * 100:.1f}%, {b['item_rate'] * 100:.1f}%, "
            f"{b['vocab_match_mentions'] * 100:.1f}% 다") in text


@pytest.mark.skipif(not (RESULTS.exists() and any((ROOT / "outputs" / "responses").glob("*_final.jsonl"))),
                    reason="채점 결과 또는 응답 파일 없음(응답은 노트 내용이 섞여 저장소에 올리지 않는다)")
def test_rendered_tables_match_readme(readme):
    """README 결과표 구간이 렌더러 출력과 같음. 문단을 README 에서만 고치면 다시 그릴 때 옛 문장으로
    돌아감."""
    import os
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_results_tables.py")], capture_output=True,
                       text=True, encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"), cwd=ROOT)
    assert r.returncode == 0, r.stderr[-500:]
    section = readme.split("<!-- 결과표 시작 -->", 1)[1].split("<!-- 결과표 끝 -->", 1)[0]
    assert section.strip() == r.stdout.strip()
