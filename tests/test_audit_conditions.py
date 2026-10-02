"""감사 조건의 유효성과 암기 판정 규칙 검증.

사전등록(docs/prereg.md)은 조건마다 다른 것을 재려 함. 그 전제가 실제 데이터에서
성립하는지, 그리고 판정 규칙이 전제에 맞게 구현됐는지 확인함.

핵심 사실(표본 100건 실측):

    empty          정답 잔존  0.0%   -> 순수 암기 검사로 유효
    other_patient  정답 잔존 16.5%   -> 흔한 약이 남의 노트에도 있어 오염됨
    no_section     정답 잔존 59.7%   -> 이미 문서화된 한계 (독해 검사)
    shuffled       정답 잔존 99.3%   -> 구조만 파괴, 답은 그대로 (의도대로)

그래서 암기 판정은 `empty` 만으로 함. `other_patient` 를 섞으면 "우연한 겹침"을
"암기"로 오판함.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompts import (  # noqa: E402
    PROMPT_VARIANTS,
    build_prompt,
    header_only,
    shuffle_sentences,
)

MEMORISATION_STRONG = 0.50
MEMORISATION_PRESENT = 0.25


def _verdict(empty_ratio: float) -> str:
    """run_phase4_evaluate.py 의 암기 판정 규칙."""
    if empty_ratio >= MEMORISATION_STRONG:
        return "암기 강함"
    if empty_ratio >= MEMORISATION_PRESENT:
        return "암기 있음"
    return "암기 경미"


# ── 판정 규칙 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "empty_ratio, expected",
    [(0.0, "암기 경미"), (0.24, "암기 경미"), (0.25, "암기 있음"),
     (0.49, "암기 있음"), (0.50, "암기 강함"), (0.90, "암기 강함")],
)
def test_memorisation_verdict_boundaries(empty_ratio, expected):
    assert _verdict(empty_ratio) == expected


def test_verdict_ignores_other_patient():
    """other_patient 가 아무리 높아도 판정은 empty 만 봄.

    other_patient 는 정답의 16.5%가 남의 노트에 이미 존재해 오염돼 있음.
    이걸 판정에 넣으면 본문을 제대로 읽은 모델을 '암기'로 몰 수 있음.
    """
    # empty 는 낮은데 other_patient 만 높은 상황
    empty_ratio, other_ratio = 0.05, 0.95
    assert _verdict(empty_ratio) == "암기 경미"
    # 두 조건의 max 로 판정하면 '암기 강함' 이 나옴: 그 차이를 고정함
    assert _verdict(max(empty_ratio, other_ratio)) == "암기 강함"


# ── 감사 조건 생성 ─────────────────────────────────────────────────────

def test_header_only_drops_clinical_body():
    """empty 조건은 임상 내용을 남기면 안 됨: 정답 잔존 0% 의 근거."""
    note = (
        "Name: ___\nUnit No: ___\n"
        "Service: MEDICINE\n"
        "Chief Complaint:\nchest pain\n"
        "Discharge Medications:\n1. Aspirin 81 mg\n2. Furosemide 20 mg\n"
    )
    out = header_only(note)
    assert "Aspirin" not in out
    assert "Furosemide" not in out
    assert "Name:" in out, "인적사항 블록은 남아야 한다"


def test_header_only_falls_back_when_no_marker():
    """알려진 절 머리말이 없으면 앞부분만 잘라 씀(무한정 남기지 않음)."""
    note = "x" * 5000
    out = header_only(note)
    assert len(out) == 400


def test_shuffle_preserves_content_but_breaks_order():
    """shuffled 는 답을 지우지 않음: 구조만 깨야 함(정답 잔존 99% 가 정상)."""
    note = "Alpha one. Beta two. Gamma three. Delta four. Epsilon five."
    out = shuffle_sentences(note, seed=0)
    for token in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon"):
        assert token in out, "shuffled 조건이 내용을 지우면 안 된다"
    assert out != note, "순서가 실제로 바뀌어야 한다"


def test_shuffle_is_reproducible():
    note = "A one. B two. C three. D four. E five. F six."
    assert shuffle_sentences(note, seed=7) == shuffle_sentences(note, seed=7)
    assert shuffle_sentences(note, seed=7) != shuffle_sentences(note, seed=8)


# ── 프롬프트 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", PROMPT_VARIANTS)
def test_build_prompt_embeds_note_and_instruction(variant):
    note = "PATIENT NOTE BODY 12345"
    out = build_prompt(note, variant)
    assert note in out
    assert "DISCHARGE SUMMARY" in out and "END" in out
    assert out.strip(), "빈 프롬프트가 나오면 안 된다"


def test_build_prompt_rejects_unknown_variant():
    with pytest.raises(ValueError, match="알 수 없는 프롬프트 변형"):
        build_prompt("note", "does-not-exist")


def test_prompt_variants_are_distinct():
    """세 변형이 실제로 달라야 '모델별 최적 프롬프트 선택'이 의미를 가짐."""
    note = "NOTE"
    outs = {v: build_prompt(note, v) for v in PROMPT_VARIANTS}
    assert len(set(outs.values())) == len(PROMPT_VARIANTS)


def test_grounded_variant_forbids_invention():
    """grounded 변형은 '노트에 없는 약을 지어내지 말라'를 명시해야 함."""
    out = build_prompt("NOTE", "grounded")
    assert "Do not add medications" in out


# ── 프롬프트 선택의 공정성 ─────────────────────────────────────────────

def _select_best(per_variant: dict[str, float]) -> str | None:
    """run_phase3_select_prompt.py 의 선택 규칙 (동점 시 정렬 순서 우선)."""
    best, best_f1 = None, -1.0
    for variant, f1 in sorted(per_variant.items()):
        if f1 > best_f1:
            best, best_f1 = variant, f1
    return best


def test_selection_picks_highest_f1():
    assert _select_best({"minimal": 0.39, "schema": 0.83, "grounded": 0.54}) == "schema"


def test_selection_is_deterministic_on_ties():
    """동점이면 항상 같은 변형이 뽑혀야 재현이 됨(무작위 금지)."""
    tied = {"minimal": 0.8, "schema": 0.8, "grounded": 0.8}
    first = _select_best(tied)
    for _ in range(20):
        assert _select_best(tied) == first
    # 정렬 순서상 첫 변형
    assert first == "grounded"


def test_common_note_subset_makes_comparison_fair():
    """변형별 노트 집합이 다르면 공통 노트로 좁혀야 공정함.

    한 변형만 쉬운 노트를 더 갖고 있으면 그 변형이 부당하게 이김.
    """
    notes_by_variant = {
        "minimal": {"n1", "n2", "n3"},
        "schema": {"n1", "n2"},
        "grounded": {"n1", "n2", "n3", "n4"},
    }
    common = set.intersection(*notes_by_variant.values())
    assert common == {"n1", "n2"}
    # 공통으로 좁히면 모든 변형이 같은 노트 위에서 비교됨
    for v in notes_by_variant:
        assert common <= notes_by_variant[v]


def test_note_order_helper_preserves_append_order():
    """공통 노트로 좁힐 때 점수-노트 대응이 어긋나면 안 됨."""
    rows = [
        {"variant": "minimal", "note_id": "a"},
        {"variant": "schema", "note_id": "a"},
        {"variant": "minimal", "note_id": "b"},
        {"variant": "schema", "note_id": "b"},
        {"variant": "minimal", "note_id": "c"},
    ]
    order = [r["note_id"] for r in rows if r["variant"] == "minimal"]
    assert order == ["a", "b", "c"], "append 순서와 같아야 인덱스 대응이 성립한다"


def test_phase1_sample_refuses_to_silently_replace_the_sample():
    """표본이 모르는 사이에 바뀌는 것을 막는 가드가 있어야 함.

    정답 파서를 고치면 노트별 `n_gold` 가 바뀌고, `qcut` 3분위 경계에 걸린
    노트들이 층을 옮김. 층이 바뀌면 층 내부 셔플이 재배열돼 시드가 같아도
    표본이 크게 달라짐: 실측으로 300건 중 229건이 교체됨.

    이미 그 표본으로 추론을 돌린 뒤라면 응답 수천 건이 한 번에 무효가 되고,
    결과를 본 뒤 표본을 다시 뽑는 것은 사전등록의 취지에 어긋남. 그래서
    구성이 달라지면 덮어쓰지 않고 중단하며, `--force` 로만 뚫을 수 있음.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_sample.py").read_text(
        encoding="utf-8"
    )
    assert "--force" in src, "강제 덮어쓰기 플래그가 없다"
    assert "덮어쓰지 않고 중단" in src, "표본 교체를 막는 가드가 없다"
    # 가드는 반드시 write_parquet 앞에 있어야 의미가 있음
    assert src.index("덮어쓰지 않고 중단") < src.index("sample.write_parquet"), (
        "가드가 저장보다 뒤에 있으면 이미 덮어쓴 뒤다"
    )


_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "README.md"
_P4 = _ROOT / "outputs" / "phase4_results.json"


@pytest.mark.skipif(not (_README.exists() and _P4.exists()),
                    reason="README 또는 Phase 4 산출물 없음")
def test_readme_headline_numbers_match_outputs():
    """README 첫 화면의 결론 수치가 산출물과 같아야 함.

    첫 화면은 가장 먼저 읽히는 자리임. 손으로 적는 자리이므로
    대조를 코드로 둠.
    """
    text = _README.read_text(encoding="utf-8")
    d = json.loads(_P4.read_text(encoding="utf-8"))
    # README 는 소수 셋째 자리까지 적음. 16모델 표에서 넷째 자리는 표만 어지럽힘.
    best = max(d["models"].values(), key=lambda m: m["normal"]["micro_f1"])
    want = {
        "최고 micro-F1": f"{best['normal']['micro_f1']:.3f}",
        "빈도 베이스라인": f"{d['baseline']['micro_f1']:.3f}",
        "MedGemma 환각률": f"{d['models']['medgemma-4b']['hallucination_rate'] * 100:.1f}%",
        "MedGemma JSON 준수": f"{d['models']['medgemma-4b']['json_compliance'] * 100:.0f}%",
    }
    for pair in d["pairs"][:2]:
        want[pair["a"] + " 차이"] = f"{pair['diff']:.3f}".replace("-", "−")
    # 의료 특화가 이긴 유일한 짝(쌍8)의 차이도 첫 화면에 있어야 함
    won = [p for p in d["pairs"] if p["significant"] and d["models"][p["a"]]["medical"]
           and p["diff"] > 0]
    for p in won:
        want[p["a"] + " 차이"] = f"{p['diff']:+.3f}"
    missing = [f"{k}={v}" for k, v in want.items() if v not in text]
    assert not missing, "README 에 없는 산출물 수치: " + ", ".join(missing)


@pytest.mark.skipif(not (_README.exists() and _P4.exists()),
                    reason="README 또는 산출물 없음")
def test_readme_states_no_memorisation_finding_first():
    """'외워서 답하는 게 아니다'가 결론 절에 있어야 함.

    이 프로젝트의 감사 설계 전체가 그 구분을 위한 것이므로, 결과가 첫 화면에
    없으면 설계 설명만 남음.
    """
    text = _README.read_text(encoding="utf-8")
    i = text.find("## 요약")
    assert i != -1, "README 에 결론 절('## 요약')이 없다"
    head = text[i:i + 2500]
    d = json.loads(_P4.read_text(encoding="utf-8"))
    # 16모델 중 일부는 빈 노트에 몇 개를 지어내 0 이 아님. 판정은 전부 '경미'여야
    # 아래 문장("외워서 답하지 않는다")이 성립함.
    verdicts = {m["memorisation_verdict"] for m in d["models"].values()}
    assert verdicts == {"암기 경미"}, f"암기 판정이 갈렸다: {verdicts}. 결론 문장을 다시 쓸 것"
    n_zero = sum(1 for m in d["models"].values() if m["condition_micro_f1"]["empty"] == 0.0)
    assert "0.000" in head and "외워" in head, "결론 절에 암기 판정(빈 노트에서 0.000)이 없다"
    assert f"{len(d['models'])}모델 중 {n_zero}개" in head, (
        f"결론 절의 '빈 노트 0.000 인 모델 수'가 산출물({n_zero}/{len(d['models'])})과 다르다")


CONDITION_CHECK = Path(__file__).resolve().parents[1] / "outputs" / "condition_inputs_check.json"


@pytest.mark.skipif(not CONDITION_CHECK.exists(), reason="condition_inputs_check.json 없음")
def test_documented_residue_matches_repo_count():
    """남의 노트 잔존 수치(사전등록 6절, report 7절)가 저장소 스크립트로 센 값과 같음."""
    root = Path(__file__).resolve().parents[1]
    d = json.loads(CONDITION_CHECK.read_text(encoding="utf-8"))
    sub, wb = d["other_patient_substring"], d["other_patient_word_boundary"]
    report = (root / "docs" / "report.md").read_text(encoding="utf-8")
    prereg = (root / "docs" / "prereg.md").read_text(encoding="utf-8")
    assert f"정답 약물의 {sub['mean_pct']}%가 남의 노트에도" in report
    assert f"단어 경계를 요구하면 {wb['mean_pct']}% 다" in report
    assert (f"정답 약물이 하나라도 남은 노트 {sub['any_pct']:.0f}%, 평균 {sub['mean_pct']}%, 중앙값 {sub['median_pct']}%, "
            f"0% 인 노트 {sub['zero_pct']:.0f}%, 25% 이상 {sub['ge25_pct']:.0f}%") in prereg
    assert (f"{wb['any_pct']:.0f}%, {wb['mean_pct']}%, {wb['median_pct']}%, {wb['zero_pct']:.0f}%, "
            f"{wb['ge25_pct']:.0f}% 가 된다") in prereg
    assert d["empty_substring"]["any_pct"] == 0.0
    assert d["shuffled"]["same_token_multiset"] == d["shuffled"]["n"] and d["shuffled"]["identical_to_normal"] == 0
    assert d["no_section"]["header_left"] == 0


SAMPLE = Path(__file__).resolve().parents[1] / "outputs" / "phase1_sample.parquet"
FREQ = Path(__file__).resolve().parents[1] / "outputs" / "phase1_drug_frequency.json"


@pytest.mark.skipif(not (SAMPLE.exists() and FREQ.exists()), reason="표본 또는 빈도 파일 없음")
def test_every_sample_gold_drug_has_a_corpus_frequency():
    """희귀도는 정답과 같은 파서로 셈. 빈도 파일에 없는 정답 약은 빈도 0 으로 읽혀 '10회 미만' 에 들어감
    (트러블슈팅 5)."""
    import polars as pl

    from notes import normalize_drug
    freq = json.loads(FREQ.read_text(encoding="utf-8"))
    gold = pl.read_parquet(SAMPLE, columns=["gold"])["gold"]
    missing = {normalize_drug(g) for gs in gold for g in gs if normalize_drug(g) not in freq}
    assert missing == set()
