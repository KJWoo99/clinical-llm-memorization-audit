"""src/notes.py: 정답셋 파서 검증.

여기서 쓰는 예시는 전부 MIMIC-IV-Note 실제 노트에서 관찰된 형식을 그대로
옮긴 것임(환자 식별 정보는 원본에서 이미 ___ 로 비식별화돼 있음).
파서가 틀리면 정답셋이 틀리고, 그러면 아래 모든 평가가 무의미해짐.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from notes import (
    classify_item,
    drug_aliases,
    extract_drug_name,
    extract_section,
    gold_medications,
    neutral_items,
    normalize_drug,
    parse_items,
    remove_section,
)

NOTE = """
Service: MEDICINE

Discharge Medications:
1. Albuterol Inhaler 2 PUFF IH Q4H:PRN wheezing, SOB
2. Emtricitabine-Tenofovir (Truvada) 1 TAB PO DAILY
3. Furosemide 40 mg PO DAILY
RX *furosemide 40 mg 1 tablet(s) by mouth Daily Disp #*30 Tablet
Refills:*3
4. Acetaminophen 500 mg PO Q6H:PRN pain

Discharge Disposition:
Home

Discharge Diagnosis:
Ascites from Portal HTN
"""


class TestSection:
    def test_extracts_only_medication_section(self):
        sec = extract_section(NOTE)
        assert sec is not None
        assert "Albuterol" in sec
        assert "Acetaminophen" in sec
        assert "Home" not in sec
        assert "Ascites" not in sec

    def test_returns_none_when_absent(self):
        assert extract_section("Service: MEDICINE\n\nDischarge Disposition:\nHome") is None

    def test_remove_section_drops_answers_but_keeps_rest(self):
        stripped = remove_section(NOTE)
        assert "Albuterol" not in stripped
        assert "Furosemide" not in stripped
        assert "Service: MEDICINE" in stripped
        assert "Discharge Disposition" in stripped

    def test_remove_section_is_noop_when_absent(self):
        text = "Service: MEDICINE\n\nDischarge Disposition:\nHome"
        assert remove_section(text) == text


class TestItems:
    def test_numbered_items_are_split(self):
        items = parse_items(extract_section(NOTE))
        assert len(items) == 4

    def test_wrapped_continuation_stays_with_its_item(self):
        """RX/Refills 줄은 3번 항목에 붙어야 하고 새 항목이 되면 안 됨."""
        items = parse_items(extract_section(NOTE))
        assert items[2].startswith("Furosemide")
        assert "Refills" in items[2]

    def test_empty_section_gives_no_items(self):
        assert parse_items("\n\n") == []

    def test_number_inside_text_does_not_start_new_item(self):
        section = "\n1. Aspirin 81 mg PO DAILY, hold if SBP < 100\n"
        assert len(parse_items(section)) == 1


class TestDrugName:
    def test_simple_name_before_dose(self):
        assert extract_drug_name("Acetaminophen 500 mg PO Q6H:PRN pain") == "Acetaminophen"

    def test_multiword_name(self):
        assert extract_drug_name("Ipratropium Bromide Neb 1 NEB IH Q6H SOB") == "Ipratropium Bromide Neb"

    def test_parenthetical_brand_is_kept(self):
        got = extract_drug_name("Emtricitabine-Tenofovir (Truvada) 1 TAB PO DAILY")
        assert got == "Emtricitabine-Tenofovir (Truvada)"

    def test_parenthetical_containing_digits_is_kept(self):
        got = extract_drug_name("OxyCODONE (Immediate Release) 2.5-5 mg PO Q4H:PRN Pain")
        assert got == "OxyCODONE (Immediate Release)"

    def test_digits_inside_name_are_not_treated_as_dose(self):
        """'Vitamin B-12' 의 B-12 는 용량이 아니라 이름의 일부임."""
        assert extract_drug_name("Vitamin B-12 100 mcg PO DAILY") == "Vitamin B-12"

    def test_redacted_dose_terminates_name(self):
        assert extract_drug_name("Allopurinol ___ mg PO DAILY") == "Allopurinol"

    def test_old_style_sig_format(self):
        got = extract_drug_name(
            "Docusate Sodium 100 mg Capsule Sig: One (1) Capsule PO BID (2 times a day)."
        )
        assert got == "Docusate Sodium"

    def test_rx_tail_is_stripped(self):
        got = extract_drug_name(
            "Furosemide 40 mg PO DAILY RX *furosemide 40 mg 1 tablet(s) by mouth"
        )
        assert got == "Furosemide"

    def test_percentage_dose_terminates_name(self):
        assert extract_drug_name("Fluocinonide 0.05 % Cream Sig: One (1) Appl") == "Fluocinonide"


class TestNormalize:
    def test_case_and_space_insensitive(self):
        assert normalize_drug("  DOCUSATE   Sodium ") == "docusate sodium"

    def test_parenthetical_removed(self):
        assert normalize_drug("Emtricitabine-Tenofovir (Truvada)") == "emtricitabine-tenofovir"

    def test_hyphen_preserved_since_it_distinguishes_drugs(self):
        assert normalize_drug("Fluticasone-Salmeterol") == "fluticasone-salmeterol"


class TestAliases:
    def test_brand_name_is_accepted_alias(self):
        al = drug_aliases("Emtricitabine-Tenofovir (Truvada)")
        assert "emtricitabine-tenofovir" in al
        assert "truvada" in al

    def test_plain_name_has_single_alias(self):
        assert drug_aliases("Furosemide") == {"furosemide"}


class TestGold:
    def test_end_to_end(self):
        gold = gold_medications(NOTE)
        assert gold == [
            "Albuterol Inhaler",
            "Emtricitabine-Tenofovir (Truvada)",
            "Furosemide",
            "Acetaminophen",
        ]

    def test_none_when_no_section(self):
        assert gold_medications("Service: MEDICINE") is None


class TestItemClassification:
    """번호 목록에는 '퇴원 복용약'이 아닌 항목이 섞여 있음. 실측 20,000건에서
    확인된 세 유형을 정답에서 빼되 환각으로도 세지 않음."""

    def test_held_medication_is_not_gold(self):
        item = ("HELD- Examplostatin 10 mg PO DAILY Hold while your blood pressure is low. "
                "Restart Examplostatin only when your doctor tells you to.")
        assert classify_item(item) == "held"

    def test_outpatient_lab_work_is_not_medication(self):
        item = "Outpatient Lab Work Please obtain biweekly BMP to monitor K and Na."
        assert classify_item(item) == "non_medication"

    def test_sliding_scale_is_not_a_discrete_drug(self):
        item = "Sliding Scale Insulin SC Sliding Scale Breakfast Lunch Dinner Bedtime Humalog"
        assert classify_item(item) == "sliding_scale"

    def test_real_drugs_named_like_supplies_are_medications(self):
        """glucose, kit, strip, pump 는 실제 약물명에 쓰임(트러블슈팅 6-b).
        glucose 나 kit 을 용품 목록에 넣으면 실패하는지 봄."""
        for item in ("Glucose Gel 15 gram PO PRN hypoglycemia", "Glucose Tab 4 gram PO PRN",
                     "EPINEPHrine Kit 1 mg IM ONCE", "Fluorescein Strip 1 STRP OU ONCE",
                     "AndroGel 1% Pump 2 PUMP TD DAILY"):
            assert classify_item(item) == "medication", item

    def test_blood_glucose_supplies_are_not_medication(self):
        """혈당 측정 용품은 glucose 토큰이 아니라 'blood glucose' 구로 잡음. 다른 용품 토큰이 없는
        항목으로 시험해야 구 규칙이 빠진 것을 잡음."""
        assert classify_item("Blood Glucose Control Solution 1 bottle") == "non_medication"

    def test_dispense_instruction_is_not_medication(self):
        """조제 지시문만 있는 항목(측정기 상표명 + please dispense)은 약이 아님."""
        assert classify_item("Accu-Chek Aviva Plus Please dispense 1 box") == "non_medication"

    def test_ordinary_medication_is_medication(self):
        assert classify_item("Furosemide 40 mg PO DAILY") == "medication"

    def test_gold_excludes_non_medications(self):
        note = """
Discharge Medications:
1. Furosemide 40 mg PO DAILY
2. HELD- Trimethoprim 100 mg PO Q24H This medication was held.
3. Outpatient Lab Work Please obtain biweekly BMP.
4. Aspirin 81 mg PO DAILY

Discharge Disposition:
Home
"""
        assert gold_medications(note) == ["Furosemide", "Aspirin"]

    def test_neutral_items_capture_the_excluded_ones(self):
        note = """
Discharge Medications:
1. Furosemide 40 mg PO DAILY
2. HELD- Trimethoprim 100 mg PO Q24H This medication was held.

Discharge Disposition:
Home
"""
        assert neutral_items(note) == ["Trimethoprim"]


class TestUnclosedParenthesis:
    """비식별화가 닫는 괄호까지 지우면 용량이 이름에 딸려옴:
    실측 20,000건에서 'Morphine SR (MS ___ 15 mg PO Q8H' 로 확인된 결함."""

    def test_unclosed_paren_does_not_swallow_dose(self):
        assert extract_drug_name("Morphine SR (MS ___ 15 mg PO Q8H") == "Morphine SR"

    def test_closed_paren_still_kept(self):
        assert extract_drug_name("OxyCODONE (Immediate Release) 5 mg PO Q4H") == \
            "OxyCODONE (Immediate Release)"


class TestDosageFormAlias:
    """용량이 없는 약은 제형이 이름에 붙음('Multivitamin Tablet').
    제형을 떼기만 하면 'Nicotine Patch' 가 망가지므로 두 표기를 모두 인정함."""

    def test_form_stripped_variant_is_an_alias(self):
        al = drug_aliases("Multivitamin Tablet")
        assert "multivitamin tablet" in al
        assert "multivitamin" in al

    def test_form_that_is_part_of_the_name_is_preserved(self):
        al = drug_aliases("Nicotine Patch")
        assert "nicotine patch" in al

    def test_omega3_capsule(self):
        al = drug_aliases("omega-3 fatty acids Capsule")
        assert "omega-3 fatty acids" in al


class TestSupplyItems:
    """전 구간 무작위 검토에서 나온 사례(트러블슈팅 5): 인슐린 펜 니들 같은
    의료용품이 번호 목록에 섞여 있음. 복용약이 아니므로 정답에서 뺌."""

    def test_pen_needle_is_supply(self):
        item = "Pen Needle (examplo needles (single use)) 31 G ___ miscellaneous qid"
        assert classify_item(item) == "non_medication"

    def test_pen_needle_singular_lowercase(self):
        assert classify_item("pen needle, examplo 31 gauge x ___ miscellaneous QID") == \
            "non_medication"

    def test_glucometer_is_supply(self):
        assert classify_item("Glucometer Please dispense one glucose meter") == "non_medication"

    def test_lowercase_sig_is_also_cut(self):
        """'Sig' 는 대소문자가 섞여 나타남. 소문자 sig 를 못 자르면
        용법 문구가 약물명에 딸려옴."""
        assert extract_drug_name("Furosemide Oral sig unknown") == "Furosemide Oral"

    def test_bare_disp_is_also_cut(self):
        """조제 지시 'disp' 가 콜론이나 # 없이 붙어도 약물명에서 뗌(트러블슈팅 5)."""
        assert extract_drug_name("Multivitamin one tab daily disp") == "Multivitamin one tab daily"
        assert extract_drug_name("Pen Needle Disp box") == "Pen Needle"
        assert extract_drug_name("Dispersible Aspirin") == "Dispersible Aspirin"
