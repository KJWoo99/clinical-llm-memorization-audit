"""퇴원요약문에서 퇴원 약물 목록을 결정론적으로 뽑아내는 파서.

이 파일이 정답셋을 만듦. 규칙으로만 동작하며, 여기서 틀리면 아래 모든 평가가 틀어짐.

정답은 `prescriptions` 테이블이 아니라 노트의 `Discharge Medications:` 절에서 뽑음.
테이블은 입원 중 투약 전체라, 한 입원 건에서 노트의 퇴원약 8개 옆에 헤파린 예방투여,
독감백신, 염화칼륨, 생리식염수 플러시 같은 입원 전용 항목 4개가 섞임. 노트 절은
퇴원약만 담고 약물명이 약국 어휘와 일치함. 테이블은 어휘 검증에만 씀.
"""
from __future__ import annotations

import re

SECTION_HEADER = re.compile(r"^[ \t]*Discharge\s+Medications?[ \t]*:", re.IGNORECASE | re.MULTILINE)

# 약물 목록이 끝나는 지점. 실측상 97.6% 가 Discharge Disposition 으로 끝나지만
# 나머지도 모두 "줄 첫머리의 제목:" 형태라 일반 규칙을 함께 둠.
KNOWN_NEXT_SECTIONS = (
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
    "Facility",
    "Completed by",
)
_KNOWN_RE = re.compile(
    r"^[ \t]*(?:" + "|".join(re.escape(s) for s in KNOWN_NEXT_SECTIONS) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_GENERIC_NEXT_RE = re.compile(r"^[ \t]*[A-Z][A-Za-z /'\-]{2,45}:[ \t]*$", re.MULTILINE)

# "1." "2 ." 같은 번호 항목. 줄 첫머리에만 인정함.
_ITEM_RE = re.compile(r"^[ \t]*(\d{1,3})[.)][ \t]*(.*)$", re.MULTILINE)

# 항목 본문에서 버릴 꼬리표: 조제/리필 지시는 약물명이 아님.
_TAIL_MARKERS = (
    "RX *", "Disp:*", "Disp #*", "Refills:*", "Start:", "Substitution:",
)

# MIMIC 비식별화로 지워진 자리
_REDACTED = "___"

# 번호 목록에 섞여 있지만 "퇴원 시 복용할 약"이 아닌 항목들.
# 정답에서 빼되, 모델이 이걸 출력해도 환각으로 세지 않음(중립 처리):
# 노트의 목록에 실제로 적혀 있어 어느 쪽 행동도 틀렸다고 하기 어려움.
_HELD_PREFIX = re.compile(r"^\s*held\s*[-:]", re.IGNORECASE)
_NON_MEDICATION_PREFIXES = (
    "outpatient lab work",
    "outpatient physical therapy",
    "outpatient occupational therapy",
    "please",
    "home ",
    "discharge instructions",
)
# 인슐린 슬라이딩 스케일은 개별 약물이 아니라 투여 규칙표라 이름을 뽑을 수 없음.
_SLIDING_SCALE = re.compile(r"sliding\s+scale", re.IGNORECASE)

# 약물명이 이 길이를 넘으면 추출이 어긋난 것임. 실측에서 걸린 것들은
# 목록에 섞인 의료용품, 지시문이었음(목발, 보행기, 압박기기, 인슐린 주사바늘,
# 상처 드레싱). 공정하게 채점할 수 없으므로 정답에서 빼고 중립 처리함.
_MAX_NAME_TOKENS = 6

# 약물이 아니라 의료용품, 기기임. 번호 목록에 함께 적히지만 복용약이 아니므로
# 정답에서 빼고 중립 처리함(혈당측정기, 검사지, 혈압계, 도뇨관, 보행보조기 등).
# 'glucose' 는 여기 두면 안 됨: `Glucose Gel`(86건), `Glucose Tab`(7건)은
# 저혈당 치료제로 약국 어휘에 실재하는 약임(트러블슈팅 6-b). 혈당 측정 용품은 아래
# _SUPPLY_PHRASES 의 'blood glucose' 로 잡음.
#
# 반대로 'strip'/'kit'/'pump' 는 넣으면 안 됨: `Fluorescein Strip`,
# `EPINEPHrine Kit`, `AndroGel 1% Pump` 처럼 실제 약물명에 쓰임.
# 'syringe'/'needle'/'nebulizer'/'catheter' 도 약물명에 나타나지만, 실측상
# 이 토큰으로 빠지는 항목은 전부 인슐린 주사기, 펜니들, 분무기 같은 용품이라
# 그대로 둠.
_SUPPLY_WORDS = frozenset({
    "glucometer", "meter", "strips", "syringe", "syringe-needle",
    "needle", "needles", "cuff", "cath", "catheter", "walker", "crutches", "bench",
    "dressing", "supplies", "dispose", "disposable", "dispos", "device",
    "commode", "wheelchair", "cane", "nebulizer", "spacer",
    "lancet", "lancets", "monitor", "monitoring", "fingerstick", "fingersticks",
})

# 토큰 하나로는 가를 수 없고 구(句)로 봐야 하는 용품. 'blood glucose ...' 는
# 전부 혈당 측정 용품이며, 실제 약물명에는 이 두 단어가 이어 나오지 않음.
_SUPPLY_PHRASES = ("blood glucose",)
_DISPENSE_HINT = re.compile(r"please\s+dispense|as\s+directed\s+disp", re.IGNORECASE)

# 뒤에 붙는 제형 표기. 이름에서 떼어낸 형태도 별칭으로 함께 인정함
# ('Multivitamin Tablet' 과 'Multivitamin' 을 모두 맞다고 봐야 함).
# 떼어내기만 하면 'Nicotine Patch' 처럼 제형이 이름의 일부인 약이 망가지므로
# 원형과 축약형을 둘 다 남김.
_DOSAGE_FORMS = (
    "tablet", "tablets", "capsule", "capsules", "solution", "suspension",
    "syrup", "cream", "ointment", "gel", "lotion", "powder", "suppository",
    "injection", "syringe", "vial", "packet", "lozenge", "chewable",
    "oral", "delayed release", "extended release",
)


def extract_section(text: str) -> str | None:
    """노트에서 퇴원 약물 절의 본문만 잘라냄. 없으면 None."""
    m = SECTION_HEADER.search(text)
    if m is None:
        return None
    rest = text[m.end():]

    end = len(rest)
    known = _KNOWN_RE.search(rest)
    if known is not None:
        end = known.start()
    generic = _GENERIC_NEXT_RE.search(rest, 0, end)
    if generic is not None:
        end = generic.start()
    return rest[:end]


def remove_section(text: str) -> str:
    """퇴원 약물 절을 통째로 들어낸 노트를 만듦.

    '모델이 노트를 읽는가, 아니면 학습 때 본 이 데이터셋을 외웠는가'를
    가르는 감사 조건에 씀. 답이 본문에 없는데도 맞히면 암기를 의심해야 함.
    """
    m = SECTION_HEADER.search(text)
    if m is None:
        return text
    section = extract_section(text)
    if section is None:
        return text
    return text[: m.start()] + text[m.end() + len(section):]


def parse_items(section: str) -> list[str]:
    """번호 항목을 잘라 원문 그대로 돌려줌.

    항목은 여러 줄에 걸쳐 이어지므로, 다음 번호가 나오기 전까지를 한 항목으로 묶음.
    """
    matches = list(_ITEM_RE.finditer(section))
    if not matches:
        return []

    items: list[str] = []
    for i, m in enumerate(matches):
        start = m.start(2)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section)
        body = section[start:end]
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            items.append(body)
    return items


def _strip_tail(item: str) -> str:
    cut = len(item)
    for marker in _TAIL_MARKERS:
        idx = item.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return item[:cut].strip()


def classify_item(item: str) -> str:
    """항목이 퇴원 복용약인지, 정답에서 빼야 할 항목인지 가름.

    반환: "medication" | "held" | "non_medication" | "sliding_scale"
    """
    stripped = item.strip()
    if _HELD_PREFIX.match(stripped):
        return "held"
    low = stripped.lower()
    if any(low.startswith(p) for p in _NON_MEDICATION_PREFIXES):
        return "non_medication"
    if _SLIDING_SCALE.search(low):
        return "sliding_scale"
    if _DISPENSE_HINT.search(low):
        return "non_medication"
    name = extract_drug_name(stripped)
    norm = normalize_drug(name)
    if len(norm.split()) > _MAX_NAME_TOKENS:
        return "non_medication"
    if _SUPPLY_WORDS & set(norm.split()):
        return "non_medication"
    if any(ph in norm for ph in _SUPPLY_PHRASES):
        return "non_medication"
    return "medication"


def extract_drug_name(item: str) -> str:
    """항목 문자열에서 약물명만 뽑음.

    용량이 시작되는 지점(숫자로 시작하는 토큰, 또는 비식별화된 ___)에서 자름.
    'Vitamin B-12' 처럼 이름 안에 숫자가 있는 경우가 있어, 토큰이 숫자로
    *시작*할 때만 용량으로 봄: 'B-12' 는 B 로 시작하므로 이름에 남음.

    괄호 안은 제품명일 수 있어 숫자가 나와도 이름으로 취급하는데, 비식별화가
    닫는 괄호째 지워버리는 경우가 있다('Morphine SR (MS ___ 15 mg PO Q8H').
    그대로 두면 괄호가 끝나지 않아 용량까지 전부 이름에 딸려오므로,
    닫히지 않은 괄호가 남으면 그 괄호 앞에서 잘라냄.
    """
    text = _strip_tail(item)
    tokens = text.split()
    name_tokens: list[str] = []
    depth = 0
    for tok in tokens:
        if depth == 0 and (tok.startswith(_REDACTED) or re.match(r"^\d", tok)):
            break
        depth += tok.count("(") - tok.count(")")
        depth = max(depth, 0)
        name_tokens.append(tok)
        if len(name_tokens) >= 8:
            break
    name = " ".join(name_tokens)
    name = re.split(r"\bSig\b", name, maxsplit=1, flags=re.IGNORECASE)[0]
    # 조제 지시 'Disp' 는 꼬리표 'Disp:*', 'Disp #*' 꼴일 때만 떼고 있어, 콜론이나 # 없이 붙은 'disp' 가
    # 약물명에 딸려옴. 시드를 고정한 전 구간 무작위 25,000건 검증에서 3종(노트 3건)이 나옴.
    name = re.split(r"\bDisp\b", name, maxsplit=1, flags=re.IGNORECASE)[0]
    if name.count("(") > name.count(")"):
        name = name[: name.rindex("(")]
    return name.strip(" ,.;:-")


def normalize_drug(name: str) -> str:
    """비교용 정규화. 대소문자, 공백, 괄호 표기 차이를 없앰."""
    s = name.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\-/+ ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" -")


def _strip_dosage_form(norm: str) -> str:
    """정규화된 이름 끝의 제형 표기를 뗌. 뗄 게 없으면 그대로 돌려줌."""
    out = norm
    changed = True
    while changed:
        changed = False
        for form in _DOSAGE_FORMS:
            if out.endswith(" " + form):
                out = out[: -(len(form) + 1)].strip()
                changed = True
    return out


def drug_aliases(name: str) -> set[str]:
    """한 약물을 가리키는 표기들. 표기 차이로 정답을 놓치지 않기 위함.

    - 'Emtricitabine-Tenofovir (Truvada)' -> 성분명과 'truvada' 둘 다
    - 'Multivitamin Tablet' -> 제형을 뗀 'multivitamin' 도 함께
      (제형을 떼기만 하면 'Nicotine Patch' 처럼 제형이 이름의 일부인 약이
       망가지므로, 떼지 않고 별칭을 하나 더 얹는 방식을 씀)
    """
    aliases = {normalize_drug(name)}
    for inner in re.findall(r"\(([^)]*)\)", name):
        norm = normalize_drug(inner)
        if norm:
            aliases.add(norm)
    for a in list(aliases):
        stripped = _strip_dosage_form(a)
        if stripped:
            aliases.add(stripped)
    aliases.discard("")
    return aliases


def gold_medications(text: str) -> list[str] | None:
    """노트 하나의 정답 약물명 목록. 절이 없으면 None(표본에서 제외해야 함).

    보류(HELD-), 비약물 안내, 인슐린 슬라이딩 스케일 항목은 빠짐.
    그 항목들은 `neutral_items` 로 따로 받아 채점에서 중립 처리함.
    """
    section = extract_section(text)
    if section is None:
        return None
    names = []
    for item in parse_items(section):
        if classify_item(item) != "medication":
            continue
        name = extract_drug_name(item)
        if name:
            names.append(name)
    return names


def neutral_items(text: str) -> list[str]:
    """정답에서 빼되 환각으로도 세지 않을 항목들의 이름."""
    section = extract_section(text)
    if section is None:
        return []
    out = []
    for item in parse_items(section):
        kind = classify_item(item)
        if kind == "medication":
            continue
        if kind == "held":
            # 'HELD- Examplostatin 10 mg ...' -> 보류된 약 이름 자체를 중립 처리
            name = extract_drug_name(_HELD_PREFIX.sub("", item).strip())
        else:
            name = extract_drug_name(item)
        if name:
            out.append(name)
    return out
