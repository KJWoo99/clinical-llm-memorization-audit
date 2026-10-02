"""이미 만들어진 절제실험 기록에서 노트 원문을 걷어냄.

    python scripts/redact_ablation_samples.py

절제실험 기록의 `samples`(모델 응답 앞 600자)가 대상임.
모델은 퇴원기록의 약 처방 줄을 그대로 옮겨 적으므로(비식별 표시 `___` 까지),
그 응답은 MIMIC 노트 원문의 일부임. 기록 파일은 저장소에 올라가므로 DUA 에 걸림.

원문은 버리지 않음. "왜 이 점수인가"를 볼 때 필요하기 때문임. 깃에서
빠지는 `outputs/responses/ablation_samples/` 로 옮기고, 기록에는 형식 정보
(어느 경로로 건졌나, 몇 개를 뽑았나, 응답 길이)와 원문 위치만 남김.

여러 번 돌려도 됨. 이미 걷어낸 기록은 건너뜀.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "outputs" / "ablation"
RAW = ROOT / "outputs" / "responses" / "ablation_samples"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    moved = skipped = 0
    for path in sorted(ABL.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        samples = rec.get("samples", [])
        if not any("response" in s or "note_id" in s for s in samples):
            skipped += 1
            continue
        raw_rel = f"outputs/responses/ablation_samples/{path.name}"
        (ROOT / raw_rel).write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        rec["samples"] = [{"recovered_by": s.get("recovered_by"),
                           "n_parsed": s.get("n_parsed"),
                           "response_chars": len(s.get("response", ""))} for s in samples]
        rec["samples_raw"] = raw_rel
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        moved += 1
        print(f"  걷어냄: {path.name}  -> {raw_rel}")
    print(f"걷어낸 기록 {moved}건, 이미 깨끗한 기록 {skipped}건")
    return 0


if __name__ == "__main__":
    import argparse

    # 인자는 없음. 읽어 두어야 --help 가 설명만 찍고 끝남.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
