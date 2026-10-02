"""평가 대상 LLM 을 전부 받음.

목록은 `src/models.py` 에서 읽음(같은 목록을 두 곳에 적으면 한쪽만 낡음).

Llama 저장소에는 safetensors 와 별개로 original/ 아래 원본 .pth(16GB)가 있음.
코드가 쓰는 형식은 safetensors 뿐이라 받을 이유가 없음.

게이트가 걸린 저장소(Llama, Gemma 계열)는 승인과 토큰이 있어야 받아짐.
토큰은 파일에 적지 않고 `huggingface-cli login` 으로 넣어 둠.

판은 `src/models.py` 의 `REVISIONS` 로 고정함. 결과를 낸 가중치와 같은 커밋을 받음.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 인자는 없음. 그래도 읽어 두어야 `--help` 가 설명만 찍고 끝남. 읽지 않으면 --help 에도 곧바로
# 수백 GB 내려받기를 시작함.
argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.utils import GatedRepoError  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import MODELS, REVISIONS  # noqa: E402

IGNORE_LLAMA_ORIGINAL = ["original/*"]

repos = []
for spec in MODELS.values():
    if spec.hf_id not in repos:
        repos.append(spec.hf_id)

print(f"평가 대상 {len(MODELS)}개, 저장소 {len(repos)}개")

failed = []
for repo in repos:
    ignore = IGNORE_LLAMA_ORIGINAL if repo.startswith("meta-llama") else None
    print(f"\n{'=' * 60}")
    print(f"받는 중: {repo} @ {REVISIONS[repo][:12]}" + ("  (원본 .pth 제외)" if ignore else ""))
    print(f"{'=' * 60}")
    t0 = time.time()
    try:
        path = snapshot_download(repo, revision=REVISIONS[repo], ignore_patterns=ignore)
        print(f"완료: {path}  ({time.time() - t0:.0f}초)")
    except GatedRepoError:
        print(f"[건너뜀] {repo} 는 아직 라이선스 승인 대기 중입니다.")
        failed.append(repo)

print("\n전부 완료." if not failed else f"\n{len(failed)}개 건너뜀: {failed}")
