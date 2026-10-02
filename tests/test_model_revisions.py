"""내려받는 가중치와 불러오는 가중치의 판이 결과를 낸 판으로 고정돼 있는지 봄.

판을 고정하지 않으면 다시 받을 때 그날의 최신판이 옴. 세 곳(report 재현 방법의 커밋 표, `src/models.py` 의
`REVISIONS`, 내려받기 스크립트)이 어긋나지 않게 함.

커밋 해시로 받으면 `refs/main` 이 생기지 않아, 로더가 판 없이 main 을 찾으면 새 환경에서 로딩이 멈춤.
그래서 모든 `from_pretrained` 호출이 판을 넘기는지, refs 가 없는 캐시에서 실제로 불러지는지도 봄.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import MODELS, REVISIONS, local_kwargs  # noqa: E402


def test_every_model_has_a_pinned_commit():
    repos = {spec.hf_id for spec in MODELS.values()}
    assert set(REVISIONS) == repos
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in REVISIONS.values())


def test_report_commit_table_matches_revisions():
    report = (ROOT / "docs" / "report.md").read_text(encoding="utf-8")
    table = dict(re.findall(r"^\| ([\w.\-]+/[\w.\-]+) \| ([0-9a-f]{40}) \|$", report, flags=re.M))
    assert table == REVISIONS


def test_download_script_passes_revision():
    src = (ROOT / "scripts" / "download_llm_models.py").read_text(encoding="utf-8")
    calls = re.findall(r"snapshot_download\(([^)]*)\)", src)
    assert calls and all("revision=REVISIONS[repo]" in c for c in calls)


def test_every_loader_call_passes_the_pinned_revision():
    """src 와 scripts 의 from_pretrained 호출이 모두 local_kwargs(판 포함)로 부름."""
    calls = []
    for p in [*(ROOT / "src").glob("*.py"), *(ROOT / "scripts").glob("*.py")]:
        src = p.read_text(encoding="utf-8")
        calls += [(p.name, c) for c in re.findall(r"\.from_pretrained\(([^)]*)\)", src)]
    assert calls
    assert [c for c in calls if "local_kwargs(" not in c[1] and "**local" not in c[1]] == []
    assert local_kwargs("google/gemma-3-4b-it") == {"local_files_only": True,
                                                  "revision": REVISIONS["google/gemma-3-4b-it"]}


def _hub() -> Path:
    home = os.environ.get("HF_HOME")
    return Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"


@pytest.mark.skipif(not (_hub() / "models--google--gemma-3-4b-it" / "snapshots").exists(),
                    reason="HF 캐시에 Gemma 3 4B 가 없다(서버에서만 돈다)")
def test_loads_from_a_cache_without_refs_main(tmp_path):
    """커밋 해시로 받은 캐시처럼 refs/main 이 없는 캐시에서 판을 주면 불러짐."""
    pytest.importorskip("transformers")
    repo = "google/gemma-3-4b-it"
    sha = REVISIONS[repo]
    real = _hub() / "models--google--gemma-3-4b-it" / "snapshots" / sha
    snap = tmp_path / "hub" / "models--google--gemma-3-4b-it" / "snapshots" / sha
    # 가중치는 빼고 복사함. 심볼릭 링크는 윈도에서 권한이 있어야 만들어짐.
    shutil.copytree(real, snap, ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pth"))
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from models import local_kwargs; "
            "from transformers import AutoTokenizer; "
            f"AutoTokenizer.from_pretrained({repo!r}, **local_kwargs({repo!r})); print('ok')")
    env = dict(os.environ, HF_HOME=str(tmp_path), HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, "-c", code, str(ROOT / "src")], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.stdout.strip().endswith("ok"), r.stderr[-800:]
