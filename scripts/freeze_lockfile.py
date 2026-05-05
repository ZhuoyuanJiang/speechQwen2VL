#!/usr/bin/env python3
"""
Regenerate requirements.lock.txt from the currently activated env.

What it does:
    pip freeze, filter out editable git installs (handled by setup_forks.sh)
    and one-time dev tooling, write a pinned lockfile with a header.

When to run:
    After pip install / pip upgrade in the speech_qwen2vl env. The lockfile
    is the snapshot of "what's actually in my working env right now".

Usage:
    conda activate speech_qwen2vl
    python scripts/freeze_lockfile.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Tools that get pulled in by reproducibility tooling (pip-tools, conda-lock,
# uv) — not project deps. If you ever installed any of these in the project
# env to regenerate locks, exclude them so they don't pollute the lockfile.
DEV_TOOLS = {
    "pip-tools", "pip_tools", "conda-lock", "conda_lock", "uv",
    "click-default-group", "ensureconda", "pyproject-hooks", "pyproject_hooks",
    "build", "installer", "trove-classifiers", "trove_classifiers",
    "ruamel.yaml", "ruamel.yaml.clib", "appdirs", "boltons",
    "cachecontrol", "crashtest", "distlib", "dulwich", "virtualenv",
    "secretstorage", "jeepney", "keyring", "jaraco.context", "jaraco.functools",
    "jaraco.classes", "backports.tarfile", "more-itertools",
    "conda-package-streaming", "conda_package_streaming",
    "requests-toolbelt", "pkginfo", "semver", "zstandard", "zipp",
    "cryptography", "importlib-metadata", "importlib_metadata",
}


def main():
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=True,
    ).stdout

    lines = []
    skipped_editable = 0
    skipped_dev = 0
    for raw in out.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("-e ") or "@ git+" in s or "@ file://" in s:
            skipped_editable += 1
            continue
        name = s.split("==")[0].split("@")[0].strip().lower()
        if name in DEV_TOOLS:
            skipped_dev += 1
            continue
        lines.append(s)

    header = f"""\
# =============================================================================
# Speech-Qwen2VL — pip lockfile (snapshot of working env)
# =============================================================================
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Generator: scripts/freeze_lockfile.py (pip freeze, filtered)
#
# WHAT THIS FILE IS
#   Exact versions of every pip dependency in the working speech_qwen2vl env
#   at the moment of generation. Captures transitive dependencies that
#   requirements.txt does not list explicitly.
#
# WHAT IT IS NOT
#   - Hash-protected: pip-compile --generate-hashes and conda-lock both fail
#     on this stack because flash-attn must be built from source (no wheels).
#   - Self-sufficient: the forked transformers + qwen-vl-utils packages are
#     git editable installs, NOT in this file. They are installed by
#     scripts/setup_forks.sh, which pins specific commit hashes.
#
# HOW TO USE
#   For exact reproduction:
#     conda env create -f environment.yml
#     conda activate speech_qwen2vl
#     pip install -r requirements.lock.txt --extra-index-url \\
#         https://download.pytorch.org/whl/cu121
#     bash scripts/setup_forks.sh
#
#   For "give me a working env without caring about every transitive version":
#     Use requirements.txt instead.
#
# WHEN TO REGENERATE
#   After any pip install / pip upgrade in the env, re-run:
#     python scripts/freeze_lockfile.py
# =============================================================================

"""

    out_path = Path("requirements.lock.txt")
    with out_path.open("w") as f:
        f.write(header)
        f.write("\n".join(sorted(lines, key=str.lower)))
        f.write("\n")

    print(f"Wrote {out_path} ({len(lines)} packages)")
    print(f"Skipped {skipped_editable} editable git installs "
          f"(handled by scripts/setup_forks.sh)")
    print(f"Skipped {skipped_dev} reproducibility dev tools")


if __name__ == "__main__":
    main()
