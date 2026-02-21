# General Q&A — Project-Level Lessons

Cross-session questions about workflow, tooling, and common practices that aren't tied to a specific session's technical work.

## Q1: Fork repos vs vendoring code — why are commits spread across 3 repos?

**Confusion**: Our project has 3 repos: the main repo (`speechQwen2VL`), a transformers fork (`ZhuoyuanJiang/transformers`), and a Qwen2-VL fork (`ZhuoyuanJiang/Qwen3-VL`). Commit history is scattered — someone would need to check 3 repos (and specific branches) to see all our changes. Is this normal? Should we just copy the modified files into the main repo instead?

**Our setup**:
```
speechQwen2VL/                          ← main repo (GitHub: ZhuoyuanJiang/speechQwen2VL)
├── Documentation/                       commits: plans, progress docs, lessons
├── notebooks/                           commits: Jupyter notebooks
├── scripts/                             commits: setup scripts
└── forks/                               ← in .gitignore, NOT tracked by main repo
    ├── transformers/                    ← separate git repo (fork, branch: speech-qwen2vl)
    │   └── .../modeling_qwen2_vl.py     commits: model architecture changes
    └── Qwen2-VL/                        ← separate git repo (fork, branch: speech-qwen2vl)
        └── .../vision_process.py        commits: fetch_audio, process_vision_info
```

### Two approaches people use

**Approach 1: Fork mode (what we do)**

Keep modified library code in forked repos, install with `pip install -e` (editable mode).

```bash
# In setup_forks.sh — editable install means Python uses the fork's code directly
pip install -e forks/transformers
pip install -e forks/Qwen2-VL/qwen-vl-utils
```

| Pros | Cons |
|------|------|
| `pip install -e` — edit code, changes take effect immediately | Commit history spread across 3 repos |
| Preserves git relationship with upstream (can rebase/merge updates) | Fork branches not easily discoverable |
| Standard practice in HuggingFace ecosystem for model modifications | Reviewer must visit fork to see actual code diffs |

**Approach 2: Vendor mode (copy files into main repo)**

Copy the modified files directly into the main repo (e.g., `src/qwen2_vl/modeling_qwen2_vl.py`).

| Pros | Cons |
|------|------|
| All commits in one repo — easy to review | Lose git relationship with upstream transformers |
| `git log` shows complete project history | Must manually manage import paths |
| Single clone to see everything | Can't easily pull upstream bug fixes |

### What most research projects do

Most HuggingFace-based research projects use **fork mode** (Approach 1), because:
1. Editable install is essential for rapid iteration on model code
2. The upstream relationship matters — transformers updates frequently with bug fixes
3. Fork branches can be pinned to exact commits for reproducibility

The downside (scattered commits) is mitigated by **documentation in the main repo** — detailed progress docs that record every fork change with code snippets. This is exactly what our `Documentation/SessionN_Progress_*.md` files do.

### Our mitigation strategy

The main repo's `Documentation/` folder serves as the **unified view** of all work:

```
Documentation/
├── Session1_Progress_Documentation.md   ← records what happened (no fork changes in S1)
├── Session2_Progress_20260220.md        ← records every change to both forks + notebook
├── Session3_Progress_20260221.md        ← records every change to transformers fork + notebook
└── Lessons/
    ├── session1_QA.md                   ← technical lessons per session
    ├── session2_QA.md
    ├── session3_QA.md
    └── general_QA.md                    ← this file (cross-session lessons)
```

Someone reading the main repo can understand **all** changes without visiting the forks. The progress docs include full code snippets, design rationale, and before/after comparisons. The forks are there for `pip install -e` and upstream tracking, not as the primary record of work.

**Lesson**: For research projects modifying HuggingFace libraries, fork mode is standard. Compensate for scattered commits by keeping thorough documentation in the main repo that records all fork changes. The main repo's commit history documents *what was done*; the fork repos contain *the actual code*.
