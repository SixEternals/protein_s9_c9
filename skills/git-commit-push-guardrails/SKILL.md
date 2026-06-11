---
name: git-commit-push-guardrails
description: Mandatory scoped Git commit/push workflow for this repo. Use whenever an AI agent reviews, stages, commits, pushes, tags, or prepares commit instructions, especially with dirty worktrees, ignored results, checkpoints, prediction CSVs, or multi-agent handoff files.
---

# Git Commit / Push Guardrails

This skill is mandatory before any `git add`, `git commit`, `git push`, or tag operation in this repo.

## Absolute Bans

- Do not run `git add .`, `git add -A`, or broad wildcard staging.
- Do not commit or push without explicit user approval for the exact scope.
- Do not use `git push --force`, `git push -f`, `git reset --hard`, `git rebase`, or `git clean -f` unless the user explicitly asks for that exact command.
- Do not delete or overwrite `data/`, `reference/`, `results/`, checkpoints, prediction CSVs, or another agent's outputs.
- Do not stage, commit, or push anything under `reborn_doc/`; it is a local project knowledge base only.
- Do not force-add ignored large artifacts unless the user explicitly confirms the specific paths.
- Do not mix unrelated work, such as BL5 figure/docx files, BL6 configs, generated logs, or old scratch files, into the same commit.

## Preflight

Run and read all relevant output before staging:

```bash
git status --short --branch
git diff --stat
git diff
git ls-files --others --exclude-standard
git status --short --ignored
```

Also inspect the task-specific files with `rg`, `sed`, or `git diff -- <paths>`. If `results/experiments.csv` is involved, check whether it is tracked:

```bash
git ls-files results/experiments.csv
rg -n "<run-id-or-version>" results/experiments.csv
```

## Scope Classification

Before staging, produce a scope table:

- **In scope**: exact files required for the requested task.
- **Out of scope**: unrelated untracked or modified files that must stay unstaged.
- **Never upload**: every file and subdirectory under `reborn_doc/`, even when related to the task.
- **Ignored artifacts**: results, checkpoints, predictions, logs, and generated files.
- **Large files**: anything likely to be unsuitable for Git, especially `*.pt`, `*.ckpt`, `test_predictions.csv`, `gate_predictions.csv`, and 100MB+ files.

Default policy:

- Stage source code, configs, run scripts, reports, and tracked ledgers only when in scope.
- Treat all `reborn_doc/**` paths as out of scope for staging, commits, and pushes.
- Do not stage checkpoints or prediction CSVs.
- Do not force-add ignored result artifacts unless explicitly requested.
- Small summaries under `results/` are optional and require explicit user confirmation if ignored.

## Validation Before Staging

Run checks matching the touched files:

```bash
python scripts/audit_compliance.py
```

For Python scripts:

```bash
python -m py_compile path/to/script.py
```

For shell launchers:

```bash
bash -n path/to/script.sh
```

For reports, run `rg` checks for stale or forbidden wording relevant to the task. Do not commit known stale text just because it is "minor".

## Scoped Staging

Stage files by exact path only:

```bash
git add path/one path/two
```

Use `git add -f` only for an explicitly approved ignored file. `results/experiments.csv` may be tracked; if `git ls-files results/experiments.csv` returns it, use normal `git add results/experiments.csv`.

After each staging command, run:

```bash
git diff --cached --name-only
git diff --cached --stat
git diff --cached
```

Confirm the staged file list exactly matches the intended commit. If any out-of-scope file is staged, unstage only that file:

```bash
git restore --staged path/to/file
```

Do not use destructive reset commands.

## Commit Discipline

Prefer small scoped commits:

- configs and launchers
- analysis scripts
- docs and tracked ledgers
- optional small artifacts only if explicitly approved

Before each commit, re-run:

```bash
git diff --cached --name-only
```

Commit with a clear message:

```bash
git commit -m "type: concise scope"
```

After each commit:

```bash
git status --short --branch
git log --oneline -1
```

## Push Discipline

Before pushing:

```bash
git status --short --branch
git log --oneline --decorate -5
git branch --show-current
git rev-parse --abbrev-ref --symbolic-full-name @{u}
```

Push only the current intended branch:

```bash
git push origin <branch>
```

After push, report:

- commit hashes and messages
- branch pushed
- push result
- `git status --short --branch`
- out-of-scope files left uncommitted
- confirmation that no `reborn_doc/`, checkpoints, prediction CSVs, `data/`, or `reference/` files were committed

## Common Failure Patterns To Avoid

- Staging all untracked files and accidentally committing another agent's work.
- Force-adding ignored `results/` directories with checkpoints or huge prediction CSVs.
- Assuming every modified file belongs to the current task.
- Forgetting tracked ledgers such as `results/experiments.csv`.
- Uploading `reborn_doc/` because it looked like ordinary documentation.
- Committing stale report language after a later experiment changed the conclusion.
- Mixing generated figures/docx files into model-code commits.
- Pushing before verifying the staged diff.
- Claiming checks passed without running the repo's actual Python environment when needed.
- Leaving the branch ahead locally and saying the push is complete.
