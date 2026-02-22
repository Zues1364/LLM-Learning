# README Workflow (Solo, Strict)

This file explains the practical flow you should follow every day to keep history clean and stable.

## 1) One-time setup

Run once per clone:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_solo_git_workflow.ps1
```

This enables:

- Local hooks from `.githooks/`
- Commit template from `.gitmessage.txt`
- Safer git defaults (`fetch.prune`, `rebase.autoStash`)

## 2) Start New Task

Use the standardized task-branch script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-task.ps1 -Task "sua loi E suffix"
```

Fast alias (after setup):

```bash
git newtask "sua loi E suffix"
```

Branch naming policy:

- Base branch is `develop` (source branch), not a prefix in branch name.
- Working branches must be `type/slug` (for example `feat/new-advisor-flow`).
- Never create plain prefix branches like `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
- Plain prefix branches block namespaced branches (`feat/...`) in Git.

Useful variants:

1. Normal run
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-task.ps1 -Task "toi uu luong advisor"
```

2. Dry-run (no mutation)
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-task.ps1 -Task "toi uu luong advisor" -DryRun
```

3. Allow dirty working tree (only if needed)
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-task.ps1 -Task "toi uu luong advisor" -AllowDirty
```

## 3) Daily flow (normal change)

1. Sync base branch:

```bash
git checkout develop
git pull --ff-only
```

2. Create a focused branch:

```bash
git checkout -b fix/<topic>
```

3. Work + commit in small slices:

```bash
git add <files>
git commit
```

4. Push branch:

```bash
git push -u origin fix/<topic>
```

5. Merge back to `develop` with a clean single commit (squash).

## 4) What hooks enforce

## pre-commit

- Blocks direct commit to `develop/main/master`.
- Blocks debug/temp files:
  - `debug_output*.txt`
  - `test/debug_*.py`
  - `test/compare_*.py`
  - `test/demo_*.py`
- Runs whitespace checks.
- Compiles staged Python files.
- Runs targeted tests for changed critical files.
- Branch-block message suggests using `scripts/start-task.ps1` or `git newtask`.

## commit-msg

- Enforces title format:

```text
<type>(<scope>): <summary>
```

- Allowed types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Max title length: 90
- Rejects generic titles (`fix: fix`, etc.).

## pre-push

- Blocks direct push to `develop/main/master`.
- Runs strict regression suite:
  - `tests/integration/test_planner_parse_resilience.py`
  - `tests/integration/test_mcp_server_tools.py`
  - `tests/integration/test_app_ask.py`
  - `tests/unit/test_mcp_client_tools.py`
  - `tests/unit/test_open_group_credit_rules.py`
- Runs frontend build check when available.

## 5) Recommended commit style

Examples:

- `fix(schedule): enforce strict E-suffix course matching`
- `fix(app): add planner parse fallback and session single-flight lock`
- `test(mcp): add regression for schedule and open-group credit rules`

## 6) If hook blocks your action

Read the hook error first, then fix root cause.

Only bypass as emergency:

- Skip pre-commit:

```bash
SKIP_PRECOMMIT_CHECKS=1 git commit -m "..."
```

- Skip pre-push:

```bash
SKIP_PUSH_CHECKS=1 git push
```

- Allow direct commit to protected branch once:

```bash
ALLOW_DIRECT_BRANCH_COMMIT=1 git commit -m "..."
```

- Allow direct push to protected branch once:

```bash
ALLOW_PUSH_PROTECTED_BRANCH=1 git push
```

## 7) Rule of thumb

- One branch = one concern.
- One commit = one coherent unit.
- Never mix debug scripts with production changes.
- Keep regression tests with bug fixes.
