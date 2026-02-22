# Solo Git Workflow

This repo is configured for a strict one-person workflow with local quality gates.

## One-time setup

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_solo_git_workflow.ps1
```

This sets:

- `core.hooksPath = .githooks`
- `commit.template = .gitmessage.txt`
- `fetch.prune = true`
- `rebase.autoStash = true`
- `commit.verbose = true`

## Start New Task

Preferred:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-task.ps1 -Task "sua loi E suffix"
```

Alias (after setup):

```bash
git newtask "sua loi E suffix"
git newtask-dry "sua loi E suffix"
```

Behavior:
- Uses `develop` as base by default.
- Pulls with `--ff-only` by default.
- Generates branch name as `type/slug`.
- Uses `feat` when type cannot be inferred.
- Adds `-2`, `-3`, ... if branch name already exists.
- Blocks when working tree is dirty unless `-AllowDirty`.

## Commit format

Use:

```text
<type>(<scope>): <summary>
```

Allowed types:

- `feat`
- `fix`
- `refactor`
- `test`
- `docs`
- `chore`

Example:

```text
fix(schedule): enforce strict E-suffix course matching
```

## Hook behavior

### pre-commit (strict)

- Blocks direct commits to `develop/main/master` (unless override).
- Blocks common debug/temp files:
  - `debug_output*.txt`
  - `test/debug_*.py`
  - `test/compare_*.py`
  - `test/demo_*.py`
- Runs `git diff --cached --check`
- Runs `py_compile` on staged `.py` files
- Runs targeted tests based on changed files:
  - `src/app.py` -> `tests/integration/test_app_ask.py`
  - `src/mcp_server/server.py` -> `tests/integration/test_mcp_server_tools.py`
  - `src/utils.py` -> `tests/unit/test_open_group_credit_rules.py`

Skip once (if absolutely needed):

```bash
SKIP_PRECOMMIT_CHECKS=1 git commit -m "..."
```

Allow direct commit to protected branch once:

```bash
ALLOW_DIRECT_BRANCH_COMMIT=1 git commit -m "..."
```

### commit-msg

- Enforces conventional commit title format.
- Requires title length <= 90 chars.
- Rejects trailing dot in title.
- Rejects generic titles like `fix: fix`.

### pre-push (strict)

- Blocks direct push to `develop/main/master` (unless override).
- Runs core regression tests:
  - `tests/integration/test_planner_parse_resilience.py`
  - `tests/integration/test_mcp_server_tools.py`
  - `tests/integration/test_app_ask.py`
  - `tests/unit/test_mcp_client_tools.py`
  - `tests/unit/test_open_group_credit_rules.py`
- Runs frontend build check if npm exists.

Skip once:

```bash
SKIP_PUSH_CHECKS=1 git push
```

Allow direct push to protected branch once:

```bash
ALLOW_PUSH_PROTECTED_BRANCH=1 git push
```

Skip only frontend build:

```bash
SKIP_FRONTEND_CHECKS=1 git push
```

## Daily flow (recommended)

1. Start task branch:

```bash
git newtask "sua loi E suffix"
```

2. Work in small chunks, commit with strict title.
3. Push branch:
```bash
git push -u origin <created-branch>
```
4. Merge back to `develop` with one clean squash commit.
