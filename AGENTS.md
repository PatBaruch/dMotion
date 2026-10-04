# dMotion agent instructions

Always use Context7 when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

Use codebase-memory-mcp first for architecture, symbol lookup, call tracing, and change-impact analysis. Ensure this repository is indexed, verify relevant source before editing, and use text search when graph coverage is missing.

## Automatic feature workflow

These instructions apply to implementation tasks without a separate reminder.
Explanation-only, planning-only, and review-only requests do not publish changes.
An explicit user instruction to keep work local, pause, or skip publication wins.

- Start from current `origin/develop` on `feature/<name>`. Use an isolated worktree
  when this checkout contains other work. Never sweep existing changes into a task.
- Run `make setup-workflow` in a new clone. If core dependencies are missing,
  install from `uv.lock` using the core-only command in `docs/GIT_WORKFLOW.md`.
- Implement the complete requested behavior. Add meaningful tests for new behavior
  and fixes, and update relevant usage/development documentation when needed.
- Write a PR body with `## Summary`, `## Documentation`, `## Validation`, and
  `## Risks and limitations`. Record actual results; explain skipped manual checks.
- Before the final response, automatically run `scripts/finish_feature.py` with
  a conventional title, that body file, and an explicit list of task files.
  The script checks, commits, pushes, and creates/updates a PR into `develop`.
  Feature-branch pushes and PR creation/updates are authorized for these tasks;
  do not ask for a second confirmation solely to perform them.
- Watch the PR's CI while the task is active. Inspect review results. Fix actionable
  findings in scope, rerun completion, and monitor the updated commit. If review
  access, credentials, or an external result is unavailable, report the specific
  blocker and PR URL. Never claim a pending review has passed.
- Attach created or worked-on PRs to the chat with the Codex artifact tool.
- Never force-push, bypass push hooks or branch protections, or push directly to
  `main` or `develop`. Keep datasets, weights, outputs, credentials, and unrelated
  work out of commits. Check before staging; never use `git add .` or `git add -A`.
- Feature publication stops at a tested, documented PR. Merging and tagging a
  release need the user's instruction. For an authorized release/hotfix, use
  `release/<version>`/`hotfix/<name>` into `main`, then return the changes to
  `develop` through a separate PR. Do not leave the two branches inconsistent.

See `docs/GIT_WORKFLOW.md` for commands and the difference between automatic
completion, local hooks, GitHub checks, and the separate Codex review setting.

## Code Review Rules

- Keep AI label suggestions distinct from reviewed dataset truth. Do not silently
  treat unreviewed images as negatives or promote suggestions to training labels.
- Preserve recording groups across train/validation/test splits. Flag changes
  that mix sessions or claim model accuracy from core unit tests alone.
- Core checks must work without webcam/audio hardware, model downloads, or the
  optional vision dependencies. Record real-camera and model checks separately.
