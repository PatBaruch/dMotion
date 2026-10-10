# dMotion agent instructions

Use Context7 when external library/API behavior is uncertain, when
introducing or upgrading dependencies, or when setup requires current
documentation. Match the project's pinned dependency versions.

Skip Context7 for routine edits, ordinary Git operations, and questions
answerable from repository files. Reuse documentation already retrieved
for the same version and topic during the task. Keep queries focused
on unresolved questions.

Use codebase-memory-mcp first for architecture, symbol lookup, call tracing, and change-impact analysis. Ensure this repository is indexed, verify relevant source before editing, and use text search when graph coverage is missing.

## Automatic feature workflow

These instructions apply to implementation tasks without a separate reminder.
Explanation-only, planning-only, and review-only requests do not publish changes.
An explicit user instruction to keep work local, pause, or skip publication wins.

- Create a fresh `feature/<task-name>` branch from current `origin/develop` for
  every new implementation task, including a distinct task in the same chat.
  Never reuse a completed, merged, or unrelated task branch. Follow-up fixes for
  the same open task stay on its branch and PR. Use an isolated worktree when
  this checkout contains other work. Never sweep existing changes into a task.
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
  When Firstmate explicitly assigns a No-mistakes task and the trusted `main`
  configuration is active, use that pipeline instead of publishing with this
  script. Run `no-mistakes axi run --base-branch develop --intent "<user goal>"`,
  handle its decisions, and synchronize with `no-mistakes axi sync` before
  making follow-up commits. Keep the repository's PR sections and required checks.
- Watch the PR's CI while the task is active. Inspect any existing review findings. Fix actionable
  findings in scope, rerun completion, and monitor the updated commit. If CI
  access, credentials, or an external result is unavailable, report the specific
  blocker and PR URL. Never claim a pending review has passed.
- Attach created or worked-on PRs to the chat with the Codex artifact tool.
- Post repository PR comments when needed for the authorized development task.
  Do not request AI reviews unless the user explicitly asks for a particular review.
- Ordinary pushes must never rewrite history or bypass hooks. No-mistakes alone
  may rewrite its exclusively owned `feature/*` task branch with an explicit
  `--force-with-lease=<ref>:<expected-sha>` after validation. Never share that
  branch with another writer. Its internal local-gate trigger may skip the caller
  hook; delivery pushes must retain hooks. Never bypass branch protections or
  push directly to `main` or `develop`. Keep datasets, weights, outputs, credentials, and unrelated
  work out of commits. Check before staging; never use `git add .` or `git add -A`.
- Automatically create, push, and update feature, release, and hotfix PRs
  without asking again. Merge automatically only when required CI and
  security and PR policy checks pass, and existing blocking findings are resolved.
  AI review is optional and is not a merge requirement.
  Use PRs into protected develop/main branches and synchronize release
  changes back into develop. The narrow No-mistakes exception above never applies
  to protected integration branches or ordinary manual/agent pushes.
  Tagging and production deployment require separate authorization.
  See `docs/GIT_WORKFLOW.md` for commands and the difference between automatic
  completion, local hooks, GitHub checks, and optional manual code review.

## Work tracking

- Use a descriptive task branch name and conventional commit/PR titles so the
  branch, commits, and PR describe the same unit of work.
- Keep the PR description current: explain what changed, why, relevant files,
  actual validation results, and remaining limitations. Link related issues or
  PRs when they exist; do not invent tracking IDs.
- Add a concise PR progress comment when a meaningful implementation or fix is
  published, a blocker changes, or the task completes. Include the affected
  commit SHA, what was done, validation evidence, and anything still pending.
  Update the PR body to describe the final result and keep comments as its history.
- Report the branch and PR link to the user. After completion, record the merge
  result and any pending promotion separately; never call pending CI successful.
  Keep source comments focused on non-obvious reasoning or constraints.

## Code Review Rules

- Keep AI label suggestions distinct from reviewed dataset truth. Do not silently
  treat unreviewed images as negatives or promote suggestions to training labels.
- Preserve recording groups across train/validation/test splits. Flag changes
  that mix sessions or claim model accuracy from core unit tests alone.
- Core checks must work without webcam/audio hardware, model downloads, or the
  optional vision dependencies. Record real-camera and model checks separately.
- Treat AI review findings as evidence to inspect, not proof that code is safe.
  Never report an absent/pending review as passed. Do not weaken coverage,
  security gates, or PR policy to make a change pass; document narrowly justified
  exceptions and inspect changes to CI configuration explicitly.
- Keep model IDs and immutable model revisions together. Changing the selected
  model requires the user's choice; reproducibility pins for that same model do
  not authorize a different architecture or model.

See `docs/AI_RELIABILITY.md` for gate thresholds, retained evidence, and the
separate requirements for model evaluation, hardware acceptance, and delivery.
