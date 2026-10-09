# Automatic Git workflow

For implementation tasks, repository agents read `AGENTS.md` and automatically
finish with documentation, local checks, a focused commit, a pushed task branch,
and a pull request. You only need to describe the feature. Explanation and planning
requests do not publish code; an explicit instruction to keep a task local wins.

This is a repository workflow, not a timer watching every edit. The agent invokes
the completion command while working. Git hooks run on pushes, and GitHub runs
CI after publication. These instructions do not grant permissions to other
repositories or override an agent's sandbox/network controls.

## Branches

| Branch | Start from | PR destination |
| --- | --- | --- |
| `feature/<name>` | `origin/develop` | `develop` |
| `release/<version>` | `origin/develop` | `main`, then return changes to `develop` |
| `hotfix/<name>` | `origin/main` | `main`, then return changes to `develop` |

`main` contains released code and is the default branch. `develop` contains the
integration code. The previous `codex/training-workflow` branch is preserved.
Release tags such as `v0.2.0` are created only for an explicitly requested release.
Use merge commits for releases and return merges to preserve Gitflow ancestry;
do not enable a linear-history requirement on these integration branches.

## One-time local setup

GitHub CLI (`gh`), Git, make, and Python 3.11-3.13 are required. This Mac already
has GitHub CLI and the project's pinned uv runtime. In a new clone, install the
locked core dependencies and local push checks:

```sh
uv sync --locked --python 3.11
make setup-workflow
```

In the existing checkout use `.tools/bin/uv` instead of `uv` when it is not on PATH.
Do not replace a working vision environment with core-only dependencies unless
that is the environment you intend to use. The push-hook installer preserves an
existing custom hook configuration and reports any integration needed.
Git does not copy hook configuration into new clones; agents install it as part
of starting work. Worktrees of this repository share its hook configuration.

## Start and complete a feature

Use a clean checkout or isolated worktree. For example, from the repository root:

```sh
git fetch origin develop
git worktree add -b feature/image-import .worktrees/image-import origin/develop
cd .worktrees/image-import
```

Install core dependencies in that worktree. Implement the feature and prepare a
PR body from `.github/pull_request_template.md`. The agent then runs, for example:

```sh
.venv/bin/python scripts/finish_feature.py \
  --title "feat: add image import" \
  --body-file /private/tmp/image-import-pr.md \
  --files src/dmotion/cli.py tests/test_cli.py docs/TRAINING.md
```

Only select files belonging to that task. The command refuses directories and
unselected changes, verifies the branch is based on the current destination,
runs `git diff --check` and `make check`, stages the selected files, and commits
them. It pushes without force and creates or updates the branch's existing PR.
It adds `gitflow:auto`, opting in only this task's PR. Initial setup must create
that repository label; if absent, completion stops after publication and can be
rerun after repairing the label.
Rerunning after a partial failure preserves the commit and reuses an open PR.
`make finish-feature ARGS='...'` exposes the same command.

Completion does not request AI review or post review-trigger comments. Tests,
security scans, PR policy, and protected Gitflow promotion run automatically.
AI review is optional and must be explicitly requested by the user.

The PR records the tested commit and local check results. CI status
remain pending until GitHub reports them. PR descriptions must explain the change,
documentation updates, validation, and risks/limitations. The PR policy checks the
presence of substantive sections and Gitflow routing; it cannot judge whether
the prose fully explains the feature.

## Enforcement and optional review

- `.githooks/pre-push` blocks direct pushes/deletions to `main` and `develop`,
  refuses dirty worktrees or pushes of another branch's commit, rejects branch
  history rewrites, and runs `make check` before publishing code.
- GitHub's `test` check requires all six OS/Python core test runs, the locked
  dependency audit, clean package installation, and Python/Actions CodeQL scans.
  Local `make check` includes Ruff, Bandit, tests, timeouts, and a coverage floor.
  These checks require no camera, audio device, or model downloads. Reports and
  distribution artifacts are retained; see [AI reliability](AI_RELIABILITY.md).
- GitHub's `pr-policy` check validates branch routing and the PR description.
  Editing the description reruns this check. Its source is the trusted default
  branch, so changes to that policy require promotion to `main` to take effect.
  Verified Dependabot PRs use their own generated documentation.
- The protection template requires `test` and `pr-policy`, an up-to-date
  branch, resolved review
  conversations, and PR-based changes. Force pushes/deletions and administrator
  bypasses are disabled. The solo-maintainer setup uses zero mandatory approving
  reviews: GitHub does not allow the PR author to approve their own PR. Add one
  required approval when an independent reviewer is available.
  The intended configuration is stored in `.github/branch-protection.json`;
  storing that file does not apply it to GitHub. Verify the live requirements
  separately during rollout; the automation cannot enforce an omitted required check on another authorized merger.
  Required checks accept results from the verified GitHub Actions app.
- Codex has a separate account-level automatic-review setting. To prevent reviews
  from consuming tokens outside Actions, turn off automatic review for
  `PatBaruch/dMotion` in [Codex code review settings](https://chatgpt.com/codex/settings/code-review).
  Repository changes do not change that account setting. Manual reviews remain
  available when explicitly requested.

While active, the agent watches CI, reads review results, fixes actionable issues,
and reruns completion. A stopped chat is not a background repair service; GitHub
CI and the deployed Gitflow loop still run, but code fixes need an active
agent. Repository PR comments needed for the task are authorized; AI review triggers require
an explicit user request.

The current `AGENTS.md` authorizes feature, release, and hotfix publication and
gated merging without another reminder. Merge only when the required CI/security
and PR policy checks pass and existing blocking findings are resolved. Missing
AI reviews do not block merging. Tagging and
production deployment still require separate authorization.

## Trusted merge and promotion loop

`.github/workflows/gitflow-automation.yml` runs after CI, relevant trusted PR
events, protected-branch changes, manual dispatch, and every ten minutes. Its
privileged job checks out only `main` and runs `scripts/gitflow_automation.py`.
It never executes PR code or downloaded artifacts. PR strings are JSON data,
never shell commands. External Actions use pinned SHAs.

The workflow publishes only `pr-policy` for branches created by its token.
It checks `test` and `pr-policy` from the verified GitHub Actions app on the exact
head commit. It never posts `@codex review`, parses AI completion comments, or
publishes an `ai-review` check. Absent, stale, malformed, or quota-failed AI
completion messages are irrelevant to merge eligibility.

Optional reviews still carry real objections: unresolved conversations and
outstanding formal requests for changes block merging. A subsequent comment
cannot withdraw a request for changes; approval or dismissal can. Review threads
are read with cursor pagination and unavailable evidence fails closed. Removing
mandatory AI completion does not erase existing findings or resolve conversations.

Only `gitflow:auto` PRs from this repository are updated or merged. Unlabeled
PRs are left alone apart from reporting policy checks. A newer protected
base is merged into a managed task branch without rewriting history; that new
head needs fresh checks. The loop rereads current evidence immediately
before merging, supplies an atomic head-SHA guard, and relies on GitHub's branch
protections too. Conflicts, unavailable evidence, and racing commits block merging.

It creates `release/automation-<develop SHA>` into `main` for code differences.
Main commits missing from develop take priority: `hotfix/sync-<main SHA>` returns
them through a PR into `develop`. Captured immutable source/base commits preserve
ancestry, safe partial creation is resumable, and unexpected branch collisions
fail. One managed promotion/sync PR prevents duplicates. These PRs need the same
checks and resolved existing findings; no tag or deployment is created.

Token-created PRs can produce approval-required workflow runs. Such runs do not
count as executed checks or prevent the explicit dispatch. The loop explicitly
dispatches `Checks` on the managed branch with an expected SHA; it rejects a branch
that advanced before dispatch. Trusted policy checks are published on that head
too. Native `allow_auto_merge` stays disabled: the loop performs freshly gated
merges rather than placing a PR in a CI-only queue. AI review is not required. Background automation cannot implement code fixes while the agent
is stopped.

## Initial rollout and prerequisites

Default-branch workflows must reach `main` before background behavior is active.
Bootstrap through tested feature and release PRs. The active agent can inspect
or merge one PR with the same real-evidence gates:

```sh
.venv/bin/python scripts/gitflow_automation.py --repo PatBaruch/dMotion --pr 7
.venv/bin/python scripts/gitflow_automation.py --repo PatBaruch/dMotion --pr 7 --merge
```

The merge command cannot invent a passing check or bypass a protection. The protection template requires only `test` and `pr-policy`. During migration,
inspect live protections on both `main` and `develop`. If `ai-review` is still
required, remove only that context as explicitly requested by the maintainer;
preserve all other checks, strictness, and conversation-resolution requirements.
Old `ai-review` results remain historical records and are never rewritten as passes.

GitHub Actions must be permitted to create promotion PRs. GitHub bundles this as
**Allow GitHub Actions to create and approve pull requests**; enabling it also
grants approval capability, although this workflow never submits reviews and
does not require any AI reviewer. Default token permissions remain
read-only. Changing this broader setting needs maintainer approval. If disabled,
promotion creation is blocked and reported; no local token is copied into secrets.

Enable Dependabot security updates separately. Weekly Dependabot configuration
and scheduled scans become active on main. After rollout, dispatch **Gitflow automation** and **Checks** for immediate
verification and retain exact-head reports. A manual dispatch demonstrates
execution; the first actual scheduled run remains separate evidence. Repository
configuration is never a substitute for those actual results.

## Troubleshooting

- If authentication fails outside an agent sandbox, run `gh auth login --hostname
  github.com`. An agent may need network approval even with valid credentials.
- If the destination advanced, merge its latest version into the task branch,
  resolve conflicts, and rerun completion. Do not force-push.
- If publication fails after committing, fix the cause and rerun with the same
  PR body; clean committed work does not need to be staged again.
- If checks fail, fix the reported failure. Never bypass checks to publish.
- If a webcam/model check could not run, say so in Validation. Passing core tests
  does not establish recognition accuracy or hardware behavior.

## References

- [Gitflow branching model](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub branch protections](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Optional Codex GitHub reviews](https://learn.chatgpt.com/docs/third-party/github)
