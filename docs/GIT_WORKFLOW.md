# Automatic Git workflow

For implementation tasks, repository agents read `AGENTS.md` and automatically
finish with documentation, local checks, a focused commit, a pushed task branch,
and a pull request. You only need to describe the feature. Explanation and planning
requests do not publish code; an explicit instruction to keep a task local wins.

This is a repository workflow, not a timer watching every edit. The agent invokes
the completion command while working. Git hooks run on pushes, and GitHub runs
CI/review after publication. These instructions do not grant permissions to other
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

Completion requests native Codex review once for the current opted-in PR head,
using a comment with the full SHA. Existing trusted requests for that SHA are
deduplicated. An unavailable service or failed request is reported with the
already-published PR preserved; it never counts as review completion.

The PR records the tested commit and local check results. CI and review status
remain pending until GitHub reports them. PR descriptions must explain the change,
documentation updates, validation, and risks/limitations. The PR policy checks the
presence of substantive sections and Gitflow routing; it cannot judge whether
the prose fully explains the feature.

## Enforcement and review

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
- Protected branches require `test`, `pr-policy`, and `ai-review`, an up-to-date
  branch, resolved review
  conversations, and PR-based changes. Force pushes/deletions and administrator
  bypasses are disabled. The solo-maintainer setup uses zero mandatory approving
  reviews: GitHub does not allow the PR author to approve their own PR. Add one
  required approval when an independent reviewer is available.
  The exact remote configuration is stored in `.github/branch-protection.json`;
  required checks accept results from the verified GitHub Actions app.
- Codex PR review needs its separate account setting: connect `PatBaruch/dMotion`,
  enable repository code review, turn on automatic review for the desired PRs,
  and choose a trigger covering updates to the PR. Configure personal preferences
  too if the repository follows those preferences. Repository rules live in
  `AGENTS.md`. Account settings cannot be enabled by a Git push.

While active, the agent watches CI, reads review results, fixes actionable issues,
and reruns completion. A stopped chat is not a background repair service; GitHub
CI, reviews, and the deployed Gitflow loop still run, but code fixes need an active
agent. Repository PR comments and review triggers needed for the task are authorized.

The current `AGENTS.md` authorizes feature, release, and hotfix publication and
gated merging without another reminder. Merge only when the required CI/security
checks pass, AI review covers the current head commit, and blocking findings are
resolved. Missing, pending, failed, or stale reviews block merging. Tagging and
production deployment still require separate authorization.

## Trusted merge and promotion loop

`.github/workflows/gitflow-automation.yml` runs after CI, relevant trusted PR
events, protected-branch changes, manual dispatch, and every ten minutes. Its
privileged job checks out only `main` and runs `scripts/gitflow_automation.py`.
It never executes PR code or downloaded artifacts. PR strings are JSON data,
never shell commands. External Actions use pinned SHAs.

The workflow publishes `ai-review` on the exact head. It accepts a submitted
`APPROVED` or recognized `COMMENTED` review from the authenticated Codex connector
bot, matching its login, immutable user ID, type, and the current `commit_id`.
Legacy formal `COMMENTED` results must match the complete observed native heading,
reviewed-commit marker, and About Codex footer after whitespace normalization.
A heading alone cannot validate malformed JSON or additional review prose.
An empty submitted `APPROVED` review is also accepted through its formal state.
The requested machine-readable completion contract is defined in
[Codex review result schema v1](schemas/codex-review-result.schema.json).
The review request includes the exact JSON example generated by the same code
that validates the response. Only the authenticated independent Codex bot may
provide completion evidence; implementation agents must never fabricate it.

A completed clean result is a standalone message with this exact structure:

````text
<!-- dmotion-review-result:v1 -->
```json
{
  "schema_version": 1,
  "reviewed_commit": "0123456789abcdef0123456789abcdef01234567",
  "status": "completed",
  "conclusion": "clean",
  "blocking_findings": 0
}
```
````

Use the actual full 40-character reviewed commit SHA. All five fields are required;
unknown fields, duplicate JSON keys, wrong types, and unsupported versions fail.
Only `completed` + `clean` + zero blocking findings passes. Incomplete/failed
reviews use `incomplete`/`failed` status and `unavailable` conclusion; findings
are reported normally and must be resolved before clean completion. No prose may
precede or follow the JSON result, except the observed native About Codex footer.
The parser validates footer content while allowing whitespace variations.
Every authenticated bot comment except the native activity-summary table enters
event ordering. A newer malformed, markerless, unknown, or incomplete result
revokes older clean evidence rather than being ignored. Surrounding whitespace
is accepted for an otherwise valid schema message.

Requests carry a schema-version marker as well as the commit marker, so an older
request without the contract does not suppress the first schema v1 request.
A reaction, activity table, or completion request itself cannot supply evidence.
Native bot formatting is outside this repository's control: if the reviewer
ignores the requested contract, the gate remains blocked with a format error.
The JSON path removes dependence on decorative closing sentences when the
reviewer follows the contract; it does not assume that prompting guarantees it.

Existing native comments remain a strict compatibility path. They require the
exact first-line clean sentence `Codex Review: Didn't find any major issues.`,
a known observed decorative closing (or none), one `Reviewed commit` marker,
and only an optional observed footer. Unknown prose anywhere fails closed.
The recognized closings are `Swish!`, `Bravo.`, `Hooray!`, `:tada:`, `Keep it up!`, `You're on a roll.`,
`Chef's kiss.`, `Another round soon, please!`, `Already looking forward to the next diff.`, and
`What shall we delve into next?`, plus `Can't wait for the next one!`. Failure messages
and P0/P1 findings anywhere block both formats. PR #21's authenticated `Hooray!`
completion and PR #24's `Can't wait for the next one!` completion are retained as
regression fixtures, including their native footers and reviewed commits. Native
wording can change independently of this repository: a new unrecognized closing
requires inspecting the authentic response and adding a regression, while the
structured contract avoids decorative-text parsing when the integration honors it.
New integrations should use the JSON contract.

A shortened SHA must
uniquely identify the current head among the PR's commits; at least ten hexadecimal
characters are required. The newest bot result must be complete and successful.
Commit history uses GraphQL cursor pagination rather than the REST endpoint's
250-commit limit. Incomplete, changing or excessive history fails the affected
PR's gate; an evidence failure on one PR cannot prevent other eligible PRs from
being inspected and processed. Such failures are recorded in the workflow output.
Comments cannot override pending/dismissed/blocking current-head formal reviews.
Formal reviews include GraphQL update/edit timestamps, matched by REST node identity
and checked for changes between the two API reads; missing or inconsistent evidence
blocks the affected PR. Formal reviews for every commit and completion comments
are compared together by submission/edit time: a late review or an edited
older-commit result revokes an earlier clean
current-head result. Every result in the newest one-second timestamp must pass;
creation IDs cannot resolve ties across reviews, comments or edits. Review and
comment events refresh the gate; the scheduled loop also rechecks thread resolution.
Missing, pending, dismissed, stale, unrecognized, quota-failed, or P0/P1-blocking
results fail. Unresolved threads and outstanding requests for changes also block.
A toggle, reaction, or connection-error comment cannot pass.

Automatic clean reviews may leave only a thumbs-up on the PR. Since that reaction
does not identify a commit, the trusted loop requests an explicit native review
once per managed head when completion is missing or stale. The request includes
the full SHA and a deduplication marker; only requests from the verified Actions
bot or repository collaborators suppress duplicates. Draft, closed, fork and
unmanaged PRs are not requested. Known blocking findings need an active agent
to resolve them; repeated pulses do not keep requesting the same commit. Native
account/allowance errors remain blocking and are never treated as approval.

Only `gitflow:auto` PRs from this repository are updated or merged. Unlabeled
PRs are left alone apart from reporting policy/review checks. A newer protected
base is merged into a managed task branch without rewriting history; that new
head needs fresh checks and review. The loop rereads current evidence immediately
before merging, supplies an atomic head-SHA guard, and relies on GitHub's branch
protections too. Conflicts, unavailable evidence, and racing commits block merging.

It creates `release/automation-<develop SHA>` into `main` for code differences.
Main commits missing from develop take priority: `hotfix/sync-<main SHA>` returns
them through a PR into `develop`. Captured immutable source/base commits preserve
ancestry, safe partial creation is resumable, and unexpected branch collisions
fail. One managed promotion/sync PR prevents duplicates. These PRs need the same
checks and current-head review; no tag or deployment is created.

Token-created PRs can produce approval-required workflow runs. Such runs do not
count as executed checks or prevent the explicit dispatch. The loop explicitly
dispatches `Checks` on the managed branch with an expected SHA; it rejects a branch
that advanced before dispatch. Trusted policy checks are published on that head
too. Native `allow_auto_merge` stays disabled: the loop performs freshly gated
merges rather than placing a PR in a CI-only queue. Actions approvals never count
as AI review. Background automation cannot implement code fixes while the agent
is stopped.

## Initial rollout and prerequisites

Default-branch workflows must reach `main` before background behavior is active.
Bootstrap through reviewed feature and release PRs. The active agent can inspect
or merge one PR with the same real-evidence gates:

```sh
.venv/bin/python scripts/gitflow_automation.py --repo PatBaruch/dMotion --pr 7
.venv/bin/python scripts/gitflow_automation.py --repo PatBaruch/dMotion --pr 7 --merge
```

The merge command cannot invent a passing check or bypass a protection. Once
main's trusted workflow publishes the gate, add `ai-review` to both protected
branches using `.github/branch-protection.json`, preserving stricter existing
settings. Never remove an existing required check to complete rollout.

GitHub Actions must be permitted to create promotion PRs. GitHub bundles this as
**Allow GitHub Actions to create and approve pull requests**; enabling it also
grants approval capability, although this workflow never submits reviews and
accepts only the independent Codex connector. Default token permissions remain
read-only. Changing this broader setting needs maintainer approval. If disabled,
promotion creation is blocked and reported; no local token is copied into secrets.

Enable Dependabot security updates separately. Weekly Dependabot configuration
and scheduled scans become active on main. Codex needs both the saved automatic
review setting covering updates and an account GitHub connection authorized for
dMotion. If `@codex review` asks to connect an account, repair the connection at
<https://chatgpt.com/codex/cloud/settings/connectors>, then retry review.

After rollout, dispatch **Gitflow automation** and **Checks** for immediate
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
- [Codex GitHub reviews](https://learn.chatgpt.com/docs/third-party/github)
