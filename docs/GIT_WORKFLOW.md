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
Rerunning after a partial failure preserves the commit and reuses an open PR.
`make finish-feature ARGS='...'` exposes the same command.

The PR records the tested commit and local check results. CI and review status
remain pending until GitHub reports them. PR descriptions must explain the change,
documentation updates, validation, and risks/limitations. The PR policy checks the
presence of substantive sections and Gitflow routing; it cannot judge whether
the prose fully explains the feature.

## Enforcement and review

- `.githooks/pre-push` blocks direct pushes/deletions to `main` and `develop`,
  refuses dirty worktrees or pushes of another branch's commit, rejects branch
  history rewrites, and runs `make check` before publishing code.
- GitHub's `test` check runs the same Ruff lint/format and pytest commands on
  pushes and PRs. It requires no camera, audio device, or model downloads.
- GitHub's `pr-policy` check validates branch routing and the PR description.
  Editing the description reruns this check.
- Protected branches require both checks, an up-to-date branch, resolved review
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
CI and configured reviews still run, but subsequent code fixes need an active
agent. No recurring automation is installed by this workflow.

Merging is manual unless the user authorizes it. The completion command does not
enable auto-merge. AI review comments alone are not a required approval or a
guaranteed blocking check. Before merging, inspect the review and the latest CI
results. Optional GitHub auto-merge still needs explicit review gates if you want
the review outcome to block a merge.

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
