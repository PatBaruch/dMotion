# Evidence for AI-assisted reliability

The engineering claim is that AI can implement changes inside a repeatable,
checked development process. A green pipeline demonstrates that its defined
checks passed for a particular commit. It does not establish that the application
is bug-free, that the detector is accurate, or that an AI reviewer found every bug.

## Automatic development loop

1. Describe a feature to an agent working in this repository. `AGENTS.md` tells it
   to start a feature branch, preserve unrelated changes, implement the behavior,
   add relevant tests, and document the result.
2. The completion script and push hook run local lint, formatting, static security
   checks, and tests. Failed local checks prevent the normal publishing path.
3. GitHub independently runs the matrix, dependency audit, package installation,
   and CodeQL scans. The required `test` check passes only if every job succeeds;
   failed, cancelled, or skipped jobs cannot make the aggregate green.
4. Trusted Gitflow automation rechecks exact-head CI and PR policy, blocks existing
   unresolved findings and requests for changes, and merges opted-in PRs through
   protections. Release and return-sync PRs maintain `main` and `develop`.
5. AI review is optional and is requested only when the user explicitly asks.
   Automatic review requests and the `ai-review` merge gate have been removed to
   avoid repeated token consumption. Existing review findings still need resolution.

## What is checked

| Check | Scope and failure rule | Evidence |
| --- | --- | --- |
| Ruff | Python lint and formatting throughout the repository | Job log |
| pytest | Core behavior on Linux and macOS, Python 3.11, 3.12, and 3.13; each test has a 60-second timeout | JUnit XML |
| Coverage | Lines and branches in application code and Python workflow scripts; minimum 65% | Coverage XML and JSON with uncovered paths |
| Bandit | Python source and workflow scripts; medium/high findings fail | JSON report |
| Dependency audit | Every registry name/version in `uv.lock`, including optional vision and alternate platform versions; any known vulnerability or collection error fails | Audit JSON and explicit scope inventory |
| CodeQL | Python and GitHub Actions, extended security queries; scores at least 7.0 or error-level findings fail | SARIF and GitHub Security alerts |
| Distribution | Locked build tools build a wheel and source archive; wheel installs into a fresh environment with hashed, locked core dependencies; CLI help/version work outside the source tree | Distribution files and SHA-256 checksums |
| PR policy | Gitflow destination, useful title, and four completed documentation sections | `pr-policy` result |

The coverage floor reflects a measured baseline, not a target for finished
production software. Camera/UI paths are currently poorly covered and remain
visible in the report. Increase meaningful coverage over time; do not hide
uncovered application code or lower the threshold to make a feature pass.

The current YOLO26m workflow removes the CLIP Git dependency, Transformers and
Grounding DINO integration. The following records the earlier audit work.

Historically, the dependency service could not audit the pinned Git snapshot of Ultralytics CLIP
as a registry release. That former exclusion was explicitly recorded with its source;
the model files and Git dependency still need provenance/security review.
An advisory scan reports known published vulnerabilities, not all vulnerabilities.

During setup, the expanded audit found advisories affecting Transformers 4.57.6.
The optional labeling dependency was upgraded to 5.10.x (locked to 5.10.4),
and the same Grounding DINO tiny model was pinned to revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`. All locked registry versions passed
the audit after that upgrade, without vulnerability exceptions. An offline CPU
smoke check on this Mac loaded the cached model and completed prediction using
the updated libraries. Synthetic input in that check demonstrates API
compatibility, not cash recognition accuracy.

## Pipeline integrity and maintenance

External Actions are pinned to verified full commit hashes. Tool dependencies are
locked; build isolation is disabled so builds use the locked backend. Jobs have
timeouts, do not retain checkout credentials, and normally receive only read
permissions. Only CodeQL jobs receive the security-report upload permission.
The separate trusted Gitflow job receives write scopes for checks, task branches,
PRs, workflow dispatch, and protected merges, and executes only main's source.
Privileged jobs do not execute PR code. Caches are disabled to keep the trust
boundary and demonstration straightforward.

PR policy runs with `pull_request_target` from the trusted default branch and
reads the event as JSON. It never checks out the feature branch or interpolates
PR text into a shell command. Changes to this policy therefore take effect only
after promotion to `main`. Regular tests execute PR code without deployment
credentials. Workflow changes still need maintainer inspection: repository CI
configuration can change in a PR, so these gates are not tamper-proof against a
maintainer who intentionally weakens them.

Dependabot proposes weekly uv and Action updates into `develop`; those PRs run
the same checks. GitHub-authenticated Dependabot PRs use their generated changelog
body instead of the human PR template. A branch name alone does not grant that
exception. Security updates may target GitHub's default branch. They are reviewed
and tested before merging, and any direct `main` update must return to `develop`.
The weekly scheduled check on `main` also detects new advisories without a code
change. Reports are retained for 14 days; distributions for 30 days. Download
important demonstration evidence before expiry.

## Account prerequisites and rollout

Feature completion and the trusted Gitflow loop do not request AI reviews.
Disable automatic review for `PatBaruch/dMotion` in the separate
[Codex code review settings](https://chatgpt.com/codex/settings/code-review)
to stop account-triggered reviews too. A repository commit cannot change that setting.

- Default-branch policy, schedules, and Dependabot configuration activate on main.
  Require `test` and `pr-policy` on both protected branches. Remove only the retired
  `ai-review` context if it exists; preserve the security checks and other protections.
- Permit Actions to create promotion PRs. GitHub bundles this with approval capability;
  this workflow never submits reviews. Keep read-only default token permissions.
- Enable Dependabot security updates and retain actual scheduled scan/update evidence
  separately from configuration and manually dispatched runs.
- Add an independent human reviewer and require an approving review when one is
  available. The solo-maintainer setup uses zero required approvals because GitHub
  prevents authors from approving their own PRs.

An active agent fixes failed checks and any existing review findings. The deployed
Gitflow loop continues after the chat stops but cannot implement repairs. Tagging
and production deployment still need separate authorization. Never pretend that
an optional AI review passed, or copy local credentials into repository secrets.

## Demonstrate the process

Use an ordinary, useful feature with a regression test. In an isolated branch,
show the test failing before the implementation and passing afterward. Keep the
failure and correction logs. Do not publish a broken branch just to get a red
badge; the normal push hook correctly prevents that.

On the resulting PR, show the changed behavior, test, documentation, reviewed
commit, CI matrix, security reports, coverage gaps, and installed package check.
If AI review identifies an issue, show the finding and correction on the next
commit. Retain the report artifacts alongside the PR and run links. This supports
the claim that AI-assisted changes are checked by multiple independent tools.

## Separate evidence for the detector and deployment

Recognition reliability needs a frozen, human-reviewed test set with recording
groups kept separate from training and validation. Report false positives, misses,
accuracy metrics, thresholds, model/version provenance, and latency on the actual
Mac. AI label suggestions are drafts until reviewed. Unit tests and mocked model
adapters cannot supply these measurements.

Actual webcam permission, rendering, audio, warm-up, and stale-result behavior
also need a hardware acceptance run. The pipeline builds Python distributions;
it does not publish to PyPI, sign/notarize a Mac application, or deploy a service.
Distribution choice, release approval, signing credentials, and rollback tests
are required before claiming a production delivery pipeline.

## References

- [CodeQL configuration](https://docs.github.com/en/code-security/reference/code-scanning/workflow-configuration-options)
- [Safe PR policy workflows](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target)
- [pip-audit scope](https://github.com/pypa/pip-audit)
- [Codex GitHub review setup](https://learn.chatgpt.com/docs/third-party/github)
