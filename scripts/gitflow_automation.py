"""Trusted, current-commit review gating and opt-in Gitflow orchestration."""

import argparse
import json
import re
import subprocess

from pr_policy import validate_pull_request

LABEL = "gitflow:auto"
BOT_LOGIN = "chatgpt-codex-connector[bot]"
BOT_ID = 199175422
ACTIONS_APP = 15368
CHECK = "ai-review"
REQUIRED_CI = ("test", "pr-policy")
REVIEW_ERRORS = re.compile(
    r"(?:unable|failed|could not|couldn't|cannot)(?: to)? (?:complete (?:the )?)?review|"
    r"quota.{0,40}(?:limit|exhaust)|(?:rate|usage)[ -]?limit|review.{0,30}unavailable|"
    r"(?:something went wrong|review (?:was )?not (?:completed|performed)|"
    r"connect (?:your account|to github)|create a codex account)",
    re.IGNORECASE,
)


class GitHub:
    """Use gh's existing authentication without exposing or copying credentials."""

    def __init__(self, repo: str):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ValueError("Use owner/repository")
        self.repo = repo
        self.prefix = f"repos/{repo}"

    def api(self, path: str, *, method: str = "GET", data: dict | None = None):
        command = ["gh", "api", "--method", method, f"{self.prefix}/{path}"]
        if data is not None:
            command.extend(["--input", "-"])
        result = subprocess.run(
            command,
            input=json.dumps(data) if data is not None else None,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        return json.loads(result.stdout) if result.stdout.strip() else None

    def pages(self, path: str, key: str | None = None) -> list:
        items = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            result = self.api(f"{path}{separator}per_page=100&page={page}")
            batch = result[key] if key else result
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise ValueError("Pagination limit exceeded; refusing incomplete evidence")

    def threads(self, number: int) -> list:
        owner, name = self.repo.split("/")
        query = """query($owner:String!,$name:String!,$number:Int!,$cursor:String){
          repository(owner:$owner,name:$name){pullRequest(number:$number){
            reviewThreads(first:100,after:$cursor){nodes{isResolved}
              pageInfo{hasNextPage endCursor}}}}}"""
        nodes, cursor = [], None
        for _ in range(100):
            payload = {
                "query": query,
                "variables": {"owner": owner, "name": name, "number": number, "cursor": cursor},
            }
            result = subprocess.run(
                ["gh", "api", "graphql", "--input", "-"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            )
            response = json.loads(result.stdout)
            if response.get("errors"):
                raise ValueError("Review-thread query failed; refusing incomplete evidence")
            page = response["data"]["repository"]["pullRequest"]["reviewThreads"]
            nodes.extend(page["nodes"])
            if not page["pageInfo"]["hasNextPage"]:
                return nodes
            cursor = page["pageInfo"]["endCursor"]
        raise ValueError("Review-thread pagination limit exceeded")


def is_codex(item: dict) -> bool:
    user = item.get("user") or {}
    return user.get("login") == BOT_LOGIN and user.get("id") == BOT_ID and user.get("type") == "Bot"


def completion_comment_verdict(
    sha: str, comments: list[dict], commits: list[dict]
) -> tuple[bool, str]:
    """Recognize only the observed native no-findings result, never a reaction."""
    candidates = [
        item
        for item in comments
        if is_codex(item)
        and (
            (item.get("body") or "").startswith("Codex Review:")
            or REVIEW_ERRORS.search(item.get("body") or "")
        )
    ]
    if not candidates:
        return False, f"No completed Codex review for current head {sha}"
    comment = max(candidates, key=lambda item: (item.get("updated_at") or "", item["id"]))
    body = comment.get("body") or ""
    if not comment.get("created_at") or not comment.get("updated_at"):
        return False, "Codex completion comment lacks submission evidence"
    if REVIEW_ERRORS.search(body) or re.search(r"\[P[01]\]", body):
        return False, "Codex reported an unsuccessful or blocking review"
    first_line = body.splitlines()[0] if body else ""
    if not re.fullmatch(r"Codex Review: Didn't find any major issues\.(?: Swish!)?", first_line):
        return False, "Unrecognized Codex completion comment"
    references = re.findall(r"^\*\*Reviewed commit:\*\* `([0-9a-f]{10,40})`$", body, re.MULTILINE)
    if len(references) != 1:
        return False, "Codex completion must identify exactly one reviewed commit"
    reference = references[0]
    if len(reference) == 40:
        matches = {reference}
    else:
        matches = {item["sha"] for item in commits if item["sha"].startswith(reference)}
    if matches != {sha}:
        return False, "Codex reviewed commit is stale, absent, or ambiguous in this PR"
    return True, f"Codex completion comment {comment['id']} covers {sha}; no unresolved blockers"


def review_verdict(
    sha: str,
    reviews: list[dict],
    threads: list[dict],
    comments: list[dict] | None = None,
    commits: list[dict] | None = None,
) -> tuple[bool, str]:
    """A toggle, reaction, empty list, quota message, or stale review cannot pass."""
    decisive_by_author = {}
    for review in sorted(reviews, key=lambda item: item["id"]):
        # A later comment cannot withdraw an outstanding request for changes.
        if review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            decisive_by_author[review["user"]["id"]] = review
    if any(item["state"] == "CHANGES_REQUESTED" for item in decisive_by_author.values()):
        return False, "A reviewer requested changes"
    if any(not thread["isResolved"] for thread in threads):
        return False, "Unresolved review conversations remain"
    matching = [item for item in reviews if is_codex(item) and item.get("commit_id") == sha]
    if not matching:
        return completion_comment_verdict(sha, comments or [], commits or [])
    review = max(matching, key=lambda item: item["id"])
    if review["state"] not in {"APPROVED", "COMMENTED"} or not review.get("submitted_at"):
        return False, "Codex review is pending, dismissed, or blocking"
    body = review.get("body") or ""
    # A newer result must be fully validated, including pending/unrecognized bodies.
    newer = [
        comment
        for comment in comments or []
        if is_codex(comment)
        and (comment.get("updated_at") or comment.get("created_at") or "") > review["submitted_at"]
        and (
            (comment.get("body") or "").startswith("Codex Review:")
            or REVIEW_ERRORS.search(comment.get("body") or "")
        )
    ]
    if newer:
        ok, reason = completion_comment_verdict(sha, newer, commits or [])
        if not ok:
            return False, f"A newer Codex result is not complete: {reason}"
    if REVIEW_ERRORS.search(body):
        return False, "Codex reported an unsuccessful review"
    if re.search(r"\[P[01]\]", body):
        return False, "Blocking findings remain in the current review summary"
    if review["state"] == "COMMENTED" and "codex review" not in body.lower():
        return False, "Unrecognized Codex completion; an explicit reviewed result is required"
    return True, f"Codex review {review['id']} covers {sha}; no unresolved blockers"


def ci_verdict(sha: str, checks: list[dict]) -> tuple[bool, str]:
    for name in REQUIRED_CI:
        matching = [
            check
            for check in checks
            if check["name"] == name
            and check.get("head_sha") == sha
            and check.get("app", {}).get("id") == ACTIONS_APP
        ]
        if not matching or any(
            check["status"] != "completed" or check["conclusion"] != "success" for check in matching
        ):
            return False, f"Required GitHub Actions check {name} is missing or not successful"
    return True, "All required CI/security contexts passed for the current head"


def inspect_pr(github: GitHub, number: int) -> dict:
    pr = github.api(f"pulls/{number}")
    sha = pr["head"]["sha"]
    reviews = github.pages(f"pulls/{number}/reviews")
    comments = github.pages(f"issues/{number}/comments")
    commits = github.pages(f"pulls/{number}/commits")
    if len(commits) != pr.get("commits", len(commits)):
        raise ValueError("Incomplete PR commit history; refusing ambiguous review evidence")
    review_ok, review_reason = review_verdict(
        sha, reviews, github.threads(number), comments, commits
    )
    checks = github.pages(f"commits/{sha}/check-runs?filter=latest", "check_runs")
    ci_ok, ci_reason = ci_verdict(sha, checks)
    policy_errors = validate_pull_request(
        pr["base"]["ref"],
        pr["head"]["ref"],
        pr["title"],
        pr.get("body") or "",
        maintenance_bot=pr["user"]["login"] == "dependabot[bot]"
        and pr["user"].get("type") == "Bot",
    )
    return {
        "pr": pr,
        "head_sha": sha,
        "review_ok": review_ok,
        "review_reason": review_reason,
        "ci_ok": ci_ok,
        "ci_reason": ci_reason,
        "policy_errors": policy_errors,
    }


def publish_gate(github: GitHub, evidence: dict, *, policy: bool = False) -> None:
    pr, sha = evidence["pr"], evidence["head_sha"]
    name = "pr-policy" if policy else CHECK
    external_id = f"dmotion:{name}:{pr['number']}:{sha}"
    runs = github.pages(f"commits/{sha}/check-runs?check_name={name}", "check_runs")
    own = [
        run
        for run in runs
        if run.get("external_id") == external_id and run["app"]["id"] == ACTIONS_APP
    ]
    data = {
        "name": name,
        "status": "completed",
        "conclusion": "success"
        if (not evidence["policy_errors"] if policy else evidence["review_ok"])
        else "failure",
        "external_id": external_id,
        "details_url": pr["html_url"],
        "output": {
            "title": "PR documentation and Gitflow" if policy else "Latest-commit AI review",
            "summary": ("\n".join(evidence["policy_errors"]) or "PR policy passed")
            if policy
            else evidence["review_reason"],
        },
    }
    if own:
        github.api(
            f"check-runs/{max(own, key=lambda run: run['id'])['id']}", method="PATCH", data=data
        )
    else:
        github.api("check-runs", method="POST", data={**data, "head_sha": sha})


class NotReady(ValueError):
    """A safe wait, rather than a transport failure or permission to bypass a gate."""


def merge_pr(github: GitHub, number: int) -> dict:
    # Recollect reviews, threads, checks, and head immediately before the atomic SHA guard.
    evidence = inspect_pr(github, number)
    pr = evidence["pr"]
    if (
        pr["state"] != "open"
        or pr.get("draft")
        or (pr["head"].get("repo") or {}).get("full_name") != github.repo
        or not evidence["review_ok"]
        or not evidence["ci_ok"]
        or evidence["policy_errors"]
        or pr.get("mergeable") is not True
        or pr.get("mergeable_state") != "clean"
    ):
        raise NotReady(
            f"PR #{number} is not ready ({pr.get('mergeable_state')}): "
            f"{evidence['review_reason']}; {evidence['ci_reason']}; {evidence['policy_errors']}"
        )
    result = github.api(
        f"pulls/{number}/merge",
        method="PUT",
        data={"sha": evidence["head_sha"], "merge_method": "merge"},
    )
    if not result.get("merged"):
        raise ValueError(f"GitHub refused PR #{number}'s protected merge")
    return result


def ensure_ci(github: GitHub, pr: dict) -> None:
    sha = pr["head"]["sha"]
    runs = github.pages(f"actions/runs?head_sha={sha}", "workflow_runs")
    if any(
        run["name"] == "Checks"
        and run["status"] in {"queued", "in_progress", "completed"}
        and run.get("conclusion") != "action_required"
        for run in runs
    ):
        return
    github.api(
        "actions/workflows/checks.yml/dispatches",
        method="POST",
        data={"ref": pr["head"]["ref"], "inputs": {"expected_sha": sha}},
    )


def update_base(github: GitHub, pr: dict) -> dict:
    """Integrate a newer protected base without rewriting the task branch."""
    base_sha = pr["base"]["sha"]
    comparison = github.api(f"compare/{base_sha}...{pr['head']['sha']}")
    if comparison["behind_by"]:
        github.api("merges", method="POST", data={"base": pr["head"]["ref"], "head": base_sha})
        return github.api(f"pulls/{pr['number']}")
    return pr


def managed_body(kind: str, source_sha: str) -> str:
    return (
        f"## Summary\n{kind} at source commit `{source_sha}` through protected Gitflow branches.\n"
        "\n## Documentation\nFollow docs/GIT_WORKFLOW.md; retain the source SHA for provenance.\n\n"
        "## Validation\nRequired CI/security and current-head AI review must pass before merging. "
        "The trusted automation dispatches checks explicitly for token-created branches.\n\n"
        "## Risks and limitations\nNo tags or deployment. Conflicts and missing/failed review "
        "block this PR; unresolved findings require an active implementing agent.\n"
    )


def create_promotion(github: GitHub, open_prs: list[dict]) -> dict | None:
    # Back-sync takes priority. Never rewrite a published task branch or duplicate a PR.
    if any(
        LABEL in {label["name"] for label in pr["labels"]}
        and pr["head"]["ref"].startswith(("release/automation-", "hotfix/sync-"))
        for pr in open_prs
    ):
        return None
    main = github.api("git/ref/heads/main")["object"]["sha"]
    develop = github.api("git/ref/heads/develop")["object"]["sha"]
    comparison = github.api(f"compare/{main}...{develop}")
    if comparison["behind_by"]:
        branch, source, target, integrate = f"hotfix/sync-{main}", main, "develop", develop
        kind = "Synchronize main back into develop"
    elif comparison.get("files"):
        branch, source, target, integrate = (f"release/automation-{develop}", develop, "main", main)
        kind = "Promote develop to main"
    else:
        return None
    # Recover a branch created before an interrupted run. Unexpected collisions fail closed.
    marker = f"<!-- dmotion-gitflow:{source}:{integrate} -->"
    existing = next((pr for pr in open_prs if pr["head"]["ref"] == branch), None)
    if existing:
        if existing["base"]["ref"] != target or marker not in (existing.get("body") or ""):
            raise ValueError(f"Unrelated PR already uses {branch}; refusing to change it")
        github.api(f"issues/{existing['number']}/labels", method="POST", data={"labels": [LABEL]})
        ensure_ci(github, existing)
        return existing
    refs = github.api(f"git/matching-refs/heads/{branch}")
    exact = [ref for ref in refs if ref["ref"] == f"refs/heads/{branch}"]
    if exact:
        head = exact[0]["object"]["sha"]
        if head != source:
            commit = github.api(f"git/commits/{head}")
            if {parent["sha"] for parent in commit["parents"]} != {source, integrate}:
                raise ValueError(f"Unexpected published branch collision: {branch}")
    else:
        github.api("git/refs", method="POST", data={"ref": f"refs/heads/{branch}", "sha": source})
    github.api("merges", method="POST", data={"base": branch, "head": integrate})
    pr = github.api(
        "pulls",
        method="POST",
        data={
            "head": branch,
            "base": target,
            "title": f"ci: {kind.lower()} ({source[:12]})",
            "body": managed_body(kind, source) + f"\n{marker}\n",
        },
    )
    github.api(f"issues/{pr['number']}/labels", method="POST", data={"labels": [LABEL]})
    ensure_ci(github, pr)
    return pr


def pulse(github: GitHub) -> list[dict]:
    results = []
    open_prs = github.pages("pulls?state=open")
    for pr in open_prs:
        managed = LABEL in {label["name"] for label in pr["labels"]}
        same_repo = (pr["head"].get("repo") or {}).get("full_name") == github.repo
        if managed and not pr.get("draft") and same_repo:
            update_base(github, pr)
        evidence = inspect_pr(github, pr["number"])
        publish_gate(github, evidence)
        publish_gate(github, evidence, policy=True)
        if managed and not pr.get("draft") and same_repo:
            ensure_ci(github, evidence["pr"])
            if evidence["review_ok"] and evidence["ci_ok"] and not evidence["policy_errors"]:
                try:
                    result = merge_pr(github, pr["number"])
                    results.append({"number": pr["number"], "merged": result["sha"]})
                except NotReady as error:
                    results.append({"number": pr["number"], "waiting": str(error)})
                continue
        results.append(
            {
                "number": pr["number"],
                "head": evidence["head_sha"],
                "review": evidence["review_reason"],
            }
        )
    created = create_promotion(github, github.pages("pulls?state=open"))
    if created:
        results.append({"created": created["html_url"]})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int, help="Read current evidence for one PR")
    parser.add_argument("--merge", action="store_true", help="Merge only the inspected PR if ready")
    parser.add_argument("--pulse", action="store_true", help="Run trusted background orchestration")
    args = parser.parse_args()
    if args.pr and args.pulse:
        parser.error("Choose --pr or --pulse, not both")
    if args.merge and (not args.pr or args.pulse):
        parser.error("--merge requires --pr and cannot be combined with --pulse")
    github = GitHub(args.repo)
    if args.pulse:
        result = pulse(github)
    elif args.merge:
        result = merge_pr(github, args.pr)
    elif args.pr:
        result = inspect_pr(github, args.pr)
        result["pr"] = {key: result["pr"][key] for key in ("number", "html_url", "state")}
    else:
        parser.error("Choose --pr or --pulse")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
