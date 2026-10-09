"""Run checks, push one completed task, and create or update its documented PR."""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from pr_policy import default_base, validate_pull_request


def run(*command: str, capture: bool = False) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def changed_paths() -> set[str]:
    tracked = run("git", "diff", "--no-renames", "--name-only", "-z", "HEAD", capture=True)
    untracked = run("git", "ls-files", "--others", "--exclude-standard", "-z", capture=True)
    return {path for path in (tracked + untracked).split("\0") if path}


def validate_files(root: Path, files: list[str], changes: set[str]) -> list[str]:
    selected = []
    for file in files:
        path = Path(file)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"Use an explicit repository-relative file path: {file}")
        if path.is_dir():
            raise ValueError(f"Select individual files, not entire directories: {file}")
        if not (root / path).resolve().is_relative_to(root):
            raise ValueError(f"The file points outside the repository: {file}")
        selected.append(path.as_posix())
    unrelated = changes - set(selected)
    if unrelated:
        raise ValueError(
            "Unselected changes exist. Use an isolated worktree; do not include unrelated work: "
            + ", ".join(sorted(unrelated))
        )
    return selected


def body_with_evidence(body: str, sha: str) -> str:
    evidence = (
        "<!-- dmotion-validation:start -->\n"
        f"Local `make check` and `git diff --check` passed before pushing commit `{sha}`.\n"
        "GitHub CI runs separately; their status is shown on this PR.\n"
        "<!-- dmotion-validation:end -->"
    )
    body = re.sub(
        r"<!-- dmotion-validation:start -->.*?<!-- dmotion-validation:end -->\s*",
        "",
        body,
        flags=re.DOTALL,
    )
    return re.sub(r"(?m)^## Validation[ \t]*\n", f"## Validation\n\n{evidence}\n", body, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body-file", required=True, type=Path)
    parser.add_argument("--base", choices=("main", "develop"))
    parser.add_argument("--commit-message")
    parser.add_argument("--files", nargs="*", default=[])
    args = parser.parse_args()
    try:
        root = Path(run("git", "rev-parse", "--show-toplevel", capture=True)).resolve()
        if Path.cwd().resolve() != root:
            raise ValueError("Run finish-feature from the repository root.")
        branch = run("git", "branch", "--show-current", capture=True)
        base = args.base or default_base(branch)
        # Even a back-merge must use its own release/hotfix branch for this command.
        default_base(branch)
        body = args.body_file.read_text()
        errors = validate_pull_request(base, branch, args.title, body)
        if errors:
            raise ValueError("\n".join(errors))
        selected = validate_files(root, args.files, changed_paths())
        run("gh", "auth", "status")
        repo = run(
            "gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner", capture=True
        )
        run("git", "fetch", "origin", base)
        run("git", "merge-base", "--is-ancestor", f"origin/{base}", "HEAD")
        run("git", "diff", "--check")
        run("git", "diff", "--cached", "--check")
        run("make", "check")
        if selected:
            run("git", "add", "--", *selected)
        if run("git", "diff", "--cached", "--name-only", capture=True):
            run("git", "commit", "-m", args.commit_message or args.title)
        if changed_paths():
            raise ValueError("The working tree changed during validation. Check it before pushing.")
        sha = run("git", "rev-parse", "HEAD", capture=True)
        # The repository pre-push hook validates the exact committed tree again.
        run("git", "push", "--set-upstream", "origin", f"HEAD:refs/heads/{branch}")
        prs = json.loads(
            run(
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--head",
                branch,
                "--base",
                base,
                "--state",
                "open",
                "--json",
                "number,url",
                capture=True,
            )
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md") as document:
            document.write(body_with_evidence(body, sha))
            document.flush()
            if prs:
                number = str(prs[0]["number"])
                run(
                    "gh",
                    "pr",
                    "edit",
                    number,
                    "--repo",
                    repo,
                    "--title",
                    args.title,
                    "--body-file",
                    document.name,
                )
                url = prs[0]["url"]
            else:
                url = run(
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    repo,
                    "--head",
                    branch,
                    "--base",
                    base,
                    "--title",
                    args.title,
                    "--body-file",
                    document.name,
                    capture=True,
                )
        # Opt in only this task's PR. Existing unrelated PRs remain unmanaged.
        run("gh", "pr", "edit", url, "--repo", repo, "--add-label", "gitflow:auto")
        print(f"\nFeature pushed: {sha}\nPull request: {url}")
        print("CI is pending until GitHub reports its results. No AI review is requested.")
        print(f"Monitor CI: gh pr checks {url} --watch --interval 15 --fail-fast")
        print("Merge only after required CI/security and PR policy checks pass.")
        print("Outstanding requests for changes and unresolved conversations prevent merging.")
        print("Trusted Gitflow automation monitors this labeled PR after deployment to main.")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"finish-feature stopped: {error}")
        print("Resolve the reported issue and rerun; existing commits and PRs are preserved.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
