"""Reject direct protected-branch pushes and validate the actual commit being pushed."""

import subprocess
import sys

ZERO = "0" * 40


def validate_updates(lines: list[str], head: str, dirty: bool) -> bool:
    needs_checks = False
    for line in lines:
        _local_ref, local_sha, remote_ref, _remote_sha = line.split()
        if remote_ref in ("refs/heads/main", "refs/heads/develop"):
            raise ValueError("main and develop accept pull requests only; push a task branch.")
        if local_sha == ZERO:
            continue  # Deleting a task branch does not publish code.
        if local_sha != head:
            raise ValueError(
                "Check out the branch being pushed so checks validate its actual code."
            )
        needs_checks = True
    if needs_checks and dirty:
        raise ValueError(
            "Commit the task in a clean worktree before pushing; preserve unrelated work."
        )
    return needs_checks


def main() -> int:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
            ).strip()
        )
        lines = []
        updates = []
        for line in sys.stdin.read().splitlines():
            if not line.strip():
                continue
            local_ref, local_sha, remote_ref, remote_sha = line.split()
            updates.append((local_sha, remote_ref, remote_sha))
            if local_sha != ZERO and remote_ref.startswith("refs/tags/"):
                local_sha = subprocess.check_output(
                    ["git", "rev-parse", f"{local_ref}^{{commit}}"], text=True
                ).strip()
            lines.append(f"{local_ref} {local_sha} {remote_ref} {remote_sha}")
        if validate_updates(lines, head, dirty):
            for local_sha, remote_ref, remote_sha in updates:
                if remote_ref.startswith("refs/heads/") and ZERO not in (local_sha, remote_sha):
                    subprocess.run(
                        ["git", "merge-base", "--is-ancestor", remote_sha, local_sha], check=True
                    )
            subprocess.run(["make", "check"], check=True)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Push blocked: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
