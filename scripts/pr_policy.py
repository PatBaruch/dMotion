"""Validate Gitflow routing and the documentation in a pull request."""

import argparse
import json
import re
from pathlib import Path

SECTIONS = ("Summary", "Documentation", "Validation", "Risks and limitations")
TITLE = re.compile(r"^(feat|fix|docs|test|refactor|perf|build|ci|chore|revert)(\([^)]+\))?!?: \S")


def default_base(branch: str) -> str:
    if branch.startswith("feature/"):
        return "develop"
    if branch.startswith(("release/", "hotfix/")):
        return "main"
    raise ValueError("Use feature/<name>, release/<version>, or hotfix/<name>.")


def validate_pull_request(base: str, head: str, title: str, body: str) -> list[str]:
    errors = []
    if base == "main":
        valid_route = head.startswith(("release/", "hotfix/"))
    elif base == "develop":
        valid_route = head == "main" or head.startswith(("feature/", "release/", "hotfix/"))
    else:
        valid_route = False
    if not valid_route:
        errors.append(
            "Features target develop; releases/hotfixes target main and return to develop."
        )
    if head.endswith("/"):
        errors.append("The branch needs a name after its prefix.")
    if not TITLE.match(title):
        errors.append(
            "Use a descriptive title such as 'feat: add image import' or 'ci: add checks'."
        )
    # Templates contain guidance in comments; that guidance is not documentation.
    visible = re.sub(r"<!--.*?-->", "", body or "", flags=re.DOTALL)
    for section in SECTIONS:
        match = re.search(
            rf"^## {re.escape(section)}\s*\n(.*?)(?=^## |\Z)",
            visible,
            flags=re.MULTILINE | re.DOTALL,
        )
        content = match.group(1).strip() if match else ""
        if len(content) < 12 or re.search(r"\b(TODO|TBD)\b", content, flags=re.IGNORECASE):
            errors.append(
                f"Fill in '## {section}' with actual results or a reason it does not apply."
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", type=Path, help="GitHub pull_request event JSON")
    args = parser.parse_args()
    event = json.loads(args.event.read_text())
    pr = event["pull_request"]
    errors = validate_pull_request(pr["base"]["ref"], pr["head"]["ref"], pr["title"], pr["body"])
    for error in errors:
        print(f"PR policy: {error}")
    if not errors:
        print("Gitflow routing and PR documentation passed.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
