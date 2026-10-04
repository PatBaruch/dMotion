"""Audit all registry packages in uv.lock without installing optional ML stacks."""

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path


def audit_inventory(lock: dict) -> tuple[list[str], list[dict]]:
    requirements = set()
    excluded = []
    for package in lock["package"]:
        source = package["source"]
        if "registry" in source:
            requirements.add(f"{package['name']}=={package['version']}")
        elif package["name"] == "dmotion" and source.get("editable") == ".":
            excluded.append({"name": "dmotion", "reason": "First-party source checked by CI"})
        elif "git" in source:
            # Advisory databases identify registry versions, not Git snapshots.
            excluded.append(
                {
                    "name": package["name"],
                    "reason": "Git source; manual audit needed",
                    "source": source["git"],
                }
            )
        else:
            raise ValueError(f"Unrecognized dependency source: {package['name']}")
    if not requirements:
        raise ValueError("No registry packages found in lockfile")
    return sorted(requirements), excluded


def requirement_groups(requirements: list[str]) -> list[list[str]]:
    """pip-audit rejects duplicate names; audit alternate platform versions too."""
    groups: list[list[str]] = []
    names: list[set[str]] = []
    for requirement in requirements:
        name = requirement.split("==", 1)[0]
        for index, seen in enumerate(names):
            if name not in seen:
                groups[index].append(requirement)
                seen.add(name)
                break
        else:
            groups.append([requirement])
            names.append({name})
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    requirements, excluded = audit_inventory(tomllib.loads(args.lock.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(requirements) + "\n")
    args.output.with_suffix(".scope.json").write_text(
        json.dumps({"registry_versions": len(requirements), "excluded": excluded}, indent=2) + "\n"
    )
    print(
        f"Auditing {len(requirements)} locked registry versions, including optional dependencies."
    )
    for item in excluded:
        print(f"Audit scope: {item['name']}: {item['reason']}")
    if args.audit:
        failed = False
        for index, group in enumerate(requirement_groups(requirements)):
            path = args.output.with_name(f"requirements-{index}.txt")
            path.write_text("\n".join(group) + "\n")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--strict",
                    "--disable-pip",
                    "--no-deps",
                    "-r",
                    str(path),
                    "--progress-spinner",
                    "off",
                    "--timeout",
                    "30",
                    "-f",
                    "json",
                    "-o",
                    str(path.with_name(f"dependency-audit-{index}.json")),
                ],
                check=False,
            )
            failed = failed or result.returncode != 0
        return int(failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
