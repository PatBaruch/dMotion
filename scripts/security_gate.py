"""Fail on high/critical CodeQL findings, including errors without a score."""

import argparse
import json
from pathlib import Path


def blocking_findings(report: dict) -> list[str]:
    if report.get("version") != "2.1.0" or not report.get("runs"):
        raise ValueError("Missing or unsupported SARIF analysis")
    findings = []
    for run in report["runs"]:
        rules = run["tool"]["driver"].get("rules", [])
        by_id = {rule["id"]: rule for rule in rules}
        for result in run.get("results", []):
            rule = by_id.get(result.get("ruleId"), {})
            if not rule and "ruleIndex" in result:
                rule = rules[result["ruleIndex"]]
            score = float(rule.get("properties", {}).get("security-severity", 0))
            level = result.get(
                "level", rule.get("defaultConfiguration", {}).get("level", "warning")
            )
            if score >= 7 or level == "error":
                findings.append(f"{result.get('ruleId', 'unknown')}: {result['message']['text']}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files = list(args.directory.glob("*.sarif"))
    if not files:
        raise ValueError("No SARIF report found; an absent scan cannot pass")
    findings = []
    for file in files:
        findings.extend(blocking_findings(json.loads(file.read_text())))
    for finding in findings:
        print(f"Security gate: {finding}")
    print(f"Analyzed {len(files)} SARIF report(s); {len(findings)} blocking finding(s).")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
