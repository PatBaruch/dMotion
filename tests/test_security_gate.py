"""Regression tests that demonstrate the security gate rejects bad evidence."""

import importlib.util
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[1] / "scripts" / "security_gate.py"
spec = importlib.util.spec_from_file_location("security_gate", source)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def report(score, level="warning"):
    return {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "rules": [
                            {
                                "id": "python/unsafe-input",
                                "properties": {"security-severity": str(score)},
                            }
                        ]
                    }
                },
                "results": [
                    {
                        "ruleId": "python/unsafe-input",
                        "level": level,
                        "message": {"text": "Unsafe input reaches a command"},
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize("score", [7, 9, 10])
def test_high_and_critical_findings_fail_even_when_scan_itself_succeeds(score):
    assert gate.blocking_findings(report(score))


def test_medium_finding_is_reported_by_codeql_but_not_a_high_severity_blocker():
    assert not gate.blocking_findings(report(6.9))
    assert gate.blocking_findings(report(0, level="error"))


def test_rule_index_without_rule_id_is_also_checked():
    data = report(9)
    finding = data["runs"][0]["results"][0]
    del finding["ruleId"]
    finding["ruleIndex"] = 0
    assert gate.blocking_findings(data)


@pytest.mark.parametrize("data", [{}, {"version": "2.1.0", "runs": []}])
def test_missing_scan_is_never_treated_as_success(data):
    with pytest.raises(ValueError, match="SARIF"):
        gate.blocking_findings(data)


def test_missing_sarif_files_fail_closed(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["security_gate.py", str(tmp_path)])
    with pytest.raises(ValueError, match="No SARIF"):
        gate.main()
