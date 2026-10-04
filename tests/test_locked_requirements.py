"""Keep the dependency audit's optional-stack coverage and exclusions explicit."""

import importlib.util
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[1] / "scripts" / "locked_requirements.py"
spec = importlib.util.spec_from_file_location("locked_requirements", source)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_optional_and_alternate_registry_versions_are_audited_without_installation():
    lock = {
        "package": [
            {"name": "torch", "version": "2.0", "source": {"registry": "https://pypi.org/simple"}},
            {"name": "torch", "version": "2.1", "source": {"registry": "https://pypi.org/simple"}},
            {"name": "dmotion", "version": "0.1", "source": {"editable": "."}},
            {"name": "clip", "version": "1.0", "source": {"git": "https://example.org/clip#abc"}},
        ]
    }
    requirements, excluded = inventory.audit_inventory(lock)
    assert requirements == ["torch==2.0", "torch==2.1"]
    assert {entry["name"] for entry in excluded} == {"dmotion", "clip"}
    assert excluded[1]["source"] == "https://example.org/clip#abc"


def test_unknown_dependency_source_does_not_silently_escape_the_audit():
    with pytest.raises(ValueError, match="Unrecognized"):
        inventory.audit_inventory(
            {
                "package": [
                    {"name": "other", "version": "1.0", "source": {"path": "/private/package"}},
                ]
            }
        )


def test_audit_groups_retain_every_platform_version_without_duplicate_names():
    requirements = ["numpy==2.0", "numpy==2.1", "torch==2.0", "torch==2.1", "tool==1.0"]
    groups = inventory.requirement_groups(requirements)
    assert sorted(item for group in groups for item in group) == sorted(requirements)
    for group in groups:
        names = [item.split("==", 1)[0] for item in group]
        assert len(names) == len(set(names))
