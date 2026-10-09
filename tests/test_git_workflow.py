"""Checks for publication safety and meaningful PR requirements."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def load_script(name):
    source = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = load_script("pr_policy")
finish = load_script("finish_feature")
push = load_script("pre_push")

BODY = """## Summary
Import a still image through the existing detector command.
## Documentation
Updated the training guide with the new command and its limitations.
## Validation
Automated local results are recorded by the completion script.
## Risks and limitations
Camera behavior was not exercised by this change.
"""


@pytest.mark.parametrize(
    "base,head",
    [
        ("develop", "feature/import"),
        ("main", "release/0.2.0"),
        ("main", "hotfix/crash"),
        ("develop", "main"),
        ("develop", "hotfix/crash"),
    ],
)
def test_valid_gitflow_routes(base, head):
    assert policy.validate_pull_request(base, head, "feat: add image import", BODY) == []


@pytest.mark.parametrize(
    "base,head",
    [
        ("main", "feature/import"),
        ("main", "develop"),
        ("develop", "codex/work"),
        ("develop", "feature/"),
    ],
)
def test_wrong_destination_or_branch_is_rejected(base, head):
    assert policy.validate_pull_request(base, head, "feat: add image import", BODY)


def test_template_comments_are_not_documentation():
    template = "\n".join(
        f"## {section}\n<!-- Write the actual details. -->" for section in policy.SECTIONS
    )
    errors = policy.validate_pull_request("develop", "feature/import", "feat: import", template)
    assert len(errors) == 4


def test_dependabot_exception_requires_verified_bot_metadata():
    assert policy.validate_pull_request("develop", "dependabot/uv/tool", "Bump tool", BODY)
    assert not policy.validate_pull_request(
        "develop",
        "dependabot/uv/tool",
        "Bump tool",
        "Updates tool with its changelog.",
        maintenance_bot=True,
    )
    assert policy.validate_pull_request(
        "develop", "feature/no-policy", "invalid", "", maintenance_bot=True
    )


def test_placeholder_in_a_required_section_is_rejected():
    body = BODY.replace("Updated the training guide", "TODO: update the training guide")
    assert any(
        "Documentation" in error
        for error in policy.validate_pull_request("develop", "feature/import", "feat: import", body)
    )


def test_finish_refuses_unrelated_changes(tmp_path):
    with pytest.raises(ValueError, match="Unselected changes"):
        finish.validate_files(tmp_path, ["feature.py"], {"feature.py", "other-task.py"})


def test_finish_refuses_bulk_staging_and_external_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    for path in ["src", "../private.txt", str(tmp_path / "feature.py")]:
        with pytest.raises(ValueError):
            finish.validate_files(tmp_path, [path], set())


def test_symlink_cannot_publish_a_file_outside_the_repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "private.txt").symlink_to(tmp_path / "private.txt")
    with pytest.raises(ValueError, match="outside"):
        finish.validate_files(root, ["private.txt"], set())


@pytest.mark.parametrize("branch", ["main", "develop"])
def test_push_hook_blocks_protected_branch_creation_update_and_deletion(branch):
    for sha in ["a" * 40, push.ZERO]:
        update = f"refs/heads/{branch} {sha} refs/heads/{branch} {push.ZERO}"
        with pytest.raises(ValueError, match="pull requests only"):
            push.validate_updates([update], "a" * 40, False)


def test_push_hook_refuses_dirty_or_wrong_checkout():
    update = f"refs/heads/feature/import {'a' * 40} refs/heads/feature/import {push.ZERO}"
    with pytest.raises(ValueError, match="clean worktree"):
        push.validate_updates([update], "a" * 40, True)
    with pytest.raises(ValueError, match="actual code"):
        push.validate_updates([update], "b" * 40, False)
    assert push.validate_updates([update], "a" * 40, False)


def test_task_branch_deletion_does_not_run_code_checks():
    update = f"(delete) {push.ZERO} refs/heads/feature/import {'a' * 40}"
    assert not push.validate_updates([update], "b" * 40, True)


def test_validation_evidence_is_replaced_for_the_latest_commit():
    first = finish.body_with_evidence(BODY, "first-sha")
    second = finish.body_with_evidence(first, "second-sha")
    assert "first-sha" not in second
    assert "second-sha" in second
    assert second.count("<!-- dmotion-validation:start -->") == 1
    assert "CI runs separately" in second


def test_completion_publishes_and_labels_without_requesting_ai_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    document = tmp_path / "body.md"
    document.write_text(BODY)
    monkeypatch.setattr(
        sys, "argv", ["finish_feature.py", "--title", "feat: import", "--body-file", str(document)]
    )
    monkeypatch.setattr(finish, "changed_paths", lambda: set())
    commands = []

    def successful_run(*command, capture=False):
        commands.append(command)
        if command == ("git", "rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if command == ("git", "branch", "--show-current"):
            return "feature/import"
        if command == ("git", "rev-parse", "HEAD"):
            return "a" * 40
        if command[:3] == ("gh", "repo", "view"):
            return "owner/repo"
        if command[:3] == ("gh", "pr", "list"):
            return '[{"number":7,"url":"https://github.com/owner/repo/pull/7"}]'
        return ""

    monkeypatch.setattr(finish, "run", successful_run)
    assert finish.main() == 0
    label = next(i for i, command in enumerate(commands) if "--add-label" in command)
    assert commands[label][:3] == ("gh", "pr", "edit")
    assert not any("--request-review" in command for command in commands)
    assert not any("scripts/gitflow_automation.py" in command for command in commands)


def test_failed_checks_never_stage_commit_push_or_open_a_pr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    document = tmp_path / "body.md"
    document.write_text(BODY)
    monkeypatch.setattr(
        sys, "argv", ["finish_feature.py", "--title", "feat: import", "--body-file", str(document)]
    )
    monkeypatch.setattr(finish, "changed_paths", lambda: set())
    commands = []

    def failing_run(*command, capture=False):
        commands.append(command)
        if command == ("git", "rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if command == ("git", "branch", "--show-current"):
            return "feature/import"
        if command[:3] == ("gh", "repo", "view"):
            return "owner/repo"
        if command == ("make", "check"):
            raise subprocess.CalledProcessError(1, command)
        return ""

    monkeypatch.setattr(finish, "run", failing_run)
    assert finish.main() == 1
    assert not any(
        command[:2] in [("git", "add"), ("git", "commit"), ("git", "push")]
        or command[:3] == ("gh", "pr", "create")
        for command in commands
    )


@pytest.mark.parametrize("hook_environment", [False, True])
def test_pre_push_rejects_failed_checks_against_a_real_local_remote(
    tmp_path, monkeypatch, hook_environment
):
    # This exercises Git's actual hook dispatch, without credentials or a network.
    import shutil

    local_git_variables = set(
        subprocess.check_output(["git", "rev-parse", "--local-env-vars"], text=True).splitlines()
    )

    def foreign_environment():
        # Git exports repository-local variables to hooks; foreign repositories
        # must not inherit them. Also remove numbered command-line config entries.
        return {
            key: value
            for key, value in os.environ.items()
            if key not in local_git_variables and not key.startswith("GIT_CONFIG_")
        }

    caller = tmp_path / "caller"
    subprocess.run(
        ["git", "init", str(caller)], check=True, capture_output=True, env=foreign_environment()
    )
    original_config = (caller / ".git/config").read_bytes()
    if hook_environment:
        monkeypatch.setenv("GIT_DIR", str(caller / ".git"))
        monkeypatch.setenv("GIT_COMMON_DIR", str(caller / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(caller))
        monkeypatch.setenv("GIT_INDEX_FILE", str(caller / ".git/index"))
    environment = foreign_environment()
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    root.mkdir()

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=root, check=check, capture_output=True, text=True, env=environment
        )

    git("init", "-b", "feature/check-hook")
    git("config", "user.name", "Workflow test")
    git("config", "user.email", "test@example.invalid")
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True, env=environment
    )
    git("remote", "add", "origin", str(remote))
    scripts = Path(__file__).resolve().parents[1]
    for name in [".githooks/pre-push", "scripts/pre_push.py", "scripts/setup-workflow.sh"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scripts / name, target)
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    (root / ".gitignore").write_text(".venv/\n")
    (root / "Makefile").write_text("check:\n\t@exit 1\n")
    subprocess.run(
        ["sh", "scripts/setup-workflow.sh"],
        cwd=root,
        check=True,
        capture_output=True,
        env=environment,
    )
    git(
        "add",
        "Makefile",
        ".gitignore",
        ".githooks/pre-push",
        "scripts/pre_push.py",
        "scripts/setup-workflow.sh",
    )
    git("commit", "-m", "ci: test actual push guard")
    assert git("push", "origin", "HEAD", check=False).returncode != 0
    assert not git("ls-remote", "--heads", "origin").stdout.strip()
    (root / "Makefile").write_text("check:\n\t@exit 0\n")
    git("add", "Makefile")
    git("commit", "-m", "ci: passing checks")
    assert git("push", "origin", "HEAD").returncode == 0
    blocked = git("push", "origin", "HEAD:refs/heads/main", check=False)
    assert blocked.returncode != 0
    assert "pull requests only" in blocked.stderr

    assert (caller / ".git/config").read_bytes() == original_config
