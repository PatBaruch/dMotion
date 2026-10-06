"""Exercise review evidence, protected merges, and resumable Gitflow orchestration."""

import copy
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest
from test_git_workflow import BODY, load_script

flow = load_script("gitflow_automation")
SHA = "a" * 40
BASE = "b" * 40


def review(**changes):
    return {
        "id": 12,
        "user": {"id": flow.BOT_ID, "login": flow.BOT_LOGIN, "type": "Bot"},
        "commit_id": SHA,
        "state": "COMMENTED",
        "submitted_at": "2026-10-04T12:00:00Z",
        "updated_at": changes.get("submitted_at", "2026-10-04T12:00:00Z"),
        "body": "Codex Review: no blocking findings.",
        **changes,
    }


def pr(**changes):
    return {
        "number": 7,
        "html_url": "https://github.com/owner/repo/pull/7",
        "head": {"sha": SHA, "ref": "feature/fix", "repo": {"full_name": "owner/repo"}},
        "base": {"sha": BASE, "ref": "develop"},
        "user": {"login": "owner", "type": "User"},
        "title": "ci: enforce reviewed merges",
        "body": BODY,
        "state": "open",
        "draft": False,
        "mergeable": True,
        "mergeable_state": "clean",
        "labels": [{"name": flow.LABEL}],
        **changes,
    }


def checks():
    return [
        {
            "name": name,
            "head_sha": SHA,
            "status": "completed",
            "conclusion": "success",
            "app": {"id": flow.ACTIONS_APP},
        }
        for name in flow.REQUIRED_CI
    ]


class FakeGitHub:
    repo = "owner/repo"

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []
        self.review_threads = []

    def api(self, path, *, method="GET", data=None):
        self.calls.append((path, method, data))
        result = self.routes[method, path]
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)

    def pages(self, path, key=None):
        return self.api(path)

    def threads(self, number):
        return self.review_threads

    def commits(self, number):
        return self.api(f"pulls/{number}/commits")

    def reviews(self, number):
        return self.api(f"pulls/{number}/reviews")


def inspection_client(**changes):
    pull = pr(**changes)
    return FakeGitHub(
        {
            ("GET", "pulls/7"): pull,
            ("GET", "pulls/7/reviews"): [review()],
            ("GET", "issues/7/comments"): [],
            ("GET", "pulls/7/commits"): [{"sha": SHA}],
            ("GET", f"commits/{SHA}/check-runs?filter=latest"): checks(),
            ("PUT", "pulls/7/merge"): {"merged": True, "sha": "c" * 40},
        }
    )


def test_only_completed_authenticated_current_review_passes():
    assert flow.review_verdict(SHA, [review()], [])[0]
    assert flow.review_verdict(SHA, [review(state="APPROVED", body="")], [])[0]


@pytest.mark.parametrize(
    "reviews,threads",
    [
        ([], []),
        ([review(commit_id=BASE)], []),
        ([review(state="PENDING")], []),
        ([review(state="DISMISSED")], []),
        ([review(state="CHANGES_REQUESTED")], []),
        ([review(submitted_at=None)], []),
        ([review(updated_at=None)], []),
        ([review(user={"id": 2, "login": flow.BOT_LOGIN, "type": "Bot"})], []),
        ([review(user={"id": flow.BOT_ID, "login": flow.BOT_LOGIN, "type": "User"})], []),
        ([review(body="Codex Review: quota exhausted, cannot complete review")], []),
        ([review(body="Codex Review: usage limit reached")], []),
        ([review(body="Codex Review: couldn't review due to a service error")], []),
        ([review(body="Codex Review: something went wrong")], []),
        ([review(body="Codex Review: review was not performed")], []),
        ([review(body="Codex Review: [P1] Security issue")], []),
        ([review(body="To use Codex here, connect to GitHub")], []),
        ([review(body="")], []),
        ([review()], [{"isResolved": False}]),
        ([review(), review(id=13, state="DISMISSED")], []),
        ([review(), review(id=13, user={"id": 3}, state="CHANGES_REQUESTED")], []),
    ],
)
def test_absent_stale_spoofed_failed_or_blocking_reviews_fail(reviews, threads):
    assert not flow.review_verdict(SHA, reviews, threads)[0]


def test_resolved_threads_and_superseded_human_requests_allow_review():
    reviews = [
        review(),
        review(id=13, user={"id": 3}, state="CHANGES_REQUESTED"),
        review(id=14, user={"id": 3}, state="APPROVED"),
    ]
    assert flow.review_verdict(SHA, reviews, [{"isResolved": True}])[0]


def test_a_comment_does_not_withdraw_a_request_for_changes():
    reviews = [
        review(),
        review(id=13, user={"id": 3}, state="CHANGES_REQUESTED"),
        review(id=14, user={"id": 3}, state="COMMENTED"),
    ]
    assert not flow.review_verdict(SHA, reviews, [])[0]


def completion(**changes):
    # The native bot's actual clean-result format observed on dMotion PR #9.
    return {
        "id": 5984578402,
        "user": {"id": flow.BOT_ID, "login": flow.BOT_LOGIN, "type": "Bot"},
        "created_at": "2026-10-04T21:28:27Z",
        "updated_at": "2026-10-04T21:28:27Z",
        "body": f"Codex Review: Didn't find any major issues. Swish!\n\n"
        f"**Reviewed commit:** `{SHA[:10]}`\n\n<details>About Codex</details>",
        **changes,
    }


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        " Swish!",
        " Already looking forward to the next diff.",
        " What shall we delve into next?",
    ],
)
def test_native_clean_comment_binds_unambiguous_pr_commit_and_reaches_merge_gate(suffix):
    commits = [{"sha": SHA}, {"sha": BASE}]
    comment = completion(body=completion()["body"].replace(" Swish!", suffix))
    assert flow.review_verdict(SHA, [], [], [comment], commits)[0]
    client = inspection_client()
    client.routes["GET", "pulls/7/reviews"] = []
    client.routes["GET", "issues/7/comments"] = [comment]
    assert flow.inspect_pr(client, 7)["review_ok"]
    assert flow.merge_pr(client, 7)["merged"]


@pytest.mark.parametrize(
    "suffix",
    ["[P1] Fix this", " [P0] Fix this", " Review was not completed", " quota exhausted"],
)
def test_clean_prefix_with_unsuccessful_or_blocking_suffix_cannot_pass(suffix):
    comment = completion(body=completion()["body"].replace(" Swish!", suffix))
    assert not flow.review_verdict(SHA, [], [], [comment], [{"sha": SHA}])[0]


@pytest.mark.parametrize(
    "first_line",
    [
        "Quoted: Codex Review: Didn't find any major issues.",
        "> Codex Review: Didn't find any major issues.",
        "Codex Review: Didn't find any major issues",
        "Codex Review: Didn't find any major issues.Swish!",
    ],
)
def test_clean_result_requires_exact_sentence_and_suffix_boundary(first_line):
    body = completion()["body"].replace(
        "Codex Review: Didn't find any major issues. Swish!", first_line
    )
    assert not flow.review_verdict(SHA, [], [], [completion(body=body)], [{"sha": SHA}])[0]


def test_incomplete_commit_history_blocks_inspection_before_merge():
    client = inspection_client(commits=2)
    with pytest.raises(ValueError, match="Incomplete PR commit history"):
        flow.merge_pr(client, 7)
    assert not any(method == "PUT" for _, method, _ in client.calls)


@pytest.mark.parametrize(
    "comment,commits",
    [
        (completion(user={"id": 1, "login": flow.BOT_LOGIN, "type": "Bot"}), [{"sha": SHA}]),
        (completion(user={"id": flow.BOT_ID, "login": "other", "type": "Bot"}), [{"sha": SHA}]),
        (
            completion(user={"id": flow.BOT_ID, "login": flow.BOT_LOGIN, "type": "User"}),
            [{"sha": SHA}],
        ),
        (completion(created_at=None), [{"sha": SHA}]),
        (completion(updated_at=None), [{"sha": SHA}]),
        (completion(body="Codex Review: running"), [{"sha": SHA}]),
        (completion(body="👍"), [{"sha": SHA}]),
        (
            completion(body="To use Codex here, create a Codex account and connect to github"),
            [{"sha": SHA}],
        ),
        (completion(body=completion()["body"] + "\n[P1] Fix this"), [{"sha": SHA}]),
        (completion(body=completion()["body"] + "\nReview was not completed"), [{"sha": SHA}]),
        (
            completion(body=completion()["body"].replace(SHA[:10], BASE[:10])),
            [{"sha": BASE}, {"sha": SHA}],
        ),
        (completion(body=completion()["body"].replace(SHA[:10], SHA[:7])), [{"sha": SHA}]),
        (completion(body=completion()["body"] + f"\n**Reviewed commit:** `{SHA}`"), [{"sha": SHA}]),
        (completion(), []),
        (completion(), [{"sha": BASE}]),
        (completion(), [{"sha": SHA}, {"sha": SHA[:10] + "c" * 30}]),
    ],
)
def test_invalid_or_ambiguous_completion_comments_cannot_pass(comment, commits):
    assert not flow.review_verdict(SHA, [], [], [comment], commits)[0]


def test_exact_comment_sha_and_latest_bot_result_revocation():
    full = completion(body=completion()["body"].replace(SHA[:10], SHA))
    assert flow.review_verdict(SHA, [], [], [full])[0]
    error = completion(
        id=5984578403,
        updated_at="2026-10-04T21:30:00Z",
        body="Codex Review: quota exhausted, cannot complete review",
    )
    assert not flow.review_verdict(SHA, [], [], [full, error])[0]
    # Editing an older result must also invalidate its former completion.
    edited = completion(id=1, updated_at="2026-10-04T21:31:00Z", body="Codex Review: running")
    assert not flow.review_verdict(SHA, [], [], [full, edited])[0]
    assert not flow.review_verdict(SHA, [review()], [], [error])[0]


def test_clean_comment_cannot_override_formal_blockers_or_unresolved_threads():
    comments, commits = [completion()], [{"sha": SHA}]
    for state in ["PENDING", "DISMISSED", "CHANGES_REQUESTED"]:
        assert not flow.review_verdict(SHA, [review(state=state)], [], comments, commits)[0]
    assert not flow.review_verdict(SHA, [], [{"isResolved": False}], comments, commits)[0]
    human = review(user={"id": 3}, state="CHANGES_REQUESTED")
    assert not flow.review_verdict(SHA, [human], [], comments, commits)[0]


@pytest.mark.parametrize(
    "comment",
    [
        completion(body="Codex Review: running"),
        completion(body=completion()["body"].replace(SHA[:10], BASE[:10])),
        completion(updated_at=None),
        completion(body="Codex Review: unknown result"),
    ],
)
def test_newer_incomplete_result_revokes_a_valid_formal_review(comment):
    assert not flow.review_verdict(SHA, [review()], [], [comment], [{"sha": SHA}])[0]
    assert not flow.review_verdict(
        SHA, [review(state="APPROVED", body="")], [], [comment], [{"sha": SHA}]
    )[0]


def test_newer_explicit_clean_result_can_confirm_a_valid_formal_review():
    assert flow.review_verdict(SHA, [review()], [], [completion()], [{"sha": SHA}])[0]


@pytest.mark.parametrize("body", ["Codex Review: running", "Codex Review: unknown result"])
def test_same_second_incomplete_comment_cannot_be_ordered_before_formal_completion(body):
    timestamp = review()["submitted_at"]
    comment = completion(body=body, created_at=timestamp, updated_at=timestamp)
    assert not flow.review_verdict(SHA, [review()], [], [comment], [{"sha": SHA}])[0]


def test_earlier_failed_attempt_is_superseded_by_a_completed_formal_review():
    comment = completion(
        body="Codex Review: running",
        created_at="2026-10-04T11:59:59Z",
        updated_at="2026-10-04T11:59:59Z",
    )
    assert flow.review_verdict(SHA, [review()], [], [comment], [{"sha": SHA}])[0]


def test_same_second_edited_comment_cannot_be_ordered_by_creation_id():
    clean = completion()
    edited = completion(id=1, body="Codex Review: running")
    assert not flow.review_verdict(SHA, [], [], [clean, edited], [{"sha": SHA}])[0]
    assert not flow.review_verdict(SHA, [review()], [], [clean, edited], [{"sha": SHA}])[0]


@pytest.mark.parametrize("state", ["COMMENTED", "APPROVED", "PENDING", "DISMISSED"])
@pytest.mark.parametrize("current_result", ["formal", "comment"])
def test_late_formal_review_of_previous_head_revokes_current_completion(state, current_result):
    stale = review(id=13, commit_id=BASE, state=state, submitted_at="2026-10-04T21:29:00Z")
    reviews = [review(), stale] if current_result == "formal" else [stale]
    comments = [] if current_result == "formal" else [completion()]
    assert not flow.review_verdict(SHA, reviews, [], comments, [{"sha": SHA}])[0]


@pytest.mark.parametrize("current_result", ["formal", "comment"])
def test_current_completion_supersedes_earlier_stale_formal_result(current_result):
    stale = review(id=999, commit_id=BASE, submitted_at="2026-10-04T11:59:59Z")
    reviews = [stale, review()] if current_result == "formal" else [stale]
    comments = [] if current_result == "formal" else [completion()]
    assert flow.review_verdict(SHA, reviews, [], comments, [{"sha": SHA}])[0]


@pytest.mark.parametrize("current_result", ["formal", "comment"])
def test_same_second_stale_formal_result_blocks_even_with_lower_id(current_result):
    timestamp = review()["submitted_at"]
    stale = review(id=1, commit_id=BASE)
    reviews = [review(), stale] if current_result == "formal" else [stale]
    comments = (
        []
        if current_result == "formal"
        else [completion(created_at=timestamp, updated_at=timestamp)]
    )
    assert not flow.review_verdict(SHA, reviews, [], comments, [{"sha": SHA}])[0]


@pytest.mark.parametrize("head", [SHA, BASE])
@pytest.mark.parametrize("field", ["updated_at", "last_edited_at"])
def test_late_edit_of_formal_review_revokes_newer_clean_completion(head, field):
    edited = review(
        id=1,
        commit_id=head,
        body="Codex Review: [P1] Edited blocker",
        **{field: "2026-10-04T21:29:00Z"},
    )
    assert not flow.review_verdict(SHA, [edited, review()], [], [completion()], [{"sha": SHA}])[0]


def test_same_second_formal_edit_and_clean_comment_cannot_be_ordered_by_id():
    edited = review(
        id=1, body="Codex Review: [P1] Edited blocker", updated_at=completion()["updated_at"]
    )
    assert not flow.review_verdict(SHA, [edited], [], [completion()], [{"sha": SHA}])[0]


@pytest.mark.parametrize("change", [{"head_sha": BASE}, {"app": {"id": 2}}])
def test_ci_requires_exact_head_and_verified_actions_app(change):
    runs = checks()
    runs[0].update(change)
    assert not flow.ci_verdict(SHA, runs)[0]


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", "neutral", None])
def test_missing_or_unsuccessful_required_ci_blocks(result):
    assert not flow.ci_verdict(SHA, checks()[1:])[0]
    runs = checks()
    runs[0]["conclusion"] = result
    assert not flow.ci_verdict(SHA, runs)[0]
    runs[0].update(conclusion="success", status="in_progress")
    assert not flow.ci_verdict(SHA, runs)[0]


def test_merge_recollects_evidence_and_uses_atomic_head_guard():
    client = inspection_client()
    assert flow.merge_pr(client, 7)["merged"]
    assert client.calls[-1] == ("pulls/7/merge", "PUT", {"sha": SHA, "merge_method": "merge"})
    client.routes["GET", "pulls/7/reviews"] = [review(state="DISMISSED")]
    client.calls.clear()
    with pytest.raises(flow.NotReady):
        flow.merge_pr(client, 7)
    assert not any(method == "PUT" for _, method, _ in client.calls)


@pytest.mark.parametrize(
    "changes",
    [
        {"draft": True},
        {"state": "closed"},
        {"mergeable": None},
        {"mergeable_state": "behind"},
        {"body": "missing sections"},
        {"head": {"sha": SHA, "ref": "feature/fix", "repo": {"full_name": "fork/repo"}}},
    ],
)
def test_unsafe_merge_states_never_call_merge_api(changes):
    client = inspection_client(**changes)
    with pytest.raises(flow.NotReady):
        flow.merge_pr(client, 7)
    assert not any(method == "PUT" for _, method, _ in client.calls)


def test_github_refusal_and_head_races_propagate_without_bypass():
    client = inspection_client()
    client.routes["PUT", "pulls/7/merge"] = {"merged": False}
    with pytest.raises(ValueError, match="refused"):
        flow.merge_pr(client, 7)
    client.routes["PUT", "pulls/7/merge"] = subprocess.CalledProcessError(1, "gh", stderr="409")
    with pytest.raises(subprocess.CalledProcessError):
        flow.merge_pr(client, 7)


def test_gate_updates_only_its_own_check_and_revokes_old_success():
    client = inspection_client()
    evidence = flow.inspect_pr(client, 7)
    path = f"commits/{SHA}/check-runs?check_name=ai-review"
    client.routes["GET", path] = []
    client.routes["POST", "check-runs"] = {}
    flow.publish_gate(client, evidence)
    posted = client.calls[-1][2]
    assert posted["head_sha"] == SHA and posted["conclusion"] == "success"
    client.routes["GET", path] = [
        {"id": 1, "external_id": posted["external_id"], "app": {"id": flow.ACTIONS_APP}},
        {"id": 2, "external_id": "unrelated", "app": {"id": flow.ACTIONS_APP}},
    ]
    client.routes["PATCH", "check-runs/1"] = {}
    evidence.update(review_ok=False, review_reason="Dismissed")
    flow.publish_gate(client, evidence)
    assert client.calls[-1][0] == "check-runs/1"
    assert client.calls[-1][2]["conclusion"] == "failure"
    client.routes["GET", f"commits/{SHA}/check-runs?check_name=pr-policy"] = []
    evidence["policy_errors"] = ["No documentation"]
    flow.publish_gate(client, evidence, policy=True)
    assert client.calls[-1][2]["conclusion"] == "failure"


def test_token_created_pr_gets_explicit_ci_dispatch_without_repeated_runs():
    path = f"actions/runs?head_sha={SHA}"
    client = FakeGitHub(
        {
            ("GET", path): [{"name": "Checks", "status": "waiting"}],
            ("POST", "actions/workflows/checks.yml/dispatches"): None,
        }
    )
    flow.ensure_ci(client, pr())
    assert client.calls[-1][2] == {"ref": "feature/fix", "inputs": {"expected_sha": SHA}}
    for status in ["queued", "in_progress", "completed"]:
        client.routes["GET", path] = [{"name": "Checks", "status": status}]
        client.calls.clear()
        flow.ensure_ci(client, pr())
        assert len(client.calls) == 1


def test_approval_required_run_does_not_suppress_expected_sha_dispatch():
    path = f"actions/runs?head_sha={SHA}"
    client = FakeGitHub(
        {
            ("GET", path): [
                {"name": "Checks", "status": "completed", "conclusion": "action_required"}
            ],
            ("POST", "actions/workflows/checks.yml/dispatches"): None,
        }
    )
    flow.ensure_ci(client, pr())
    assert client.calls[-1][2]["inputs"]["expected_sha"] == SHA
    client.routes["GET", path].append({"name": "Checks", "status": "queued"})
    client.calls.clear()
    flow.ensure_ci(client, pr())
    assert len(client.calls) == 1


def test_new_base_is_merged_into_task_without_force_push():
    client = FakeGitHub(
        {
            ("GET", f"compare/{BASE}...{SHA}"): {"behind_by": 1},
            ("POST", "merges"): {},
            ("GET", "pulls/7"): pr(),
        }
    )
    flow.update_base(client, pr())
    assert client.calls[1] == ("merges", "POST", {"base": "feature/fix", "head": BASE})
    client.routes["GET", f"compare/{BASE}...{SHA}"] = {"behind_by": 0}
    client.calls.clear()
    flow.update_base(client, pr())
    assert len(client.calls) == 1


@pytest.mark.parametrize("behind", [0, 1])
@pytest.mark.parametrize("interrupted", [False, True])
def test_promotions_use_captured_commits_and_recover_an_interrupted_branch(behind, interrupted):
    branch = f"hotfix/sync-{BASE}" if behind else f"release/automation-{SHA}"
    source, integrate, target = (BASE, SHA, "develop") if behind else (SHA, BASE, "main")
    created = pr(head={"ref": branch, "sha": source}, base={"ref": target})
    client = FakeGitHub(
        {
            ("GET", "git/ref/heads/main"): {"object": {"sha": BASE}},
            ("GET", "git/ref/heads/develop"): {"object": {"sha": SHA}},
            ("GET", f"compare/{BASE}...{SHA}"): {"behind_by": behind, "files": ["file"]},
            ("GET", f"git/matching-refs/heads/{branch}"): [
                {"ref": f"refs/heads/{branch}", "object": {"sha": source}}
            ]
            if interrupted
            else [],
            ("POST", "git/refs"): {},
            ("POST", "merges"): {},
            ("POST", "pulls"): created,
            ("POST", "issues/7/labels"): {},
            ("GET", f"actions/runs?head_sha={source}"): [],
            ("POST", "actions/workflows/checks.yml/dispatches"): None,
        }
    )
    assert flow.create_promotion(client, []) == created
    assert ("merges", "POST", {"base": branch, "head": integrate}) in client.calls
    assert any(path == "git/refs" for path, _, _ in client.calls) is not interrupted
    body = next(data["body"] for path, _, data in client.calls if path == "pulls")
    assert not flow.validate_pull_request(target, branch, "ci: promote", body)
    # Existing managed promotions never create duplicates.
    client.calls.clear()
    assert flow.create_promotion(client, [created]) is None
    assert not client.calls


def test_aligned_branches_do_not_create_empty_release_prs():
    client = FakeGitHub(
        {
            ("GET", "git/ref/heads/main"): {"object": {"sha": BASE}},
            ("GET", "git/ref/heads/develop"): {"object": {"sha": SHA}},
            ("GET", f"compare/{BASE}...{SHA}"): {"behind_by": 0, "files": []},
        }
    )
    assert flow.create_promotion(client, []) is None
    assert all(method == "GET" for _, method, _ in client.calls)


def test_pulse_preserves_unrelated_and_draft_prs(monkeypatch):
    client = FakeGitHub({("GET", "pulls?state=open"): [pr(labels=[]), pr(number=8, draft=True)]})
    inspect = Mock(return_value={"pr": pr(), "head_sha": SHA, "review_reason": "pending"})
    monkeypatch.setattr(flow, "inspect_pr", inspect)
    publish = Mock()
    monkeypatch.setattr(flow, "publish_gate", publish)
    monkeypatch.setattr(flow, "create_promotion", lambda *_: None)
    monkeypatch.setattr(flow, "update_base", Mock(side_effect=AssertionError("updated unrelated")))
    monkeypatch.setattr(flow, "merge_pr", Mock(side_effect=AssertionError("merged unrelated")))
    assert len(flow.pulse(client)) == 2
    assert publish.call_count == 4


@pytest.mark.parametrize("raced", [False, True])
def test_managed_pulse_merges_only_after_a_fresh_read_and_handles_readiness_wait(
    monkeypatch, raced
):
    client = FakeGitHub({("GET", "pulls?state=open"): [pr()]})
    evidence = {
        "pr": pr(),
        "head_sha": SHA,
        "review_reason": "complete",
        "review_ok": True,
        "ci_ok": True,
        "policy_errors": [],
    }
    monkeypatch.setattr(flow, "inspect_pr", Mock(return_value=evidence))
    monkeypatch.setattr(flow, "publish_gate", Mock())
    monkeypatch.setattr(flow, "update_base", Mock())
    dispatch = Mock()
    monkeypatch.setattr(flow, "ensure_ci", dispatch)
    merge = (
        Mock(side_effect=flow.NotReady("head advanced"))
        if raced
        else Mock(return_value={"sha": BASE})
    )
    monkeypatch.setattr(flow, "merge_pr", merge)
    monkeypatch.setattr(flow, "create_promotion", lambda *_: pr())
    results = flow.pulse(client)
    assert "waiting" in results[0] if raced else results[0]["merged"] == BASE
    assert results[1]["created"].endswith("/7")
    merge.assert_called_once_with(client, 7)
    dispatch.assert_called_once()


def test_unexpected_branch_collision_never_creates_or_rewrites_pr():
    branch = f"release/automation-{SHA}"
    client = FakeGitHub(
        {
            ("GET", "git/ref/heads/main"): {"object": {"sha": BASE}},
            ("GET", "git/ref/heads/develop"): {"object": {"sha": SHA}},
            ("GET", f"compare/{BASE}...{SHA}"): {"behind_by": 0, "files": ["file"]},
            ("GET", f"git/matching-refs/heads/{branch}"): [
                {"ref": f"refs/heads/{branch}", "object": {"sha": "c" * 40}}
            ],
            ("GET", f"git/commits/{'c' * 40}"): {"parents": [{"sha": "d" * 40}]},
        }
    )
    with pytest.raises(ValueError, match="collision"):
        flow.create_promotion(client, [])
    assert all(method == "GET" for _, method, _ in client.calls)


def test_api_uses_json_stdin_and_complete_pagination(monkeypatch):
    response = subprocess.CompletedProcess([], 0, stdout='{"accepted": true}')
    run = Mock(return_value=response)
    monkeypatch.setattr(flow.subprocess, "run", run)
    client = flow.GitHub("owner/repo")
    assert client.api("pulls", method="POST", data={"body": "`$(private)`\nnew line"})
    assert json.loads(run.call_args.kwargs["input"])["body"].endswith("\nnew line")
    api = Mock(side_effect=[{"items": [1] * 100}, {"items": [2]}])
    monkeypatch.setattr(client, "api", api)
    assert len(client.pages("example?filter=latest", "items")) == 101
    assert api.call_args_list[1].args[0].endswith("&per_page=100&page=2")
    api.side_effect = None
    api.return_value = [1] * 100
    with pytest.raises(ValueError, match="Pagination"):
        client.pages("example")
    with pytest.raises(ValueError):
        flow.GitHub("../bad/repo")


def test_review_thread_pagination_errors_fail_closed(monkeypatch):
    def response(next_page):
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviewThreads": {
                                    "nodes": [{"isResolved": True}],
                                    "pageInfo": {"hasNextPage": next_page, "endCursor": "cursor"},
                                }
                            }
                        }
                    }
                }
            ),
        )

    run = Mock(side_effect=[response(True), response(False)])
    monkeypatch.setattr(flow.subprocess, "run", run)
    assert len(flow.GitHub("owner/repo").threads(7)) == 2
    assert json.loads(run.call_args.kwargs["input"])["variables"]["cursor"] == "cursor"
    run.side_effect = None
    run.return_value = subprocess.CompletedProcess([], 0, stdout='{"errors": ["denied"]}')
    with pytest.raises(ValueError, match="query failed"):
        flow.GitHub("owner/repo").threads(7)


def review_nodes(items):
    return subprocess.CompletedProcess(
        [],
        0,
        stdout=json.dumps(
            {
                "data": {
                    "nodes": [
                        {
                            "id": item["node_id"],
                            "updatedAt": item["updated_at"],
                            "lastEditedAt": item.get("last_edited_at"),
                            "submittedAt": item["submitted_at"],
                            "state": item["state"],
                            "body": item["body"],
                            "commit": {"oid": item["commit_id"]},
                        }
                        for item in items
                    ]
                }
            }
        ),
    )


def test_review_reader_batches_nodes_and_retains_edit_evidence(monkeypatch):
    items = [
        review(
            id=index,
            node_id=f"node-{index}",
            updated_at="2026-10-04T21:29:00Z",
            last_edited_at="2026-10-04T21:28:00Z",
        )
        for index in range(101)
    ]
    rest = [
        {key: value for key, value in item.items() if key not in {"updated_at", "last_edited_at"}}
        for item in items
    ]
    client = flow.GitHub("owner/repo")
    monkeypatch.setattr(client, "pages", Mock(return_value=rest))
    run = Mock(side_effect=[review_nodes(items[:100]), review_nodes(items[100:])])
    monkeypatch.setattr(flow.subprocess, "run", run)
    assert client.reviews(7) == items
    assert json.loads(run.call_args.kwargs["input"])["variables"]["ids"] == ["node-100"]


@pytest.mark.parametrize(
    "change",
    [
        {"body": "Changed during read"},
        {"state": "DISMISSED"},
        {"commit_id": BASE},
        {"submitted_at": "2026-10-04T13:00:00Z"},
        {"updated_at": None},
        {"node_id": "other"},
    ],
)
def test_review_reader_rejects_changed_or_incomplete_edit_evidence(monkeypatch, change):
    item = review(node_id="node-12")
    client = flow.GitHub("owner/repo")
    monkeypatch.setattr(client, "pages", Mock(return_value=[item]))
    monkeypatch.setattr(
        flow.subprocess, "run", Mock(return_value=review_nodes([{**item, **change}]))
    )
    with pytest.raises(ValueError):
        client.reviews(7)


@pytest.mark.parametrize(
    "stdout", ['{"errors":["denied"]}', '{"data":{"nodes":[null]}}', '{"data":{"nodes":[]}}']
)
def test_review_reader_fails_closed_for_unavailable_nodes(monkeypatch, stdout):
    client = flow.GitHub("owner/repo")
    monkeypatch.setattr(client, "pages", Mock(return_value=[review(node_id="node-12")]))
    monkeypatch.setattr(
        flow.subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, stdout=stdout))
    )
    with pytest.raises(ValueError):
        client.reviews(7)


def test_review_reader_requires_node_identity_but_skips_graphql_for_no_reviews(monkeypatch):
    client = flow.GitHub("owner/repo")
    pages = Mock(return_value=[review()])
    run = Mock()
    monkeypatch.setattr(client, "pages", pages)
    monkeypatch.setattr(flow.subprocess, "run", run)
    with pytest.raises(ValueError, match="identities"):
        client.reviews(7)
    pages.return_value = []
    assert client.reviews(7) == []
    run.assert_not_called()


def commit_page(start, stop, total, cursor=None):
    return subprocess.CompletedProcess(
        [],
        0,
        stdout=json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "commits": {
                                "totalCount": total,
                                "nodes": [
                                    {"commit": {"oid": f"{index:040x}"}}
                                    for index in range(start, stop)
                                ],
                                "pageInfo": {
                                    "hasNextPage": cursor is not None,
                                    "endCursor": cursor,
                                },
                            }
                        }
                    }
                }
            }
        ),
    )


def test_graphql_reads_complete_pr_history_beyond_rest_cap(monkeypatch):
    run = Mock(
        side_effect=[
            commit_page(0, 100, 251, "page1"),
            commit_page(100, 200, 251, "page2"),
            commit_page(200, 251, 251),
        ]
    )
    monkeypatch.setattr(flow.subprocess, "run", run)
    commits = flow.GitHub("owner/repo").commits(7)
    assert len(commits) == 251 and commits[-1]["sha"] == f"{250:040x}"
    assert json.loads(run.call_args.kwargs["input"])["variables"]["cursor"] == "page2"


@pytest.mark.parametrize(
    "pages",
    [
        [commit_page(0, 1, 2)],
        [commit_page(0, 1, 2, "page1"), commit_page(0, 1, 2)],
        [commit_page(0, 1, 2, "page1"), commit_page(1, 2, 3)],
        [commit_page(0, 1, 3, "page1"), commit_page(1, 2, 3, "page1")],
        [subprocess.CompletedProcess([], 0, stdout='{"errors": ["denied"]}')],
    ],
)
def test_graphql_history_rejects_incomplete_duplicate_raced_or_failed_pages(monkeypatch, pages):
    monkeypatch.setattr(flow.subprocess, "run", Mock(side_effect=pages))
    with pytest.raises(ValueError):
        flow.GitHub("owner/repo").commits(7)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Incomplete history"),
        subprocess.CalledProcessError(1, ["gh", "api"]),
        subprocess.TimeoutExpired(["gh", "api"], 60),
    ],
)
def test_one_pr_evidence_failure_revokes_its_gate_and_other_prs_still_progress(monkeypatch, error):
    blocked, healthy = pr(number=6), pr()
    client = FakeGitHub({("GET", "pulls?state=open"): [blocked, healthy]})
    evidence = {
        "pr": healthy,
        "head_sha": SHA,
        "review_reason": "complete",
        "review_ok": True,
        "ci_ok": True,
        "policy_errors": [],
    }
    monkeypatch.setattr(flow, "inspect_pr", Mock(side_effect=[error, evidence]))
    publish = Mock()
    monkeypatch.setattr(flow, "publish_gate", publish)
    monkeypatch.setattr(flow, "update_base", Mock())
    monkeypatch.setattr(flow, "ensure_ci", Mock())
    merge = Mock(return_value={"sha": BASE})
    monkeypatch.setattr(flow, "merge_pr", merge)
    monkeypatch.setattr(flow, "create_promotion", lambda *_: None)
    results = flow.pulse(client)
    assert results[0]["number"] == 6 and str(error) in results[0]["blocked"]
    assert not publish.call_args_list[0].args[1]["review_ok"]
    assert results[1]["merged"] == BASE
    merge.assert_called_once_with(client, 7)


def test_timed_out_gate_publication_does_not_stop_later_prs(monkeypatch):
    client = FakeGitHub({("GET", "pulls?state=open"): [pr(number=6), pr()]})
    monkeypatch.setattr(
        flow,
        "process_pr",
        Mock(
            side_effect=[
                subprocess.TimeoutExpired(["gh", "api"], 60),
                {"number": 7, "merged": BASE},
            ]
        ),
    )
    monkeypatch.setattr(
        flow, "publish_gate", Mock(side_effect=subprocess.TimeoutExpired(["gh", "api"], 60))
    )
    monkeypatch.setattr(flow, "create_promotion", Mock(return_value=None))
    results = flow.pulse(client)
    assert "timed out" in results[0]["blocked"]
    assert "timed out" in results[0]["gate_unavailable"]
    assert results[1]["merged"] == BASE


def request_evidence(client):
    client.routes["GET", "pulls/7/reviews"] = []
    client.routes["POST", "issues/7/comments"] = {"id": 42}
    return flow.inspect_pr(client, 7)


def test_review_request_binds_head_and_is_deduplicated_for_trusted_requester():
    client = inspection_client()
    evidence = request_evidence(client)
    assert flow.ensure_review(client, evidence) == {"id": 42}
    body = client.calls[-1][2]["body"]
    assert body.startswith("@codex review\n") and SHA in body
    assert f"<!-- dmotion-codex-review:{SHA} -->" in body
    for user, association in [
        ({"id": flow.ACTIONS_BOT_ID, "login": "github-actions[bot]", "type": "Bot"}, "NONE"),
        ({"id": 123, "login": "owner", "type": "User"}, "OWNER"),
    ]:
        client.routes["GET", "issues/7/comments"] = [
            {"body": body, "user": user, "author_association": association}
        ]
        client.calls.clear()
        assert flow.ensure_review(client, evidence) is None
        assert all(method == "GET" for _, method, _ in client.calls)


@pytest.mark.parametrize(
    "user,association",
    [
        ({"id": 1, "login": "github-actions[bot]", "type": "Bot"}, "NONE"),
        ({"id": 1, "login": "outside", "type": "User"}, "NONE"),
    ],
)
def test_untrusted_marker_cannot_suppress_native_review_request(user, association):
    client = inspection_client()
    evidence = request_evidence(client)
    client.routes["GET", "issues/7/comments"] = [
        {
            "body": f"<!-- dmotion-codex-review:{SHA} -->",
            "user": user,
            "author_association": association,
        }
    ]
    assert flow.ensure_review(client, evidence)["id"] == 42


@pytest.mark.parametrize(
    "change",
    [
        {"draft": True},
        {"labels": []},
        {"state": "closed"},
        {"head": {"sha": SHA, "ref": "feature/fix", "repo": {"full_name": "fork/repo"}}},
    ],
)
def test_review_requests_preserve_drafts_closed_unrelated_and_fork_prs(change):
    client = inspection_client(**change)
    evidence = request_evidence(client)
    client.calls.clear()
    assert flow.ensure_review(client, evidence) is None
    assert not client.calls


@pytest.mark.parametrize(
    "change",
    [
        {"review_ok": True},
        {"policy_errors": ["invalid route"]},
        {"review_reason": "Unresolved review conversations remain"},
        {"review_reason": "A reviewer requested changes"},
        {"review_reason": "Blocking findings remain in the current review summary"},
    ],
)
def test_completed_review_and_actionable_blockers_do_not_cause_repeated_requests(change):
    client = inspection_client()
    evidence = {**request_evidence(client), **change}
    client.calls.clear()
    assert flow.ensure_review(client, evidence) is None
    assert not client.calls


def test_quoted_marker_without_a_native_request_cannot_suppress_review():
    client = inspection_client()
    evidence = request_evidence(client)
    client.routes["GET", "issues/7/comments"] = [
        {
            "body": f"Quoted marker <!-- dmotion-codex-review:{SHA} -->",
            "user": {"id": 123, "type": "User"},
            "author_association": "OWNER",
        }
    ]
    assert flow.ensure_review(client, evidence)["id"] == 42


@pytest.mark.parametrize("change", [{"draft": True}, {"labels": []}])
def test_review_request_rechecks_draft_and_opt_in_before_posting(change):
    client = inspection_client()
    evidence = request_evidence(client)
    client.routes["GET", "pulls/7"].update(change)
    with pytest.raises(flow.NotReady):
        flow.ensure_review(client, evidence)
    assert all(method == "GET" for _, method, _ in client.calls)


def test_new_head_gets_its_own_review_request_and_raced_head_is_not_requested():
    client = inspection_client()
    evidence = request_evidence(client)
    client.routes["GET", "issues/7/comments"] = [
        {
            "body": f"<!-- dmotion-codex-review:{BASE} -->",
            "user": {"id": 123, "type": "User"},
            "author_association": "OWNER",
        }
    ]
    assert flow.ensure_review(client, evidence)["id"] == 42
    client.routes["GET", "pulls/7"]["head"]["sha"] = BASE
    client.calls.clear()
    with pytest.raises(flow.NotReady, match="advanced"):
        flow.ensure_review(client, evidence)
    assert all(method == "GET" for _, method, _ in client.calls)


def test_explicit_request_command_posts_without_publishing_gate_or_merging(monkeypatch, capsys):
    client = inspection_client()
    request_evidence(client)
    monkeypatch.setattr(flow, "GitHub", lambda _: client)
    monkeypatch.setattr(
        sys, "argv", ["flow", "--repo", "owner/repo", "--pr", "7", "--request-review"]
    )
    assert flow.main() == 0
    assert json.loads(capsys.readouterr().out)["review_request"] == 42
    assert [path for path, method, _ in client.calls if method != "GET"] == ["issues/7/comments"]


def test_read_only_command_does_not_publish_or_merge(monkeypatch, capsys):
    client = inspection_client()
    monkeypatch.setattr(flow, "GitHub", lambda _: client)
    monkeypatch.setattr(sys, "argv", ["flow", "--repo", "owner/repo", "--pr", "7"])
    assert flow.main() == 0
    assert json.loads(capsys.readouterr().out)["review_ok"]
    assert all(method == "GET" for _, method, _ in client.calls)
