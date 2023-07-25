# pylint: disable=wrong-or-nonexistent-copyright-notice

"""Code to interact with GitHub API to label and auto-merge pull requests."""

import datetime
import json
import os
import sys
import time
import traceback
from typing import Callable, Optional, List, Any, Dict, Set, Union

from google.cloud import secretmanager_v1beta1

from dev_tools.github_repository import GithubRepository

GITHUB_REPO_NAME = 'cirq'
GITHUB_REPO_ORGANIZATION = 'quantumlib'
ACCESS_TOKEN_ENV_VARIABLE = 'CIRQ_BOT_GITHUB_ACCESS_TOKEN'

POLLING_PERIOD = datetime.timedelta(seconds=10)
USER_AUTO_MERGE_LABEL = 'automerge'
HEAD_AUTO_MERGE_LABEL = 'front_of_queue_automerge'
AUTO_MERGE_LABELS = [USER_AUTO_MERGE_LABEL, HEAD_AUTO_MERGE_LABEL]
RECENTLY_MODIFIED_THRESHOLD = datetime.timedelta(seconds=30)

PR_SIZE_LABELS = ['size: U', 'size: XS', 'size: S', 'size: M', 'size: L', 'size: XL']
PR_SIZES = [0, 10, 50, 250, 1000, 1 << 30]


def get_pr_size_label(tot_changes: int) -> str:
    i = 0
    ret = ''
    while i < len(PR_SIZES):
        if tot_changes < PR_SIZES[i]:
            ret = PR_SIZE_LABELS[i]
            break
        i += 1
    return ret


def is_recent_date(date: datetime.datetime) -> bool:
    d = datetime.datetime.utcnow() - date
    return d < RECENTLY_MODIFIED_THRESHOLD


class CannotAutomergeError(RuntimeError):
    def __init__(self, *args, may_be_temporary: bool = False):
        super().__init__(*args)
        self.may_be_temporary = may_be_temporary


class PullRequestDetails:
    def __init__(self, payload: Any, repo: GithubRepository) -> None:
        self.payload = payload
        self.repo = repo

    @staticmethod
    def from_github(repo: GithubRepository, pull_id: int) -> 'PullRequestDetails':
        """Retrieves a single pull request.

        References:
            https://developer.github.com/v3/pulls/#get-a-single-pull-request

        Args:
            repo: The github repo to get the pull request from.
            pull_id: The id of the pull request.

        Raises:
            RuntimeError: If the request does not return status 200 (success).
        """
        url = f"https://api.github.com/repos/{repo.organization}/{repo.name}/pulls/{pull_id}"

        response = repo.get(url)

        if response.status_code != 200:
            raise RuntimeError(
                f'Pull check failed. Code: {response.status_code}. Content: {response.content!r}.'
            )

        payload = json.JSONDecoder().decode(response.content.decode())
        return PullRequestDetails(payload, repo)

    @property
    def remote_repo(self) -> GithubRepository:
        """Return the GithubRepository corresponding to this pull request."""
        return GithubRepository(
            organization=self.payload['head']['repo']['owner']['login'],
            name=self.payload['head']['repo']['name'],
            access_token=self.repo.access_token,
        )

    def is_on_fork(self) -> bool:
        local = (self.repo.organization.lower(), self.repo.name.lower())
        remote = (self.remote_repo.organization.lower(), self.remote_repo.name.lower())
        return local != remote

    def has_label(self, desired_label: str) -> bool:
        return any(label['name'] == desired_label for label in self.payload['labels'])

    @property
    def last_updated(self) -> datetime.datetime:
        return datetime.datetime.strptime(self.payload['updated_at'], '%Y-%m-%dT%H:%M:%SZ')

    @property
    def modified_recently(self) -> bool:
        return is_recent_date(self.last_updated)

    @property
    def marked_automergeable(self) -> bool:
        return any(self.has_label(label) for label in AUTO_MERGE_LABELS)

    @property
    def marked_size(self) -> bool:
        return any(self.has_label(label) for label in PR_SIZE_LABELS)

    @property
    def pull_id(self) -> int:
        return self.payload['number']

    @property
    def branch_name(self) -> str:
        return self.payload['head']['ref']

    @property
    def base_branch_name(self) -> str:
        return self.payload['base']['ref']

    @property
    def branch_sha(self) -> str:
        return self.payload['head']['sha']

    @property
    def title(self) -> str:
        return self.payload['title']

    @property
    def body(self) -> str:
        return self.payload['body']

    @property
    def additions(self) -> int:
        return int(self.payload['additions'])

    @property
    def deletions(self) -> int:
        return int(self.payload['deletions'])

    @property
    def tot_changes(self) -> int:
        return self.deletions + self.additions


def check_collaborator_has_write(
    repo: GithubRepository, username: str
) -> Optional[CannotAutomergeError]:
    """Checks whether the given user is a collaborator (admin and write access).

    References:
        https://developer.github.com/v3/issues/events/#list-events-for-an-issue

    Args:
        repo: The github repo to check.
        username: The github username to check whether the user is a collaborator.

    Returns:
        CannotAutomergeError if the user does not have admin and write permissions and so
            cannot use automerge, None otherwise.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """
    url = (
        f"https://api.github.com/repos/{repo.organization}/"
        f"{repo.name}/collaborators/{username}/permission"
    )

    response = repo.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            f'Collaborator check failed. Code: {response.status_code}. '
            f'Content: {response.content!r}.'
        )

    payload = json.JSONDecoder().decode(response.content.decode())
    if payload['permission'] not in ['admin', 'write']:
        return CannotAutomergeError('Only collaborators with write permission can use automerge.')

    return None


def get_all(repo: GithubRepository, url_func: Callable[[int], str]) -> List[Any]:
    """Get all results, accounting for pagination.

    Args:
        repo: The github repo to call GET on.
        url_func: A function from an integer page number to the url to get the result for that page.

    Returns:
        A list of the results by page.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """
    results: List[Any] = []
    page = 0
    has_next = True
    while has_next:
        url = url_func(page)
        response = repo.get(url)

        if response.status_code != 200:
            raise RuntimeError(
                f'Request failed to {url}. Code: {response.status_code}.'
                f' Content: {response.content!r}.'
            )

        payload = json.JSONDecoder().decode(response.content.decode())
        results += payload
        has_next = 'link' in response.headers and 'rel="next"' in response.headers['link']
        page += 1
    return results


def check_auto_merge_labeler(
    repo: GithubRepository, pull_id: int
) -> Optional[CannotAutomergeError]:
    """Checks whether the given pull request had an automerge id and user who added it was admin.

    References:
        https://developer.github.com/v3/issues/events/#list-events-for-an-issue

    Args:
        repo: The github repo to check.
        pull_id: The github pull id to check.

    Returns:
        CannotAutomergeError if the automerge iid is missing or the user who added is not an admin.
    """
    events = get_all(
        repo,
        lambda page: (
            f"https://api.github.com/repos/{repo.organization}/"
            f"{repo.name}/issues/{pull_id}/events?per_page=100&page={page}"
        ),
    )

    relevant = [
        event
        for event in events
        if event['event'] == 'labeled' and event['label']['name'] in AUTO_MERGE_LABELS
    ]
    if not relevant:
        return CannotAutomergeError('"automerge" label was never added.')

    return check_collaborator_has_write(repo, relevant[-1]['actor']['login'])


def add_comment(repo: GithubRepository, pull_id: int, text: str) -> None:
    """Add a comment to a pull request.

    References:
        https://developer.github.com/v3/issues/comments/#create-a-comment

    Arg:
        rep: The github repo whose pull request should have a comment added to.
        pull_id: The id of the pull request to comment on.
        text: The text of the comment.

    Raises:
        RuntimeError: If the request does not return status 201 (created).
    """
    url = (
        f"https://api.github.com/repos/{repo.organization}/"
        f"{repo.name}/issues/{pull_id}/comments"
    )
    data = {'body': text}
    response = repo.post(url, json=data)

    if response.status_code != 201:
        raise RuntimeError(
            f'Add comment failed. Code: {response.status_code}. Content: {response.content!r}.'
        )


def edit_comment(repo: GithubRepository, text: str, comment_id: int) -> None:
    """Edits an existing github comment.

    References:
        https://developer.github.com/v3/issues/comments/#edit-a-comment

    Args:
        repo: The github repo that contains the comment.
        text: The new comment text.
        comment_id: The id of the comment to edit.

    Raises:
            RuntimeError: If the request does not return status 200 (success).
    """
    url = (
        f"https://api.github.com/repos/{repo.organization}/"
        f"{repo.name}/issues/comments/{comment_id}"
    )
    data = {'body': text}
    response = repo.patch(url, json=data)

    if response.status_code != 200:
        raise RuntimeError(
            f'Edit comment failed. Code: {response.status_code}. Content: {response.content!r}.'
        )


def get_branch_details(repo: GithubRepository, branch: str) -> Any:
    """Get details about a github branch.

    References:
        https://developer.github.com/v3/repos/branches/#get-branch

    Args:
        repo: The github repo that has the branch.
        branch: The name of the branch.

    Returns:
        The raw response to the query to get details.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """
    url = f"https://api.github.com/repos/{repo.organization}/{repo.name}/branches/{branch}"
    response = repo.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            f'Failed to get branch details. Code: {response.status_code}. '
            f'Content: {response.content!r}.'
        )

    return json.JSONDecoder().decode(response.content.decode())


def get_pr_statuses(pr: PullRequestDetails) -> List[Dict[str, Any]]:
    """List the commit statuses of a specific pull request.

    References:
        https://developer.github.com/v3/repos/statuses/#list-statuses-for-a-specific-ref

    Args:
        pr: The pull request details.

    Returns:
        The raw response to the request.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """

    url = (
        f"https://api.github.com/repos/{pr.repo.organization}/"
        f"{pr.repo.name}/commits/{pr.branch_sha}/statuses"
    )
    response = pr.repo.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            f'Get statuses failed. Code: {response.status_code}. Content: {response.content!r}.'
        )

    return json.JSONDecoder().decode(response.content.decode())


def get_pr_check_status(pr: PullRequestDetails) -> Any:
    """Get the combined status for a pull request.

    References:
        https://developer.github.com/v3/repos/statuses/#get-the-combined-status-for-a-specific-ref

    Args:
        pr: The pull request details.

    Returns:
        The raw response to the request.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """

    url = (
        f"https://api.github.com/repos/{pr.repo.organization}/"
        f"{pr.repo.name}/commits/{pr.branch_sha}/status"
    )
    response = pr.repo.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            f'Get status failed. Code: {response.status_code}. Content: {response.content!r}.'
        )

    return json.JSONDecoder().decode(response.content.decode())


def classify_pr_status_check_state(pr: PullRequestDetails) -> Optional[bool]:
    """Classify the pull request status.

    Args:
        pr: The pull request whose status should be checked.

    Returns:
        True if the status is successful, False if the status has failed, and None if the
        status is pending.

    Raises:
        RuntimeError: If the status state is of an unknown type.
    """
    has_failed = False
    has_pending = False

    check_status = get_pr_check_status(pr)
    state = check_status['state']
    if state == 'failure':
        has_failed = True
    elif state == 'pending':
        has_pending = True
    elif state != 'success':
        raise RuntimeError(f'Unrecognized status state: {state!r}')

    check_data = get_pr_checks(pr)
    for check in check_data['check_runs']:
        if check['status'] != 'completed':
            has_pending = True
        elif check['conclusion'] != 'success':
            has_failed = True

    if has_failed:
        return False
    if has_pending:
        return None
    return True


def classify_pr_synced_state(pr: PullRequestDetails) -> Optional[bool]:
    """Get the mergeable state of the pull request.

    References:
        https://developer.github.com/v3/pulls/#get-a-single-pull-request
        https://developer.github.com/v4/enum/mergestatestatus/

    Args:
        pr: The pull request to query for mergable state.

    Returns:
        True if the classification is clean, False if it is behind, and None otherwise.
    """
    state = pr.payload['mergeable_state'].lower()
    classification = {'behind': False, 'clean': True}
    return classification.get(state, None)


def get_pr_review_status(pr: PullRequestDetails, per_page: int = 100) -> Any:
    """Gets the review status of the pull request.

    References:
        https://developer.github.com/v3/pulls/reviews/#list-reviews-on-a-pull-request

    Args:
        pr: The pull reuqest whose review status will be checked.
        per_page: The number of results to return per page.

    Returns:
        The full response from the review query.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """
    url = (
        f"https://api.github.com/repos/{pr.repo.organization}/{pr.repo.name}"
        f"/pulls/{pr.pull_id}/reviews"
        f"?per_page={per_page}"
    )
    response = pr.repo.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            f'Get review failed. Code: {response.status_code}. Content: {response.content!r}.'
        )

    return json.JSONDecoder().decode(response.content.decode())


def get_pr_checks(pr: PullRequestDetails) -> Dict[str, Any]:
    """List checks for a pull request.

    References:
        https://developer.github.com/v3/checks/runs/#list-check-runs-for-a-specific-ref

    Args:
        pr: The pull request to get checks for.

    Returns:
        The raw response of the request.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """
    url = (
        f"https://api.github.com/repos/{pr.repo.organization}/{pr.repo.name}"
        f"/commits/{pr.branch_sha}/check-runs?per_page=100"
    )
    response = pr.repo.get(url, headers={'Accept': 'application/vnd.github.antiope-preview+json'})

    if response.status_code != 200:
        raise RuntimeError(
            f'Get check-runs failed. Code: {response.status_code}. Content: {response.content!r}.'
        )

    return json.JSONDecoder().decode(response.content.decode())


_last_print_was_tick = False
_tick_count = 0


def log(*args):
    global _last_print_was_tick
    if _last_print_was_tick:
        print()
    _last_print_was_tick = False
    print(*args)


def wait_for_polling_period():
    global _last_print_was_tick
    global _tick_count
    _last_print_was_tick = True
    print('.', end='', flush=True)
    _tick_count += 1
    if _tick_count == 100:
        print()
        _tick_count = 0
    time.sleep(POLLING_PERIOD.total_seconds())


def absent_status_checks(pr: PullRequestDetails, master_data: Optional[Any] = None) -> Set[str]:
    if pr.base_branch_name == 'master' and master_data is not None:
        branch_data = master_data
    else:
        branch_data = get_branch_details(pr.repo, pr.base_branch_name)
    status_data = get_pr_statuses(pr)
    check_data = get_pr_checks(pr)

    statuses_present = {status['context'] for status in status_data}
    checks_present = {check['name'] for check in check_data['check_runs']}
    reqs = branch_data['protection']['required_status_checks']['contexts']
    return set(reqs) - statuses_present - checks_present


def get_repo_ref(repo: GithubRepository, ref: str) -> Dict[str, Any]:
    """Get a given github reference.

    References:
        https://developer.github.com/v3/git/refs/#get-a-reference

    Args:
        repo: The github repo to get the reference from.
        ref: The id of the reference.

    Returns:
        The raw response of the request for the reference..

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """

    url = f"https://api.github.com/repos/{repo.organization}/{repo.name}/git/refs/{ref}"
    response = repo.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            f'Refs get failed. Code: {response.status_code}. Content: {response.content!r}.'
        )
    payload = json.JSONDecoder().decode(response.content.decode())
    return payload


def get_master_sha(repo: GithubRepository) -> str:
    """Get the sha hash for the given repo."""
    ref = get_repo_ref(repo, 'heads/master')
    return ref['object']['sha']


def list_pr_comments(repo: GithubRepository, pull_id: int) -> List[Dict[str, Any]]:
    """List comments for a given pull request.

    References:
        https://developer.github.com/v3/issues/comments/#list-comments-on-an-issue

    Args:
        repo: The github repo for the pull request.
        pull_id: The id of the pull request.

    Returns:
        A list of the raw responses for the pull requests.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """
    url = (
        f"https://api.github.com/repos/{repo.organization}/"
        f"{repo.name}/issues/{pull_id}/comments"
    )
    response = repo.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            f'Comments get failed. Code: {response.status_code}. Content: {response.content!r}.'
        )
    payload = json.JSONDecoder().decode(response.content.decode())
    return payload


def delete_comment(repo: GithubRepository, comment_id: int) -> None:
    """Delete a comment.

    References:
        https://developer.github.com/v3/issues/comments/#delete-a-comment

    Args:
        repo: The github repo where the comment lives.
        comment_id: The id of the comment to delete.

    Raises:
        RuntimeError: If the request does not return status 204 (no content).
    """
    url = (
        f"https://api.github.com/repos/{repo.organization}/"
        f"{repo.name}/issues/comments/{comment_id}"
    )
    response = repo.delete(url)
    if response.status_code != 204:
        raise RuntimeError(
            f'Comment delete failed. Code: {response.status_code}. Content: {response.content!r}.'
        )


def update_branch(pr: PullRequestDetails) -> Union[bool, CannotAutomergeError]:
    """Equivalent to hitting the 'update branch' button on a PR.

    As of Feb 2020 this API feature is still in beta. Note that currently, if
    you attempt to update branch when already synced to master, a vacuous merge
    commit will be created.

    References:
        https://developer.github.com/v3/pulls/#update-a-pull-request-branch

    Args:
        pr: The pull request to update.

    Returns:
        True if the update was successful and CannotAutomergeError if it is not possible to
        perform the update.
    """
    url = (
        f"https://api.github.com/repos/{pr.repo.organization}/{pr.repo.name}"
        f"/pulls/{pr.pull_id}/update-branch"
    )
    data = {'expected_head_sha': pr.branch_sha}
    response = pr.repo.put(
        url,
        json=data,
        # Opt into BETA feature.
        headers={'Accept': 'application/vnd.github.lydian-preview+json'},
    )

    if response.status_code == 422:
        return CannotAutomergeError(
            "Failed to update branch (incorrect expected_head_sha).", may_be_temporary=True
        )
    if response.status_code != 202:
        return CannotAutomergeError(
            f"Unrecognized update-branch status code ({response.status_code})."
        )

    return True


def attempt_sync_with_master(pr: PullRequestDetails) -> Union[bool, CannotAutomergeError]:
    """Sync a pull request with the master branch.

    References:
        https://developer.github.com/v3/repos/merging/#perform-a-merge

    Args:
        pr: The pull request to sync.

    Returns:
        True if the sync was successful and CannotAutomergeError if it was not possible to sync.

    Raises:
        RuntimeError: If the merge request returned a failed response.
    """
    master_sha = get_master_sha(pr.repo)
    remote = pr.remote_repo
    url = f"https://api.github.com/repos/{remote.organization}/{remote.name}/merges"
    data = {
        'base': pr.branch_name,
        'head': master_sha,
        'commit_message': 'Update branch (automerge)',
    }
    response = pr.remote_repo.post(url, json=data)

    if response.status_code == 201:
        # Merge succeeded.
        log(f'Synced #{pr.pull_id} ({pr.title!r}) with master.')
        return True

    if response.status_code == 204:
        # Already merged.
        return False

    if response.status_code == 409:
        # Merge conflict.
        return CannotAutomergeError("There's a merge conflict.")

    if response.status_code == 403:
        # Permission denied.
        return CannotAutomergeError(
            "Spurious failure. Github API requires me to be an admin on the "
            "fork repository to merge master into the PR branch. Hit "
            "'Update Branch' for me before trying again."
        )

    raise RuntimeError(
        'Sync with master failed for unknown reason. '
        f'Code: {response.status_code}. Content: {response.content!r}.'
    )


def attempt_squash_merge(pr: PullRequestDetails) -> Union[bool, CannotAutomergeError]:
    """Perform a squash merge on a pull request.

    References:
        https://developer.github.com/v3/pulls/#merge-a-pull-request-merge-button

    Args:
        pr: The pull request to squash merge.

    Returns:
        True if the squash merge was successful and CannotAutomergeError if the square merge
        was not possible

    Raises:
        RuntimeError: If the request to merge returned a failed merge response.
    """
    url = (
        f"https://api.github.com/repos/{pr.repo.organization}/"
        f"{pr.repo.name}/pulls/{pr.pull_id}/merge"
    )
    data = {
        'commit_title': f'{pr.title} (#{pr.pull_id})',
        'commit_message': pr.body or '',
        'sha': pr.branch_sha,
        'merge_method': 'squash',
    }
    response = pr.repo.put(url, json=data)

    if response.status_code == 200:
        # Merge succeeded.
        log(f'Merged PR#{pr.pull_id} ({pr.title!r}):\n{indent(pr.body)}\n')
        return True

    if response.status_code == 405:
        return CannotAutomergeError("Pull Request is not mergeable.")

    if response.status_code == 409:
        # Need to sync.
        return False

    raise RuntimeError(
        f'Merge failed. Code: {response.status_code}. Content: {response.content!r}.'
    )


def auto_delete_pr_branch(pr: PullRequestDetails) -> bool:
    """Delete a branch.

    References:
        https://developer.github.com/v3/git/refs/#delete-a-reference

    Args:
        pr: The pull request to delete.

    Returns:
        True of the delete was successful, False otherwise.

    Raises:
        RuntimeError: If the request does not return status 204 (no content).
    """

    open_pulls = list_open_pull_requests(pr.repo, base_branch=pr.branch_name)
    if any(open_pulls):
        log(f'Not deleting branch {pr.branch_name!r}. It is used elsewhere.')
        return False

    remote = pr.remote_repo
    if pr.is_on_fork():
        log(
            f'Not deleting branch {pr.branch_name!r}. It belongs to a fork '
            f'({pr.remote_repo.organization}/{pr.remote_repo.name}).'
        )
        return False

    url = (
        f"https://api.github.com/repos/{remote.organization}/"
        f"{remote.name}/git/refs/heads/{pr.branch_name}"
    )
    response = pr.repo.delete(url)

    if response.status_code == 204:
        # Delete succeeded.
        log(f'Deleted branch {pr.branch_name!r}.')
        return True

    log(f'Delete failed. Code: {response.status_code}. Content: {response.content!r}.')
    return False


def branch_data_modified_recently(payload: Any) -> bool:
    """Whether the branch was modified recently."""
    modified_date = datetime.datetime.strptime(
        payload['commit']['commit']['committer']['date'], '%Y-%m-%dT%H:%M:%SZ'
    )
    return is_recent_date(modified_date)


def add_labels_to_pr(repo: GithubRepository, pull_id: int, *labels: str) -> None:
    """Add lables to a pull request.

    References:
        https://developer.github.com/v3/issues/labels/#add-labels-to-an-issue

    Args:
        repo: The github repo where the pull request lives.
        pull_id: The id of the pull request.
        *labels: The labels to add to the pull request.

    Raises:
        RuntimeError: If the request to add labels returned anything other than success.
    """
    url = f"https://api.github.com/repos/{repo.organization}/{repo.name}/issues/{pull_id}/labels"
    response = repo.post(url, json=list(labels))

    if response.status_code != 200:
        raise RuntimeError(
            f'Add labels failed. Code: {response.status_code}. Content: {response.content!r}.'
        )


def remove_label_from_pr(repo: GithubRepository, pull_id: int, label: str) -> bool:
    """Removes a label from a pull request.

    References:
        https://developer.github.com/v3/issues/labels/#remove-a-label-from-an-issue

    Args:
        repo: The github repo for the pull request.
        pull_id: The id for the pull request.
        label: The label to remove.

    Raises:
        RuntimeError: If the request does not return status 200 (success).

    Returns:
        True if the label existed and was deleted. False if the label did not exist.
    """
    url = (
        f"https://api.github.com/repos/{repo.organization}/"
        f"{repo.name}/issues/{pull_id}/labels/{label}"
    )
    response = repo.delete(url)

    if response.status_code == 404:
        payload = json.JSONDecoder().decode(response.content.decode())
        if payload['message'] == 'Label does not exist':
            return False

    if response.status_code == 200:
        # Removed the label.
        return True

    raise RuntimeError(
        f'Label remove failed. Code: {response.status_code}. Content: {response.content!r}.'
    )


def list_open_pull_requests(
    repo: GithubRepository, base_branch: Optional[str] = None, per_page: int = 100
) -> List[PullRequestDetails]:
    """List open pull requests.

    Args:
        repo: The github repo for the pull requests.
        base_branch: The branch for which to request pull requests.
        per_page: The number of results to obtain per page.

    Returns:
        A list of the pull requests.

    Raises:
        RuntimeError: If the request does not return status 200 (success).
    """
    url = (
        f"https://api.github.com/repos/{repo.organization}/{repo.name}/pulls"
        f"?per_page={per_page}"
    )
    data = {'state': 'open'}
    if base_branch is not None:
        data['base'] = base_branch
    response = repo.get(url, json=data)

    if response.status_code != 200:
        raise RuntimeError(
            f'List pulls failed. Code: {response.status_code}. Content: {response.content!r}.'
        )

    pulls = json.JSONDecoder().decode(response.content.decode())
    results = [PullRequestDetails(pull, repo) for pull in pulls]

    # Filtering via the API doesn't seem to work, so we do it ourselves.
    if base_branch is not None:
        results = [result for result in results if result.base_branch_name == base_branch]
    return results


def find_auto_mergeable_prs(repo: GithubRepository) -> List[int]:
    open_prs = list_open_pull_requests(repo)
    auto_mergeable_prs = [pr for pr in open_prs if pr.marked_automergeable]
    return [pr.payload['number'] for pr in auto_mergeable_prs]


def find_problem_with_automergeability_of_pr(
    pr: PullRequestDetails, master_branch_data: Any
) -> Optional[CannotAutomergeError]:
    # Sanity.
    if pr.payload['state'] != 'open':
        return CannotAutomergeError('Not an open pull request.')
    if pr.base_branch_name != 'master':
        return CannotAutomergeError('Can only automerge into master.')
    if pr.payload['mergeable_state'] == 'dirty':
        return CannotAutomergeError('There are merge conflicts.')

    # If a user removes the automerge label, remove the head label for them.
    if pr.has_label(HEAD_AUTO_MERGE_LABEL) and not pr.has_label(USER_AUTO_MERGE_LABEL):
        return CannotAutomergeError(
            f'The {USER_AUTO_MERGE_LABEL} label was removed.', may_be_temporary=True
        )

    # Only collaborators with write access can use the automerge labels.
    label_problem = check_auto_merge_labeler(pr.repo, pr.pull_id)
    if label_problem is not None:
        return label_problem

    # Check review status.
    review_status = get_pr_review_status(pr)
    if not any(review['state'] == 'APPROVED' for review in review_status):
        return CannotAutomergeError('No approved review.')
    if any(review['state'] == 'REQUEST_CHANGES' for review in review_status):
        return CannotAutomergeError('A review is requesting changes.')

    # Any failing status checks?
    status_check_state = classify_pr_status_check_state(pr)
    if status_check_state is False:
        return CannotAutomergeError('A status check is failing.')

    # Some issues can only be detected after waiting a bit.
    if not pr.modified_recently:
        # Nothing is setting a required status check.
        missing_statuses = absent_status_checks(pr, master_branch_data)
        if missing_statuses:
            return CannotAutomergeError(
                'A required status check is not present.\n\n'
                f'Missing statuses: {sorted(missing_statuses)!r}'
            )

        # Can't figure out how to make it merge.
        if pr.payload['mergeable_state'] == 'blocked':
            if status_check_state is True:
                return CannotAutomergeError(
                    "Merging is blocked (I don't understand why).", may_be_temporary=True
                )
        if pr.payload['mergeable'] is False:
            return CannotAutomergeError(
                "PR isn't classified as mergeable (I don't understand why).", may_be_temporary=True
            )

    return None


def cannot_merge_pr(pr: PullRequestDetails, reason: CannotAutomergeError):
    log(f'Cancelled automerge of PR#{pr.pull_id} ({pr.title!r}): {reason.args[0]}')

    add_comment(pr.repo, pr.pull_id, f'Automerge cancelled: {reason}')

    for label in AUTO_MERGE_LABELS:
        if pr.has_label(label):
            remove_label_from_pr(pr.repo, pr.pull_id, label)


def drop_temporary(
    pr: PullRequestDetails,
    problem: Optional[CannotAutomergeError],
    prev_seen_times: Dict[int, datetime.datetime],
    next_seen_times: Dict[int, datetime.datetime],
) -> Optional[CannotAutomergeError]:
    """Filters out problems that may be temporary."""

    if problem is not None and problem.may_be_temporary:
        since = prev_seen_times.get(pr.pull_id, datetime.datetime.utcnow())
        if is_recent_date(since):
            next_seen_times[pr.pull_id] = since
            return None

    return problem


def gather_auto_mergeable_prs(
    repo: GithubRepository, problem_seen_times: Dict[int, datetime.datetime]
) -> List[PullRequestDetails]:
    result = []
    raw_prs = list_open_pull_requests(repo)
    master_branch_data = get_branch_details(repo, 'master')
    if branch_data_modified_recently(master_branch_data):
        return []

    prev_seen_times = dict(problem_seen_times)
    problem_seen_times.clear()
    for raw_pr in raw_prs:
        if not raw_pr.marked_automergeable:
            continue

        # Looking up a single PR gives more data, e.g. the 'mergeable' entry.
        pr = PullRequestDetails.from_github(repo, raw_pr.pull_id)
        problem = find_problem_with_automergeability_of_pr(pr, master_branch_data)
        if problem is None:
            result.append(pr)

        persistent_problem = drop_temporary(
            pr, problem, prev_seen_times=prev_seen_times, next_seen_times=problem_seen_times
        )
        if persistent_problem is not None:
            cannot_merge_pr(pr, persistent_problem)

    return result


def merge_desirability(pr: PullRequestDetails) -> Any:
    synced = classify_pr_synced_state(pr) is True
    tested = synced and (classify_pr_status_check_state(pr) is True)
    forked = pr.is_on_fork()

    # 1. Prefer to merge already-synced PRs. This minimizes the number of builds
    #    performed by travis.
    # 2. Prefer to merge synced PRs from forks. This minimizes manual labor;
    #    currently the bot can't resync these PRs. Secondarily, avoid unsynced
    #    PRs from forks until necessary because they will fail when hit.
    # 3. Prefer to merge PRs where the status checks have already completed.
    #    This is just faster, because the next build can be started sooner.
    # 4. Use seniority as a tie breaker.

    # Desired order is:
    #   TF
    #   SF
    #   T_
    #   S_
    #   __
    #   _F
    # (S = synced, T = tested, F = forked.)

    if forked:
        if tested:
            rank = 5
        elif synced:
            rank = 4
        else:
            rank = 0
    else:
        if tested:
            rank = 3
        elif synced:
            rank = 2
        else:
            rank = 1

    return rank, -pr.pull_id


def pick_head_pr(active_prs: List[PullRequestDetails]) -> Optional[PullRequestDetails]:
    if not active_prs:
        return None

    for pr in sorted(active_prs, key=merge_desirability, reverse=True):
        if pr.has_label(HEAD_AUTO_MERGE_LABEL):
            return pr

    promoted = max(active_prs, key=merge_desirability)
    log(f'Front of queue: PR#{promoted.pull_id} ({promoted.title!r})')
    add_labels_to_pr(promoted.repo, promoted.pull_id, HEAD_AUTO_MERGE_LABEL)
    return promoted


def merge_duty_cycle(
    repo: GithubRepository, persistent_temporary_problems: Dict[int, datetime.datetime]
):
    """Checks and applies auto merge labeling operations."""
    active_prs = gather_auto_mergeable_prs(repo, persistent_temporary_problems)
    head_pr = pick_head_pr(active_prs)
    if head_pr is None:
        return

    state = classify_pr_synced_state(head_pr)
    if state is False:
        result = update_branch(head_pr)
    elif state is True:
        result = attempt_squash_merge(head_pr)
        if result is True:
            auto_delete_pr_branch(head_pr)
            for label in AUTO_MERGE_LABELS:
                remove_label_from_pr(repo, head_pr.pull_id, label)
    else:
        # `gather_auto_mergeable_prs` is responsible for this case.
        result = False

    if isinstance(result, CannotAutomergeError):
        cannot_merge_pr(head_pr, result)


def label_duty_cycle(repo: GithubRepository):
    """Checks and applies size labeling operations."""
    open_prs = list_open_pull_requests(repo)
    size_unlabeled_prs = [pr for pr in open_prs if not pr.marked_size]

    for pr in size_unlabeled_prs:
        full_pr_data = PullRequestDetails.from_github(repo, pr.pull_id)
        new_label = get_pr_size_label(full_pr_data.tot_changes)
        log(f'Adding size label {new_label} to #{full_pr_data.pull_id} ({full_pr_data.title!r}).')
        add_labels_to_pr(repo, pr.pull_id, new_label)


def indent(text: str) -> str:
    return '    ' + text.replace('\n', '\n    ')


def main() -> None:
    access_token = os.getenv(ACCESS_TOKEN_ENV_VARIABLE)
    if not access_token:
        project_id = 'cirq-infra'
        print(f'{ACCESS_TOKEN_ENV_VARIABLE} not set. Trying secret manager.', file=sys.stderr)
        client = secretmanager_v1beta1.SecretManagerServiceClient()
        secret_name = f'projects/{project_id}/secrets/cirq-bot-api-key/versions/1'
        response = client.access_secret_version(name=secret_name)
        access_token = response.payload.data.decode('UTF-8')

    repo = GithubRepository(
        organization=GITHUB_REPO_ORGANIZATION, name=GITHUB_REPO_NAME, access_token=access_token
    )

    log('Watching for automergeable PRs.')
    problem_seen_times: Dict[int, datetime.datetime] = {}
    while True:
        try:
            merge_duty_cycle(repo, problem_seen_times)
            label_duty_cycle(repo)
        except Exception:  # Anything but a keyboard interrupt / system exit.
            traceback.print_exc()
        wait_for_polling_period()


if __name__ == '__main__':
    main()
