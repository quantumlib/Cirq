from typing import Optional, List, Any, Dict

import json
import os
import time
import sys

import requests

from dev_tools.github_repository import GithubRepository


CLA_ACCESS_TOKEN = None


class CurrentStuckGoToNextError(RuntimeError):
    pass


class PullRequestDetails:
    def __init__(self, payload: Any, repo: GithubRepository) -> None:
        self.payload = payload
        self.repo = repo

    @staticmethod
    def from_github(repo: GithubRepository,
                    pull_id: int) -> 'PullRequestDetails':
        """
        References:
            https://developer.github.com/v3/pulls/#get-a-single-pull-request
        """
        url = ("https://api.github.com/repos/{}/{}/pulls/{}"
               "?access_token={}".format(repo.organization,
                                         repo.name,
                                         pull_id,
                                         repo.access_token))

        response = requests.get(url)

        if response.status_code != 200:
            raise RuntimeError(
                'Pull check failed. Code: {}. Content: {}.'.format(
                    response.status_code, response.content))

        payload = json.JSONDecoder().decode(response.content.decode())
        return PullRequestDetails(payload, repo)

    @property
    def remote_repo(self):
        return GithubRepository(
            organization=self.payload['head']['repo']['owner']['login'],
            name=self.payload['head']['repo']['name'],
            access_token=self.repo.access_token)

    @property
    def marked_automergeable(self) -> bool:
        return any(label['name'] == 'automerge'
                   for label in self.payload['labels'])

    @property
    def marked_cla_no(self) -> bool:
        return any(label['name'] == 'cla: no'
                   for label in self.payload['labels'])

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


def find_existing_status_comment(repo: GithubRepository, pull_id: int
                                 ) -> Optional[Dict[str, Any]]:
    expected_user = 'CirqBot'
    expected_text = '(automerge): PENDING'

    comments = list_pr_comments(repo, pull_id)
    for comment in comments:
        if comment['user']['login'] == expected_user:
            if expected_text in comment['body']:
                return comment

    return None


def add_comment(repo: GithubRepository, pull_id: int, text: str) -> None:
    """
    References:
        https://developer.github.com/v3/issues/comments/#create-a-comment
    """
    url = ("https://api.github.com/repos/{}/{}/issues/{}/comments"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pull_id,
                                     repo.access_token))
    data = {
        'body': text
    }
    response = requests.post(url, json=data)

    if response.status_code != 201:
        raise RuntimeError('Add comment failed. Code: {}. Content: {}.'.format(
            response.status_code, response.content))


def edit_comment(repo: GithubRepository, text: str, comment_id: int) -> None:
    """
    References:
        https://developer.github.com/v3/issues/comments/#edit-a-comment
    """
    url = ("https://api.github.com/repos/{}/{}/issues/comments/{}"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     comment_id,
                                     repo.access_token))
    data = {
        'body': text
    }
    response = requests.patch(url, json=data)

    if response.status_code != 200:
        raise RuntimeError('Edit comment failed. Code: {}. Content: {}.'.format(
            response.status_code, response.content))


def leave_status_comment(repo: GithubRepository,
                         pull_id: int,
                         success: Optional[bool],
                         state_description: str) -> None:
    cur = find_existing_status_comment(repo, pull_id)
    status = 'PENDING' if success is None else 'DONE' if success else 'FAILED'
    body = '(automerge): {} ({})'.format(status, state_description)
    if cur is None:
        add_comment(repo, pull_id, body)
    else:
        edit_comment(repo, body, cur['id'])


def get_pr_check_status(pr: PullRequestDetails) -> Any:
    """
    References:
        https://developer.github.com/v3/repos/statuses/#get-the-combined-status-for-a-specific-ref
    """

    url = ("https://api.github.com/repos/{}/{}/commits/{}/status"
           "?access_token={}".format(pr.repo.organization,
                                     pr.repo.name,
                                     pr.branch_sha,
                                     pr.repo.access_token))
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            'Get status failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))

    return json.JSONDecoder().decode(response.content.decode())


def get_pr_review_status(pr: PullRequestDetails) -> Any:
    """
    References:
        https://developer.github.com/v3/pulls/reviews/#list-reviews-on-a-pull-request
    """
    url = ("https://api.github.com/repos/{}/{}/pulls/{}/reviews"
           "?access_token={}".format(pr.repo.organization,
                                     pr.repo.name,
                                     pr.pull_id,
                                     pr.repo.access_token))
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            'Get review failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))

    return json.JSONDecoder().decode(response.content.decode())


def wait_a_tick():
    print('.', end='', flush=True)
    time.sleep(5)


def wait_for_status(repo: GithubRepository,
                    pull_id: int,
                    prev_pr: Optional[PullRequestDetails],
                    fail_if_same: bool,
                    ) -> PullRequestDetails:
    while True:
        pr = PullRequestDetails.from_github(repo, pull_id)
        if pr.payload['state'] != 'open':
            raise CurrentStuckGoToNextError('Not an open PR.')

        if prev_pr and pr.branch_sha == prev_pr.branch_sha:
            if fail_if_same:
                raise CurrentStuckGoToNextError("I think I'm stuck in a loop.")
            wait_a_tick()
            continue

        if not pr.marked_automergeable:
            raise CurrentStuckGoToNextError('"automerge" label was removed.')
        if pr.base_branch_name != 'master':
            raise CurrentStuckGoToNextError('Failed to merge into master.')

        review_status = get_pr_review_status(pr)
        if not any(review['state'] == 'APPROVED' for review in review_status):
            raise CurrentStuckGoToNextError('No approved review.')
        if any(review['state'] == 'REQUEST_CHANGES'
               for review in review_status):
            raise CurrentStuckGoToNextError('A review is requesting changes.')

        check_status = get_pr_check_status(pr)
        if check_status['state'] == 'pending':
            wait_a_tick()
            continue
        if check_status['state'] != 'success':
            raise CurrentStuckGoToNextError('A status check is failing.')

        return pr


def get_repo_ref(repo: GithubRepository, ref: str) -> Dict[str, Any]:
    """
    References:
        https://developer.github.com/v3/git/refs/#get-a-reference
    """

    url = ("https://api.github.com/repos/{}/{}/git/refs/{}"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     ref,
                                     repo.access_token))
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            'Refs get failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))
    payload = json.JSONDecoder().decode(response.content.decode())
    return payload


def get_master_sha(repo: GithubRepository) -> str:
    ref = get_repo_ref(repo, 'heads/master')
    return ref['object']['sha']


def watch_for_spurious_cla_failure(pr: PullRequestDetails) -> bool:
    # It's not spurious if it was already there!
    if pr.marked_cla_no:
        return False

    print('Waiting for spurious CLA failure..', flush=True, end='')
    for _ in range(6):
        wait_a_tick()
        new_pr = PullRequestDetails.from_github(pr.repo, pr.pull_id)
        if new_pr.marked_cla_no:
            print('\nCaught spurious failure occurring.')
            return True

    print('\nGooglebot appears to be sleeping peacefully.')
    return False


def watch_for_cla_restore(pr: PullRequestDetails):
    print('Waiting for CLA restore..', flush=True, end='')
    for _ in range(6):
        wait_a_tick()
        new_pr = PullRequestDetails.from_github(pr.repo, pr.pull_id)
        if not new_pr.marked_cla_no:
            print('\nCLA restored.')
            return

    raise CurrentStuckGoToNextError('CLA stuck on no.')


def list_pr_comments(repo: GithubRepository, pull_id: int
                     ) -> List[Dict[str, Any]]:
    """
    References:
        https://developer.github.com/v3/issues/comments/#list-comments-on-an-issue
    """
    url = ("https://api.github.com/repos/{}/{}/issues/{}/comments"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pull_id,
                                     repo.access_token))
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            'Comments get failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))
    payload = json.JSONDecoder().decode(response.content.decode())
    return payload


def delete_comment(repo: GithubRepository, comment_id: int) -> None:
    """
    References:
        https://developer.github.com/v3/issues/comments/#delete-a-comment
    """
    url = ("https://api.github.com/repos/{}/{}/issues/comments/{}"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     comment_id,
                                     repo.access_token))
    response = requests.delete(url)
    if response.status_code != 204:
        raise RuntimeError(
            'Comment delete failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))


def find_spurious_coauthor_comment_id(pr: PullRequestDetails) -> Optional[int]:
    expected_user = 'googlebot'
    expected_text = ('one or more commits were authored or co-authored '
                     'by someone other than the pull request submitter')

    comments = list_pr_comments(pr.repo, pr.pull_id)
    for comment in comments:
        if comment['user']['login'] == expected_user:
            if expected_text in comment['body']:
                return comment['id']

    return None


def find_spurious_fixed_comment_id(pr: PullRequestDetails) -> Optional[int]:
    expected_user = 'googlebot'
    expected_text = 'A Googler has manually verified that the CLAs look good.'

    comments = list_pr_comments(pr.repo, pr.pull_id)
    for comment in comments:
        if comment['user']['login'] == expected_user:
            if expected_text in comment['body']:
                return comment['id']

    return None


def fight_cla_bot(pr: PullRequestDetails) -> None:
    """The cla bot is *not* happy when someone updates others' PRs.

    This bounces the 'cla: no' label back to 'cla: yes'. Only works if the
    access token being used corresponds to a registered Google employee account.
    Otherwise cla bot will just restore the label and message.
    """
    if not watch_for_spurious_cla_failure(pr):
        return

    # Manually indicate that this is fine.
    add_labels_to_pr(pr.repo,
                     pr.pull_id,
                     'cla: yes',
                     override_token=CLA_ACCESS_TOKEN)

    spurious_comment_id = find_spurious_coauthor_comment_id(pr)
    if spurious_comment_id is not None:
        print("Deleting spurious co-author comment from googlebot.")
        delete_comment(pr.repo, spurious_comment_id)

    watch_for_cla_restore(pr)

    spurious_comment_id_2 = find_spurious_fixed_comment_id(pr)
    if spurious_comment_id_2 is not None:
        print("Deleting spurious repair comment from googlebot.")
        delete_comment(pr.repo, spurious_comment_id_2)


def sync_with_master(pr: PullRequestDetails) -> bool:
    """
    References:
        https://developer.github.com/v3/repos/merging/#perform-a-merge
    """
    master_sha = get_master_sha(pr.repo)
    remote = pr.remote_repo
    url = ("https://api.github.com/repos/{}/{}/merges"
           "?access_token={}".format(remote.organization,
                                     remote.name,
                                     remote.access_token))
    data = {
        'base': pr.branch_name,
        'head': master_sha,
        'commit_message': 'Update branch (automerge)'.format(pr.branch_name)
    }
    response = requests.post(url, json=data)

    if response.status_code == 201:
        # Merge succeeded.
        fight_cla_bot(pr)
        return True

    if response.status_code == 204:
        # Already merged.
        return False

    raise RuntimeError('Sync with master failed. Code: {}. Content: {}.'.format(
        response.status_code, response.content))


def squash_merge(pr: PullRequestDetails) -> bool:
    """
    References:
        https://developer.github.com/v3/pulls/#merge-a-pull-request-merge-button
    """
    url = ("https://api.github.com/repos/{}/{}/pulls/{}/merge"
           "?access_token={}".format(pr.repo.organization,
                                     pr.repo.name,
                                     pr.pull_id,
                                     pr.repo.access_token))
    data = {
        'commit_title': '{} (#{})'.format(pr.title, pr.pull_id),
        'commit_message': pr.body,
        'sha': pr.branch_sha,
        'merge_method': 'squash'
    }
    response = requests.put(url, json=data)

    if response.status_code == 200:
        # Merge succeeded.
        return True

    if response.status_code == 405:
        # Not allowed. Maybe checks need to be run again?
        return False

    if response.status_code == 409:
        # Head was modified. We should try again.
        return False

    raise RuntimeError('Merge failed. Code: {}. Content: {}.'.format(
        response.status_code, response.content))


def auto_delete_pr_branch(pr: PullRequestDetails) -> bool:
    """
    References:
        https://developer.github.com/v3/git/refs/#delete-a-reference
    """

    open_pulls = list_open_pull_requests(pr.repo, base_branch=pr.branch_name)
    if any(open_pulls):
        print('Not deleting branch {!r}. It is used elsewhere.'.format(
            pr.branch_name))
        return False

    remote = pr.remote_repo
    url = ("https://api.github.com/repos/{}/{}/git/refs/heads/{}"
           "?access_token={}".format(remote.organization,
                                     remote.name,
                                     pr.branch_name,
                                     remote.access_token))
    response = requests.delete(url)

    if response.status_code == 204:
        # Delete succeeded.
        print('Deleted branch {!r}.'.format(pr.branch_name))
        return True

    print('Delete failed. Code: {}. Content: {}.'.format(
        response.status_code, response.content))
    return False


def add_labels_to_pr(repo: GithubRepository,
                     pull_id: int,
                     *labels: str,
                     override_token: str = None) -> None:
    """
    References:
        https://developer.github.com/v3/issues/labels/#add-labels-to-an-issue
    """
    url = ("https://api.github.com/repos/{}/{}/issues/{}/labels"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pull_id,
                                     override_token or repo.access_token))
    response = requests.post(url, json=list(labels))

    if response.status_code != 200:
        raise RuntimeError(
            'Add labels failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))


def remove_label_from_pr(repo: GithubRepository,
                         pull_id: int,
                         label: str) -> bool:
    """
    References:
        https://developer.github.com/v3/issues/labels/#remove-a-label-from-an-issue
    """
    url = ("https://api.github.com/repos/{}/{}/issues/{}/labels/{}"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pull_id,
                                     label,
                                     repo.access_token))
    response = requests.delete(url)

    if response.status_code == 404:
        payload = json.JSONDecoder().decode(response.content.decode())
        if payload['message'] == 'Label does not exist':
            return False

    if response.status_code == 200:
        # Removed the label.
        return True

    raise RuntimeError(
        'Label remove failed. Code: {}. Content: {}.'.format(
            response.status_code, response.content))


def list_open_pull_requests(repo: GithubRepository,
                            base_branch: Optional[str] = None
                            ) -> List[PullRequestDetails]:
    url = ("https://api.github.com/repos/{}/{}/pulls"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     repo.access_token))
    data = {
        'state': 'open',
    }
    if base_branch is not None:
        data['base'] = base_branch
    response = requests.get(url, json=data)

    if response.status_code != 200:
        raise RuntimeError(
            'List pulls failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))

    pulls = json.JSONDecoder().decode(response.content.decode())
    results = [PullRequestDetails(pull, repo) for pull in pulls]

    # Filtering via the API doesn't seem to work, so we do it ourselves.
    if base_branch is not None:
        results = [result for result in results
                   if result.base_branch_name == base_branch]
    return results


def find_auto_mergeable_prs(repo: GithubRepository) -> List[int]:
    open_prs = list_open_pull_requests(repo)
    auto_mergeable_prs = [pr for pr in open_prs if pr.marked_automergeable]
    return [pr.payload['number'] for pr in auto_mergeable_prs]


def auto_merge_pull_request(repo: GithubRepository, pull_id: int) -> None:
    r"""
    START                out of date
    |         .---------------------------------.
    |         |                                 |
    \         V          yes           yes      |     yes               any
    `-> WAIT_FOR_CHECKS -----> SYNC ---------> MERGE ------> TRY_DELETE --> DONE
              |                 |
              | checks fail     | merge conflict
              \                 \
              `-----------------`------------> TERMINAL_FAILURE
    """

    print('Auto-merging PR#{}'.format(pull_id))
    prev_pr = None
    fail_if_same = False
    while True:
        print('Waiting for checks to succeed..', end='', flush=True)
        pr = wait_for_status(repo, pull_id, prev_pr, fail_if_same)

        print('\nChecks succeeded. Checking if synced...')
        if sync_with_master(pr):
            print('Had to resync with master.')
            prev_pr = pr
            fail_if_same = False
            continue

        print('Still synced. Attempting merge...')
        if not squash_merge(pr):
            print('Merge not allowed. Starting over again...')
            prev_pr = pr
            fail_if_same = True
            continue

        print('Merged successfully!')
        break

    auto_delete_pr_branch(pr)
    print('Done merging "{} (#{})".'.format(pr.title, pull_id))
    print()


def auto_merge_multiple_pull_requests(repo: GithubRepository,
                                      pull_ids: List[int]) -> None:
    print('Automerging multiple PRs: {}'.format(pull_ids))
    for pull_id in pull_ids:
        try:
            leave_status_comment(repo,
                                 pull_id,
                                 None,
                                 'syncing and waiting...')
            auto_merge_pull_request(repo, pull_id)
            leave_status_comment(repo,
                                 pull_id,
                                 True,
                                 'merged successfully')
        except CurrentStuckGoToNextError as ex:
            print('#!\nPR#{} is stuck: {}'.format(pull_id, ex.args))
            print('Continuing to next.')
            leave_status_comment(repo,
                                 pull_id,
                                 False,
                                 ex.args[0])
        except Exception:
            leave_status_comment(repo, pull_id, False, 'Unexpected error.')
            raise
        finally:
            remove_label_from_pr(repo, pull_id, 'automerge')
    print('Finished attempting to automerge {}.'.format(pull_ids))


def watch_for_auto_mergeable_pull_requests(repo: GithubRepository):
    while True:
        print('Watching for automergeable PRs..', end='', flush=True)
        while True:
            auto_ids = find_auto_mergeable_prs(repo)
            if auto_ids:
                print('\nFound automergeable PRs: {}'.format(auto_ids))
                break
            wait_a_tick()
        auto_merge_multiple_pull_requests(repo, auto_ids)


def main():
    global cla_access_token
    pull_ids = [int(e) for e in sys.argv[1:]]
    access_token = os.getenv('CIRQ_BOT_GITHUB_ACCESS_TOKEN')
    cla_access_token = os.getenv('CIRQ_BOT_ALT_CLA_GITHUB_ACCESS_TOKEN')
    if not access_token:
        print('CIRQ_BOT_GITHUB_ACCESS_TOKEN not set.', file=sys.stderr)
        sys.exit(1)
    if not cla_access_token:
        print('CIRQ_BOT_ALT_CLA_GITHUB_ACCESS_TOKEN not set.', file=sys.stderr)
        sys.exit(1)

    repo = GithubRepository(
        organization='quantumlib',
        name='cirq',
        access_token=access_token)

    if pull_ids:
        auto_merge_multiple_pull_requests(repo, pull_ids)
    else:
        watch_for_auto_mergeable_pull_requests(repo)


if __name__ == '__main__':
    main()
