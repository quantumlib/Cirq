from typing import Optional, List, Any

import json
import os
import time
import sys

import requests

from dev_tools.github_repository import GithubRepository


class PullRequestDetails:
    def __init__(self, payload) -> None:
        self.payload = payload

    @staticmethod
    def from_github(repo: GithubRepository,
                    pull_id: int) -> 'PullRequestDetails':
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
        return PullRequestDetails(payload)

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


def get_pr_check_status(repo: GithubRepository, pr: PullRequestDetails) -> Any:
    url = ("https://api.github.com/repos/{}/{}/commits/{}/status"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pr.branch_sha,
                                     repo.access_token))
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            'Get status failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))

    return json.JSONDecoder().decode(response.content.decode())


def get_pr_review_status(repo: GithubRepository, pr: PullRequestDetails) -> Any:
    url = ("https://api.github.com/repos/{}/{}/pulls/{}/reviews"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pr.pull_id,
                                     repo.access_token))
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            'Get review failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))

    return json.JSONDecoder().decode(response.content.decode())


def wait_for_status(repo: GithubRepository,
                    pull_id: int,
                    prev_pr: Optional[PullRequestDetails],
                    fail_if_same: bool) -> PullRequestDetails:
    def wait_a_tick():
        print('.', end='', flush=True)
        time.sleep(5)

    while True:
        pr = PullRequestDetails.from_github(repo, pull_id)
        if pr.payload['state'] != 'open':
            raise RuntimeError('Not an open PR! {}'.format(pr.payload))

        if prev_pr and pr.branch_sha == prev_pr.branch_sha:
            if fail_if_same:
                raise RuntimeError('Doing the same thing twice while expecting'
                                   ' different results.')
            wait_a_tick()
            continue

        if pr.base_branch_name != 'master':
            raise RuntimeError('PR not merging into master: {}.'.format(
                pr.payload))

        review_status = get_pr_review_status(repo, pr)
        if not any(review['state'] == 'APPROVED' for review in review_status):
            raise RuntimeError(
                'No approved review: {}'.format(review_status))

        check_status = get_pr_check_status(repo, pr)
        if check_status['state'] == 'pending':
            wait_a_tick()
            continue
        if check_status['state'] != 'success':
            raise RuntimeError(
                'A status check is failing: {}'.format(check_status))

        return pr


def sync_with_master(repo: GithubRepository, pr: PullRequestDetails) -> bool:
    url = ("https://api.github.com/repos/{}/{}/merges"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     repo.access_token))
    data = {
        'base': pr.branch_name,
        'head': 'master'
    }
    response = requests.post(url, json=data)

    if response.status_code == 201:
        # Merge succeeded.
        return True

    if response.status_code == 204:
        # Already merged.
        return False

    raise RuntimeError('Sync with master failed. Code: {}. Content: {}.'.format(
        response.status_code, response.content))


def squash_merge(repo: GithubRepository, pr: PullRequestDetails
                 ) -> bool:
    url = ("https://api.github.com/repos/{}/{}/pulls/{}/merge"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pr.pull_id,
                                     repo.access_token))
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

    raise RuntimeError('Merge failed. Code: {}. Content: {}.'.format(
        response.status_code, response.content))


def auto_delete_pr_branch(repo: GithubRepository,
                          pr: PullRequestDetails) -> bool:
    open_pulls = list_open_pull_requests(repo, base_branch=pr.branch_name)
    if any(open_pulls):
        print('Not deleting branch {!r}. It is used elsewhere.'.format(
            pr.branch_name))
        return False

    url = ("https://api.github.com/repos/{}/{}/git/refs/heads/{}"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pr.branch_name,
                                     repo.access_token))
    response = requests.delete(url)

    if response.status_code == 204:
        # Delete succeeded.
        print('Deleted branch {!r}.'.format(pr.branch_name))
        return True

    print('Delete failed. Code: {}. Content: {}.'.format(
        response.status_code, response.content))
    return False


def list_open_pull_requests(repo: GithubRepository, base_branch: str
                            ) -> List[PullRequestDetails]:
    url = ("https://api.github.com/repos/{}/{}/pulls"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     repo.access_token))
    data = {
        'state': 'open',
        'base': base_branch
    }
    response = requests.get(url, json=data)

    if response.status_code != 200:
        raise RuntimeError(
            'List pulls failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))

    pulls = json.JSONDecoder().decode(response.content.decode())
    results = [PullRequestDetails(pull) for pull in pulls]

    # Filtering via the API doesn't seem to work, so we do it ourselves.
    return [result for result in results
            if result.base_branch_name == base_branch]


def auto_merge_pull_request(repo: GithubRepository, pull_id: int) -> None:
    """
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

    prev_pr = None
    fail_if_same = False
    while True:
        print('Waiting for checks to succeed..', end='', flush=True)
        pr = wait_for_status(repo, pull_id, prev_pr, fail_if_same)

        print('\nChecks succeeded. Checking if synced...')
        if sync_with_master(repo, pr):
            print('Had to resync with master.')
            prev_pr = pr
            fail_if_same = False
            continue

        print('Still synced. Attempting merge...')
        if not squash_merge(repo, pr):
            print('Merge not allowed. Starting over again...')
            prev_pr = pr
            fail_if_same = True
            continue

        print('Merged successfully!')
        break

    auto_delete_pr_branch(repo, pr)
    print('Done.')


def main():
    if len(sys.argv) < 2:
        print('No pull request numbers given.', file=sys.stderr)
        sys.exit(1)

    pull_ids = [int(e) for e in sys.argv[1:]]
    access_token = os.getenv('CIRQ_GITHUB_PR_ACCESS_TOKEN')
    if not access_token:
        print('CIRQ_GITHUB_PR_ACCESS_TOKEN not set.', file=sys.stderr)
        sys.exit(1)

    repo = GithubRepository(
        organization='quantumlib',
        name='cirq',
        access_token=access_token)

    for pull_id in pull_ids:
        auto_merge_pull_request(repo, pull_id)


if __name__ == '__main__':
    main()
