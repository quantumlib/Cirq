from typing import Optional, List, Any

import json
import os
import time
import sys

import requests

from dev_tools.github_repository import GithubRepository


class CurrentStuckGoToNextError(RuntimeError):
    pass


class PullRequestDetails:
    def __init__(self, payload: Any, repo: GithubRepository) -> None:
        self.payload = payload
        self.repo = repo

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
    def pull_id(self) -> int:
        return self.payload['number']

    @property
    def branch_name(self) -> str:
        return self.payload['head']['ref']

    @property
    def branch_label(self) -> str:
        return self.payload['head']['label']

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


def get_pr_check_status(pr: PullRequestDetails) -> Any:
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
                    fail_if_same: bool) -> PullRequestDetails:
    while True:
        pr = PullRequestDetails.from_github(repo, pull_id)
        if pr.payload['state'] != 'open':
            raise CurrentStuckGoToNextError(
                'Not an open PR! {}'.format(pr.payload))

        if prev_pr and pr.branch_sha == prev_pr.branch_sha:
            if fail_if_same:
                raise CurrentStuckGoToNextError(
                    'Doing the same thing twice while expecting '
                    'different results.')
            wait_a_tick()
            continue

        if not pr.marked_automergeable:
            raise CurrentStuckGoToNextError(
                'Not labelled with "automerge".')
        if pr.base_branch_name != 'master':
            raise CurrentStuckGoToNextError(
                'PR not merging into master: {}.'.format(pr.payload))

        review_status = get_pr_review_status(pr)
        if not any(review['state'] == 'APPROVED' for review in review_status):
            raise CurrentStuckGoToNextError(
                'No approved review: {}'.format(review_status))
        if any(review['state'] == 'REQUEST_CHANGES'
               for review in review_status):
            raise CurrentStuckGoToNextError(
                'Review requesting changes: {}'.format(review_status))

        check_status = get_pr_check_status(pr)
        if check_status['state'] == 'pending':
            wait_a_tick()
            continue
        if check_status['state'] != 'success':
            raise CurrentStuckGoToNextError(
                'A status check is failing: {}'.format(check_status))

        return pr


def get_refs(repo: GithubRepository) -> List[Any]:
    url = ("https://api.github.com/repos/{}/{}/git/refs"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     repo.access_token))
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            'Refs get failed. Code: {}. Content: {}.'.format(
                response.status_code, response.content))
    payload = json.JSONDecoder().decode(response.content.decode())
    return payload


def get_master_sha(repo: GithubRepository) -> str:
    refs = get_refs(repo)
    matches = [ref for ref in refs if ref['ref'] == 'refs/heads/master']
    if len(matches) != 1:
        raise RuntimeError('Wrong number of masters.')
    return matches[0]['object']['sha']


def sync_with_master(pr: PullRequestDetails) -> bool:
    master_sha = get_master_sha(pr.repo)
    remote = pr.remote_repo
    url = ("https://api.github.com/repos/{}/{}/merges"
           "?access_token={}".format(remote.organization,
                                     remote.name,
                                     remote.access_token))
    data = {
        'base': pr.branch_name,
        'head': master_sha,
        'message': 'Update branch [automerge]'.format(pr.branch_name)
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


def squash_merge(pr: PullRequestDetails) -> bool:
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

    raise RuntimeError('Merge failed. Code: {}. Content: {}.'.format(
        response.status_code, response.content))


def auto_delete_pr_branch(pr: PullRequestDetails) -> bool:
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


def remove_automerge_label(repo: GithubRepository, pull_id: int) -> None:
    pr = PullRequestDetails.from_github(repo, pull_id)
    if not pr.marked_automergeable:
        return

    labels = pr.payload['labels']
    new_labels = [label['name']
                  for label in labels
                  if label['name'] != 'automerge']

    url = ("https://api.github.com/repos/{}/{}/issues/{}"
           "?access_token={}".format(repo.organization,
                                     repo.name,
                                     pull_id,
                                     repo.access_token))
    data = {
        'labels': new_labels
    }
    response = requests.patch(url, json=data)

    if response.status_code != 200:
        raise RuntimeError(
            'Label change failed. Code: {}. Content: {}.'.format(
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
            auto_merge_pull_request(repo, pull_id)
            remove_automerge_label(repo, pull_id)
        except CurrentStuckGoToNextError as ex:
            print('#!\nPR#{} is stuck: {}'.format(pull_id, ex.args))
            remove_automerge_label(repo, pull_id)
            print('Continuing to next.')
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
    pull_ids = [int(e) for e in sys.argv[1:]]
    access_token = os.getenv('CIRQ_GITHUB_PR_ACCESS_TOKEN')
    if not access_token:
        print('CIRQ_GITHUB_PR_ACCESS_TOKEN not set.', file=sys.stderr)
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
