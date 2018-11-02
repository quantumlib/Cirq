# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prettifies output from pyflame, so that it looks clearer in Chrome."""

from typing import Dict, Any, Sequence, List

import json
import sys


def main():
    root = json.load(sys.stdin)

    # Ensure all nodes have 'children' list.
    for node in root['nodes']:
        if 'children' not in node:
            node['children'] = []

    # Prettify.
    clean_out_idles(root)
    merge_redundant_nodes(root)
    shorten_names(root)

    json.dump(root, sys.stdout)


def id_of_node_with_name(root: Dict[str, Any], name: str) -> int:
    nodes = root['nodes']
    for node in nodes:
        func_name = node['callFrame']['functionName']
        if func_name == name:
            return node['id']
    raise ValueError('no such node')


def determine_merge_rewrites(root: Dict[str, Any]) -> Dict[int, int]:
    """Figures out which nodes should be merged by 'merge_identicals'."""

    root_id = id_of_node_with_name(root, '(root)')
    idle_id = id_of_node_with_name(root, '(idle)')
    id_rewrites = {idle_id: root_id}

    # Determine the parent of each node. (Note: assumes single parents.)
    nodes = root['nodes']
    parents = {node['id']: None for node in nodes}
    for node in nodes:
        parent_id = node['id']
        parent_id = id_rewrites.get(parent_id, parent_id)
        for child in node['children']:
            parents[child] = parent_id

    # Compute the ancestry of every node as a grand|parent|child string.
    # (Ignore line information when doing so; only care about the function.)
    def ancestry_key(node):
        child_id = node['id']
        if child_id not in ancestries:
            p = parents.get(child_id, None)
            leaf = name_without_line_number(node['callFrame']['functionName'])
            if p is None:
                ancestries[child_id] = leaf
            else:
                ancestries[child_id] = ancestries[p] + '|' + leaf
        return ancestries[child_id]
    ancestries = {}

    # Nodes with identical ancestries should be merged.
    name_to_id = {}
    for node in nodes:
        node_id = node['id']
        key = ancestry_key(node)
        if key in name_to_id:
            id_rewrites[node_id] = name_to_id[key]
        else:
            name_to_id[key] = node_id

    return id_rewrites


def merge_redundant_nodes(root: Dict[str, Any]) -> None:
    """Merges nodes that refer to the same stack state.

    pyflame likes to generate multiple nodes for single stack states, but
    this makes the flame chart view in Chrome useless. We have to figure out
    which nodes should be merged, to make the chart look nice.
    """

    rewrites = determine_merge_rewrites(root)

    nodes = root['nodes']
    id_to_index = {node['id']: i for i, node in enumerate(nodes)}

    # Transfer profiler hits from merge-source to merge-destination.
    for node in nodes:
        node_id = node['id']
        if node_id in rewrites:
            new_id = rewrites[node_id]
            new_node = nodes[id_to_index[new_id]]
            new_node['hitCount'] += node['hitCount']
            new_node['children'].extend(node['children'])

    # Delete nodes that have now been merged into other nodes.
    nodes[:] = [node for node in nodes if node['id'] not in rewrites]

    # Rewrite references to now-deleted nodes to point at the correct targets.
    samples = root['samples']
    samples[:] = rewritten_list(samples, rewrites)
    for node in nodes:
        node_id = node['id']
        children = node['children']
        children[:] = sorted(set(c
                                 for c in rewritten_list(children, rewrites)
                                 if c != node_id))


def clean_out_idles(root: Dict[str, Any]) -> None:
    """Removes '(idle)' spam by extending preceeding samples.

    pyflame is supposed to do this automatically when you pass the '-x' switch,
    but as far as I can tell it simply doesn't.
    """
    idle_id = id_of_node_with_name(root, '(idle)')
    samples = root['samples']
    prev_sample = 0
    for i, sample in enumerate(samples):
        if sample == idle_id:
            samples[i] = prev_sample
        else:
            prev_sample = sample


def shorten_names(root: Dict[str, Any]) -> None:
    """Replaces overly-specific function names with shorter ones.

    pyflame likes to be very specific, listing full path and function name and
    line number information. But this is really overly specific; we reduce it
    to just the file and function.
    """
    nodes = root['nodes']
    for node in nodes:
        frame = node['callFrame']
        func_name = frame['functionName']
        lineless_name = name_without_line_number(func_name)
        frame['functionName'] = lineless_name.split('/')[-1]


def rewritten_list(target: Sequence[int],
                   rewrites: Dict[int, int]
                   ) -> List[int]:
    return [rewrites.get(e, e) for e in target]


def name_without_line_number(name: str) -> str:
    """Removes the trailing line number from a node name."""
    parts = name.split(':')
    if len(parts) > 1:
        return ':'.join(parts[:-1])
    return name


if __name__ == '__main__':
    main()
