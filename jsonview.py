"""
Abbreviate long-lists in JSON output
====================================
"""

import sys
import json
import rich


def truncate(d, limit):
    if limit is None:
        return d

    if isinstance(d, dict):
        return {k: truncate(v, limit) for k, v in d.items()}

    if hasattr(d, "__len__") and len(d) > limit:
        return list(d[:limit]) + [f"... {len(d) - limit} more items"]

    return d


def jsonview(file_name, limit=None):
    with open(file_name, encoding='utf-8') as f:
        data = json.load(f)
    rich.print(truncate(data, limit))


if __name__ == "__main__":

    try:
        limit = int(sys.argv[2])
    except IndexError:
        limit = 5

    jsonview(sys.argv[1], limit)
