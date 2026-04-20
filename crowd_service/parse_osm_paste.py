#!/usr/bin/env python3
"""
Convert raw OSM way-page paste(s) into the JSON used by the crowd pipeline.

When you open a Way in OSM's web UI and look at its "Nodes" section, you
see something like:

    Nodes
    86 nodes
    11941921312
    11941921313
    9025467231 (part of ways highway=footway …)
    11941921314 (part of way building=yes …)
    293550911
    …

Paste that block into a text file (one block per street), separated by a
header line of the form `# <Street Name>`. This script extracts the node
IDs per street (preserving order) and writes/merges them into
`crowd_service/user_polygons.json`.

Example paste file `crowd_service/pastes.txt`:

    # Wood Street
    Nodes
    45 nodes
    1234567890
    1234567891
    …

    # Marland Street
    Nodes
    50 nodes
    …

Usage:
    python crowd_service/parse_osm_paste.py \
        -i crowd_service/pastes.txt \
        -o crowd_service/user_polygons.json
"""

import argparse
import json
import re
from pathlib import Path


HEADER_RE = re.compile(r"^\s*#\s*(.+?)\s*$")
# A node ID line begins with a long integer (OSM node IDs are 6-11+ digits).
# Allow trailing " (part of …)" annotations.
NODE_RE = re.compile(r"^\s*(\d{6,})(?:\s*\(.*\))?\s*$")


def parse_paste_file(path: Path) -> dict[str, list[int]]:
    """Return {street_name: [node_id, ...]} from a paste file."""
    out: dict[str, list[int]] = {}
    current: str | None = None
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = HEADER_RE.match(line)
        if m:
            current = m.group(1).strip()
            out.setdefault(current, [])
            continue
        if current is None:
            continue  # ignore preamble before the first header
        nm = NODE_RE.match(line)
        if nm:
            out[current].append(int(nm.group(1)))
    # Strip empties and dedupe (preserving first occurrence) in case OSM
    # reports the same node twice (closing vertex of a closed way).
    clean = {}
    for k, ids in out.items():
        if not ids:
            continue
        seen = set()
        deduped = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        clean[k] = deduped
    return clean


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-i", "--input", required=True,
                    help="Text file with OSM paste blocks (one per street)")
    ap.add_argument("-o", "--output",
                    default="crowd_service/user_polygons.json",
                    help="JSON output (default: crowd_service/user_polygons.json)")
    ap.add_argument("--merge", action="store_true", default=True,
                    help="Merge with existing JSON (default: on). Use --overwrite to replace.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite the output JSON instead of merging")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    new = parse_paste_file(in_path)

    if out_path.exists() and not args.overwrite:
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    existing.update(new)  # user-supplied (new) wins on collision
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2))
    total_nodes = sum(len(v) for v in existing.values())
    print(f"Wrote {out_path}  ({len(existing)} streets, {total_nodes} node ids)")
    for name, ids in existing.items():
        marker = "✓" if name in new else " "
        print(f"  {marker}  {name:<40s} {len(ids):>4d} nodes")


if __name__ == "__main__":
    main()
