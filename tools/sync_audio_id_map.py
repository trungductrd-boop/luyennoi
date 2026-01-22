#!/usr/bin/env python3
"""
Sync `data/samples/audio_id_map.json` into the persisted `helpers.PERSISTED_STORE['samples']`.
- Adds or updates entries with `filename`, `lesson_id`: None, `vocab_id`: None.
- Safe: supports `--dry-run` to preview changes without writing.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import helpers

AUDIO_MAP = Path("data/samples/audio_id_map.json")


def normalize_basename(p: str) -> str:
    return os.path.basename(p) if p else p


def main(dry_run: bool = False):
    if not AUDIO_MAP.exists():
        print(f"ERROR: {AUDIO_MAP} not found")
        return 2

    try:
        m = json.loads(AUDIO_MAP.read_text(encoding="utf-8"))
    except Exception as e:
        print("ERROR: failed to read audio_id_map.json:", e)
        return 2

    helpers.load_persisted_store()
    samples = helpers.PERSISTED_STORE.setdefault("samples", {})

    added = 0
    updated = 0
    unchanged = 0
    missing_files = []

    for sid, mapped in m.items():
        fname = normalize_basename(mapped)
        if not fname:
            print("Skipping empty mapping for", sid)
            continue
        entry = samples.get(sid)
        if entry:
            if entry.get("filename") != fname:
                print(f"Will update sample {sid}: {entry.get('filename')} -> {fname}")
                if not dry_run:
                    entry["filename"] = fname
                    samples[sid] = entry
                updated += 1
            else:
                unchanged += 1
        else:
            print(f"Will add sample {sid}: {fname}")
            if not dry_run:
                samples[sid] = {"filename": fname, "lesson_id": None, "vocab_id": None}
            added += 1

        # check file presence on disk
        sample_path = os.path.join(helpers.SAMPLES_DIR, fname)
        if not os.path.exists(sample_path):
            missing_files.append(fname)

    print("\nSummary:")
    print(f"  entries_processed: {len(m)}")
    print(f"  added: {added}")
    print(f"  updated: {updated}")
    print(f"  unchanged: {unchanged}")
    print(f"  missing_files_on_disk: {len(missing_files)}")
    if missing_files:
        print("  Missing files (may need to copy assets into data/samples):")
        for fn in sorted(set(missing_files)):
            print("   -", fn)

    if dry_run:
        print("\nDry-run mode, no changes written.")
        return 0

    # backup persisted store then save
    try:
        bak = helpers.VOCAB_STORE_FILE + ".bak"
        shutil.copy2(helpers.VOCAB_STORE_FILE, bak)
        print(f"Backup saved to {bak}")
    except Exception:
        print("Warning: failed to create backup of persisted store")

    try:
        helpers.save_persisted_store()
        print("Persisted store updated: ", helpers.VOCAB_STORE_FILE)
    except Exception as e:
        print("ERROR: failed to save persisted store:", e)
        return 2

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sync audio_id_map.json into persisted samples store")
    p.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = p.parse_args()
    rc = main(dry_run=args.dry_run)
    raise SystemExit(rc)
