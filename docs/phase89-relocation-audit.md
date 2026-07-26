# Phase 8.9 relocation audit

The first live Phase 8.9 audit showed that the current SQLite index contains both stale rows and mounted files that are not indexed.

A normal indexer run is not a safe reconciliation mechanism because the current indexer inserts and updates discovered files but does not remove rows whose files are absent. It can also create a new row and stable ID for a file that was merely moved.

## Read-only relocation analysis

`phase89_relocation_audit.py` consumes the full JSON output from `phase89_audit.py` and compares:

- current-family DB paths whose file is absent from the mount;
- mounted files absent from the DB;
- legacy DB paths whose top-level family no longer exists on the mount.

It only declares a unique relocation candidate when exactly one stale path and exactly one missing mounted path share the same case-insensitive filename. It does not inspect safetensors content and does not write to SQLite.

Run on Nibbler:

```bash
python3 /tmp/phase89_relocation_audit.py \
  --audit-json /home/clink/docker/lora_builder/data/phase89_audit.json \
  --json /home/clink/docker/lora_builder/data/phase89_relocation_audit.json
```

## First live result, 26 July 2026

- Stale rows within currently mounted families: 691
- Mounted files absent from the DB: 334
- Unique exact-filename relocation candidates: 23
- Ambiguous exact-filename candidates: 0
- Legacy/unmounted-family DB rows: 114

Observed relocation transitions:

- 20 from WAN2.2 to WAN2.1
- 2 within WAN2.2, from Action to Body
- 1 within Z-Image, from Action to Body

Legacy/unmounted DB families:

- Hunyuna_15: 72
- SD: 38
- Wan Video 2.2 T2V-A14B: 2
- Flux.1 D: 1
- SDXL-Lightning: 1

## Safety conclusion

Do not run an uncontrolled full rescan or delete stale rows yet.

The 23 unique filename matches are candidates for a future stable-ID-preserving path migration, not automatic proof that the files are identical. A write-capable reconciliation phase must include:

1. database backup;
2. dry-run plan;
3. stable-ID collision checks;
4. optional stronger identity checks such as file size or hash;
5. explicit approval before applying updates or deletions;
6. a post-change read-only audit.

The remaining unmatched rows and files require separate review or controlled indexing after model-family mappings are defined.
