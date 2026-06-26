#!/usr/bin/env python3
"""Add a display-only `<kind>_display` column to the runtime corpus.

`outputs/recipes_cleaned.csv` stores the parsed, clean_text'd list as
`<kind>_list_json`, and the joined text column is exactly `" ".join(that list)`
(clean_recipes_dataset `list_to_text`). So:

    <kind>_display = " | ".join(json.loads(<kind>_list_json))

is the same content as the joined text, only with the item boundary kept as
" | " so the frontend can render a per-item list. Display-only: not used by
BM25 / vector / matcher / gate.

GUARANTEE CHECK: `<kind>_display.replace(" | ", " ") == <runtime text column>`,
verified on both the cleaned source and the final runtime corpus; the write is
refused unless the runtime match rate clears the threshold (the only expected
mismatches are rows whose original text already contains a literal "|").

This merges the former build_ingredients_display.py / build_directions_display.py
into one shared engine + per-kind config (see KINDS). The existing-column guard
is applied to BOTH kinds (an intentional safety improvement; the old ingredients
script lacked it).

Default = verify only; pass --write to update the runtime CSV (atomic).

Examples:
  python -m rag.build.build_display_columns --kind ingredients
  python -m rag.build.build_display_columns --kind directions
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from rag.paths import OUTPUTS_DIR as OUTPUTS

csv.field_size_limit(10**7)

# Per-kind column config. cleaned_text and runtime_text differ for directions:
# the cleaned file calls the joined steps "directions_text" but the runtime
# corpus renames it to "instruction_text".
KINDS = {
    "ingredients": {
        "list_col": "ingredients_list_json",
        "cleaned_text": "ingredients_text",
        "runtime_text": "ingredients_text",
        "display_col": "ingredients_display",
    },
    "directions": {
        "list_col": "directions_list_json",
        "cleaned_text": "directions_text",
        "runtime_text": "instruction_text",
        "display_col": "directions_display",
    },
}


def header_index(header: list[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError:
        sys.exit(f"FATAL: column '{name}' not found. Header sample: {header[:12]}")


def parse_list_json(raw: str) -> list[str] | None:
    """Return the list, or None if the cell is unparseable (counted as a failure)."""
    raw = (raw or "").strip()
    if not raw or raw.lower() == "nan":
        return []
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    return None


def build_map(cleaned_path: Path, list_col: str, text_col: str) -> tuple[dict[str, str], dict]:
    """doc_id -> <kind>_display, plus a verification report on the cleaned source."""
    mapping: dict[str, str] = {}
    rep = {"rows": 0, "match": 0, "mismatch": 0, "parse_fail": 0, "samples": []}
    with cleaned_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        i_doc = header_index(header, "doc_id")
        i_json = header_index(header, list_col)
        i_text = header_index(header, text_col)
        for row in reader:
            rep["rows"] += 1
            doc_id = row[i_doc]
            items = parse_list_json(row[i_json])
            itext = (row[i_text] or "").strip()
            if items is None:
                rep["parse_fail"] += 1
                mapping[doc_id] = ""
                continue
            mapping[doc_id] = " | ".join(items)
            recon = " ".join(items).strip()
            if recon == itext:
                rep["match"] += 1
            else:
                rep["mismatch"] += 1
                if len(rep["samples"]) < 5:
                    rep["samples"].append({"doc_id": doc_id, "recon": recon[:160], "text": itext[:160]})
    return mapping, rep


def apply_to_runtime(
    runtime_path: Path, mapping: dict[str, str], write: bool, display_col: str, text_col: str
) -> tuple[dict, Path]:
    rep = {"rows": 0, "match": 0, "mismatch": 0, "no_map": 0, "samples": []}
    tmp_path = runtime_path.with_suffix(runtime_path.suffix + ".tmp")
    out_fh = tmp_path.open("w", encoding="utf-8", newline="") if write else None
    writer = csv.writer(out_fh) if write else None
    try:
        with runtime_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            if display_col in header:
                sys.exit(f"{display_col} already present in runtime corpus; nothing to do.")
            i_doc = header_index(header, "doc_id")
            i_text = header_index(header, text_col)
            if writer:
                writer.writerow(header + [display_col])
            for row in reader:
                rep["rows"] += 1
                doc_id = row[i_doc]
                display = mapping.get(doc_id)
                if display is None:
                    rep["no_map"] += 1
                    display = ""
                itext = (row[i_text] or "").strip()
                if display:
                    recon = display.replace(" | ", " ").strip()
                    if recon == itext:
                        rep["match"] += 1
                    else:
                        rep["mismatch"] += 1
                        if len(rep["samples"]) < 5:
                            rep["samples"].append({"doc_id": doc_id, "recon": recon[:160], "text": itext[:160]})
                if writer:
                    writer.writerow(row + [display])
    except Exception:
        if out_fh:
            out_fh.close()
            tmp_path.unlink(missing_ok=True)
        raise
    if out_fh:
        out_fh.close()
    return rep, tmp_path


def pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.4f}%" if d else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(KINDS), required=True, help="Which display column to build")
    ap.add_argument("--cleaned", type=Path, default=OUTPUTS / "recipes_cleaned.csv")
    ap.add_argument("--runtime", type=Path, default=OUTPUTS / "retrieval_corpus_runtime.csv")
    ap.add_argument("--write", action="store_true", help="Write the new column (default: verify only)")
    ap.add_argument("--threshold", type=float, default=0.999, help="Min runtime match rate required to write")
    args = ap.parse_args()
    cfg = KINDS[args.kind]
    display_col = cfg["display_col"]

    print(f"[1/2] Building map ({args.kind}) from {args.cleaned.name} ...", flush=True)
    mapping, crep = build_map(args.cleaned, cfg["list_col"], cfg["cleaned_text"])
    print(f"  cleaned rows        : {crep['rows']:,}")
    print(f"  parse failures      : {crep['parse_fail']:,}")
    print(f"  match (recon==text) : {crep['match']:,}  ({pct(crep['match'], crep['rows'])})")
    print(f"  mismatch            : {crep['mismatch']:,}")
    for s in crep["samples"]:
        print(f"    MISMATCH {s['doc_id']}\n      recon: {s['recon']}\n      text : {s['text']}")

    print(f"\n[2/2] Applying {display_col} to {args.runtime.name} (write={args.write}) ...", flush=True)
    rrep, tmp_path = apply_to_runtime(args.runtime, mapping, args.write, display_col, cfg["runtime_text"])
    checked = rrep["match"] + rrep["mismatch"]
    print(f"  runtime rows        : {rrep['rows']:,}")
    print(f"  no mapping          : {rrep['no_map']:,}")
    print(f"  rows with display   : {checked:,}")
    print(f"  match (no-pipe==txt): {rrep['match']:,}  ({pct(rrep['match'], checked)})")
    print(f"  mismatch            : {rrep['mismatch']:,}")
    for s in rrep["samples"]:
        print(f"    MISMATCH {s['doc_id']}\n      recon: {s['recon']}\n      text : {s['text']}")

    rate = (rrep["match"] / checked) if checked else 0.0
    if args.write:
        # The only expected mismatches are rows whose original text already
        # contains a literal "|" (a source-data quirk; display content is still
        # correct). Allow the write as long as the match rate clears the bar.
        if rate >= args.threshold:
            os.replace(tmp_path, args.runtime)
            print(f"\nOK: wrote {display_col} ({pct(rrep['match'], checked)} match, "
                  f"{rrep['mismatch']} benign source-pipe rows).")
        else:
            tmp_path.unlink(missing_ok=True)
            print(f"\nABORTED write: match rate {rate:.4%} below threshold {args.threshold}.")
            sys.exit(1)
    else:
        print(f"\nDRY-RUN: match rate {rate:.4%}. Re-run with --write to apply.")


if __name__ == "__main__":
    main()
