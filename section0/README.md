# Section 0: Feasibility

Goal: verify, with numbers, that the data can support the project before any design is locked.

| Question | Script |
|---|---|
| History depth, gaps, rows per month, schema stability | `s0_inventory.py` |
| Column semantics, nulls, trip linkage, cancellations, hub coverage, connection volume and miss rates | `s0_processed.py` |
| Can DB's own prognosis be reconstructed at 60/30/10 min lead times? Time zones? | `s0_raw.py` |

## Run (from the repo root, PowerShell)

```powershell
py -m pip install -r requirements-section0.txt
py section0/s0_inventory.py
py section0/s0_processed.py --month 2026-08
py section0/s0_raw.py --day 2026-08-12
```

Downloads go to `section0/data/` (git-ignored). Reports go to `section0/out/*.txt`.

## Assumptions to verify in the output

- Hub EVA numbers (section 8 of the processed report runs a name check).
- A null change time means "no change known", i.e. on time (section 5 checks this).
- Raw request timestamps are UTC and processed timestamps are Europe/Berlin local time (the time zone sanity lines in the raw report check this; rerun with `--req-ts-tz` or `--processed-tz` if they show a shift of 60 or 120 minutes).
