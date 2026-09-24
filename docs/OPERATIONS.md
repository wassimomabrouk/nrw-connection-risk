# Operations

How the data collector runs in production, how it is monitored, and how to recover from failures.

## Architecture

| Component | Where | Purpose |
|---|---|---|
| Collector (`nrw-collector.service`) | Oracle Cloud Always Free VM (Ampere A1, 1 OCPU, 6 GB, Frankfurt), Ubuntu 24.04 | Polls the DB Timetables API for the five hubs and writes the raw and parsed layers |
| Raw layer | `data/collector/raw/source=*/date=*/hour=*.jsonl.gz` | Every API response, untouched. **Source of truth.** |
| Parsed layer | `data/collector/parsed/source=*/date=*/*.parquet` | One row per stop, event and observation. Derived; can be rebuilt from raw |
| Heartbeat | `data/collector/heartbeat.json` | Last success per endpoint, success and failure counts |
| Monitoring | healthchecks.io, two checks | `nrw-collector`: ping every 5 min while API calls succeed, alert after 15 min of silence. `nrw-backup`: ping after each successful daily backup, alert after 30 hours |
| Backup (`nrw-backup.timer`) | Google Drive via rclone | Daily copy of the raw layer at about 03:30 UTC. `copy` semantics: deletions on the server never propagate |

Unit files are versioned in [`deploy/`](../deploy). Secrets live only in `.env` on the server (mode 600) and in `~/.config/rclone/rclone.conf` (mode 600); neither is committed.

## Everyday commands

All commands run from a local shell; replace `KEY` and `HOST` with the SSH key path and the server address.

```
ssh -i KEY ubuntu@HOST "systemctl is-active nrw-collector"                       # is it running?
ssh -i KEY ubuntu@HOST "cat ~/nrw-connection-risk/data/collector/heartbeat.json"   # live counters
ssh -i KEY ubuntu@HOST "journalctl -u nrw-collector -n 50 --no-pager"             # recent logs
ssh -i KEY ubuntu@HOST "systemctl list-timers nrw-backup.timer --no-pager"        # next backup
ssh -i KEY ubuntu@HOST "rclone size gdrive:nrw-connection-risk-backup/raw"        # backup size
```

## Deploying a code change

```
ssh -i KEY ubuntu@HOST "cd ~/nrw-connection-risk && git pull && .venv/bin/pip install -e . && .venv/bin/python -m pytest -q && sudo systemctl restart nrw-collector"
```

The service handles SIGTERM: it finishes the current request and flushes buffered rows before stopping, so a restart loses no data. The gap is a few seconds; `fchg` (full state every 30 minutes) repairs anything `rchg` missed.

## Failure scenarios

| Symptom | Likely cause | Action |
|---|---|---|
| `nrw-collector` alert email | Server stopped (for example Oracle idle reclamation), crash loop, or API outage | Check the instance state in the Oracle console and start it if stopped. Otherwise read `journalctl`. An API outage resolves itself; the collector retries every cycle |
| `nrw-backup` alert email | Google token revoked or expired, Drive full, network | Run `sudo systemctl start nrw-backup.service` and read `journalctl -u nrw-backup`. If the token is invalid, run `rclone config reconnect gdrive:` on a machine with a browser and copy `rclone.conf` to the server again |
| Rising `failure_count` in the heartbeat | DB API errors or rate limiting | The client retries with backoff and stays at 50 of the allowed 60 calls per minute. Persistent HTTP 401 or 403 means the API key or subscription is invalid |

## Restore

1. Copy the raw layer back from the backup:
   `rclone copy gdrive:nrw-connection-risk-backup/raw data/collector/raw`
2. Rebuild the parsed layer from raw:
   `python tools/rebuild_parsed.py --out data/collector`
3. Check the result: `python tools/collector_status.py`

Data collected between the last backup and the failure exists only on the server's disk. An Oracle idle stop keeps the disk, so nothing is lost in that case; only a terminated instance loses up to one day of data.
