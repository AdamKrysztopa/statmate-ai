#!/usr/bin/env bash
# Backup database + results/logs directory into tmp_devcontainer/backups
set -euo pipefail
STAMP=$(date +"%Y%m%d-%H%M%S")
DEST="tmp_devcontainer/backups"
mkdir -p "$DEST"
tar -czf "$DEST/statmate-backup-$STAMP.tgz" data/ || {
  echo "Backup failed" >&2
  exit 1
}
echo "Backup written to $DEST/statmate-backup-$STAMP.tgz"
