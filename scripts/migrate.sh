#!/usr/bin/env bash
# Wrapper around `alembic upgrade head` with sensible defaults.
#
# Usage:
#   ./scripts/migrate.sh                  # upgrade to head
#   ./scripts/migrate.sh downgrade -1     # roll back one revision
#   ./scripts/migrate.sh current          # show current revision
#
# Environment variables:
#   HAL_DATABASE_URL       overrides the async URL used by alembic env.py
#   HAL_SYNC_DATABASE_URL  overrides the sync URL in alembic.ini

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if ! command -v alembic >/dev/null 2>&1; then
    echo "error: alembic is not installed (try: pip install alembic)" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    exec alembic upgrade head
fi

exec alembic "$@"
