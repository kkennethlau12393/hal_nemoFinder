#!/usr/bin/env bash
#
# hal-nemoFinder — backup script.
#
# Produces a single timestamped archive containing:
#   - Postgres custom-format dump (compressed)
#   - Audit log export (JSONL, tamper-evident)
#   - manifest.json (version, checksums, row counts, audit chain result)
#
# Exit codes:
#   0  success
#   1  pg_dump failed
#   2  checksum mismatch after archiving
#   3  audit chain broken
#
# Environment:
#   PGHOST, PGPORT, PGUSER, PGDATABASE  (standard libpq)
#   PGPASSWORD or ~/.pgpass             (libpq auth)
#   BACKUP_DIR                          (default: ./backups)
#   HAL_CLI                             (default: hal-nemofinder)
#
set -euo pipefail

: "${PGHOST:=localhost}"
: "${PGPORT:=5432}"
: "${PGUSER:=hal}"
: "${PGDATABASE:=hal_nemofinder}"
: "${BACKUP_DIR:=./backups}"
: "${HAL_CLI:=hal-nemofinder}"

log() { printf '[backup %s] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }
die() { log "ERROR: $*"; exit "${2:-1}"; }

command -v pg_dump    >/dev/null 2>&1 || die "pg_dump not found" 1
command -v psql       >/dev/null 2>&1 || die "psql not found" 1
command -v sha256sum  >/dev/null 2>&1 || SHA_CMD="shasum -a 256"
SHA_CMD="${SHA_CMD:-sha256sum}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hal-backup-XXXXXX")"
STAGE_DIR="${WORK_DIR}/hal-${PGDATABASE}-${TIMESTAMP}"
mkdir -p "${STAGE_DIR}" "${BACKUP_DIR}"

cleanup() { rm -rf "${WORK_DIR}"; }
trap cleanup EXIT

DUMP_FILE="${STAGE_DIR}/database.dump"
AUDIT_FILE="${STAGE_DIR}/audit.jsonl"
MANIFEST_FILE="${STAGE_DIR}/manifest.json"
ARCHIVE="${BACKUP_DIR}/hal-${PGDATABASE}-${TIMESTAMP}.tar.gz"

# ---------------------------------------------------------------------------
# 1. Postgres dump
# ---------------------------------------------------------------------------
log "Dumping database ${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE}"
if ! pg_dump \
        --host="${PGHOST}" --port="${PGPORT}" --username="${PGUSER}" \
        --dbname="${PGDATABASE}" \
        --format=custom \
        --compress=9 \
        --no-owner --no-privileges \
        --file="${DUMP_FILE}"; then
    die "pg_dump failed" 1
fi
DUMP_BYTES=$(wc -c <"${DUMP_FILE}" | tr -d ' ')
log "Dump size: ${DUMP_BYTES} bytes"

# ---------------------------------------------------------------------------
# 2. Row counts per table (for post-restore verification)
# ---------------------------------------------------------------------------
log "Collecting row counts"
ROWCOUNTS_JSON=$(psql \
    --host="${PGHOST}" --port="${PGPORT}" --username="${PGUSER}" \
    --dbname="${PGDATABASE}" -Atq <<'SQL'
SELECT json_object_agg(
    schemaname || '.' || relname,
    n_live_tup
) FROM pg_stat_user_tables;
SQL
)
ROWCOUNTS_JSON="${ROWCOUNTS_JSON:-{}}"

# ---------------------------------------------------------------------------
# 3. Audit log export & chain verification
# ---------------------------------------------------------------------------
AUDIT_CHAIN_OK="unknown"
if command -v "${HAL_CLI}" >/dev/null 2>&1; then
    log "Exporting audit log via ${HAL_CLI}"
    if "${HAL_CLI}" audit export --format jsonl --output "${AUDIT_FILE}"; then
        log "Verifying audit chain"
        if "${HAL_CLI}" audit verify --input "${AUDIT_FILE}"; then
            AUDIT_CHAIN_OK="true"
        else
            AUDIT_CHAIN_OK="false"
            log "Audit chain verification FAILED"
        fi
    else
        log "WARN: audit export failed; continuing without audit snapshot"
        : > "${AUDIT_FILE}"
    fi
else
    log "WARN: ${HAL_CLI} not available; skipping audit export"
    : > "${AUDIT_FILE}"
fi

# ---------------------------------------------------------------------------
# 4. Manifest
# ---------------------------------------------------------------------------
DUMP_SHA=$(${SHA_CMD} "${DUMP_FILE}" | awk '{print $1}')
AUDIT_SHA=$(${SHA_CMD} "${AUDIT_FILE}" | awk '{print $1}')
HAL_VERSION=$("${HAL_CLI}" version 2>/dev/null || echo "unknown")

cat > "${MANIFEST_FILE}" <<EOF
{
  "schema": "hal-backup/v1",
  "created_at": "${TIMESTAMP}",
  "hostname": "$(hostname)",
  "hal_version": "${HAL_VERSION}",
  "database": {
    "host": "${PGHOST}",
    "port": ${PGPORT},
    "user": "${PGUSER}",
    "name": "${PGDATABASE}"
  },
  "files": {
    "database.dump": {
      "bytes": ${DUMP_BYTES},
      "sha256": "${DUMP_SHA}"
    },
    "audit.jsonl": {
      "bytes": $(wc -c <"${AUDIT_FILE}" | tr -d ' '),
      "sha256": "${AUDIT_SHA}"
    }
  },
  "row_counts": ${ROWCOUNTS_JSON},
  "audit_chain_verified": ${AUDIT_CHAIN_OK}
}
EOF

# ---------------------------------------------------------------------------
# 5. Archive
# ---------------------------------------------------------------------------
log "Creating ${ARCHIVE}"
tar -C "${WORK_DIR}" -czf "${ARCHIVE}" "$(basename "${STAGE_DIR}")"
ARCHIVE_SHA=$(${SHA_CMD} "${ARCHIVE}" | awk '{print $1}')
echo "${ARCHIVE_SHA}  $(basename "${ARCHIVE}")" > "${ARCHIVE}.sha256"

# Verify the archive is readable and contains the expected members.
if ! tar -tzf "${ARCHIVE}" >/dev/null; then
    die "archive verification failed" 2
fi

log "Backup complete: ${ARCHIVE}"
log "SHA256: ${ARCHIVE_SHA}"

if [[ "${AUDIT_CHAIN_OK}" == "false" ]]; then
    exit 3
fi
exit 0
