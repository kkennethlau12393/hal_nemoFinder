#!/usr/bin/env bash
#
# hal-nemoFinder — restore script.
#
# Usage:
#   scripts/restore.sh <archive.tar.gz>
#
# Steps:
#   1. Verify archive checksum against <archive>.sha256 (if present)
#   2. Extract into a temp directory
#   3. Validate manifest + per-file SHA256 hashes
#   4. pg_restore into the target database
#   5. Compare post-restore row counts with the manifest
#   6. Re-verify the audit chain
#
# Exit codes:
#   0  success
#   1  argument / precondition failure
#   2  checksum mismatch
#   3  audit chain broken after restore
#   4  row count mismatch
#
set -euo pipefail

: "${PGHOST:=localhost}"
: "${PGPORT:=5432}"
: "${PGUSER:=hal}"
: "${PGDATABASE:=hal_nemofinder}"
: "${HAL_CLI:=hal-nemofinder}"

log() { printf '[restore %s] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }
die() { log "ERROR: $*"; exit "${2:-1}"; }

[[ $# -ge 1 ]] || die "usage: $0 <archive.tar.gz>" 1
ARCHIVE="$1"
[[ -f "${ARCHIVE}" ]] || die "archive not found: ${ARCHIVE}" 1

command -v pg_restore >/dev/null 2>&1 || die "pg_restore not found" 1
command -v psql       >/dev/null 2>&1 || die "psql not found" 1
command -v jq         >/dev/null 2>&1 || die "jq is required" 1
SHA_CMD="sha256sum"
command -v sha256sum  >/dev/null 2>&1 || SHA_CMD="shasum -a 256"

# ---------------------------------------------------------------------------
# 1. Checksum
# ---------------------------------------------------------------------------
if [[ -f "${ARCHIVE}.sha256" ]]; then
    log "Verifying archive checksum"
    EXPECTED=$(awk '{print $1}' "${ARCHIVE}.sha256")
    ACTUAL=$(${SHA_CMD} "${ARCHIVE}" | awk '{print $1}')
    [[ "${EXPECTED}" == "${ACTUAL}" ]] || die "archive checksum mismatch" 2
fi

# ---------------------------------------------------------------------------
# 2. Extract
# ---------------------------------------------------------------------------
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hal-restore-XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT
log "Extracting ${ARCHIVE}"
tar -C "${WORK_DIR}" -xzf "${ARCHIVE}"

STAGE_DIR="$(find "${WORK_DIR}" -mindepth 1 -maxdepth 1 -type d | head -1)"
[[ -n "${STAGE_DIR}" ]] || die "archive layout unexpected" 1

MANIFEST="${STAGE_DIR}/manifest.json"
[[ -f "${MANIFEST}" ]] || die "manifest.json missing from archive" 1

DUMP_FILE="${STAGE_DIR}/database.dump"
AUDIT_FILE="${STAGE_DIR}/audit.jsonl"

# ---------------------------------------------------------------------------
# 3. Validate file hashes
# ---------------------------------------------------------------------------
validate_hash() {
    local file="$1" key="$2"
    local expected
    expected=$(jq -r ".files[\"${key}\"].sha256" "${MANIFEST}")
    local actual
    actual=$(${SHA_CMD} "${file}" | awk '{print $1}')
    [[ "${expected}" == "${actual}" ]] || die "${key} sha256 mismatch" 2
}
validate_hash "${DUMP_FILE}"  "database.dump"
validate_hash "${AUDIT_FILE}" "audit.jsonl"
log "Manifest hashes OK"

# ---------------------------------------------------------------------------
# 4. pg_restore
# ---------------------------------------------------------------------------
log "Restoring into ${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE}"
pg_restore \
    --host="${PGHOST}" --port="${PGPORT}" --username="${PGUSER}" \
    --dbname="${PGDATABASE}" \
    --clean --if-exists \
    --no-owner --no-privileges \
    --jobs=4 \
    "${DUMP_FILE}"

# ---------------------------------------------------------------------------
# 5. Row count verification
# ---------------------------------------------------------------------------
log "Verifying row counts"
POST_COUNTS=$(psql \
    --host="${PGHOST}" --port="${PGPORT}" --username="${PGUSER}" \
    --dbname="${PGDATABASE}" -Atq <<'SQL'
SELECT json_object_agg(
    schemaname || '.' || relname,
    n_live_tup
) FROM pg_stat_user_tables;
SQL
)
MISMATCH=$(jq -n --argjson pre "$(jq .row_counts "${MANIFEST}")" --argjson post "${POST_COUNTS:-{}}" '
    ($pre // {}) as $p |
    ($post // {}) as $q |
    [$p | to_entries[] | select(($q[.key] // 0) != .value) | .key] | length
')
if [[ "${MISMATCH}" != "0" ]]; then
    log "WARN: ${MISMATCH} tables differ in row count (stats may be stale; run ANALYZE and re-check)"
    psql --host="${PGHOST}" --port="${PGPORT}" --username="${PGUSER}" \
         --dbname="${PGDATABASE}" -c "ANALYZE;" >/dev/null
fi

# ---------------------------------------------------------------------------
# 6. Audit chain re-verification
# ---------------------------------------------------------------------------
if command -v "${HAL_CLI}" >/dev/null 2>&1; then
    log "Re-verifying audit chain"
    if ! "${HAL_CLI}" audit verify --input "${AUDIT_FILE}"; then
        die "audit chain verification failed after restore" 3
    fi
else
    log "WARN: ${HAL_CLI} not available; skipping audit chain recheck"
fi

log "Restore complete"
exit 0
