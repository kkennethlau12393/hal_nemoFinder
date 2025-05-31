#!/usr/bin/env bash
#
# Generate a CycloneDX SBOM for hal-nemofinder.
#
# Usage:
#   scripts/generate_sbom.sh [version]
#
# Output:
#   sbom/sbom-<version>.cdx.json
#
# Prefers `syft` (container + filesystem aware, broader coverage).
# Falls back to `cyclonedx-py` for a pure Python SBOM.
#
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VERSION="${1:-$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])' 2>/dev/null || echo "0.0.0")}"
OUT_DIR="${REPO_ROOT}/sbom"
OUT_FILE="${OUT_DIR}/sbom-${VERSION}.cdx.json"

mkdir -p "${OUT_DIR}"

log() { printf '[sbom] %s\n' "$*" >&2; }

if command -v syft >/dev/null 2>&1; then
    log "Using syft $(syft version 2>/dev/null | head -1)"
    syft "dir:${REPO_ROOT}" \
        --output "cyclonedx-json=${OUT_FILE}" \
        --source-name hal-nemofinder \
        --source-version "${VERSION}"
elif command -v cyclonedx-py >/dev/null 2>&1; then
    log "Using cyclonedx-py"
    cyclonedx-py environment \
        --output-format JSON \
        --output-file "${OUT_FILE}" \
        --spec-version 1.5
elif python -m cyclonedx_py --help >/dev/null 2>&1; then
    log "Using python -m cyclonedx_py"
    python -m cyclonedx_py environment \
        --output-format JSON \
        --output-file "${OUT_FILE}"
else
    cat >&2 <<EOF
[sbom] ERROR: no SBOM generator found.

Install one of:
    pipx install cyclonedx-bom            # Python tool
    brew install syft                     # or download from https://github.com/anchore/syft
    curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
EOF
    exit 1
fi

# Sanity-check the output.
if ! python -c "import json,sys; json.load(open('${OUT_FILE}'))" 2>/dev/null; then
    log "ERROR: generated SBOM is not valid JSON"
    exit 2
fi

SIZE=$(wc -c < "${OUT_FILE}" | tr -d ' ')
COMPONENTS=$(python -c "import json; d=json.load(open('${OUT_FILE}')); print(len(d.get('components',[])))")
log "Wrote ${OUT_FILE} (${SIZE} bytes, ${COMPONENTS} components)"
