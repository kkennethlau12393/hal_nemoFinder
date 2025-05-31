# hal-nemoFinder — Disaster Recovery Plan

Owner: Platform SRE
Review cadence: quarterly (including a live drill)

## 1. Targets

| Metric | Target | Justification |
|---|---|---|
| **RTO** (Recovery Time Objective) | **4 hours** | Non-interactive scientific workload; half-day downtime tolerated. |
| **RPO** (Recovery Point Objective) | **15 minutes** | Continuous WAL archival + nightly base backup. |
| **Audit RPO** | **0** | Audit log is synchronously mirrored to append-only storage. |
| **Critical-path restore drill** | Quarterly | Documented & timed. |

## 2. Scope

| Component | State | DR strategy |
|---|---|---|
| API pods | Stateless | Helm re-deploy |
| Celery workers | Stateless | Helm re-deploy |
| PostgreSQL (with pgvector) | **Stateful** | HA primary/replica + WAL-G to S3/GCS |
| Redis (broker + cache) | Broker = semi-durable; cache = ephemeral | Sentinel or managed HA; cache rebuilt on startup |
| Object storage (KB caches, artefacts) | Stateful | Bucket versioning + cross-region replication |
| Audit log | **Critical, immutable** | Synchronous write to append-only store, daily offline snapshots |
| Secrets (HMAC, OIDC, DB creds) | Stateful | Vault / external-secrets; BYOK in KMS; backup per vendor guidance |

## 3. Backup strategy

### 3.1 Postgres

- **Base backup**: nightly `scripts/backup.sh` via CronJob → S3
  (`s3://hal-backups/db/YYYY/MM/DD/`).
- **WAL archive**: continuous push (e.g. WAL-G or pgBackRest) with
  15 min archive_timeout.
- **Retention**:
  - Daily for 30 days
  - Weekly for 12 weeks
  - Monthly for 7 years (GxP)
- **Integrity**: every backup run writes `manifest.json` with
  per-table row counts, SHA-256 hashes and the audit-chain
  verification result (exit code 3 if the chain is broken).

### 3.2 Redis

- The Celery **broker** database (`redis://.../1`) is snapshotted
  via AOF + RDB every 5 minutes.  Losing the broker means losing
  in-flight tasks; the application is designed to retry idempotently.
- The **cache** database (`redis://.../0`) is explicitly **not**
  backed up; it is rebuilt from Postgres + external KBs on startup.

### 3.3 Audit log

- Every `AuditRecord` write is mirrored to an append-only store
  (WORM S3, Azure Immutable Blob, or an external ledger) via the
  `audit_sink` configuration.
- Daily offline snapshot: `hal-nemofinder audit export` →
  object storage with versioning + legal hold.

### 3.4 Secrets

- Vault snapshots every 6 h to a separate account / project.
- HMAC keys archived for 7 years (see RUNBOOK § 8).

## 4. Restore procedure

### 4.1 Database

```bash
# 1. Pick the most recent usable base backup
aws s3 ls s3://hal-backups/db/ --recursive | tail -20

# 2. Pull & restore
aws s3 cp s3://hal-backups/db/2026/04/13/hal-hal_nemofinder-20260413T040000Z.tar.gz .
scripts/restore.sh hal-hal_nemofinder-20260413T040000Z.tar.gz

# 3. Replay WAL to the target PITR point (if using WAL-G)
wal-g backup-fetch /var/lib/postgresql/data LATEST
wal-g wal-fetch <wal> /var/lib/postgresql/data/pg_wal/<wal>
# ...or rely on Patroni / pgBackRest recovery.conf.
```

### 4.2 Application (Helm)

```bash
helm install hal helm/hal-nemofinder -n hal -f values-prod.yaml --atomic
kubectl -n hal rollout status deploy/hal-hal-nemofinder-api
```

### 4.3 Audit chain recheck

```bash
kubectl -n hal exec deploy/hal-hal-nemofinder-api -- \
    hal-nemofinder audit verify
```

A failure here means the tamper-evident log was broken during the
incident window — this is a **P0 security event**. Page the
security officer before bringing the service back online.

## 5. Postgres failover

Assumes a streaming replica in a second AZ managed by Patroni /
cloud provider.

| Step | Command | Expected |
|---|---|---|
| 1. Detect | Patroni cluster has one healthy replica, primary `DOWN` | |
| 2. Promote | `patronictl -c /etc/patroni.yml failover` | Replica becomes primary |
| 3. Update Helm secret | Point `externalDatabase.host` at the new primary (or rely on VIP/DNS) | |
| 4. Restart API & workers | `kubectl rollout restart` | Pods reconnect |
| 5. Re-verify | Health endpoints green, audit chain intact | |

## 6. Redis failover

- Sentinel / managed failover is transparent: Celery reconnects
  with exponential backoff.
- After failover, confirm no queue is orphaned:
  ```bash
  redis-cli LLEN analysis
  redis-cli LLEN verification
  ```
  If either is stuck, `rollout restart` the worker deployment.

## 7. Disaster drill checklist (run quarterly)

- [ ] Spin up a recovery namespace in a secondary region
- [ ] Restore yesterday's base backup into an isolated Postgres
- [ ] `scripts/restore.sh` completes with exit code 0
- [ ] Row counts match manifest
- [ ] `hal-nemofinder audit verify` succeeds
- [ ] `helm install` into the recovery namespace
- [ ] Smoke test: `curl /api/v1/health/ready` → 200
- [ ] Submit a known-good analysis; result matches production
- [ ] Time the end-to-end exercise; record vs. the 4 h RTO
- [ ] File drill report and address any gaps

## 8. Communication

During a real incident:

1. Open an incident channel (`#inc-hal-<date>`).
2. Post status to the status page.
3. Designate IC, comms, and scribe roles.
4. Executive updates every 30 min.
5. Post-incident review within 5 business days; action items
   tracked to closure.
