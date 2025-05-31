# hal-nemoFinder — Operations Runbook

Audience: SRE / DevOps on-call.
Scope: Kubernetes deployment via the bundled Helm chart.

---

## 1. Deploy

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm dependency update helm/hal-nemofinder
helm install hal helm/hal-nemofinder \
    -n hal --create-namespace \
    -f values-prod.yaml \
    --atomic --timeout 15m
```

Verify:

```bash
kubectl -n hal rollout status deploy/hal-hal-nemofinder-api
kubectl -n hal rollout status deploy/hal-hal-nemofinder-worker
kubectl -n hal run --rm -it curl --image=curlimages/curl -- \
    curl -sSf http://hal-hal-nemofinder-api:8000/api/v1/health/ready
```

## 2. Upgrade

```bash
helm repo update
helm upgrade hal helm/hal-nemofinder -n hal \
    -f values-prod.yaml \
    --atomic --timeout 15m
```

The `pre-upgrade` hook runs `alembic upgrade head`.  `--atomic`
rolls the release back automatically if any resource fails to
become ready.

## 3. Rollback

```bash
helm history hal -n hal
helm rollback hal <REVISION> -n hal --wait
```

Database migrations are forward-only.  If rolling back across a
schema change:

1. Restore the database from the last pre-upgrade backup (§ 7).
2. `helm rollback` to the earlier revision.
3. Re-verify audit chain with `hal-nemofinder audit verify`.

## 4. Scale

API (horizontal, stateless):

```bash
kubectl -n hal scale deploy/hal-hal-nemofinder-api --replicas=10
# or edit the HPA:
kubectl -n hal patch hpa hal-hal-nemofinder-api --type merge \
    -p '{"spec":{"minReplicas":5,"maxReplicas":40}}'
```

Workers (scale with queue depth, not CPU):

```bash
kubectl -n hal scale deploy/hal-hal-nemofinder-worker --replicas=20
```

Scale limits are bounded by Postgres connection capacity
(`max_connections`) and Redis memory. Budget roughly
`2 × (api.replicas × 10 + worker.replicas × concurrency)` connections.

## 5. Restart a stuck worker

```bash
kubectl -n hal get pods -l app.kubernetes.io/component=worker
kubectl -n hal logs <pod> --tail=200
kubectl -n hal exec <pod> -- hal-nemofinder worker inspect --active
kubectl -n hal delete pod <pod>     # rolling replacement
```

If **all** workers are stuck, inspect Redis for a broken broker
state (the queue name is `analysis` / `verification`):

```bash
kubectl -n hal exec deploy/hal-redis-master -- \
    redis-cli -a "$REDIS_PASSWORD" LLEN analysis
```

If depth keeps growing with 0 consumers, flush in-flight task
acknowledgements from the unacked Redis hash and restart all
workers at once (`kubectl rollout restart`).

## 6. Investigate a failing verifier

```bash
kubectl -n hal exec deploy/hal-hal-nemofinder-api -- \
    hal-nemofinder list verifiers

kubectl -n hal logs deploy/hal-hal-nemofinder-worker \
    --since=30m | grep -E 'verifier=|ERROR' | less
```

Typical failure modes:

| Symptom                              | Cause                                  | Remedy |
|--------------------------------------|----------------------------------------|--------|
| `429 Too Many Requests` from PubChem | upstream throttling                    | Raise `HAL_CACHE_TTL_SECONDS`, back off. |
| RDKit `Sanitize` errors              | pathological SMILES in the input       | Quarantine claim; file an extractor bug. |
| `pgvector` dimension mismatch        | model change without migration         | Rebuild embeddings via `hal-nemofinder embeddings rebuild`. |
| Verifier times out                   | slow external KB                       | Tune `HAL_VERIFICATION_TIMEOUT_SECONDS`. |

## 7. Restore from backup

```bash
# 1. Locate the archive
ls -lh /backups/hal-hal_nemofinder-*.tar.gz

# 2. Stop the API & worker (drain connections)
kubectl -n hal scale deploy/hal-hal-nemofinder-api --replicas=0
kubectl -n hal scale deploy/hal-hal-nemofinder-worker --replicas=0

# 3. Run the restore script inside a privileged maintenance pod
kubectl -n hal exec -it deploy/hal-hal-nemofinder-api -- \
    scripts/restore.sh /backups/hal-hal_nemofinder-20260413T040000Z.tar.gz

# 4. Scale the workloads back up
kubectl -n hal scale deploy/hal-hal-nemofinder-api --replicas=3
kubectl -n hal scale deploy/hal-hal-nemofinder-worker --replicas=5

# 5. Re-verify audit chain and smoke-test
kubectl -n hal exec deploy/hal-hal-nemofinder-api -- \
    hal-nemofinder audit verify
```

## 8. Rotate `AUDIT_HMAC_KEY`

The audit log is HMAC-chained; rotating the key must be a two-phase
operation so the old segments remain verifiable.

```bash
# 1. Freeze new audit writes
kubectl -n hal annotate deploy hal-hal-nemofinder-api \
    hal.nemofinder/audit-freeze=true --overwrite
kubectl -n hal rollout restart deploy/hal-hal-nemofinder-api

# 2. Export & verify the current chain
kubectl -n hal exec deploy/hal-hal-nemofinder-api -- \
    hal-nemofinder audit export --output /tmp/audit-pre-rotate.jsonl
kubectl -n hal exec deploy/hal-hal-nemofinder-api -- \
    hal-nemofinder audit verify --input /tmp/audit-pre-rotate.jsonl

# 3. Seal the current segment (adds a FINAL record HMAC'd with the OLD key)
kubectl -n hal exec deploy/hal-hal-nemofinder-api -- \
    hal-nemofinder audit seal

# 4. Generate the new key and update the secret (external-secrets store)
NEW_KEY=$(openssl rand -base64 48)
kubectl -n hal create secret generic hal-audit-new \
    --from-literal=AUDIT_HMAC_KEY="$NEW_KEY" --dry-run=client -o yaml \
    | kubectl apply -f -

# 5. Update the Helm release to point at the new secret
helm upgrade hal helm/hal-nemofinder -n hal -f values-prod.yaml \
    --set audit.hmacKeySecret=hal-audit-new --atomic

# 6. Unfreeze
kubectl -n hal annotate deploy hal-hal-nemofinder-api \
    hal.nemofinder/audit-freeze- --overwrite
kubectl -n hal rollout restart deploy/hal-hal-nemofinder-api

# 7. Archive the OLD secret for the retention period (7 years for GxP).
```

**Never** delete the old key before the last audit segment it signed
falls out of the retention window.

## 9. Common errors

| Error message | Likely cause | Fix |
|---|---|---|
| `alembic.util.CommandError: Can't locate revision` | chart upgraded past a deleted migration | `helm rollback`, then upgrade one minor at a time |
| `sqlalchemy.exc.OperationalError: SSL connection has been closed` | PgBouncer pool exhaustion | Scale PG pool or reduce worker concurrency |
| `redis.exceptions.ConnectionError: Connection refused` | Redis restarted, Celery not reconnected | `kubectl rollout restart deploy/hal-hal-nemofinder-worker` |
| `fastapi.exceptions.HTTPException: 503 not ready` | Startup probe, migration still running | Wait; check `job/hal-hal-nemofinder-migrate-*` logs |
| `Verifier reliability overrides ignored` | malformed YAML in config | Validate with `hal-nemofinder config dump` |
| `PluginLoadError: module not found` | PVC not mounted / wrong entry-point | Check `plugins.pvc` and `plugins.modules` values |

## 10. Monitoring — Grafana dashboard checklist

Watch these panels on the `hal-nemofinder` dashboard:

- **p99 latency, /api/v1/health/***  < 500 ms
- **p99 latency, /api/v1/analyze (sync)** < 2 s
- **HTTP 5xx rate**  < 0.1 %
- **Celery queue length (`analysis`, `verification`)**  < 50
- **Worker task success rate**  > 99.5 %
- **Postgres connection pool utilisation**  < 80 %
- **Bayesian posterior distribution** — the `hal_claim_posterior`
  histogram should stay roughly bimodal; a flattening of the
  distribution is a strong signal that verifier reliability has
  degraded (stale reference data, failing KB lookups).
- **`hal_verifier_latency_seconds`** per verifier
- **`hal_audit_chain_verified_total`** — must tick up every hour

If any red threshold is crossed, follow the alert runbook link
in AlertManager.
