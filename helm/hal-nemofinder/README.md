# hal-nemofinder Helm chart

Production-grade Helm chart for the open-source
[hal-nemoFinder](https://github.com/hal-nemofinder/hal-nemofinder)
hallucination-detection framework for AI-driven drug discovery.

- Multi-pod API deployment with HPA, PDB and pod anti-affinity
- Celery worker deployment with independent scaling (custom metric + CPU fallback)
- Pre-install/pre-upgrade Alembic migration hook
- Optional Bitnami PostgreSQL (pgvector flavour) and Redis subcharts
- NetworkPolicy, ServiceMonitor, Ingress — all opt-in
- PodSecurity "restricted" compatible (non-root, read-only rootfs, seccomp)

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Kubernetes  | >= 1.25 |
| Helm        | >= 3.12 |
| `kubectl`   | matching your cluster |
| `helm dependency update` has been run |
| (optional) prometheus-operator for ServiceMonitor |
| (optional) cert-manager for TLS |
| (optional) external-secrets for production secret management |

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm dependency update helm/hal-nemofinder
```

## Installation

Quick start (dev / evaluation):

```bash
helm install hal helm/hal-nemofinder \
  --namespace hal --create-namespace \
  --set postgresql.auth.password=hal \
  --set redis.auth.password=hal
```

Production:

```bash
helm install hal helm/hal-nemofinder \
  --namespace hal --create-namespace \
  -f values-prod.yaml
```

where `values-prod.yaml` disables the embedded dependencies and supplies
external secret references:

```yaml
postgresql:
  enabled: false
redis:
  enabled: false
externalDatabase:
  host: postgres.prod.internal
  existingSecret: hal-db
externalRedis:
  existingSecret: hal-redis
  existingSecretUrlKey: REDIS_URL
auth:
  required: true
  oidc:
    issuer: https://login.example.com/
    clientId: hal-nemofinder
    audience: hal
    existingSecret: hal-oidc
audit:
  enabled: true
  hmacKeySecret: hal-audit
networkPolicy:
  enabled: true
serviceMonitor:
  enabled: true
ingress:
  enabled: true
  hosts:
    - host: hal.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: hal-tls
      hosts:
        - hal.example.com
```

## Configuration reference

| Key | Default | Description |
|---|---|---|
| `image.repository` | `ghcr.io/hal-nemofinder/hal-nemofinder` | Container image repository. |
| `image.tag` | `0.1.0` | Image tag — pin to a released version. |
| `image.pullPolicy` | `IfNotPresent` | Image pull policy. |
| `imagePullSecrets` | `[]` | Pull secrets for private registries. |
| `api.replicaCount` | `3` | Replicas when autoscaling is disabled. |
| `api.resources.requests.cpu` | `500m` | API CPU request. |
| `api.resources.requests.memory` | `512Mi` | API memory request. |
| `api.resources.limits.cpu` | `2000m` | API CPU limit. |
| `api.resources.limits.memory` | `2Gi` | API memory limit. |
| `api.autoscaling.enabled` | `true` | Enable HPA for API. |
| `api.autoscaling.minReplicas` | `3` | HPA minimum. |
| `api.autoscaling.maxReplicas` | `20` | HPA maximum. |
| `api.autoscaling.targetCPUUtilizationPercentage` | `70` | Target CPU. |
| `api.podDisruptionBudget.minAvailable` | `2` | PDB minAvailable. |
| `api.service.type` | `ClusterIP` | API service type. |
| `api.service.port` | `8000` | API service port. |
| `worker.replicaCount` | `5` | Worker replicas when autoscaling is disabled. |
| `worker.queues` | `analysis,verification` | Celery queue list. |
| `worker.concurrency` | `4` | Celery worker concurrency. |
| `worker.resources.requests.cpu` | `1000m` | Worker CPU request. |
| `worker.resources.requests.memory` | `1Gi` | Worker memory request. |
| `worker.resources.limits.cpu` | `4000m` | Worker CPU limit. |
| `worker.resources.limits.memory` | `4Gi` | Worker memory limit. |
| `worker.autoscaling.enabled` | `true` | Enable HPA for worker. |
| `worker.autoscaling.minReplicas` | `2` | Worker HPA minimum. |
| `worker.autoscaling.maxReplicas` | `50` | Worker HPA maximum. |
| `worker.autoscaling.customMetric.enabled` | `false` | Use custom queue-depth metric. |
| `postgresql.enabled` | `true` | Install the embedded Bitnami PostgreSQL subchart. |
| `postgresql.auth.username` | `hal` | Embedded PG user. |
| `postgresql.auth.database` | `hal_nemofinder` | Embedded PG database. |
| `postgresql.image.repository` | `pgvector/pgvector` | pgvector-enabled image. |
| `postgresql.image.tag` | `pg16` | pgvector image tag. |
| `redis.enabled` | `true` | Install the embedded Bitnami Redis subchart. |
| `redis.auth.enabled` | `true` | Require Redis password. |
| `externalDatabase.host` | `""` | External Postgres host. |
| `externalDatabase.existingSecret` | `""` | Secret with DB password. |
| `externalRedis.url` | `""` | External Redis URL. |
| `envFromSecret` | `""` | Secret name providing extra `HAL_` env vars. |
| `auth.required` | `true` | Enable OIDC / API key authentication. |
| `audit.enabled` | `true` | Enable tamper-evident audit log. |
| `audit.hmacKeySecret` | `""` | Secret containing `AUDIT_HMAC_KEY`. |
| `plugins.enabled` | `false` | Load third-party verifier plugins. |
| `plugins.pvc` | `""` | PVC mounted at `/plugins`. |
| `ingress.enabled` | `false` | Create an Ingress. |
| `ingress.className` | `nginx` | IngressClass. |
| `networkPolicy.enabled` | `false` | Enforce default-deny + allow policies. |
| `serviceMonitor.enabled` | `false` | Emit a Prometheus Operator ServiceMonitor. |
| `podSecurityContext` | see `values.yaml` | Pod-level security context. |
| `containerSecurityContext` | see `values.yaml` | Container-level security context. |

See `values.yaml` for the complete, annotated list.

## Upgrade

```bash
helm repo update
helm upgrade hal helm/hal-nemofinder -n hal -f values-prod.yaml \
  --atomic --timeout 10m
```

The `pre-upgrade` hook runs `alembic upgrade head` before the new pods
are rolled out.  If the migration job fails, `--atomic` rolls the
release back automatically.

## Rollback

```bash
helm history hal -n hal
helm rollback hal <revision> -n hal --wait
```

Database migrations are **forward-only**; if a rollback is required
after a destructive schema change, restore from backup first
(see `docs/operations/DISASTER_RECOVERY.md`).

## Production deployment checklist

- [ ] `postgresql.enabled=false`, point at a managed HA Postgres
- [ ] `redis.enabled=false`, point at a managed HA Redis (TLS)
- [ ] All secrets sourced from an external secret store
  (Vault / AWS / GCP / Azure) via external-secrets or CSI
- [ ] `auth.required=true`, OIDC configured
- [ ] `audit.enabled=true`, `AUDIT_HMAC_KEY` rotated on the
  schedule described in the runbook
- [ ] `networkPolicy.enabled=true`
- [ ] `serviceMonitor.enabled=true` and AlertManager routes in place
- [ ] Resource requests/limits reviewed against load-test baseline
- [ ] HPA maximums sized to cluster quotas
- [ ] PodDisruptionBudget acceptable for your maintenance windows
- [ ] Ingress protected by a WAF and rate-limited upstream
- [ ] `scripts/backup.sh` scheduled as a CronJob or external backup tool
- [ ] Disaster recovery drill executed at least once per quarter

## Uninstall

```bash
helm uninstall hal -n hal
kubectl delete pvc -n hal -l app.kubernetes.io/instance=hal   # optional
```
