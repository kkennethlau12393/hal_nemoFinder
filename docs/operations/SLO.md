# hal-nemoFinder — Service Level Objectives

This document captures the example SLOs operators should adapt to
their own production deployment.  Values assume a mid-sized
pharma workload (~5 k analyses / day, ~50 k claims / day).

## 1. SLIs & SLOs

| # | SLI | SLO target | Window |
|---|-----|-----------|--------|
| 1 | API availability (2xx+3xx+4xx < 500) | **99.9%** | 30d rolling |
| 2 | `/api/v1/health/*` p99 latency | **< 500 ms** | 5m window, 28d SLO |
| 3 | `/api/v1/analyze` (sync accept) p99 latency | **< 2 s** | 5m window, 28d SLO |
| 4 | Job completion p95, jobs with ≤ 20 claims | **< 60 s** | 1h window, 28d SLO |
| 5 | Verifier error rate (per verifier) | **< 1%** | 1h window, 7d SLO |
| 6 | Audit chain integrity | **100%** | real-time |

### Availability budget

```
error_budget_30d = (1 - 0.999) * 30 * 24 * 60   # ≈ 43.2 min/month
```

Burn-rate alert policy (Google SRE workbook):

| Severity | Window | Threshold | Action |
|---|---|---|---|
| Page  | 1h + 5m | **14.4× burn** | wake on-call |
| Page  | 6h + 30m | **6× burn**   | wake on-call |
| Ticket | 24h + 2h | **1× burn**   | next business day |

## 2. Prometheus recording rules

```yaml
groups:
  - name: hal-nemofinder.recording
    interval: 30s
    rules:
      - record: job:hal_http_requests_total:rate5m
        expr: sum by (job, code) (rate(hal_http_requests_total[5m]))

      - record: job:hal_http_request_errors:ratio_5m
        expr: |
          sum by (job) (rate(hal_http_requests_total{code=~"5.."}[5m]))
            /
          sum by (job) (rate(hal_http_requests_total[5m]))

      - record: job:hal_http_request_duration_seconds:p99_5m
        expr: |
          histogram_quantile(0.99,
            sum by (le, job, handler) (rate(hal_http_request_duration_seconds_bucket[5m])))
```

## 3. Alert rules

```yaml
groups:
  - name: hal-nemofinder.alerts
    rules:
      # ---- Availability burn ------------------------------------------------
      - alert: HalErrorBudgetBurnFast
        expr: job:hal_http_request_errors:ratio_5m > (14.4 * 0.001)
        for: 5m
        labels:
          severity: page
        annotations:
          summary: "hal-nemofinder burning error budget 14.4x"
          runbook: https://github.com/hal-nemofinder/hal-nemofinder/blob/main/docs/operations/RUNBOOK.md

      - alert: HalErrorBudgetBurnSlow
        expr: job:hal_http_request_errors:ratio_5m > (6 * 0.001)
        for: 30m
        labels:
          severity: page

      # ---- Latency ----------------------------------------------------------
      - alert: HalHealthLatencyHigh
        expr: job:hal_http_request_duration_seconds:p99_5m{handler=~"/api/v1/health/.*"} > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Health endpoint p99 > 500ms"

      - alert: HalAnalyzeLatencyHigh
        expr: job:hal_http_request_duration_seconds:p99_5m{handler="/api/v1/analyze"} > 2
        for: 10m
        labels:
          severity: warning

      # ---- Workers ----------------------------------------------------------
      - alert: HalQueueBacklog
        expr: celery_queue_length{queue=~"analysis|verification"} > 500
        for: 10m
        labels:
          severity: page
        annotations:
          summary: "Celery queue {{ $labels.queue }} backlog > 500"

      - alert: HalWorkerTaskFailures
        expr: rate(celery_task_failed_total[10m]) > 0.05
        for: 15m
        labels:
          severity: warning

      # ---- Audit ------------------------------------------------------------
      - alert: HalAuditChainBroken
        expr: increase(hal_audit_chain_broken_total[5m]) > 0
        labels:
          severity: page
        annotations:
          summary: "Audit chain verification failed"
          description: "Tamper-evident audit log has been broken. Freeze writes and investigate immediately."

      # ---- Infrastructure ---------------------------------------------------
      - alert: HalPodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total{namespace="hal"}[15m]) > 0.2
        for: 10m
        labels:
          severity: warning
```

## 4. Review cadence

- SLO attainment reviewed at every monthly service review.
- Error budgets that burn > 50% in a calendar month trigger an
  engineering freeze on new features until the SLI recovers.
- Alert thresholds are reviewed quarterly against production load.
