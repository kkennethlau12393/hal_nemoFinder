# Deploying hal_nemoFinder at Acme Corp

A pragmatic, week-one rollout plan for getting hal_nemoFinder into
Acme's drug-discovery stack without waiting on framework changes.

## Day 1 — Install and smoke test

```bash
# On the hal-nemofinder host (or inside the Acme container image)
python -m venv /opt/hal/venv
/opt/hal/venv/bin/pip install hal-nemofinder acme-hal-plugins
/opt/hal/venv/bin/hal-nemofinder version
/opt/hal/venv/bin/hal-nemofinder list verifiers
```

You should see `acme_internal_library` and `acme_assay_data` listed
under "plugin" — loaded automatically via the entry points declared
in `acme-hal-plugins`.

## Day 2 — Point the framework at Acme's internal mirrors

Drop `hal_nemofinder.yaml` into `/etc/hal_nemofinder/`:

```yaml
plugins:
  modules:
    - acme_plugins.verifiers
    - acme_plugins.kb_clients

kb_client_overrides:
  pubchem: acme_plugins.kb_clients.AcmeCompoundDB
  chembl: acme_plugins.kb_clients.AcmeBioactivityDB
```

Then set `HAL_CONFIG_FILE=/etc/hal_nemofinder/hal_nemofinder.yaml` in
the systemd unit / Kubernetes deployment and restart the service.

Validate the config file before rolling out:

```bash
hal-nemofinder config validate /etc/hal_nemofinder/hal_nemofinder.yaml
hal-nemofinder config show
```

## Day 3 — Curate a regression set from historical AI outputs

Pull the last quarter of flagged hallucinations out of your review
system and convert them to a JSONL file. The schema lives at
`src/core/regression_set_schema.json`; use it with any JSON Schema
validator in CI to guarantee the file stays valid.

Save to `/opt/acme/regression/validated_claims.jsonl` and add the
path to your YAML:

```yaml
regression_set_paths:
  - /opt/acme/regression/validated_claims.jsonl
  - /opt/acme/regression/recent_hallucinations.json
```

## Day 4 — Calibrate and tune reliability

Run calibration and dump metrics to disk:

```bash
hal-nemofinder calibrate --output /var/log/hal/calibration-$(date +%F).json
```

Inspect the reliability diagram. If a verifier is over-confident
(e.g. predicted mean 0.9, actual mean 0.7 in the top bin), derive new
sensitivity / specificity values and drop them into
`verifier_reliability_overrides`:

```yaml
verifier_reliability_overrides:
  acme_internal_library:
    sensitivity: 0.96
    specificity: 0.99
```

Re-run calibration until the ECE is under your team's threshold
(Acme targets <= 0.05).

## Day 5 — Staging rollout with real traffic

1. Deploy hal_nemoFinder behind a 5% traffic mirror from the primary
   LLM serving stack.
2. Pipe verdicts into your observability stack (Datadog / Grafana)
   via the metrics JSON exported by `hal-nemofinder calibrate` and
   the `/metrics` Prometheus endpoint served by `hal-nemofinder serve`.
3. Compare the system's flagged claims against the review team's
   manual verdicts for two weeks before flipping to 100% traffic.

## Ongoing operations

* **Weekly recalibration.** Schedule `hal-nemofinder calibrate` in
  Airflow and alert if ECE or Brier score regresses by > 10%.
* **Quarterly regression-set refresh.** Add any new hallucinations
  caught in review to the JSONL. Tag them with `source:
  "acme_q<n>_review"` so you can track drift.
* **Plugin updates.** Ship new Acme verifiers by bumping the version
  of `acme-hal-plugins` and redeploying; no framework changes needed.
