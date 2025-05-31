# Extending hal_nemoFinder with your proprietary data in 15 minutes

This folder is a complete, runnable worked example of how a pharma
company (here, fictional "Acme Corp") bolts its own verifiers,
knowledge bases, and regression sets onto hal_nemoFinder **without
modifying a single line of framework code**.

Everything here is opinionated toward production deployment. Copy it,
rename `acme_plugins` to your own namespace, and you are done.

## The three deployment patterns

hal_nemoFinder supports three independent plugin discovery mechanisms.
You can mix and match them per deployment.

### 1. Python entry points (recommended for distribution)

Ship your plugin as a regular pip-installable package. In your
`pyproject.toml`:

```toml
[project.entry-points."hal_nemofinder.verifiers"]
acme_internal_library = "acme_plugins.verifiers:AcmeInternalLibraryVerifier"
acme_assay_data = "acme_plugins.verifiers:AcmeAssayDataVerifier"

[project.entry-points."hal_nemofinder.knowledge_clients"]
acme_compound_db = "acme_plugins.kb_clients:AcmeCompoundDB"
```

After `pip install acme-hal-plugins` the framework auto-imports these
at startup. Nothing else to configure.

### 2. Module-path config (recommended for in-repo extensions)

If your plugin lives in the same repo as your deployment (no pip
package), list the dotted module path in `hal_nemofinder.yaml`:

```yaml
plugins:
  modules:
    - acme_plugins.verifiers
    - acme_plugins.kb_clients
```

### 3. File-path config (recommended for quick experiments)

For notebooks, ad-hoc scripts, or rescue patches:

```yaml
plugins:
  files:
    - /opt/acme/custom_extractor.py
```

All three mechanisms cooperate — entry points are loaded first, then
modules, then files. Anything that fails to load is logged in a
red panel; the rest keeps working.

## Calibrating against your own data

Regression sets are plain `.jsonl` (or `.json` / `.yaml`) files.
Acme's historical hallucinations live in
`acme_plugins/regression_sets/validated_claims.jsonl`. Point
hal_nemoFinder at them:

```yaml
regression_set_paths:
  - /opt/acme/regression/validated_claims.jsonl
```

Then run:

```bash
hal-nemofinder calibrate --output /var/log/hal/calibration.json
```

The CLI prints accuracy, precision, recall, F1, Brier score, ECE, a
reliability diagram, and per-claim-type / per-source breakdowns — and
writes a JSON report you can feed into your observability stack.

Use the ECE and Brier score to decide whether your verifier reliability
priors need recalibrating. When they do, drop the new values into
`verifier_reliability_overrides` in your YAML.

## Swapping a built-in KB client for your internal mirror

Built-in clients like PubChem are registered by short name. To replace
the PubChem client with Acme's internal mirror on day one — with no
code change to hal_nemoFinder — set:

```yaml
kb_client_overrides:
  pubchem: acme_plugins.kb_clients.AcmeCompoundDB
```

`AcmeCompoundDB` must implement the same method surface as
`PubChemClient` (see `acme_plugins/kb_clients.py`). The plugin loader
hot-swaps it at startup and every downstream verifier picks it up
transparently.

## Running this example locally

```bash
# from the root of hal_nemoFinder
pip install -e .
pip install -e examples/customer_plugins          # installs acme_plugins via its pyproject
export HAL_CONFIG_FILE=$(pwd)/examples/customer_plugins/hal_nemofinder.yaml
hal-nemofinder list verifiers
hal-nemofinder calibrate --regression-set examples/customer_plugins/acme_plugins/regression_sets/validated_claims.jsonl
hal-nemofinder verify "Compound ACME-123 has IC50 of 4.2 nM against EGFR."
```

See `deployment.md` for a production rollout timeline.
