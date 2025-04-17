# hal_nemoFinder

Open-source hallucination detection framework for AI-driven drug discovery.

## Quick Start

```bash
docker-compose up -d
python -m seed.seed_chembl
curl -X POST http://localhost:8000/api/v1/analyze -H "Content-Type: application/json" -d '{"text": "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has a molecular weight of 220.3 Da."}'
```

See `/docs` for the full API documentation.
