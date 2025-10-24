# CloudWise Senior AWS Cost Optimization Engineer — Technical Assessment

This repo contains a complete, runnable submission with:
- **Part 1:** Cost anomaly detection algorithm (`part1_cost_anomaly_detection.py`)
- **Part 2:** FastAPI endpoint integration (`part2_fastapi_integration.py`)
- **Part 3:** System architecture (`part3_system_architecture.md`)
- **requirements.txt** and basic usage instructions

> Designed to run **offline** with synthetic data so reviewers can execute it without AWS credentials. Production integrations can be plugged in by swapping the data source.

---

## Quick Start

```bash
python3 -V   # 3.10+ recommended
pip install -r requirements.txt

# Part 1 - demo
python part1_cost_anomaly_detection.py

# Part 2 - API
uvicorn part2_fastapi_integration:app --reload --port 8000
# Then POST to:
# http://localhost:8000/api/v1/accounts/123456789012/cost-optimization
```

### Example Request
```bash
curl -s -X POST "http://localhost:8000/api/v1/accounts/123456789012/cost-optimization"   -H "Content-Type: application/json"   -d '{
    "analysis_period_days": 30,
    "include_services": ["EC2","RDS","S3"],
    "optimization_types": ["rightsizing","lifecycle","ri"],
    "min_monthly_savings": 10
  }' | jq .
```

---

## Notes on the Algorithm

- Detects:
  - **Underutilized compute** (< 40% avg util) → right-size or off-hours schedule
  - **Steady EC2 spend** → **RI/Savings Plan** coverage
  - **RDS cost spikes** → review Multi-AZ/replicas/right-size
  - **S3 lifecycle** (STANDARD → IA/GLACIER)
  - **Dev environment waste** → scheduler
- Includes growth guards (avoid downsizing rapidly growing services) and weekend usage heuristics.
- Produces **severity**, **confidence**, and **estimated monthly savings**.

---

## Error Handling & Performance (API)

- **AuthZ**: simple user stub with allowed accounts
- **Validation**: Pydantic schema (days 7–90, service/type enums, 12‑digit account ID)
- **Circuit breaker**: protects upstream data fetches
- **Caching**: in‑memory TTL cache
- **Async**: `async def` with simulated I/O

---

## Testing

- You can run the Part 1 script to see a JSON result. For full unit tests, add `pytest` and import `detect_cost_anomalies` with custom datasets to assert outputs.

---

## Production Hook Points

- Replace `fetch_cost_data` in the API with real **Cost Explorer** + **Compute Optimizer** calls.
- Swap pricing constants with live **AWS Pricing API** or a pricing service.
