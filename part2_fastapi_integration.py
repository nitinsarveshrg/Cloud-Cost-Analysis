#!/usr/bin/env python3
"""
CloudWise Senior AWS Cost Optimization Engineer - Part 2
FastAPI Integration

- POST /api/v1/accounts/{account_id}/cost-optimization
- Validates inputs, stubs AWS/user auth, calls Part 1 algorithm, and returns
  structured OptimizationOpportunity items.
- Includes simple circuit breaker, TTL cache, and structured logging.
"""
from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException, Path, status, Body
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import hashlib
import asyncio
import logging
import re
from functools import lru_cache
import time

# Import algorithm (local file)
from part1_cost_anomaly_detection import detect_cost_anomalies

app = FastAPI(title="CloudWise Cost Optimization API", version="1.0.0")

# --------------- Models ---------------
SUPPORTED_SERVICES = {"EC2", "RDS", "S3"}
SUPPORTED_TYPES = {"rightsizing", "lifecycle", "ri"}
# Enforced deadline for upstream cost fetch (used with asyncio.wait_for)
FETCH_TIMEOUT_SECONDS = 8.0

class User(BaseModel):
    """Authenticated user principal used for authZ checks."""
    id: str
    email: str
    accounts: List[str]

async def get_current_user() -> User:
    # In production: verify JWT/session and look up entitlements.
    return User(id="user-1", email="user@example.com", accounts=["123456789012", "111122223333"])

class OptimizationOpportunity(BaseModel):
    """DTO describing a single cost optimization opportunity."""
    id: str
    title: str
    description: str
    service: str
    optimization_type: str
    estimated_monthly_savings: float
    implementation_effort: str  # "low", "medium", "high"
    confidence_score: float
    implementation_steps: List[str]
    risks: List[str]

class CostOptimizationRequest(BaseModel):
    """Request payload for generating cost optimization insights."""
    analysis_period_days: int = Field(30, ge=7, le=90)
    include_services: Optional[List[str]] = None
    optimization_types: Optional[List[str]] = None
    min_monthly_savings: Optional[float] = Field(10.0, ge=0)

    @field_validator("include_services")
    @classmethod
    def _validate_services(cls, v):
        if v is None:
            return v
        invalid = [s for s in v if s.upper() not in SUPPORTED_SERVICES]
        if invalid:
            raise ValueError(f"Unsupported services: {invalid}. Supported: {sorted(SUPPORTED_SERVICES)}")
        return [s.upper() for s in v]

    @field_validator("optimization_types")
    @classmethod
    def _validate_types(cls, v):
        if v is None:
            return v
        invalid = [t for t in v if t.lower() not in SUPPORTED_TYPES]
        if invalid:
            raise ValueError(f"Unsupported optimization_types: {invalid}. Supported: {sorted(SUPPORTED_TYPES)}")
        return [t.lower() for t in v]

class CostOptimizationResponse(BaseModel):
    """Response payload containing opportunities and summary metrics."""
    account_id: str
    analysis_date: datetime
    current_monthly_spend: float
    optimization_opportunities: List[OptimizationOpportunity]
    total_potential_savings: float
    implementation_timeline: Dict[str, List[OptimizationOpportunity]]  # "immediate", "30_days", "90_days"

# --------------- Utilities ---------------

ACCOUNT_ID_RE = re.compile(r"^\d{12}$")

class CircuitBreaker:
    """Minimal synchronous circuit breaker to protect upstream calls."""
    def __init__(self, failure_threshold:int=3, reset_timeout:int=30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = "closed"
        self.last_failure = None

    def on_success(self):
        self.failures = 0
        self.state = "closed"

    def on_failure(self):
        self.failures += 1
        self.last_failure = datetime.utcnow()
        if self.failures >= self.failure_threshold:
            self.state = "open"

    def can_call(self) -> bool:
        if self.state == "open":
            if self.last_failure and (datetime.utcnow() - self.last_failure).total_seconds() >= self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        return True

breaker = CircuitBreaker()

# Simple in-memory TTL cache
class TTLCache:
    """Very small in-memory TTL cache for demonstration purposes."""
    def __init__(self, ttl_seconds:int=120):
        self.ttl = ttl_seconds
        self._data: Dict[str, tuple[datetime, dict]] = {}

    def get(self, key: str):
        item = self._data.get(key)
        if not item:
            return None
        ts, value = item
        if (datetime.utcnow() - ts).total_seconds() > self.ttl:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: dict):
        self._data[key] = (datetime.utcnow(), value)

cache = TTLCache(ttl_seconds=180)

logger = logging.getLogger("cloudwise")
logging.basicConfig(level=logging.INFO)

# Fake "fetch_cost_data" to keep the app runnable offline.

async def fetch_cost_data(account_id: str, days: int = 30, services: Optional[List[str]] = None) -> dict:
    """
    Fetch cost data for an account.

    In production this would call AWS Cost Explorer or Compute Optimizer APIs.
    For the take-home test, this function returns synthetic cost data
    so that the application can run completely offline on any machine.

    Args:
        account_id (str): AWS account ID (validated upstream)
        days (int): Number of days in the analysis period (default 30)
        services (Optional[List[str]]): Optional list like ["EC2","RDS","S3"] to filter results.

    Returns:
        dict: Mocked cost data structure matching detect_cost_anomalies() schema.
    """
    import random
    import datetime as dt

    # bound days for realism
    days = max(7, min(90, int(days)))

    # Generate a date range (for realism)
    start_date = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    end_date = dt.date.today().isoformat()

    # Simulated daily EC2/RDS/S3 costs with small variation
    ec2_costs = [round(120 + random.uniform(-5, 5), 2) for _ in range(days)]
    # Make RDS show a clear jump halfway to trigger the spike rule occasionally
    rds_costs = [round(45 + random.uniform(-2, 2), 2) for _ in range(days // 2)] + \
                [round(90 + random.uniform(-3, 3), 2) for _ in range(days - days // 2)]
    s3_costs = [round(25 + random.uniform(-1, 1), 2) for _ in range(days)]

    # ---- build base dataset FIRST ----
    data = {
        "account_id": account_id,
        "time_period": f"{start_date}_to_{end_date}",
        "service_costs": [
            {
                "service": "EC2-Instance",
                "daily_costs": ec2_costs,
                "instance_types": {
                    "t3.large": {"count": 3, "utilization_avg": 25.0, "daily_cost": 95.20},
                    "m5.xlarge": {"count": 2, "utilization_avg": 45.0, "daily_cost": 85.30},
                },
            },
            {
                "service": "RDS",
                "daily_costs": rds_costs,
                "instances": {
                    "db.r5.large": {"count": 1, "utilization_avg": 60.0, "daily_cost": 45.20}
                },
            },
            {
                "service": "S3",
                "daily_costs": s3_costs,
                "storage_classes": {
                    "STANDARD": {"gb": 850, "monthly_cost": 195.50},
                    "STANDARD_IA": {"gb": 0, "monthly_cost": 0.00},
                    "GLACIER": {"gb": 0, "monthly_cost": 0.00},
                },
            },
        ],
        "usage_patterns": {
            "peak_hours": "09:00-17:00",
            "weekend_usage_drop": 0.40,
            "development_instances": 5,
        },
    }

    # ---- then optionally filter by requested services ----
    if services:
        services_upper = {s.upper() for s in services}
        filtered = []
        for s in data["service_costs"]:
            prefix = s["service"].split("-")[0].upper()  # "EC2-Instance" -> "EC2"
            if prefix in services_upper:
                filtered.append(s)
        data["service_costs"] = filtered

    return data

def _map_anomaly_to_opportunity(a: dict) -> OptimizationOpportunity:
    """Map a Part 1 anomaly dict to an OptimizationOpportunity model."""
    # Map Part1 anomaly types to optimization_type
    tmap = {
        "underutilized_compute": "rightsizing",
        "ri_or_sp_opportunity": "ri",
        "storage_lifecycle_opportunity": "lifecycle",
        "database_cost_spike": "rightsizing",
        "development_resource_waste": "rightsizing",
    }
    opt_type = tmap.get(a["type"], "rightsizing")
    title = f"{a['service']}: {a['recommendation']}"
    return OptimizationOpportunity(
        id=hashlib.md5((title + a["description"]).encode()).hexdigest()[:12],
        title=title,
        description=a["description"],
        service=a["service"].split("-")[0].upper(),
        optimization_type=opt_type,
        estimated_monthly_savings=a["estimated_monthly_savings"],
        implementation_effort=a["implementation_effort"],
        confidence_score=a["confidence_score"],
        implementation_steps=[
            "Validate workload performance requirements",
            "Apply change in a non-prod environment",
            "Roll out with automated rollback",
            "Monitor CloudWatch metrics post-change"
        ],
        risks=[
            "Potential performance degradation if right-sized too aggressively",
            "Access pattern changes could reduce S3 lifecycle savings",
        ]
    )

# --------------- Endpoint ---------------
@app.post("/api/v1/accounts/{account_id}/cost-optimization", response_model=CostOptimizationResponse)
async def get_cost_optimization_insights(
    account_id: str = Path(..., description="12-digit AWS account ID", pattern=r"^\d{12}$"),
    request: CostOptimizationRequest = Body(...),
    current_user: User = Depends(get_current_user),
) -> CostOptimizationResponse:
    """
    Generate cost optimization insights for a customer's AWS account.
    """
    # AuthZ
    if account_id not in current_user.accounts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User not authorized for this account")

    # Circuit breaker
    if not breaker.can_call():
        raise HTTPException(status_code=503, detail="Upstream temporarily unavailable (circuit open)")

    # Cache key
    cache_key = f"{account_id}:{request.analysis_period_days}:{request.include_services}:{request.optimization_types}:{request.min_monthly_savings}"
    cached = cache.get(cache_key)
    if cached:
        logger.info("Serving from cache")
        return cached
    
    start = time.monotonic()
    ok = False

    try:
        # NOTE: fetch_cost_data now accepts include_services (see next block)
        cost_data = await asyncio.wait_for(
    fetch_cost_data(account_id, request.analysis_period_days, request.include_services),
    timeout=FETCH_TIMEOUT_SECONDS,
)
        analysis = detect_cost_anomalies(cost_data)
        opps = [_map_anomaly_to_opportunity(a) for a in analysis["anomalies_detected"]]

        # Filter by type and min savings
        if request.optimization_types:
            opps = [o for o in opps if o.optimization_type in request.optimization_types]
        if request.min_monthly_savings:
            opps = [o for o in opps if o.estimated_monthly_savings >= request.min_monthly_savings]

        # Simple timeline bucketing by effort/savings
        immediate = [o for o in opps if o.implementation_effort == "low"]
        in_30 = [o for o in opps if o.implementation_effort == "medium"]
        in_90 = [o for o in opps if o.implementation_effort == "high"]

        resp = CostOptimizationResponse(
            account_id=account_id,
            analysis_date=datetime.utcnow(),
            current_monthly_spend=sum(
                sum(svc.get("daily_costs", [])) for svc in cost_data.get("service_costs", [])
            ),  # rough: sum of daily costs over window
            optimization_opportunities=opps,
            total_potential_savings=analysis["total_potential_savings"],
            implementation_timeline={
                "immediate": immediate,
                "30_days": in_30,
                "90_days": in_90,
            },
        )
        breaker.on_success()
        cache.set(cache_key, resp)
        ok = True
        return resp
    except asyncio.TimeoutError:
        breaker.on_failure()
        raise HTTPException(status_code=504, detail="Timed out fetching cost data")
    except HTTPException:
        breaker.on_failure()
        raise
    except Exception as e:
        breaker.on_failure()
        logger.exception("Unhandled error")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        dur_ms = int((time.monotonic() - start) * 1000)
        logger.info("endpoint=cost_optimization ok=%s dur_ms=%d account=%s", ok, dur_ms, account_id)


# --------------- Local dev ---------------
# Run: uvicorn part2_fastapi_integration:app --reload --port 8000
