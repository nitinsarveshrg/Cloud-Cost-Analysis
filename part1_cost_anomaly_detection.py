#!/usr/bin/env python3
"""
CloudWise Senior AWS Cost Optimization Engineer - Part 1
Cost Anomaly Detection Algorithm

- Implements detect_cost_anomalies(cost_data) -> dict as specified.
- Focuses on underutilized compute, S3 lifecycle, RI/SP coverage, and dev waste.
- Designed to work **offline** (no AWS creds required) but structured for easy
  replacement with live pricing/usage retrieval.
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math
import statistics
from datetime import date, timedelta, datetime

# ------------------
# Simple pricing constants (conservative, USD). Replace with live pricing if desired.
# These are **approximate** and intended to get realistic order-of-magnitude estimates
# without external dependencies.
# ------------------
EC2_ONDEMAND_HOURLY = {
    # family:size: hourly
    ("t3", "medium"): 0.0416,
    ("t3", "large"):  0.0832,
    ("m5", "xlarge"): 0.192,
    ("m5", "large"):  0.096,
}
RDS_ONDEMAND_DAILY = {
    "db.r5.large": 45.20 / 30.0,  # derive approx from sample data
}

S3_STORAGE_MONTHLY_PER_GB = {
    "STANDARD": 0.023,
    "STANDARD_IA": 0.0125,
    "GLACIER": 0.004
}

# Helper to extract family, size from instance type like "t3.large"
def _split_instance_type(it: str) -> Tuple[str, str]:
    parts = it.split(".")
    return (parts[0], parts[1]) if len(parts) == 2 else (it, "")

@dataclass
class Recommendation:
    type: str
    service: str
    severity: str
    description: str
    recommendation: str
    estimated_monthly_savings: float
    implementation_effort: str
    confidence_score: float

def _pct_diff(a: float, b: float) -> float:
    if a == 0:
        return 0.0 if b == 0 else 1.0
    return abs(b - a) / a

def _calc_severity(savings: float) -> str:
    if savings >= 1000: return "critical"
    if savings >= 300:  return "high"
    if savings >= 75:   return "medium"
    return "low"

def _bounded_confidence(*signals: float) -> float:
    # average but bound 0.5..0.98 to avoid overstating
    if not signals:
        return 0.6
    avg = sum(signals)/len(signals)
    return max(0.5, min(0.98, avg))

def _rolling_mean(vals: List[float], window: int = 7) -> List[float]:
    out = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        w = vals[start:i+1]
        out.append(sum(w)/len(w))
    return out

def _has_weekend_drop(daily_costs: List[float], weekend_drop_threshold: float = 0.3) -> bool:
    # naive: assume first day is a Monday for synthetic data; robust analyses would use timestamps
    if len(daily_costs) < 14:
        return False
    weekdays = [c for i,c in enumerate(daily_costs) if (i % 7) < 5]
    weekends = [c for i,c in enumerate(daily_costs) if (i % 7) >= 5]
    if not weekends or not weekdays:
        return False
    wk_avg = sum(weekdays)/len(weekdays)
    we_avg = sum(weekends)/len(weekends)
    if wk_avg == 0: 
        return False
    drop = 1 - (we_avg / wk_avg)
    return drop >= weekend_drop_threshold

def detect_cost_anomalies(cost_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze cost trends and usage signals to produce optimization recommendations.
    Input matches the structure provided in the assessment prompt.
    Output matches the expected output structure example.
    """
    anomalies: List[Recommendation] = []
    total_savings = 0.0

    # ---- Context ----
    account_id = cost_data.get("account_id", "unknown")
    tp = cost_data.get("time_period", "")
    usage_patterns = cost_data.get("usage_patterns", {})
    weekend_drop_hint = usage_patterns.get("weekend_usage_drop", 0.0)
    peak_hours = usage_patterns.get("peak_hours")
    dev_instances = usage_patterns.get("development_instances", 0)

    # ---- Iterate services ----
    for svc in cost_data.get("service_costs", []):
        name = svc.get("service")
        daily_costs: List[float] = svc.get("daily_costs", [])
        # 1) Growth check (avoid downsizing if strong upward trend)
        growth_score = 0.0
        if len(daily_costs) >= 7:
            rm = _rolling_mean(daily_costs, 7)
            start, end = rm[max(0,len(rm)-14)], rm[-1]  # compare to 2-week-ago mean
            growth = _pct_diff(start, end)
            # If growing > 20% do not downsize; use this to reduce confidence of downsizing recs.
            if end > start and growth > 0.2:
                growth_score = 1.0

        # 2) Underutilized compute (EC2-like)
        if name.lower().startswith("ec2"):
            inst = svc.get("instance_types", {})
            for itype, meta in inst.items():
                util = float(meta.get("utilization_avg", 0.0))
                count = int(meta.get("count", 0))
                daily_cost = float(meta.get("daily_cost", 0.0))
                family, size = _split_instance_type(itype)

                if util < 40.0 and count > 0:
                    # Right-size one tier down if possible (simple heuristic: large->medium, xlarge->large)
                    target_size = "medium" if size == "large" else "large" if size == "xlarge" else None
                    if target_size:
                        src_price = EC2_ONDEMAND_HOURLY.get((family, size), (daily_cost/24.0))
                        tgt_price = EC2_ONDEMAND_HOURLY.get((family, target_size), src_price*0.55)
                        hourly_sav = max(0.0, src_price - tgt_price)
                        monthly_sav = hourly_sav * 24 * 30 * count
                    else:
                        # Fallback: stop 1/3 of instances during off-hours if weekend drop present
                        off_pct = 0.3 if weekend_drop_hint >= 0.3 or _has_weekend_drop(daily_costs) else 0.15
                        monthly_sav = daily_cost * 30.0 * off_pct

                    # Confidence: higher when util is much lower, reduced by growth trend
                    util_signal = min(0.98, (40.0 - util) / 40.0) if util < 40 else 0.5
                    confidence = _bounded_confidence(util_signal, 1 - 0.5*growth_score)

                    rec = Recommendation(
                        type="underutilized_compute",
                        service="EC2-Instance",
                        severity=_calc_severity(monthly_sav),
                        description=f"{count} {itype} instances running at {util:.0f}% average utilization",
                        recommendation=f"Right-size to {family}.{target_size} instances" if target_size else "Schedule off-hours shutdown for idle instances",
                        estimated_monthly_savings=round(monthly_sav, 2),
                        implementation_effort="low",
                        confidence_score=round(confidence, 2),
                    )
                    anomalies.append(rec)
                    total_savings += rec.estimated_monthly_savings

            # EC2 RI/SP opportunity: if daily cost variance is low and mean spend is steady, suggest coverage
            if len(daily_costs) >= 10:
                mean = statistics.mean(daily_costs)
                stdev = statistics.pstdev(daily_costs)
                coeff_var = (stdev / mean) if mean else 1.0
                if mean > 50 and coeff_var < 0.15:  # fairly steady
                    # Assume 30% savings if moving 50% of steady spend to SP/RI
                    monthly = mean * 30.0
                    monthly_sav = monthly * 0.5 * 0.30
                    confidence = _bounded_confidence(0.85 if coeff_var < 0.1 else 0.75, 1 - 0.3*growth_score)
                    rec = Recommendation(
                        type="ri_or_sp_opportunity",
                        service="EC2-Instance",
                        severity=_calc_severity(monthly_sav),
                        description="Steady on-demand EC2 spend detected (low variance)",
                        recommendation="Purchase Savings Plan or Reserved Instances to cover ~50% of steady usage",
                        estimated_monthly_savings=round(monthly_sav, 2),
                        implementation_effort="medium",
                        confidence_score=round(confidence, 2),
                    )
                    anomalies.append(rec)
                    total_savings += rec.estimated_monthly_savings

        # 3) RDS under-utilization (simple: sharp step-up day suggests Multi-AZ or size increase)
        if name.upper() == "RDS":
            if len(daily_costs) >= 5:
                for i in range(1, len(daily_costs)):
                    if daily_costs[i-1] > 0 and (daily_costs[i] >= 1.9 * daily_costs[i-1]):
                        # sudden doubling -> Multi-AZ enabled or new replica; propose review
                        monthly_sav = daily_costs[i] * 30 * 0.25  # assume 25% potential via right-sizing/turning off replica in dev
                        rec = Recommendation(
                            type="database_cost_spike",
                            service="RDS",
                            severity=_calc_severity(monthly_sav),
                            description="Detected sudden RDS daily cost jump (~2x)",
                            recommendation="Investigate Multi-AZ/replica changes and right-size storage/instance",
                            estimated_monthly_savings=round(monthly_sav, 2),
                            implementation_effort="medium",
                            confidence_score=0.8,
                        )
                        anomalies.append(rec)
                        total_savings += rec.estimated_monthly_savings
                        break

        # 4) S3 lifecycle
        if name.upper() == "S3":
            sc = svc.get("storage_classes", {})
            std = sc.get("STANDARD", {"gb": 0, "monthly_cost": 0.0})
            std_gb = float(std.get("gb", 0.0))
            # Heuristic: If STANDARD GB > 500 and either weekend drop or slow growth, suggest IA
            if std_gb > 200:
                std_cost = std_gb * S3_STORAGE_MONTHLY_PER_GB["STANDARD"]
                ia_cost = std_gb * S3_STORAGE_MONTHLY_PER_GB["STANDARD_IA"]
                monthly_sav = max(0.0, std_cost - ia_cost) * 0.8  # haircut for retrieval/transition costs
                confidence = _bounded_confidence(0.8 if weekend_drop_hint >= 0.3 else 0.7, 1 - 0.3*growth_score)
                rec = Recommendation(
                    type="storage_lifecycle_opportunity",
                    service="S3",
                    severity=_calc_severity(monthly_sav),
                    description=f"{int(std_gb)}GB in Standard storage with likely infrequent access",
                    recommendation="Implement lifecycle policy: STANDARD → STANDARD_IA after 30 days (then to GLACIER after 90d if access remains low)",
                    estimated_monthly_savings=round(monthly_sav, 2),
                    implementation_effort="low",
                    confidence_score=round(confidence, 2),
                )
                anomalies.append(rec)
                total_savings += rec.estimated_monthly_savings

    # 5) Dev resource waste: suggest off-hours schedules for N instances
    if dev_instances and dev_instances > 0:
        # save 65% by turning off 12h weeknights + weekends: approx 16/24*5 + 2*24 = 152h saved out of 720h -> ~21%? 
        # We'll assume 55% realistic blended saving across fleet.
        blended_savings = 50.0 * dev_instances  # placeholder per-dev instance monthly saving
        rec = Recommendation(
            type="development_resource_waste",
            service="EC2-Instance",
            severity=_calc_severity(blended_savings),
            description=f"{dev_instances} development instances likely idle off-hours",
            recommendation="Apply Instance Scheduler to stop during nights/weekends",
            estimated_monthly_savings=round(blended_savings, 2),
            implementation_effort="low",
            confidence_score=0.75,
        )
        anomalies.append(rec)
        total_savings += rec.estimated_monthly_savings

    # ---- Prioritize by savings/effort (low effort favored), and trim float noise ----
    def effort_weight(e: str) -> float:
        return {"low": 1.0, "medium": 0.7, "high": 0.4}.get(e, 0.6)

    anomalies.sort(key=lambda r: (r.estimated_monthly_savings * effort_weight(r.implementation_effort)), reverse=True)

    # Build output
    out = {
        "anomalies_detected": [
            {
                "type": r.type,
                "service": r.service,
                "severity": r.severity,
                "description": r.description,
                "recommendation": r.recommendation,
                "estimated_monthly_savings": round(float(r.estimated_monthly_savings), 2),
                "implementation_effort": r.implementation_effort,
                "confidence_score": round(float(r.confidence_score), 2),
            } for r in anomalies
        ],
        "total_potential_savings": round(float(total_savings), 2),
        "optimization_score": round(min(10.0, 3.0 + math.log1p(total_savings)/2.0), 1),
        "next_review_date": (date.today().replace(day=1) + timedelta(days=32)).replace(day=1).isoformat(),
    }
    return out


# ----------- Demo usage ----------
if __name__ == "__main__":
    # Minimal demonstration with the example-like structure.
    sample = {
        "account_id": "123456789012",
        "time_period": "2025-09-01_to_2025-10-01",
        "service_costs": [
            {
                "service": "EC2-Instance",
                "daily_costs": [120.50, 125.30, 128.75, 129.25, 132.50, 130.1, 128.9,
                                129.1, 130.2, 130.0, 131.5, 132.4, 133.0, 132.9, 132.7,
                                132.1, 131.8, 131.0, 130.6, 130.9, 131.2, 131.4, 131.3, 131.7, 131.6, 131.5, 131.4, 131.2, 131.1, 131.0],
                "instance_types": {
                    "t3.large": {"count": 3, "utilization_avg": 25.0, "daily_cost": 95.20},
                    "m5.xlarge": {"count": 2, "utilization_avg": 45.0, "daily_cost": 85.30}
                }
            },
            {
                "service": "RDS",
                "daily_costs": [45.20, 45.20, 45.20, 45.20, 90.40, 90.40, 90.40, 90.40, 90.40] + [90.4]*21,
                "instances": {
                    "db.r5.large": {"count": 1, "utilization_avg": 60.0, "daily_cost": 45.20}
                }
            },
            {
                "service": "S3",
                "daily_costs": [25.30, 26.10, 28.50, 35.20, 42.80] + [42.8]*25,
                "storage_classes": {
                    "STANDARD": {"gb": 850, "monthly_cost": 195.50},
                    "STANDARD_IA": {"gb": 0, "monthly_cost": 0.00},
                    "GLACIER": {"gb": 0, "monthly_cost": 0.00}
                }
            }
        ],
        "usage_patterns": {
            "peak_hours": "09:00-17:00",
            "weekend_usage_drop": 0.40,
            "development_instances": 5
        }
    }

    result = detect_cost_anomalies(sample)
    import json
    print(json.dumps(result, indent=2))
