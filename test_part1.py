import json
from part1_cost_anomaly_detection import detect_cost_anomalies

def test_underutilized_detects():
    data = {
        "account_id": "123456789012",
        "time_period": "t",
        "service_costs": [
            {"service":"EC2-Instance",
             "daily_costs":[100]*30,
             "instance_types":{"t3.large":{"count":2,"utilization_avg":20.0,"daily_cost":60.0}}}
        ],
        "usage_patterns":{"development_instances":0}
    }
    out = detect_cost_anomalies(data)
    assert out["anomalies_detected"], "Should detect at least one anomaly"
    assert out["total_potential_savings"] > 0


def test_s3_lifecycle_opportunity():
    data = {
        "account_id": "123456789012",
        "time_period": "t",
        "service_costs": [
            {"service":"S3",
             "daily_costs":[40]*30,
             "storage_classes":{"STANDARD":{"gb": 500, "monthly_cost": 11.5}}}
        ],
        "usage_patterns":{}
    }
    out = detect_cost_anomalies(data)
    assert any(a['type']=='storage_lifecycle_opportunity' for a in out['anomalies_detected'])

def test_ri_opportunity_when_steady():
    data = {
        "account_id": "123456789012",
        "time_period": "t",
        "service_costs": [
            {"service":"EC2-Instance",
             "daily_costs":[100.0 + (i%3)*0.1 for i in range(30)],
             "instance_types":{}}
        ],
        "usage_patterns":{}
    }
    out = detect_cost_anomalies(data)
    assert any(a['type']=='ri_or_sp_opportunity' for a in out['anomalies_detected'])
