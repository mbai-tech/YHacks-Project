import pytest

from src.web.water_risk_service import DEFAULT_SCENARIO, WaterRiskService


@pytest.fixture(scope="module")
def service():
    return WaterRiskService("data")


def test_coordinate_search_returns_candidate(service):
    results = service.search("41.3083, -72.9279")
    assert results
    assert results[0]["kind"] == "coordinates"


def test_scenario_reduces_total_risk_for_sample_location(service):
    location = service.search("New Haven")[0]
    risk = service.get_risk(location["latitude"], location["longitude"], name=location["name"])
    scenario = service.apply_scenario(
        location["latitude"],
        location["longitude"],
        DEFAULT_SCENARIO,
        baseline={
            "factor_scores": risk["baseline"],
            "nearby_sources": risk["nearby_sources"],
            "context": risk["context"],
            "recommendations": risk["recommendations"],
            "explanation": risk["explanation"],
        },
        name=location["name"],
    )
    assert scenario["scenario"]["total_score"] <= risk["baseline"]["total_score"]
