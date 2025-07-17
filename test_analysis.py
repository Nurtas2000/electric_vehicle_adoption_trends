import pytest
from src.analysis.forecasting import EVForecaster
from src.utils.data_loader import EVDataLoader

@pytest.fixture
def sample_data():
    loader = EVDataLoader()
    return loader.load_ev_data().sample(100)

def test_forecaster_init(sample_data):
    forecaster = EVForecaster(sample_data)
    assert forecaster.data.equals(sample_data)
    
def test_forecast_horizon(sample_data):
    forecaster = EVForecaster(sample_data)
    forecast = forecaster.generate_forecast(horizon=3)
    assert len(forecast) == 3
    assert all(isinstance(x, float) for x in forecast.values())
    
def test_geo_merge(sample_data):
    loader = EVDataLoader()
    merged = loader.merge_geo_ev_data(sample_data)
    assert 'geometry' in merged.columns
    assert len(merged) > 0
