import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class EVForecaster:
    def __init__(self, data: pd.DataFrame):
        """Initialize with EV adoption data"""
        self.data = data
        self.prepare_time_series()
        
    def prepare_time_series(self):
        """Aggregate data to yearly time series"""
        self.ts_data = self.data.groupby('Year')['EV_Sales'].sum()
        
    def generate_forecast(self, horizon: int = 3) -> pd.Series:
        """Generate ARIMA forecast"""
        model = ARIMA(self.ts_data, order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=horizon)
        return forecast.predicted_mean
    
    def validate_model(self, test_years: int = 2) -> float:
        """Calculate MAE on holdout period"""
        train = self.ts_data.iloc[:-test_years]
        test = self.ts_data.iloc[-test_years:]
        
        model = ARIMA(train, order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=test_years)
        
        return mean_absolute_error(test, forecast.predicted_mean)
