#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Climate trend forecasting module for AI for Sustainable Ecosystems project.
This script implements time-series forecasting models for climate data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
import logging
import joblib

# Time-series forecasting libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("climate_forecast.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("climate_forecast")

# Load environment variables
load_dotenv()

# Set paths
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))


def load_climate_data(file_path=None):
    """
    Load climate data from file.
    
    Args:
        file_path (str, optional): Path to climate data file. If None, use default.
    
    Returns:
        pd.DataFrame: Climate data
    """
    if file_path is None:
        file_path = DATA_DIR / "climate" / "global_temperature.csv"
    
    logger.info(f"Loading climate data from: {file_path}")
    
    try:
        # Try loading the data
        df = pd.read_csv(file_path)
        
        # Check if file exists but is empty template
        if len(df) == 0:
            return create_sample_climate_data()
        
        # Ensure date column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        logger.info(f"Successfully loaded climate data with {len(df)} records")
        return df
    
    except FileNotFoundError:
        logger.warning(f"Climate data file not found: {file_path}")
        logger.info("Creating sample climate data for demonstration")
        return create_sample_climate_data()
    
    except Exception as e:
        logger.error(f"Error loading climate data: {str(e)}")
        logger.info("Creating sample climate data for demonstration")
        return create_sample_climate_data()


def create_sample_climate_data():
    """
    Create sample climate data for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample climate data
    """
    # Create date range from 1950 to 2020 (monthly data)
    date_rng = pd.date_range(start='1950-01-01', end='2020-12-31', freq='M')
    
    # Create base temperature (seasonal pattern + overall warming trend)
    n = len(date_rng)
    time_index = np.arange(n)
    
    # Seasonality: yearly cycle with higher temperatures in summer
    month_effect = 0.6 * np.sin(2 * np.pi * time_index / 12)
    
    # Long-term trend: warming over time (0.02°C per year)
    trend = 0.02 * time_index / 12
    
    # Starting temperature (global average)
    base_temp = 14.0
    
    # Combine components
    temps = base_temp + trend + month_effect
    
    # Add random noise
    noise = np.random.normal(0, 0.3, n)
    temps = temps + noise
    
    # El Niño events (periodic warming every ~5 years)
    el_nino_years = [1957, 1965, 1972, 1982, 1987, 1991, 1997, 2002, 2009, 2015]
    for year in el_nino_years:
        # Get indices for the El Niño year
        el_nino_indices = [i for i, date in enumerate(date_rng) if date.year == year]
        # Add El Niño warming
        temps[el_nino_indices] += np.random.uniform(0.3, 0.7, len(el_nino_indices))
    
    # Create DataFrame
    df = pd.DataFrame(data={
        'temperature': temps,
        'temperature_anomaly': temps - base_temp
    }, index=date_rng)
    
    logger.info(f"Created sample climate data with {len(df)} records")
    return df


def plot_climate_data(df, column='temperature', title=None):
    """
    Plot climate data time series.
    
    Args:
        df (pd.DataFrame): Climate data
        column (str): Column to plot
        title (str, optional): Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column], color='tab:red')
    
    # Smooth trend line (12-month moving average)
    plt.plot(df.index, df[column].rolling(window=12).mean(), color='black', 
             linestyle='-', linewidth=2, label='12-month Moving Average')
    
    if title is None:
        title = f"{column.replace('_', ' ').title()} Over Time"
    
    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(f'{column.replace("_", " ").title()} (°C)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{column}_timeseries.png")
    plt.close()


def check_stationarity(series):
    """
    Check stationarity of time series using Augmented Dickey-Fuller test.
    
    Args:
        series: Time series data
        
    Returns:
        bool: True if stationary, False otherwise
    """
    result = adfuller(series.dropna())
    
    logger.info("Augmented Dickey-Fuller Test:")
    logger.info(f"ADF Statistic: {result[0]}")
    logger.info(f"p-value: {result[1]}")
    
    for key, value in result[4].items():
        logger.info(f"Critical Value ({key}): {value}")
    
    # If p-value is less than 0.05, series is stationary
    is_stationary = result[1] < 0.05
    logger.info(f"Series is {'stationary' if is_stationary else 'non-stationary'}")
    
    return is_stationary


def build_arima_model(data, column='temperature'):
    """
    Build ARIMA model for climate data.
    
    Args:
        data (pd.DataFrame): Climate data
        column (str): Column to model
        
    Returns:
        tuple: Fitted model, predictions, parameters
    """
    logger.info(f"Building ARIMA model for {column}")
    
    # Check stationarity
    is_stationary = check_stationarity(data[column])
    
    # Convert to Series if DataFrame
    series = data[column] if isinstance(data, pd.DataFrame) else data
    
    # Split data for training and testing
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # Auto-determine ARIMA parameters using pmdarima
    auto_model = pm.auto_arima(
        train,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        m=12,  # Monthly seasonality
        seasonal=True,
        d=None,  # Auto-determine differencing
        D=None,  # Auto-determine seasonal differencing
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    # Get parameters
    params = auto_model.get_params()
    logger.info(f"Best ARIMA parameters: {params['order']}")
    if 'seasonal_order' in params:
        logger.info(f"Best seasonal parameters: {params['seasonal_order']}")
    
    # Build and fit SARIMAX model with best parameters
    model = SARIMAX(
        train,
        order=params['order'],
        seasonal_order=params.get('seasonal_order', (0, 0, 0, 0)),
        enforce_stationarity=not is_stationary,
        enforce_invertibility=False
    )
    
    fitted_model = model.fit(disp=False)
    logger.info("Model summary:")
    logger.info(fitted_model.summary())
    
    # Make predictions
    forecast_steps = len(test)
    forecast = fitted_model.get_forecast(steps=forecast_steps)
    predicted_mean = forecast.predicted_mean
    
    # Calculate error metrics
    mae = mean_absolute_error(test, predicted_mean)
    rmse = np.sqrt(mean_squared_error(test, predicted_mean))
    
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"Root Mean Squared Error: {rmse}")
    
    return fitted_model, predicted_mean, params


def forecast_future_climate(model, steps=120, original_data=None):
    """
    Forecast future climate using trained model.
    
    Args:
        model: Fitted time series model
        steps (int): Number of steps to forecast
        original_data: Original data for plotting
        
    Returns:
        pd.Series: Forecasted values
    """
    logger.info(f"Forecasting {steps} steps into the future")
    
    # Generate forecast
    forecast = model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # If original data is provided, create a nice visualization
    if original_data is not None:
        plt.figure(figsize=(14, 7))
        
        # Plot original data
        plt.plot(original_data.index, original_data, label='Historical Data')
        
        # Plot forecast
        forecast_start = original_data.index[-1] + pd.DateOffset(months=1)
        forecast_index = pd.date_range(start=forecast_start, periods=steps, freq='M')
        
        plt.plot(forecast_index, forecast_mean, color='red', label='Forecast')
        plt.fill_between(forecast_index, 
                        forecast_ci.iloc[:, 0], 
                        forecast_ci.iloc[:, 1], 
                        color='pink', alpha=0.3)
        
        plt.title('Climate Forecast', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        fig_dir = RESULTS_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_dir / "climate_forecast.png")
        plt.close()
    
    # Create DataFrame with forecast results
    forecast_df = pd.DataFrame({
        'forecast': forecast_mean,
        'lower_bound': forecast_ci.iloc[:, 0],
        'upper_bound': forecast_ci.iloc[:, 1]
    })
    
    return forecast_df


def save_model(model, filename='climate_arima_model.pkl'):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model to save
        filename (str): Filename for saved model
    """
    model_dir = MODELS_DIR / "analytics"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / filename
    logger.info(f"Saving model to {model_path}")
    
    joblib.dump(model, model_path)


def load_model(filename='climate_arima_model.pkl'):
    """
    Load trained model from disk.
    
    Args:
        filename (str): Filename of saved model
        
    Returns:
        Loaded model
    """
    model_path = MODELS_DIR / "analytics" / filename
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def main():
    """Main function to run the climate forecasting pipeline."""
    logger.info("Starting climate forecasting pipeline")
    
    # Load climate data
    climate_data = load_climate_data()
    
    # Plot the data
    plot_climate_data(climate_data, column='temperature')
    plot_climate_data(climate_data, column='temperature_anomaly')
    
    # Build ARIMA model
    model, predictions, params = build_arima_model(climate_data)
    
    # Save the model
    save_model(model)
    
    # Forecast future climate
    forecast = forecast_future_climate(model, steps=120, original_data=climate_data['temperature'])
    
    # Save forecast results
    results_dir = RESULTS_DIR / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(results_dir / "climate_forecast.csv")
    
    logger.info("Climate forecasting pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 