# StockPredictor
A real-time stock predictor that uses Kite MCP to fetch live market data and generate short-term price predictions using custom analytics.

# Stock Price Predictor using Kite MCP and Machine Learning

A comprehensive Python-based stock price prediction system that fetches real-time and historical market data using **Kite MCP** (Zerodha's Market Capture Protocol) or falls back to **Yahoo Finance** when necessary. It supports multiple predictive models including **Linear Regression**, **Random Forest**, and **LSTM Neural Networks**, and combines them into an **ensemble prediction** for increased accuracy.

---

## Features

* ✅ Integrates with **Kite MCP** API for live data
* ✅ Supports **Yahoo Finance** as fallback
* ✅ Performs **feature engineering** with technical indicators (SMA, RSI, MACD)
* ✅ Adds **sentiment analysis** using price momentum or NLTK's VADER (if available)
* ✅ Trains and evaluates **Linear Regression**, **Random Forest**, and **LSTM** models
* ✅ Automatically plots predictions and saves results
* ✅ Builds an **ensemble model** combining predictions from all available models

---

## Dependencies

### Required

* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `yfinance`
* `requests`

### Optional (for enhanced functionality)

* `nltk` (for sentiment analysis)
* `TA-Lib` (for advanced technical indicators)
* `tensorflow` (for LSTM model)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-predictor-kite.git
cd stock-predictor-kite
```

### 2. Install Dependencies

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> If you do not want to use optional features like TensorFlow or TA-Lib, simply ignore related warnings.

### 3. Set Environment Variables (Optional for Kite MCP)

If you want to use Kite MCP API:

```bash
export KITE_API_KEY=your_api_key
export KITE_API_SECRET=your_api_secret
export KITE_REQUEST_TOKEN=your_request_token
```

If these are not provided, the system will default to using `yfinance`.

---

## How to Use

### Run with Default Settings (Apple stock, 1 year data):

```bash
python main.py
```

### Customize Symbol and Time Period

```python
main(symbol='RELIANCE.NS', days=730)  # 2 years of Reliance stock data from NSE
```

---

## Output

* Plots of predictions (`.png`): `AAPL_Linear_Regression_prediction.png`, etc.
* Model performance metrics (`.csv`): `AAPL_metrics.csv`
* Combined prediction results (`.csv`): `AAPL_predictions.csv`

### Example Output Log

```
INFO: Successfully authenticated with Kite API
INFO: Successfully fetched data for AAPL
INFO: Linear Regression model trained successfully
INFO: Random Forest model trained successfully
INFO: LSTM model trained successfully
INFO: Ensemble model evaluated
INFO: Results saved to AAPL_predictions.csv
```

---

## Model Comparison

Each model is evaluated using:

* MSE (Mean Squared Error)
* MAE (Mean Absolute Error)
* RMSE (Root Mean Square Error)
* R² Score (Coefficient of Determination)

The model with the lowest **MSE** is considered the best performer.

---

## Notes

* If fewer than 100 data points remain after preprocessing, the system will halt.
* TensorFlow models can be memory-intensive; reduce epochs or increase RAM if needed.
* Sentiment is generated based on price change patterns if NLTK is not available.

---

## Troubleshooting

* **Authentication failed:** Ensure correct environment variables or API credentials.
* **TA-Lib import error:** Either install TA-Lib or proceed with built-in fallback methods.
* **TensorFlow not found:** Install `tensorflow` to use LSTM model.

---

## Future Improvements

* Integration with live trading via Kite Connect
* Real-time dashboard using Streamlit
* Multi-stock batch prediction
* Advanced ensemble methods (e.g., weighted average, stacking)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

* [Zerodha Kite API](https://kite.trade/)
* [Yahoo Finance](https://finance.yahoo.com/)
* [NLTK](https://www.nltk.org/)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/)
* [TensorFlow](https://www.tensorflow.org/)

---

## Contribution Guidelines

We welcome contributions. Here's how you can help:

* Submit issues and feature requests
* Fork and raise pull requests
* Contribute to documentation
* Help build a web interface or REST API

## Contact and Credits

Author: Swapnendu Sikdar
Email: swapnendusikdar@gmail.com
Institution: Jadavpur University
Field: Electrical Engineering
Interests: AI/ML, Deep Learning,Machine Learning, Computer Vision, Robotics, Smart Infrastructure
