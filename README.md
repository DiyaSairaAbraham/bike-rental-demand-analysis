# Bike Rental Demand Analysis & Forecasting 🚴‍♂️📊

A comprehensive data analytics and time-series forecasting project utilizing the **UCI Bike Sharing Dataset**. This project conducts exploratory data analysis (EDA) to uncover patterns in consumer behavior across seasons, weather conditions, and day types, and deploys a predictive validation model for fleet logistics planning.

This project was developed independently during the **Taiwan Experience Education Program (TEEP)**.

---

## 🚀 Business Objectives & Insights
* **Macro Growth:** Evaluated operational scaling, revealing an explosive **64.43% year-over-year growth** in average daily traffic from 2011 to 2012.
* **Infrastructure Utility:** Proved that the system functions as core city transit infrastructure, with **Registered Commuters** making up **81.17%** of the total utilization footprint.
* **Environmental Constraints:** Isolated short-term environmental drop-offs, identifying a **63% reduction** in daily demand during rainy or snowy weather conditions.

---

## 🔮 Predictive Modeling (Stretch Goal)
To optimize fleet capacity, a **Simple Exponential Smoothing (SES)** model (\(\alpha = 0.35\)) was deployed to project baseline operational targets. The model was trained on historical data and validated against a blind 30-day testing window (December 2012).

### Model Validation Performance:
* **Mean Absolute Error (MAE):** 1,565.96 rentals
* **Root Mean Squared Error (RMSE):** 2,120.83 rentals

*Note: Given the high variance and extreme fluctuations of winter weather in December, the model successfully captures a stable baseline trend line, providing reliable supply targets for warehouse maintenance planning.*

---

## 📁 Repository Structure
```text
├── data/
│   └── day.csv             # 731-day aggregated historical log
├── outputs/charts/         # 10 generated analytical visualizations
│   ├── daily_rentals_moving_averages.png
│   ├── rentals_by_season.png
│   ├── rentals_by_month.png
│   ├── rentals_by_day_type.png
│   ├── rentals_by_weather.png
│   ├── user_type_shares.png
│   ├── yearly_user_types_stacked.png
│   ├── rentals_boxplot_by_season.png
│   ├── demand_forecast_validation.png
│   └── forecast_with_green_highlight_box.png
├── .gitignore              # Programmatically excludes virtual environment (.venv)
└── analysis.py            # Complete unified Python pipeline
```

---

## 🛠️ How to Run the Pipeline

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd bike-rental-demand-analysis
   ```

2. **Run the script:**
   ```bash
   python analysis.py
   ```
   *The data will validate automatically, print the statistical summaries to the console, and update all 10 pristine plots inside the `outputs/charts/` directory.*
