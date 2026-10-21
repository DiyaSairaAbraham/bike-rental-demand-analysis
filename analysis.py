from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.patches as patches

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "day.csv"
CHART_DIR = PROJECT_DIR / "outputs" / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

SEASON_LABELS = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER_LABELS = {1: "Clear", 2: "Mist/Cloudy", 3: "Light Rain/Snow", 4: "Heavy Rain/Snow"}
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

SEASON_ORDER = ["Spring", "Summer", "Fall", "Winter"]
WEATHER_ORDER = ["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain/Snow"]
DAY_ORDER = ["Non-working day", "Working day"]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}\nEnsure day.csv is inside the data folder.")

df = pd.read_csv(DATA_PATH)

required_columns = {
    "dteday", "season", "yr", "mnth", "workingday",
    "weathersit", "casual", "registered", "cnt",
}

missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

df["dteday"] = pd.to_datetime(df["dteday"], errors="raise")
df = df.sort_values("dteday").reset_index(drop=True)

if df["dteday"].duplicated().any():
    raise ValueError("Duplicate dates found in the daily dataset.")

original_missing = df.isnull().sum()
if original_missing.any():
    print("\nWARNING: Missing values found in the original dataset:\n", original_missing[original_missing > 0])
else:
    print("Original dataset validation: no missing values.")

if not (df["casual"] + df["registered"]).equals(df["cnt"]):
    raise ValueError("Validation failed: casual + registered does not equal cnt.")

if (df[["casual", "registered", "cnt"]] < 0).any().any():
    raise ValueError("Negative rental counts were found.")

df["moving_avg_7"] = df["cnt"].rolling(window=7).mean()
df["moving_avg_30"] = df["cnt"].rolling(window=30).mean()

df["season_name"] = df["season"].map(SEASON_LABELS)
df["weather_name"] = df["weathersit"].map(WEATHER_LABELS)
df["month_name"] = df["mnth"].map(MONTH_LABELS)
df["day_type"] = df["workingday"].map({0: "Non-working day", 1: "Working day"})

print("\n" + "=" * 65)
print("BIKE RENTAL DEMAND ANALYSIS")
print("=" * 65)
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['dteday'].min().date()} to {df['dteday'].max().date()}")
print(f"Total daily records: {len(df)}")
print("Rental count validation: PASSED")


# ============================================================
# 3. DAILY RENTALS AND MOVING AVERAGES
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["dteday"], df["cnt"], label="Daily rentals", alpha=0.4)
ax.plot(df["dteday"], df["moving_avg_7"], label="7-day moving average", linewidth=2)
ax.plot(df["dteday"], df["moving_avg_30"], label="30-day moving average", linewidth=2)
ax.set_title("Daily Bike Rentals with Moving Averages (2011–2012)")
ax.set_xlabel("Date")
ax.set_ylabel("Total Rentals")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(CHART_DIR / "daily_rentals_moving_averages.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 4. AVERAGE DAILY RENTALS BY SEASON
# ============================================================

season_demand = df.groupby("season_name", observed=True)["cnt"].mean().reindex(SEASON_ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
season_demand.plot(kind="bar", ax=ax)
ax.set_title("Average Daily Bike Rentals by Season")
ax.set_xlabel("Season")
ax.set_ylabel("Average Daily Rentals")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(CHART_DIR / "rentals_by_season.png", dpi=300)
plt.close(fig)


# ============================================================
# 5. AVERAGE DAILY RENTALS BY MONTH
# ============================================================

monthly_demand = df.groupby("mnth")["cnt"].mean().reindex(range(1, 13))
fig, ax = plt.subplots(figsize=(10, 5))
monthly_demand.plot(kind="bar", ax=ax)
ax.set_title("Average Daily Bike Rentals by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Average Daily Rentals")
ax.set_xticks(range(12), [MONTH_LABELS[m] for m in range(1, 13)], rotation=0)
fig.tight_layout()
fig.savefig(CHART_DIR / "rentals_by_month.png", dpi=300)
plt.close(fig)


# ============================================================
# 6. WORKING DAYS VS NON-WORKING DAYS
# ============================================================

workingday_demand = df.groupby("day_type", observed=True)["cnt"].mean().reindex(DAY_ORDER)
fig, ax = plt.subplots(figsize=(7, 5))
workingday_demand.plot(kind="bar", ax=ax)
ax.set_title("Average Bike Rentals: Working vs Non-working Days")
ax.set_xlabel("Day Type")
ax.set_ylabel("Average Daily Rentals")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(CHART_DIR / "rentals_by_day_type.png", dpi=300)
plt.close(fig)


# ============================================================
# 7. AVERAGE DAILY RENTALS BY WEATHER
# ============================================================

weather_demand = df.groupby("weather_name", observed=True)["cnt"].mean().reindex(WEATHER_ORDER).dropna()
fig, ax = plt.subplots(figsize=(9, 5))
weather_demand.plot(kind="bar", ax=ax)
ax.set_title("Average Daily Bike Rentals by Weather")
ax.set_xlabel("Weather Situation")
ax.set_ylabel("Average Daily Rentals")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(CHART_DIR / "rentals_by_weather.png", dpi=300)
plt.close(fig)


# ============================================================
# 8. OVERALL CASUAL VS REGISTERED USER SHARES
# ============================================================

user_totals = df[["casual", "registered"]].sum()
user_shares = user_totals / user_totals.sum() * 100
fig, ax = plt.subplots(figsize=(7, 5))
user_shares.rename({"casual": "Casual", "registered": "Registered"}).plot(kind="bar", ax=ax)
ax.set_title("Share of Total Bike Rentals by User Type")
ax.set_xlabel("User Type")
ax.set_ylabel("Share of Total Rentals (%)")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(CHART_DIR / "user_type_shares.png", dpi=300)
plt.close(fig)


# ============================================================
# 9. CASUAL VS REGISTERED BY YEAR — STACKED BAR CHART
# ============================================================

yearly_users = df.groupby("yr")[["casual", "registered"]].sum().reindex([0, 1])
yearly_users.index = yearly_users.index.map({0: 2011, 1: 2012})
fig, ax = plt.subplots(figsize=(8, 5))
yearly_users.rename(columns={"casual": "Casual", "registered": "Registered"}).plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Casual and Registered Rentals by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Total Rentals")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="User Type")
fig.tight_layout()
fig.savefig(CHART_DIR / "yearly_user_types_stacked.png", dpi=300)
plt.close(fig)


# ============================================================
# 10. YEAR-OVER-YEAR GROWTH IN AVERAGE DAILY RENTALS
# ============================================================

yearly_demand = df.groupby("yr")["cnt"].mean()
print("\nAverage daily rentals by year:")
for y_code, y_label in [(0, 2011), (1, 2012)]:
    print(f"{y_label}: {yearly_demand.loc[y_code]:.2f}")
growth = (yearly_demand.loc[1] - yearly_demand.loc[0]) / yearly_demand.loc[0] * 100
print(f"Year-over-year growth: {growth:.2f}%")


# ============================================================
# 11. BOX PLOT — DISTRIBUTION BY SEASON
# ============================================================

boxplot_data = [df.loc[df["season_name"] == season, "cnt"].dropna().to_numpy() for season in SEASON_ORDER]
fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(boxplot_data, tick_labels=SEASON_ORDER)
ax.set_title("Distribution of Daily Bike Rentals by Season")
ax.set_xlabel("Season")
ax.set_ylabel("Total Daily Rentals")
fig.tight_layout()
fig.savefig(CHART_DIR / "rentals_boxplot_by_season.png", dpi=300)
plt.close(fig)


# ============================================================
# 12. FORECAST MODEL TRAIN, EVALUATE & CHART (FIXED DATES)
# ============================================================

print("\n" + "=" * 65)
print("RUNNING FORECASTING MODEL VALIDATION (STRETCH GOAL)")
print("=" * 65)

# Split data: Last 30 days of the dataset (December 2012) form the Test/Validation window
train_df = df.iloc[:-30].reset_index(drop=True)
test_df = df.iloc[-30:].reset_index(drop=True)

alpha = 0.35
smoothed_history = np.zeros(len(df))

# 1. Track history & fit alpha level up until the last 30 days
current_level = train_df["cnt"].iloc[0]
for i in range(len(train_df)):
    current_level = alpha * train_df["cnt"].iloc[i] + (1 - alpha) * current_level
    smoothed_history[i] = current_level

# 2. Hold the level steady to project a flat horizon across the last 30 test days of 2012
forecast_val = current_level
for i in range(len(train_df), len(df)):
    smoothed_history[i] = forecast_val

# Extract actuals vs forecast array for error metrics
test_actual = test_df["cnt"].to_numpy()
test_predicted = smoothed_history[-30:]

mae = np.mean(np.abs(test_actual - test_predicted))
rmse = np.sqrt(np.mean((test_actual - test_predicted) ** 2))

print(f"Validation Horizon: Last 30 Days of dataset ({test_df['dteday'].min().date()} to {test_df['dteday'].max().date()})")
# ============================================================
# 12. FORECAST MODEL TRAIN & EVALUATE (FIXED & ALIGNED)
# ============================================================

print("\n" + "=" * 65)
print("RUNNING FORECASTING MODEL VALIDATION (STRETCH GOAL)")
print("=" * 65)

# Split data: The final 30 days of the dataset (December 2012) form our Test window
train_df = df.iloc[:-30].reset_index(drop=True)
test_df = df.iloc[-30:].reset_index(drop=True)

alpha = 0.35
smoothed_history = np.zeros(len(df))

# 1. Track history up until the last 30 days
current_level = train_df["cnt"].iloc[0]
for i in range(len(train_df)):
    current_level = alpha * train_df["cnt"].iloc[i] + (1 - alpha) * current_level
    smoothed_history[i] = current_level

# 2. Hold the level steady to project a flat horizon across the last 30 test days of 2012
forecast_val = current_level
for i in range(len(train_df), len(df)):
    smoothed_history[i] = forecast_val

# Extract actuals vs forecast array for error metrics
test_actual = test_df["cnt"].to_numpy()
test_predicted = smoothed_history[-30:]

mae = np.mean(np.abs(test_actual - test_predicted))
rmse = np.sqrt(np.mean((test_actual - test_predicted) ** 2))

print(f"Validation Horizon: Last 30 Days of dataset ({test_df['dteday'].min().date()} to {test_df['dteday'].max().date()})")
print(f"👉 Mean Absolute Error (MAE): {mae:.2f} rentals")
print(f"👉 Root Mean Squared Error (RMSE): {rmse:.2f} rentals")

# 3. Save the Zoomed-In 90-Day Focus Chart (Matches your Chart 3 exactly)
fig, ax = plt.subplots(figsize=(12, 6))
history_slice = df.iloc[-90:]
ax.plot(history_slice["dteday"], history_slice["cnt"], label="Actual Demand History", color="steelblue", linewidth=2)
ax.plot(test_df["dteday"], test_predicted, label=f"Validation Forecast Line (α={alpha})", color="crimson", linestyle="--", linewidth=2.5)
ax.set_title("30-Day Bike Rental Demand Forecast Validation (Zoomed)")
ax.set_xlabel("Date")
ax.set_ylabel("Total Rentals")
ax.legend(loc="upper left")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(CHART_DIR / "demand_forecast_validation.png", dpi=300)
plt.close(fig)


# ============================================================
# 13. FULL HISTORY TIMELINE WITH CORRECTED GREEN HIGHLIGHT BOX
# ============================================================

fig, ax = plt.subplots(figsize=(15, 6))

# Plot all actual spikes across 2011-2012
ax.plot(df["dteday"], df["cnt"], label="Actual Demand Spikes", color="lightsteelblue", alpha=0.5)

# Plot training tracking phase (from start up to the last 30 days)
ax.plot(train_df["dteday"], smoothed_history[:-30], label=f"Exponential Smoothing Fit Path", color="crimson", linewidth=2)

# Plot validation flat trend line (over the final 30 days of December 2012)
ax.plot(test_df["dteday"], test_predicted, label="Validation Forecast Horizon", color="crimson", linestyle="--", linewidth=3)

# Draw the Green Highlight Box strictly around December 2012 (the final 30 days)
box_left = matplotlib.dates.date2num(test_df["dteday"].iloc[0] - pd.Timedelta(days=2))
box_width = matplotlib.dates.date2num(test_df["dteday"].iloc[-1]) - box_left + 2
box_bottom = -100
box_height = 8800

rect = patches.Rectangle(
    (box_left, box_bottom), box_width, box_height,
    linewidth=2, edgecolor="forestgreen", facecolor="honeydew", alpha=0.3, label="Testing Validation Window (Dec 2012)"
)
ax.add_patch(rect)

# Chart layout adjustments
ax.set_title("Two-Year Bike History with Aligned Testing Validation Focus Box (2011–2012)")
ax.set_xlabel("Date")
ax.set_ylabel("Total Rentals")
ax.set_ylim(-200, 9000)
ax.legend(loc="upper left")
ax.grid(alpha=0.15)

fig.tight_layout()
fig.savefig(CHART_DIR / "forecast_with_green_highlight_box.png", dpi=300)
plt.close(fig)
