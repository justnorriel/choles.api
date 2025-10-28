from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import mysql.connector
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dateutil.relativedelta import relativedelta
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Catering Forecast API",
    description="AI-powered sales forecasting for catering reservations",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    print("🚀 Catering Forecast API is starting up...")
    print(f"📊 Database: {DB_CONFIG['database']} at {DB_CONFIG['host']}")
    print("🔗 Available endpoints:")
    print("   - GET /predict-monthly-sales?months_ahead=12")
    print("   - GET /health")
    print("   - GET /docs (Swagger UI)")
    print("✅ API is ready!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "user": os.environ.get("DB_USER", "u981606973_cholesAdmin"),
    "password": os.environ.get("DB_PASSWORD", "q~K:SNCwU]F1"),
    "database": os.environ.get("DB_NAME", "u981606973_choles_db")
}


def fetch_monthly_sales():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print(f"✅ Database connection successful to {DB_CONFIG['database']}")

        query = """
            SELECT DATE_FORMAT(event_date, '%Y-%m') AS month,
                   SUM(downpayment_price + balance) AS total_sales
            FROM reservation
            WHERE event_date IS NOT NULL
            GROUP BY month
            ORDER BY month ASC
        """

        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"📊 Fetched {len(df)} months of sales data")

        # Convert YYYY-MM to datetime
        df['month'] = pd.to_datetime(df['month'], format="%Y-%m")

        return df
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {err}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")


def linear_forecast(series: pd.Series, months_ahead: int) -> pd.Series:

    X = [[i] for i in range(len(series))]
    y = series.values
    model = LinearRegression()
    model.fit(X, y)
    future_X = [[i] for i in range(len(series), len(series) + months_ahead)]
    preds = model.predict(future_X)

    last_month = series.index[-1]
    future_index = [last_month + relativedelta(months=+i+1) for i in range(months_ahead)]
    return pd.Series(preds, index=future_index)

def holt_winters_forecast(series: pd.Series, months_ahead: int) -> pd.Series:
    try:
        model = ExponentialSmoothing(series,
                                     seasonal_periods=12,
                                     trend='add',
                                     seasonal='add',
                                     initialization_method="estimated").fit(optimized=True)
        preds = model.forecast(months_ahead)
        return preds
    except Exception:
        return linear_forecast(series, months_ahead)

def auto_forecast(series: pd.Series, months_ahead: int) -> (pd.Series, str):
    n = len(series)
    if n >= 24:
        preds = holt_winters_forecast(series, months_ahead)
        return preds, "holt_winters"
    else:
        preds = linear_forecast(series, months_ahead)
        return preds, "linear_regression"

class ForecastResponse(BaseModel):
    history: List[Dict[str, Any]]
    forecast: List[Dict[str, Any]]
    model: str
    months_ahead: int
@app.get("/predict-monthly-sales", response_model=ForecastResponse)
def predict_monthly_sales(months_ahead: int = Query(3, ge=1, le=36)):
    df = fetch_monthly_sales()
    if df.empty:
        raise HTTPException(status_code=404, detail="No reservation data found in the database.")

    # ✅ Ensure month is index for forecasting
    df = df.set_index('month')

    series = df['total_sales']
    preds, model_name = auto_forecast(series, months_ahead)

    # history data formatting
    history = [{"month": dt.strftime("%Y-%m"), "total_sales": float(val)}
               for dt, val in series.items()]

    # predicted data formatting
    forecast = [{"month": dt.strftime("%Y-%m"), "predicted_total_sales": float(val)}
                for dt, val in preds.items()]

    return {
        "history": history,
        "forecast": forecast,
        "model": model_name,
        "months_ahead": months_ahead
    }


@app.get("/health")
def health():
    return {"status": "ok"}
