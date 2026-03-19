#!/usr/bin/env python3
"""
SilverSignal API Server — FastAPI + Uvicorn
Endpoints:
  GET  /api/signals         → all ticker signals
  GET  /api/signals/{ticker} → single ticker signal
  POST /api/refresh          → trigger data refresh
  GET  /api/refresh/status   → last refresh status
  GET  /api/ohlc/{ticker}/{timeframe}  → OHLC + indicators for charts
  GET  /api/regime           → market regime
"""

import sys
import os
import json
import threading
from datetime import datetime

# Add backend dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config import ALL_TICKERS, TICKER_GROUPS
from datastore import MarketDataStore
from scoring import compute_market_regime

app = FastAPI(title="SilverSignal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data store
store = MarketDataStore()

# Refresh state
refresh_state = {
    "is_refreshing": False,
    "progress": [],
    "last_completed": None,
    "last_updated": None,
}
refresh_lock = threading.Lock()


# ═══════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════

@app.get("/api/signals")
def get_all_signals():
    """Get scored signals for all tickers + market regime."""
    signals = store.get_all_signals()

    # Rank within groups
    for group_name, tickers in TICKER_GROUPS.items():
        group_signals = [(t, signals[t]) for t in tickers if t in signals and "error" not in signals[t]]
        group_signals.sort(key=lambda x: x[1].get("dailyScore", 0), reverse=True)
        for rank, (ticker, _) in enumerate(group_signals, 1):
            signals[ticker]["rank"] = rank

    regime = compute_market_regime(signals)

    return {
        "signals": signals,
        "regime": regime,
        "lastUpdated": refresh_state.get("last_updated") or datetime.now().isoformat(),
        "tickerCount": len([s for s in signals.values() if "error" not in s]),
    }


@app.get("/api/signals/{ticker}")
def get_ticker_signal(ticker: str):
    """Get signal for a single ticker."""
    ticker = ticker.upper()
    if ticker not in ALL_TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in universe")
    return store.get_signals(ticker)


@app.post("/api/refresh")
def trigger_refresh():
    """Trigger a full data refresh (async)."""
    with refresh_lock:
        if refresh_state["is_refreshing"]:
            return {"status": "already_refreshing", "progress": refresh_state["progress"]}

        refresh_state["is_refreshing"] = True
        refresh_state["progress"] = []

    def _do_refresh():
        def callback(ticker, status):
            with refresh_lock:
                refresh_state["progress"].append({
                    "ticker": ticker,
                    "status": status.get("status", "unknown"),
                    "message": status.get("message", ""),
                })

        try:
            store.refresh_all(callback=callback)
        except Exception as e:
            with refresh_lock:
                refresh_state["progress"].append({
                    "ticker": "SYSTEM",
                    "status": "error",
                    "message": str(e),
                })
        finally:
            with refresh_lock:
                refresh_state["is_refreshing"] = False
                refresh_state["last_completed"] = datetime.now().isoformat()
                refresh_state["last_updated"] = datetime.now().isoformat()

    thread = threading.Thread(target=_do_refresh, daemon=True)
    thread.start()

    return {"status": "started", "message": "Refresh initiated for all tickers"}


@app.get("/api/refresh/status")
def get_refresh_status():
    """Get current refresh status."""
    with refresh_lock:
        return {
            "isRefreshing": refresh_state["is_refreshing"],
            "progress": refresh_state["progress"],
            "lastCompleted": refresh_state.get("last_completed"),
            "totalTickers": len(ALL_TICKERS),
            "completedTickers": len(refresh_state["progress"]),
        }


@app.get("/api/ohlc/{ticker}/{timeframe}")
def get_ohlc(ticker: str, timeframe: str = "daily"):
    """Get OHLC + indicator data for chart rendering."""
    ticker = ticker.upper()
    if ticker not in ALL_TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in universe")

    bars = 252 if timeframe == "daily" else 104
    data = store.get_ohlc(ticker, bars=bars)

    if not data:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    return {"ticker": ticker, "timeframe": timeframe, "bars": len(data), "data": data}


@app.get("/api/regime")
def get_regime():
    """Get market regime."""
    signals = store.get_all_signals()
    return compute_market_regime(signals)


@app.get("/api/health")
def health():
    return {"status": "ok", "tickers": len(ALL_TICKERS)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
