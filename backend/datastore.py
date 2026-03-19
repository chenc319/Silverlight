"""
SQLite memory bank with delta-fetch logic.
Downloads OHLC from yfinance, stores in SQLite, computes indicators.
"""

import sqlite3
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

from config import ALL_TICKERS, BENCHMARK, DB_PATH
from indicators import compute_all_indicators
from scoring import score_ticker


class MarketDataStore:
    """SQLite-backed market data store with delta-fetch."""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create tables for each ticker if they don't exist."""
        conn = self._get_conn()
        for ticker in ALL_TICKERS:
            table = self._table_name(ticker)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    date TEXT PRIMARY KEY,
                    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                    td_setup_buy INTEGER DEFAULT 0, td_setup_sell INTEGER DEFAULT 0,
                    td_countdown_buy INTEGER DEFAULT 0, td_countdown_sell INTEGER DEFAULT 0,
                    td_setup_count INTEGER DEFAULT 0, td_countdown_count INTEGER DEFAULT 0,
                    rsi14 REAL DEFAULT 50, stoch_k REAL DEFAULT 50, stoch_d REAL DEFAULT 50,
                    macd_line REAL DEFAULT 0, macd_signal REAL DEFAULT 0, macd_hist REAL DEFAULT 0,
                    bb_upper REAL DEFAULT 0, bb_mid REAL DEFAULT 0, bb_lower REAL DEFAULT 0,
                    rel_to_spy REAL DEFAULT 0
                )
            """)
        conn.commit()
        conn.close()

    def _table_name(self, ticker: str) -> str:
        return f"ohlc_{ticker.replace('-', '_').replace('.', '_')}"

    def get_last_date(self, ticker: str) -> Optional[str]:
        """Get the last stored date for a ticker."""
        conn = self._get_conn()
        table = self._table_name(ticker)
        try:
            row = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()
            conn.close()
            return row[0] if row and row[0] else None
        except Exception:
            conn.close()
            return None

    def get_row_count(self, ticker: str) -> int:
        conn = self._get_conn()
        table = self._table_name(ticker)
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            conn.close()
            return 0

    def fetch_and_store(self, ticker: str) -> dict:
        """
        Delta-fetch: only download new bars from yfinance.
        If DB is empty, seed with 2 years of data.
        Returns status dict.
        """
        last_date = self.get_last_date(ticker)

        try:
            if last_date is None:
                # Seed with 2 years
                df_new = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
                if df_new.empty:
                    return {"ticker": ticker, "status": "error", "message": "No data from yfinance"}
            else:
                # Delta fetch: start from last_date + 1 day
                start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                end = datetime.now().strftime("%Y-%m-%d")
                if start >= end:
                    return {"ticker": ticker, "status": "up_to_date", "rows": self.get_row_count(ticker)}

                df_new = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if df_new.empty:
                    return {"ticker": ticker, "status": "up_to_date", "rows": self.get_row_count(ticker)}

            # Flatten multi-level columns if present
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = [c[0] if isinstance(c, tuple) else c for c in df_new.columns]

            # Normalize
            df_new.columns = [c.lower() for c in df_new.columns]
            df_new.index.name = 'date'
            df_new = df_new.reset_index()
            df_new['date'] = pd.to_datetime(df_new['date']).dt.strftime('%Y-%m-%d')

            # Store raw OHLC
            conn = self._get_conn()
            table = self._table_name(ticker)

            for _, row in df_new.iterrows():
                conn.execute(f"""
                    INSERT OR REPLACE INTO {table} (date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    row['date'],
                    float(row.get('open', 0)),
                    float(row.get('high', 0)),
                    float(row.get('low', 0)),
                    float(row.get('close', 0)),
                    int(row.get('volume', 0)),
                ))
            conn.commit()
            conn.close()

            # Now recompute indicators on full history
            self._recompute_indicators(ticker)

            return {
                "ticker": ticker,
                "status": "updated",
                "newBars": len(df_new),
                "totalRows": self.get_row_count(ticker),
            }

        except Exception as e:
            return {"ticker": ticker, "status": "error", "message": str(e)}

    def _recompute_indicators(self, ticker: str):
        """Load full history from DB, compute indicators, update DB."""
        conn = self._get_conn()
        table = self._table_name(ticker)

        df = pd.read_sql_query(
            f"SELECT date, open, high, low, close, volume FROM {table} ORDER BY date",
            conn
        )
        if df.empty:
            conn.close()
            return

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Get SPY close for relative perf
        spy_close = None
        if ticker != BENCHMARK:
            spy_table = self._table_name(BENCHMARK)
            try:
                spy_df = pd.read_sql_query(
                    f"SELECT date, close FROM {spy_table} ORDER BY date", conn
                )
                if not spy_df.empty:
                    spy_df['date'] = pd.to_datetime(spy_df['date'])
                    spy_df = spy_df.set_index('date')
                    spy_close = spy_df['close']
            except Exception:
                pass

        # Compute all indicators
        df = compute_all_indicators(df, spy_close)
        df['symbol'] = ticker

        # Update DB with computed columns
        df_reset = df.reset_index()
        for _, row in df_reset.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
            conn.execute(f"""
                UPDATE {table} SET
                    td_setup_buy = ?, td_setup_sell = ?,
                    td_countdown_buy = ?, td_countdown_sell = ?,
                    td_setup_count = ?, td_countdown_count = ?,
                    rsi14 = ?, stoch_k = ?, stoch_d = ?,
                    macd_line = ?, macd_signal = ?, macd_hist = ?,
                    bb_upper = ?, bb_mid = ?, bb_lower = ?,
                    rel_to_spy = ?
                WHERE date = ?
            """, (
                int(row.get('td_setup_buy', 0)),
                int(row.get('td_setup_sell', 0)),
                int(row.get('td_countdown_buy', 0)),
                int(row.get('td_countdown_sell', 0)),
                int(row.get('td_setup_count', 0)),
                int(row.get('td_countdown_count', 0)),
                float(row.get('rsi14', 50)),
                float(row.get('stoch_k', 50)),
                float(row.get('stoch_d', 50)),
                float(row.get('macd_line', 0)),
                float(row.get('macd_signal', 0)),
                float(row.get('macd_hist', 0)),
                float(row.get('bb_upper', 0)),
                float(row.get('bb_mid', 0)),
                float(row.get('bb_lower', 0)),
                float(row.get('rel_to_spy', 0)),
                date_str,
            ))

        conn.commit()
        conn.close()

    def get_signals(self, ticker: str) -> dict:
        """Get scored signal for a ticker from DB data."""
        conn = self._get_conn()
        table = self._table_name(ticker)

        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} ORDER BY date", conn
            )
            conn.close()

            if df.empty:
                return {"symbol": ticker, "error": "No data"}

            df['symbol'] = ticker
            return score_ticker(df)
        except Exception as e:
            conn.close()
            return {"symbol": ticker, "error": str(e)}

    def get_ohlc(self, ticker: str, bars: int = 252) -> list:
        """Get OHLC + indicator data for chart rendering."""
        conn = self._get_conn()
        table = self._table_name(ticker)

        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} ORDER BY date DESC LIMIT {bars}", conn
            )
            conn.close()

            if df.empty:
                return []

            df = df.sort_values('date')

            return df.to_dict(orient='records')
        except Exception as e:
            conn.close()
            return []

    def get_all_signals(self) -> dict:
        """Get signals for all tickers."""
        signals = {}
        for ticker in ALL_TICKERS:
            try:
                sig = self.get_signals(ticker)
                signals[ticker] = sig
            except Exception:
                signals[ticker] = {"symbol": ticker, "error": "computation failed"}
        return signals

    def refresh_all(self, callback=None):
        """Fetch and recompute for all tickers. callback(ticker, status) for progress."""
        # Fetch SPY first (needed for relative perf)
        status = self.fetch_and_store(BENCHMARK)
        if callback:
            callback(BENCHMARK, status)

        results = [status]
        for ticker in ALL_TICKERS:
            if ticker == BENCHMARK:
                continue
            try:
                s = self.fetch_and_store(ticker)
                results.append(s)
                if callback:
                    callback(ticker, s)
            except Exception as e:
                err = {"ticker": ticker, "status": "error", "message": str(e)}
                results.append(err)
                if callback:
                    callback(ticker, err)
            time.sleep(0.1)  # Rate limiting

        return results
