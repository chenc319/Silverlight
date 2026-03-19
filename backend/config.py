"""Ticker universe and constants."""

TICKER_GROUPS = {
    "INDICES": ["SPY", "QQQ", "DIA", "IWM", "SMH", "UUP", "USO", "TLT"],
    "MAG7": ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"],
    "MEGA_TECH": ["NFLX", "AVGO", "AMD", "ORCL"],
    "SECTORS": ["XLK", "XLC", "XLY", "XLF", "XLV", "XLU", "XLRE", "XLI", "XLP", "XLB", "XLE"],
}

ALL_TICKERS = (
    TICKER_GROUPS["INDICES"]
    + TICKER_GROUPS["MAG7"]
    + TICKER_GROUPS["MEGA_TECH"]
    + TICKER_GROUPS["SECTORS"]
)

BENCHMARK = "SPY"

DB_PATH = "/home/user/workspace/silversignal/data/market.db"
