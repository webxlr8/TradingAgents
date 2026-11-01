#!/usr/bin/env python3
"""Generate LLM trading signals for all 6 crypto tickers.

Writes signals.json to TRADINGAGENTS_HOME so the HTTP signals-server can
serve it to strategy-lab's llm_agent strategy.

Also maintains a trades.json journal that logs signal CHANGES (e.g.,
BTC went from Hold to Buy) for post-trade analysis.

Run: python gen_signals.py
"""
import json, os, sys, time, traceback
from datetime import datetime, timedelta

HOME_DIR = os.environ.get("TRADINGAGENTS_HOME", os.path.expanduser("~/.tradingagents"))
OUT = os.path.join(HOME_DIR, "signals.json")
TRADES_LOG = os.path.join(HOME_DIR, "trades.json")

TICKERS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "BNBUSDT": "BNB-USD",
    "XRPUSDT": "XRP-USD",
    "DOGEUSDT": "DOGE-USD",
}

# All useful analysts for crypto: market (technicals), news (sentiment),
# social (social media sentiment). Skip 'fundamentals' — it's SEC-focused.
# StockTwits 403 is non-fatal; the social analyst still contributes Reddit data.
ANALYSTS = ("market", "news", "social")


def load_json(path):
    """Load a JSON file, returning empty dict/list on error."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_json(path, data):
    """Atomically write JSON to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def log_trade(binance_sym, yahoo_ticker, old_rating, new_rating, decision, date):
    """Append a signal change to the trades journal."""
    trades = load_json(TRADES_LOG)
    if not isinstance(trades, list):
        trades = []

    entry = {
        "timestamp": time.time(),
        "datetime": datetime.utcnow().isoformat() + "Z",
        "date": date,
        "binance_symbol": binance_sym,
        "yahoo_ticker": yahoo_ticker,
        "old_signal": old_rating,
        "new_signal": new_rating,
        "decision_summary": (decision or "")[:500],
    }
    trades.append(entry)
    save_json(TRADES_LOG, trades)
    print(f"  -> TRADE JOURNAL: {yahoo_ticker} {old_rating} -> {new_rating}")


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Generating signals for {date}")
    print(f"Analysts: {ANALYSTS}")
    print(f"Output: {OUT}")
    print(f"Trades log: {TRADES_LOG}")

    # Load existing signals (to detect changes)
    signals = load_json(OUT)
    if not isinstance(signals, dict):
        signals = {}

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 2          # Thorough bull/bear debate
    config["max_risk_discuss_rounds"] = 1

    success_count = 0
    fail_count = 0

    for binance_sym, yahoo_ticker in TICKERS.items():
        print(f"\n{'='*50}\n{yahoo_ticker} ({binance_sym})\n{'='*50}")
        try:
            ta = TradingAgentsGraph(selected_analysts=ANALYSTS, config=config)
            state, decision = ta.propagate(yahoo_ticker, date, "crypto")

            rating = "Hold"
            dec_lower = (decision or "").lower()
            for word in ["underweight", "sell", "overweight", "buy"]:
                if word in dec_lower:
                    rating = word.capitalize()
                    break

            # Check for signal change
            old_entry = signals.get(binance_sym)
            old_rating = old_entry.get("rating", "None") if old_entry else "None"
            if old_rating != rating:
                log_trade(binance_sym, yahoo_ticker, old_rating, rating, decision, date)

            entry = {
                "ticker": yahoo_ticker,
                "rating": rating,
                "decision": (decision or "")[:500],
                "timestamp": time.time(),
                "date": date,
                "analysts": list(ANALYSTS),
                "debate_rounds": config["max_debate_rounds"],
            }
            signals[binance_sym] = entry
            signals[yahoo_ticker] = entry

            # Save after each ticker so partial results survive failures
            save_json(OUT, signals)

            print(f"  -> {rating}")
            success_count += 1
        except Exception as e:
            print(f"  -> FAILED: {e}")
            traceback.print_exc()
            fail_count += 1

    print(f"\nDone. {success_count}/{len(TICKERS)} succeeded, {fail_count} failed -> {OUT}")
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
