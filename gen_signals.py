#!/usr/bin/env python3
"""Generate LLM trading signals for all 6 crypto tickers.

Writes signals.json to TRADINGAGENTS_HOME so the HTTP signals-server can
serve it to strategy-lab's llm_agent strategy.

Run: python gen_signals.py
"""
import json, os, sys, time, traceback
from datetime import datetime, timedelta

OUT = os.path.join(
    os.environ.get("TRADINGAGENTS_HOME", os.path.expanduser("~/.tradingagents")),
    "signals.json",
)

TICKERS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "BNBUSDT": "BNB-USD",
    "XRPUSDT": "XRP-USD",
    "DOGEUSDT": "DOGE-USD",
}

ANALYSTS = ("market", "news")


def main():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Generating signals for {date}")

    signals = {}
    if os.path.exists(OUT):
        try:
            signals = json.loads(open(OUT).read())
        except Exception:
            pass

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1

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

            entry = {
                "ticker": yahoo_ticker,
                "rating": rating,
                "decision": (decision or "")[:500],
                "timestamp": time.time(),
                "date": date,
            }
            signals[binance_sym] = entry
            signals[yahoo_ticker] = entry

            with open(OUT, "w") as f:
                json.dump(signals, f, indent=2)

            print(f"  -> {rating}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. {len(TICKERS)} tickers -> {OUT}")


if __name__ == "__main__":
    main()
