#!/usr/bin/env python3
"""Generate LLM trading signals for all 6 crypto tickers.

Writes signals.json to TRADINGAGENTS_HOME so the HTTP signals-server can
serve it to strategy-lab's llm_agent strategy.

Also maintains a trades.json journal that logs signal CHANGES (e.g.,
BTC went from Hold to Buy) for post-trade analysis.

Key design decisions:
  - market + news analysts ONLY (social hangs on crypto tickers)
  - 1 debate round (fast, ~60-90s per ticker)
  - 60s delay between tickers to avoid Anthropic rate limits
  - 5-minute timeout per ticker (kills hung API calls)
  - Push signals to lab.wx8.org after each ticker

Run: python gen_signals.py
"""
import json, os, sys, time, traceback, signal
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

# Market + news only. Social analyst hangs on crypto (StockTwits 403, Reddit slow).
ANALYSTS = ("market", "news")

# Per-ticker timeout in seconds
TICKER_TIMEOUT = 300  # 5 minutes max per ticker


class TickerTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TickerTimeout("Ticker analysis timed out")


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


def push_signals(signals):
    """POST signals to strategy-lab so they appear immediately."""
    push_url = os.environ.get("PUSH_SIGNALS_URL", "https://lab.wx8.org/api/ai-signals")
    try:
        import urllib.request
        req = urllib.request.Request(
            push_url,
            data=json.dumps(signals).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"  Push failed: {e}")
        return False


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
    print(f"  TRADE JOURNAL: {yahoo_ticker} {old_rating} -> {new_rating}")


def main():
    # Gateway providers that route per session (e.g. opencode zen) need a
    # unique session header per run. Set it before any LLM client is created.
    if not os.environ.get("TRADINGAGENTS_OPENAI_EXTRA_HEADERS"):
        import uuid
        os.environ["TRADINGAGENTS_OPENAI_EXTRA_HEADERS"] = json.dumps(
            {"x-opencode-session": str(uuid.uuid4())}
        )

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"=== Signal Generation ===")
    print(f"Date: {date}")
    print(f"Analysts: {ANALYSTS}")
    print(f"Tickers: {list(TICKERS.keys())}")
    print(f"Timeout per ticker: {TICKER_TIMEOUT}s")
    print(f"Output: {OUT}")
    sys.stdout.flush()

    # Load existing signals (to detect changes)
    signals = load_json(OUT)
    if not isinstance(signals, dict):
        signals = {}

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1       # Fast: 1 round
    config["max_risk_discuss_rounds"] = 1

    print(f"Provider: {config.get('llm_provider')}  models: "
          f"{config.get('deep_think_llm')}/{config.get('quick_think_llm')}")
    sys.stdout.flush()

    # Fail fast on auth/billing problems instead of burning 6 tickers on them.
    try:
        from tradingagents.llm_clients.factory import create_tier_client
        probe = create_tier_client(config, "quick")
        probe.get_llm().invoke("Reply with OK")
        print("LLM preflight: OK")
    except Exception as e:
        print(f"LLM preflight FAILED: {str(e)[:300]}")
        print("Aborting: no signals generated this run.")
        sys.stdout.flush()
        sys.exit(2)
    sys.stdout.flush()

    success_count = 0
    fail_count = 0

    for i, (binance_sym, yahoo_ticker) in enumerate(TICKERS.items()):
        # Rate limit: wait 60s between tickers (skip before first)
        if i > 0:
            print(f"\n  Waiting 60s before next ticker (rate limit)...")
            sys.stdout.flush()
            time.sleep(60)

        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(TICKERS)}] {yahoo_ticker} ({binance_sym})")
        print(f"{'='*50}")
        sys.stdout.flush()
        start = time.time()

        try:
            # Set alarm for per-ticker timeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TICKER_TIMEOUT)

            ta = TradingAgentsGraph(selected_analysts=ANALYSTS, config=config)
            state, decision = ta.propagate(yahoo_ticker, date, "crypto")

            # Cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            elapsed = time.time() - start

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

            # Save after each ticker
            save_json(OUT, signals)

            # Push to strategy-lab
            pushed = push_signals(signals)
            push_str = "pushed" if pushed else "local only"
            print(f"  -> {rating} ({elapsed:.0f}s, {push_str})")
            sys.stdout.flush()

            success_count += 1

        except TickerTimeout:
            signal.alarm(0)
            elapsed = time.time() - start
            print(f"  -> TIMEOUT after {elapsed:.0f}s")
            sys.stdout.flush()
            fail_count += 1

        except Exception as e:
            signal.alarm(0)
            elapsed = time.time() - start
            print(f"  -> FAILED after {elapsed:.0f}s: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            fail_count += 1

    print(f"\n{'='*50}")
    print(f"DONE: {success_count}/{len(TICKERS)} succeeded, {fail_count} failed")
    print(f"Signals file: {OUT}")
    sys.stdout.flush()

    if fail_count > 0 and success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
