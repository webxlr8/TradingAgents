#!/usr/bin/env python3
"""Generate LLM trading signals for the tracked crypto pairs.

Writes signals.json to TRADINGAGENTS_HOME so the HTTP signals-server can
serve it to strategy-lab's llm_agent strategy, and pushes each finished
signal straight to strategy-lab so the dashboard updates live.

Also maintains a trades.json journal that logs signal CHANGES (e.g. BTC
went from Hold to Buy) for post-trade analysis.

Design notes:
  - market + news analysts only: the social analyst hangs on crypto tickers
  - 1 debate round: keeps latency and token cost down
  - tickers run in a small thread pool so a 4-hourly cycle finishes well
    inside the task timeout even when the model is slow
  - one bad ticker never blocks the others; the whole run has a wall-clock
    budget and reports what it managed to produce

Run: python gen_signals.py
"""
import json, os, sys, threading, time, traceback, uuid
from concurrent.futures import ThreadPoolExecutor
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

# Parallel workers and an overall wall-clock budget for the whole run.
WORKERS = int(os.environ.get("SIGNALS_WORKERS", "3"))
TOTAL_TIMEOUT = int(os.environ.get("SIGNALS_TOTAL_TIMEOUT", "2700"))

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg)
        sys.stdout.flush()


def load_json(path):
    """Load a JSON file, returning None on error."""
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
    """POST signals to strategy-lab so the dashboard updates immediately."""
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
        log(f"  push failed: {str(e)[:120]}")
        return False


def log_trade(binance_sym, yahoo_ticker, old_rating, new_rating, decision, date, entry_price=None):
    """Append a signal change to the trades journal."""
    trades = load_json(TRADES_LOG)
    if not isinstance(trades, list):
        trades = []

    trades.append({
        "timestamp": time.time(),
        "datetime": datetime.utcnow().isoformat() + "Z",
        "date": date,
        "binance_symbol": binance_sym,
        "yahoo_ticker": yahoo_ticker,
        "old_signal": old_rating,
        "new_signal": new_rating,
        "decision_summary": (decision or "")[:500],
        "entry_price": entry_price,
    })
    save_json(TRADES_LOG, trades)
    log(f"  TRADE JOURNAL: {yahoo_ticker} {old_rating} -> {new_rating}")


def rating_from(decision):
    """Map the LLM's free-text decision onto one of four ratings."""
    rating = "Hold"
    dec_lower = (decision or "").lower()
    for word in ["underweight", "sell", "overweight", "buy"]:
        if word in dec_lower:
            rating = word.capitalize()
            break
    return rating


def spot_price(binance_symbol):
    """Latest spot price from Binance (public endpoint, no key needed)."""
    try:
        import urllib.request
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
        with urllib.request.urlopen(url, timeout=8) as r:
            return float(json.loads(r.read())["price"])
    except Exception:
        return None


def build_report(state, decision):
    """Pick the most decision-relevant text out of the agent graph state."""
    if not isinstance(state, dict):
        return (decision or "")[:1500]

    parts = []
    debate = state.get("investment_debate_state") or {}
    if isinstance(debate, dict) and debate.get("judge_decision"):
        parts.append(str(debate["judge_decision"]).strip())
    if state.get("trader_investment_plan"):
        parts.append(str(state["trader_investment_plan"]).strip())
    risk = state.get("risk_debate_state") or {}
    if isinstance(risk, dict) and risk.get("judge_decision"):
        parts.append(str(risk["judge_decision"]).strip())
    if state.get("final_trade_decision"):
        parts.append(str(state["final_trade_decision"]).strip())
    if not parts:
        parts.append((decision or "").strip())
    text = "\n\n".join(p for p in parts if p)
    return text[:1500]


def main():
    # Gateways that route per session (e.g. opencode zen) require a unique
    # session header per run. Must be set before any LLM client is built.
    if not os.environ.get("TRADINGAGENTS_OPENAI_EXTRA_HEADERS"):
        os.environ["TRADINGAGENTS_OPENAI_EXTRA_HEADERS"] = json.dumps(
            {"x-opencode-session": str(uuid.uuid4())}
        )

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    log("=== Signal Generation ===")
    log(f"Date: {date}")
    log(f"Analysts: {ANALYSTS}")
    log(f"Tickers: {list(TICKERS.keys())}")
    log(f"Workers: {WORKERS}  total budget: {TOTAL_TIMEOUT}s")
    log(f"Output: {OUT}")

    signals = load_json(OUT)
    if not isinstance(signals, dict):
        signals = {}

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1

    log(f"Provider: {config.get('llm_provider')}  models: "
        f"{config.get('deep_think_llm')}/{config.get('quick_think_llm')}")

    # Fail fast on auth/billing problems instead of burning every ticker on them.
    try:
        from tradingagents.llm_clients.factory import create_tier_client
        probe = create_tier_client(config, "quick")
        probe.get_llm().invoke("Reply with OK")
        log("LLM preflight: OK")
    except Exception as e:
        log(f"LLM preflight FAILED: {str(e)[:300]}")
        log("Aborting: no signals generated this run.")
        sys.exit(2)

    state_lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}

    # A shared gateway will sometimes answer 429/overloaded for minutes at a
    # time. Retrying the same model then just burns the run budget, so after
    # the retries we fall back to a different model, then to the next one.
    attempts = int(os.environ.get("SIGNALS_ATTEMPTS", "3"))
    fallbacks = [m.strip() for m in
                 os.environ.get("SIGNALS_FALLBACK_MODELS", "glm-5.3,deepseek-v4-pro").split(",")
                 if m.strip()]
    model_variants = [None] + fallbacks

    def analyze(yahoo_ticker):
        import random
        last_error = None
        for model in model_variants:
            cfg = config if model is None else {**config, "deep_think_llm": model}
            label = model or config.get("deep_think_llm")
            for attempt in range(1, attempts + 1):
                try:
                    ta = TradingAgentsGraph(selected_analysts=ANALYSTS, config=cfg)
                    state, decision = ta.propagate(yahoo_ticker, date, "crypto")
                    return decision, state
                except Exception as e:
                    last_error = e
                    log(f"  {yahoo_ticker} [{label}] attempt {attempt}/{attempts} failed: {str(e)[:140]}")
                    if attempt < attempts:
                        # Back off longer and jittered: the gateway overloads
                        # in bursts, so hammering it on a fixed short delay
                        # makes it worse.
                        time.sleep(30 * attempt + random.uniform(0, 15))
        raise last_error

    def record(binance_sym, yahoo_ticker, decision, state, elapsed):
        rating = rating_from(decision)
        price = spot_price(binance_sym)
        with state_lock:
            old_entry = signals.get(binance_sym)
            old_rating = old_entry.get("rating", "None") if old_entry else "None"
            if old_rating != rating:
                log_trade(binance_sym, yahoo_ticker, old_rating, rating, decision, date, price)

            entry = {
                "ticker": yahoo_ticker,
                "rating": rating,
                "decision": (decision or "")[:500],
                "timestamp": time.time(),
                "date": date,
                "analysts": list(ANALYSTS),
                "debate_rounds": config["max_debate_rounds"],
                "entry_price": price,
                "report": build_report(state, decision),
            }
            signals[binance_sym] = entry
            signals[yahoo_ticker] = entry
            save_json(OUT, signals)
            pushed = push_signals(signals)
            counters["ok"] += 1

        log(f"  -> {yahoo_ticker} {rating} ({elapsed:.0f}s, "
            f"{'pushed' if pushed else 'local only'})")

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    futures = {pool.submit(analyze, y): (b, y) for b, y in TICKERS.items()}
    start = time.time()
    deadline = start + TOTAL_TIMEOUT
    pending = dict(futures)

    while pending and time.time() < deadline:
        for fut in list(pending):
            if not fut.done():
                continue
            binance_sym, yahoo_ticker = pending.pop(fut)
            elapsed = time.time() - start
            try:
                decision, state = fut.result()
            except Exception as e:
                with state_lock:
                    counters["fail"] += 1
                log(f"  -> {yahoo_ticker} FAILED after {elapsed:.0f}s: {str(e)[:220]}")
                traceback.print_exc()
                continue
            record(binance_sym, yahoo_ticker, decision, state, elapsed)
        if pending:
            time.sleep(5)

    for fut, (binance_sym, yahoo_ticker) in pending.items():
        with state_lock:
            counters["fail"] += 1
        log(f"  -> {yahoo_ticker} ABANDONED (run budget {TOTAL_TIMEOUT}s exhausted)")

    log("=" * 50)
    log(f"DONE: {counters['ok']}/{len(TICKERS)} succeeded, {counters['fail']} failed")
    log(f"Signals file: {OUT}")

    if counters["ok"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
