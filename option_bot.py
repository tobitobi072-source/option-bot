"""Automated big-move option bot (Lab 7 "ALL signals" version). Up to 5 trades a day, only on clear signals.

It starts by itself each weekday morning and watches SPY and QQQ minute by minute, 8:30am-2:00pm Central.
CLEAR SIGNALS (same rules as the Lab 7 test, on 5-minute bars):
  ORB+trend    price breaks the first-15-minute range in the direction of the daily trend
  ORB          price breaks the first-15-minute range (either way)
  BURST        a 5-minute bar 2.5x bigger than normal on 2x volume, beyond VWAP
  GAP-REVERSE  the day opens 0.7%+ gapped one way, then breaks the 15-minute range the other way
  VWAP-RECLAIM price stretches 1%+ from the open, then crosses back through VWAP
On a signal: buy a cheap same-day CALL (up) or PUT (down), 2 USD out of the money, up to MAX_BET of the account.
Each trade exits on its own: SELL at 2x (a standing order), or CUT at -50%, or at 2:00pm at the latest.
Then it's free to take the next clear signal. One trade at a time, at most MAX_TRADES a day. No signal = no trade.

Guardrails: stop for the day after DAILY_LOSS_STOP losing trades; stop for good below KILL_LEVEL or
below LOCK_PCT of the account's best level. Paper trading unless you deliberately switch to live.
"""
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ------------------------------------------------------------- YOUR RULES --
SYMBOLS = ["SPY", "QQQ"]
SIGNALS = ["ORB+trend", "ORB", "BURST", "GAP-REVERSE", "VWAP-RECLAIM"]
EXPIRY = "0DTE"              # same-day options (cheapest). "1DTE" = next-day options
STRIKE_OFFSET = 2            # USD out of the money (2 held up better than 5 on same-day options in Lab 7)
TARGET = 2.0                 # sell when the option is worth 2x what was paid
STOP = 0.50                  # cut the trade if it loses 50%
MAX_TRADES = 5               # most trades in one day
DAILY_LOSS_STOP = 2          # stop for the day after this many losing trades
MAX_BET = 0.50               # most of the account in one trade
START_EQUITY = 100.0
KILL_LEVEL = 50.0            # stop for good below this (USD)
LOCK_PCT = 0.40              # stop for good below 40% of the account's best level
MAX_LATE_MIN = 6             # don't chase a signal seen more than 6 minutes after it fired
LIVE_CONFIRM_PHRASE = "I understand real money is at risk"

CT = ZoneInfo("America/Chicago")
OPEN, OR_END, T_LAST, T_EXIT, T_STOP = dtime(8, 30), dtime(8, 45), dtime(13, 45), dtime(13, 58), dtime(14, 5)


def env(name, default, cast=float):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default


# ------------------------------------------------------------- signals ----
def trend_score(c):
    s20, s50, s200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    parts = [np.sign(c - s200), np.sign(s50 - s200), np.sign(c - s20), np.sign(macd - sig), np.sign(c / c.shift(20) - 1)]
    return sum(parts).where(s200.notna())


def trend_dir(daily_close):
    v = float(trend_score(daily_close).iloc[-1])
    return int(np.sign(v)) if abs(v) >= 3 else 0


def day_signals(g, trend, prev_close, signals=SIGNALS):
    """g: today's COMPLETED 5-minute bars (index = bar start, Central). Identical rules to Lab 7.
    Returns [(time to enter = next bar start, direction, price, signal name)], one per bar+direction."""
    if len(g) < 4 or g.index[0].time() != OPEN:
        return []
    c, h, l, v, o = (g[k].values.astype(float) for k in ("close", "high", "low", "volume", "open"))
    vwap = np.cumsum((h + l + c) / 3 * v) / np.maximum(np.cumsum(v), 1)
    absret = np.abs(np.diff(c, prepend=o[0]) / c)
    avg_abs = pd.Series(absret).rolling(20, min_periods=5).mean().shift(1).values
    avg_vol = pd.Series(v).rolling(20, min_periods=5).mean().shift(1).values
    or_hi, or_lo = h[:3].max(), l[:3].min()
    gap = o[0] / prev_close - 1 if prev_close else 0
    stretched_dn = stretched_up = False
    out, seen = [], set()
    for i in range(3, len(g)):
        t = g.index[i].time()
        if c[i] < o[0] * 0.99:
            stretched_dn = True
        if c[i] > o[0] * 1.01:
            stretched_up = True
        if t < OR_END or t >= T_LAST:
            continue
        hits = []
        up_break, dn_break = c[i] > or_hi >= c[i - 1], c[i] < or_lo <= c[i - 1]
        if up_break or dn_break:
            d = 1 if up_break else -1
            hits.append(("ORB", d))
            if d == trend:
                hits.append(("ORB+trend", d))
            if (gap <= -0.007 and up_break) or (gap >= 0.007 and dn_break):
                hits.append(("GAP-REVERSE", d))
        if not np.isnan(avg_abs[i]) and absret[i] > 2.5 * avg_abs[i] and v[i] > 2 * avg_vol[i]:
            d = 1 if c[i] > c[i - 1] else -1
            if (d == 1 and c[i] > vwap[i]) or (d == -1 and c[i] < vwap[i]):
                hits.append(("BURST", d))
        if stretched_dn and c[i] > vwap[i] >= c[i - 1]:
            hits.append(("VWAP-RECLAIM", 1)); stretched_dn = False
        if stretched_up and c[i] < vwap[i] <= c[i - 1]:
            hits.append(("VWAP-RECLAIM", -1)); stretched_up = False
        for name, d in hits:
            if name in signals and (i, d) not in seen:
                seen.add((i, d))
                out.append((g.index[i] + pd.Timedelta(minutes=5), d, float(c[i]), name))
    return out


# --------------------------------------------------------------- alerts ---
def github_api(method, path, body=None):
    """Talk to this repository on GitHub (only when running in GitHub Actions)."""
    import json
    import urllib.request
    token, repo = os.getenv("GITHUB_TOKEN"), os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        return None
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}{path}", method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read() or b"null")


def milestone_title(m):
    return f"Account doubled: {2 ** m}x your start"


# ------------------------------------------------------------------ bot ---
class Bot:
    def __init__(self):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.historical.option import OptionHistoricalDataClient
        key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")
        self.live = os.getenv("TRADING_MODE", "paper").lower() == "live"
        if self.live and os.getenv("LIVE_CONFIRM") != LIVE_CONFIRM_PHRASE:
            sys.exit("Live mode blocked: LIVE_CONFIRM is not set to the exact phrase.")
        self.start_eq = env("START_EQUITY", START_EQUITY)
        self.kill = env("KILL_LEVEL", KILL_LEVEL)
        self.max_bet = env("MAX_BET", MAX_BET)
        self.max_trades = env("MAX_TRADES", MAX_TRADES, int)
        self.loss_stop = env("DAILY_LOSS_STOP", DAILY_LOSS_STOP, int)
        self.expiry = os.getenv("EXPIRY") or EXPIRY
        self.offset = env("STRIKE_OFFSET", STRIKE_OFFSET, int)
        self.trading = TradingClient(key, secret, paper=not self.live)
        self.stocks = StockHistoricalDataClient(key, secret)
        self.options = OptionHistoricalDataClient(key, secret)
        self.consumed = set()        # signals already acted on (or skipped) today
        self.daily = {}              # per symbol: (trend, previous close)
        self.halted = False
        self.last_held = set()
        self.alerted = set()         # doubling alerts already sent (2x, 4x, 8x ...)
        try:
            for i in github_api("GET", "/issues?state=all&per_page=100") or []:
                self.alerted.add(i["title"])
        except Exception as e:
            print("Couldn't read past alerts:", e)
        self.log_lines = []

    def log(self, msg):
        line = f"{datetime.now(CT):%I:%M %p} {msg}"
        print(line, flush=True)
        self.log_lines.append(line)

    # ---- data
    def load_daily(self, today):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed, Adjustment
        now = datetime.now(CT)
        for sym in SYMBOLS:
            d = self.stocks.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Day, start=now - timedelta(days=420),
                end=now - timedelta(minutes=16), feed=DataFeed.SIP, adjustment=Adjustment.ALL)).df.xs(sym, level="symbol")
            d.index = pd.to_datetime(d.index.tz_convert("America/New_York").date)
            d = d[d.index < pd.Timestamp(today)]
            self.daily[sym] = (trend_dir(d["close"]), float(d["close"].iloc[-1]))
            word = {1: "UP", -1: "DOWN", 0: "unclear"}[self.daily[sym][0]]
            self.log(f"{sym} daily trend: {word}")

    def bars_today(self, sym, now):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
        df = self.stocks.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=sym, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=datetime.combine(now.date(), OPEN, CT), end=now, feed=DataFeed.IEX)).df
        if df.empty:
            return df
        g = df.xs(sym, level="symbol")
        g.index = g.index.tz_convert(CT)
        return g[(g.index.time >= OPEN) & (g.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now))]

    def quote(self, occ):
        from alpaca.data.requests import OptionLatestQuoteRequest
        from alpaca.data.enums import OptionsFeed
        q = self.options.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=occ,
                                                                          feed=OptionsFeed.INDICATIVE))[occ]
        return float(q.bid_price or 0), float(q.ask_price or 0)

    # ---- one pass (called about once a minute)
    def step(self):
        from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, AssetClass
        now = datetime.now(CT)
        today = now.date()
        if self.halted:
            return
        acct = self.trading.get_account()
        equity = float(acct.equity)
        hist = self.trading.get_portfolio_history(GetPortfolioHistoryRequest(period="all", timeframe="1D"))
        since = os.getenv("START_DATE")
        cut = datetime.fromisoformat(since).replace(tzinfo=CT).timestamp() if since else 0
        peak = max([e for t, e in zip(hist.timestamp or [], hist.equity or []) if e and t >= cut] + [equity, self.start_eq])
        held = {p.symbol: p for p in self.trading.get_all_positions() if p.asset_class == AssetClass.US_OPTION}
        orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=200,
                                                          after=datetime.combine(today, dtime(0, 0), CT)))
        orders = [o for o in orders if len(o.symbol) > 10]
        for sym in self.last_held - set(held):
            hit = [o for o in orders if o.symbol == sym and o.side == OrderSide.SELL and o.filled_at
                   and o.limit_price is not None]
            if hit:
                self.log(f"SOLD {sym} at the {TARGET:.0f}x target ({float(hit[0].filled_avg_price):.2f}) ✓")
        self.last_held = set(held)
        open_orders = [o for o in orders if o.status.value in ("new", "accepted", "pending_new", "partially_filled")]

        self.check_doubling(equity)

        # guardrail: stop for good
        floor = max(self.kill, peak * LOCK_PCT)
        if equity <= floor:
            self.close_all(held, open_orders)
            self.log(f"GUARDRAIL STOP: account {equity:,.2f} is at or below {floor:,.2f} USD. "
                     "Everything closed; the bot stays out until you review it.")
            self.halted = True
            return

        # stale entry orders: cancel if not filled within 3 minutes
        for o in open_orders:
            if o.side == OrderSide.BUY and datetime.now(timezone.utc) - o.submitted_at > timedelta(minutes=3):
                self.trading.cancel_order_by_id(o.id)
                self.log(f"Entry for {o.symbol} didn't fill in time: cancelled.")

        # manage what we hold
        for sym, p in held.items():
            buy = next((o for o in reversed(sorted(orders, key=lambda o: o.submitted_at))
                        if o.symbol == sym and o.side == OrderSide.BUY and o.filled_at), None)
            entry = float(p.avg_entry_price)
            sells = [o for o in open_orders if o.symbol == sym and o.side == OrderSide.SELL]
            bid, _ = self.quote(sym)
            reason = None
            if buy is None:
                reason = "left over from an earlier day"
            elif now.time() >= T_EXIT:
                reason = "2:00pm exit"
            elif 0 < bid <= entry * (1 - STOP):
                reason = f"stop: down {(1 - bid / entry) * 100:.0f}%"
            if reason:
                for o in sells:
                    self.trading.cancel_order_by_id(o.id)
                if sells:
                    time.sleep(2)
                self.trading.close_position(sym)
                self.log(f"SOLD {p.qty} x {sym} ({reason}); about {(bid / entry - 1) * 100:+.0f}%")
            elif not sells:
                self.place_target(sym, int(float(p.qty)), entry)
        if held or any(o.side == OrderSide.BUY for o in open_orders):
            return                                   # one trade at a time

        # today's tally
        buys = [o for o in orders if o.side == OrderSide.BUY and o.filled_at]
        losses, by_sym = 0, {}
        for o in sorted([o for o in orders if o.filled_at], key=lambda o: o.filled_at):
            if o.side == OrderSide.BUY:
                by_sym[o.symbol] = float(o.filled_avg_price)
            elif o.symbol in by_sym:
                losses += float(o.filled_avg_price) < by_sym.pop(o.symbol)
        if now.time() >= T_LAST or len(buys) >= self.max_trades or losses >= self.loss_stop:
            return
        last_exit = max([o.filled_at for o in orders if o.filled_at and o.side == OrderSide.SELL],
                        default=datetime.combine(today, OPEN, CT))

        # look for the next clear signal
        found = []
        for k, sym in enumerate(SYMBOLS):
            g = self.bars_today(sym, now)
            trend, prev = self.daily[sym]
            for t_in, d, S, name in day_signals(g, trend, prev):
                key = (sym, t_in, d)
                if key not in self.consumed and t_in > pd.Timestamp(last_exit):
                    found.append((t_in, k, sym, d, S, name))
        fresh = []
        for f in found:
            if (pd.Timestamp(now) - f[0]).total_seconds() / 60 > MAX_LATE_MIN:
                self.consumed.add((f[2], f[0], f[3]))        # too old to chase: pass over quietly
            else:
                fresh.append(f)
        if not fresh:
            return
        t_in, _, sym, d, S, name = min(fresh)
        for f in fresh:                              # same-time signals on the other symbol are passed over
            if f[0] <= t_in:
                self.consumed.add((f[2], f[0], f[3]))
        side = "CALL" if d == 1 else "PUT"
        self.log(f"SIGNAL {name}: {sym} {'UP' if d == 1 else 'DOWN'} at {t_in:%I:%M %p} → {side}  "
                 f"(trade #{len(buys) + 1} today)")
        self.enter(sym, d, S, equity, acct, today)

    def enter(self, sym, d, S, equity, acct, today):
        from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest, GetCalendarRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, ContractType
        cal = self.trading.get_calendar(GetCalendarRequest(start=today, end=today + timedelta(days=10)))
        exp = today if self.expiry == "0DTE" else next(c.date for c in cal if c.date > today)
        k = round(S) + (self.offset if d == 1 else -self.offset)
        ct = ContractType.CALL if d == 1 else ContractType.PUT
        res = self.trading.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[sym], expiration_date=exp, type=ct,
            strike_price_gte=str(k - 1), strike_price_lte=str(k + 1)))
        by_k = {float(c.strike_price): c.symbol for c in res.option_contracts or []}
        occ = next((by_k[x] for x in (k, k + 1, k - 1) if x in by_k), None)
        if not occ:
            self.log(f"No {exp} contract near strike {k}; skipped.")
            return
        _, ask = self.quote(occ)
        if ask <= 0.02:
            self.log(f"{occ}: no usable price; skipped.")
            return
        limit = round(ask * 1.03 + 0.01, 2)
        cash = float(acct.options_buying_power or acct.cash)
        n = int(min(self.max_bet * equity, cash) // (limit * 100))
        if n < 1:
            self.log(f"{occ} costs {limit * 100:,.0f} USD — more than {self.max_bet:.0%} of the account. Skipped.")
            return
        self.trading.submit_order(LimitOrderRequest(symbol=occ, qty=n, side=OrderSide.BUY, limit_price=limit,
                                                    time_in_force=TimeInForce.DAY))
        self.log(f"BUY {n} x {occ} at up to {limit:.2f} ({n * limit * 100:,.2f} USD). "
                 f"Target {TARGET:.0f}x, stop -{STOP:.0%}, out by 2pm.")

    def check_doubling(self, equity):
        m = 1
        while equity >= self.start_eq * 2 ** m:
            title = milestone_title(m)
            if title not in self.alerted:
                self.alerted.add(title)
                self.log(f"ALERT: {title}!")
                try:
                    github_api("POST", "/issues", {
                        "title": title,
                        "body": f"Your option bot's account has reached **{2 ** m}x** where it started "
                                f"(as of {datetime.now(CT):%a %b %d, %I:%M %p} Central).\n\n"
                                "The bot keeps trading as normal. Check the exact balance in your Alpaca app."})
                except Exception as e:
                    self.log(f"Couldn't send the alert: {e}")
            m += 1

    def place_target(self, occ, qty, entry):
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        tp = round(entry * TARGET, 2)
        self.trading.submit_order(LimitOrderRequest(symbol=occ, qty=qty, side=OrderSide.SELL, limit_price=tp,
                                                    time_in_force=TimeInForce.DAY))
        self.log(f"Filled {qty} x {occ} at {entry:.2f}. Standing sell order at {tp:.2f} ({TARGET:.0f}x).")

    def close_all(self, held, open_orders):
        for o in open_orders:
            self.trading.cancel_order_by_id(o.id)
        time.sleep(2 if open_orders else 0)
        for sym in held:
            self.trading.close_position(sym)

    def connection_check(self):
        """Proves the keys work: reads the account and today's market hours without trading."""
        try:
            acct = self.trading.get_account()
            clock = self.trading.get_clock()
            eq = float(acct.equity)
            self.log(f"Connection check OK: Alpaca {'LIVE' if self.live else 'paper'} account reached, "
                     f"{eq / self.start_eq * 100 - 100:+.0f}% vs start, options level {acct.options_trading_level}. "
                     f"Next market open: {clock.next_open.astimezone(CT):%a %b %d %I:%M %p} Central.")
        except Exception as e:
            self.log(f"CONNECTION CHECK FAILED: {e}. Check the two Alpaca key secrets in GitHub.")
            raise SystemExit(1)

    # ---- the day
    def run_day(self):
        now = datetime.now(CT)
        self.log(f"Option bot starting ({'LIVE' if self.live else 'paper'}).")
        if now.time() < dtime(8, 0) or now.time() >= T_STOP:
            self.log("Outside today's window (8:00am-2:05pm Central). Nothing to trade right now.")
            self.connection_check()
            return
        clock = self.trading.get_clock()
        if not clock.is_open and clock.next_open.astimezone(CT).date() != now.date():
            self.log("Market closed today.")
            return
        self.load_daily(now.date())
        while datetime.now(CT).time() < T_STOP:
            if datetime.now(CT).time() >= OPEN:
                try:
                    self.step()
                except Exception:
                    self.log("Error this pass (will retry next minute):\n" + traceback.format_exc()[-600:])
            time.sleep(60)
        try:
            self.step()                              # final pass: make sure everything is closed
        finally:
            acct = self.trading.get_account()
            eq = float(acct.equity)
            self.log(f"Day over. Account {eq / self.start_eq * 100 - 100:+.0f}% vs start.")
            path = os.getenv("GITHUB_STEP_SUMMARY")
            if path:
                open(path, "a").write("## Option bot day log\n```\n" + "\n".join(self.log_lines) + "\n```\n")


if __name__ == "__main__":
    Bot().run_day()
