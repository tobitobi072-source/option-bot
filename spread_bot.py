"""Late-day credit-spread bot (Lab 11b winner). Fully automatic, SPY + QQQ, up to 2 trades a day.

Each trading day (all times Central):
  9:00   reads the direction of the first 30 minutes (8:30-9:00) for SPY and QQQ
  1:30pm reads the direction of the day so far. If both directions AGREE and the day has moved at least 0.25%:
         up day   -> SELL a put spread  : sell the put $2 below the price, buy the put $4 below ($2 wide)
         down day -> SELL a call spread : sell the call $2 above the price, buy the call $4 above
         Same-day expiry. The spread is left to expire at 3pm (so it is not a day trade).
  2:25pm SAFETY: if the price has crossed the sold strike, buy the spread back so the account is never
         handed shares (Alpaca stops SPY/QQQ same-day orders at 2:30pm Central).
Size: RISK of the account per trade (the most a spread can lose). A small account may take ONE spread if it
risks no more than MIN_ONE_MAX of the account. Guardrail: stop for good below KILL_PCT of the start; the stop is
remembered as a GitHub issue and the bot stays out until you close that issue.
Paper trading unless you deliberately switch to live.
"""
import json
import os
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np

# ------------------------------------------------------------- YOUR RULES --
SYMBOLS = ["SPY", "QQQ"]
ENTRY = dtime(13, 30)          # sell time (Central)
ENTRY_LAST = dtime(13, 45)     # give up on unfilled orders after this
SAFETY = dtime(14, 25)         # buy back any spread whose sold strike is in the money
OFFSET = 2                     # sold strike this many dollars beyond the price
WIDTH = 2                      # spread width in dollars
MIN_MOVE = 0.0025              # day must have moved at least 0.25%
MIN_CREDIT = 0.05              # skip if the spread pays less than this per share
RISK = 0.20                    # most a spread may lose, as a share of the account
MIN_ONE_MAX = 0.25             # a small account may still take ONE spread if it risks no more than this
START_EQUITY = 1000.0
KILL_PCT = 0.50                # stop for good below 50% of the start
LIVE_CONFIRM_PHRASE = "I understand real money is at risk"

CT = ZoneInfo("America/Chicago")
OPEN, T30, T_STOP = dtime(8, 30), dtime(9, 0), dtime(15, 5)


def env(name, default, cast=float):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default


def occ(sym, day, cp, strike):
    return f"{sym}{day:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


def plan_spread(S, direction, offset=OFFSET, width=WIDTH):
    """Up day -> put credit spread below the price; down day -> call credit spread above. Returns (cp, sold, bought)."""
    k0 = round(S)
    if direction > 0:
        return "P", k0 - offset, k0 - offset - width
    return "C", k0 + offset, k0 + offset + width


def signal(open_px, px_930, px_now, min_move=MIN_MOVE):
    """+1 / -1 when the first 30 minutes and the day so far agree and the day moved at least min_move; else 0."""
    r30, rnow = px_930 / open_px - 1, px_now / open_px - 1
    if r30 == 0 or rnow == 0 or np.sign(r30) != np.sign(rnow) or abs(rnow) < min_move:
        return 0
    return 1 if rnow > 0 else -1


def contracts_for(max_loss_per_spread, equity, risk=RISK, min_one_max=MIN_ONE_MAX):
    unit = max_loss_per_spread * 100
    if unit <= 0:
        return 0
    n = int((risk * equity) // unit)
    if n == 0 and unit <= min_one_max * equity:
        n = 1
    return n


def open_legs(s_occ, l_occ):
    from alpaca.trading.requests import OptionLegRequest
    from alpaca.trading.enums import OrderSide, PositionIntent
    return [OptionLegRequest(symbol=s_occ, side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=l_occ, side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN)]


def close_legs(s_occ, l_occ):
    from alpaca.trading.requests import OptionLegRequest
    from alpaca.trading.enums import OrderSide, PositionIntent
    return [OptionLegRequest(symbol=s_occ, side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_CLOSE),
            OptionLegRequest(symbol=l_occ, side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_CLOSE)]


def github_api(method, path, body=None):
    token, repo = os.getenv("GITHUB_TOKEN"), os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        return None
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}{path}", method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read() or b"null")


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
        self.name = os.getenv("BOT_NAME") or "Spread bot"
        self.start_eq = env("START_EQUITY", START_EQUITY)
        self.risk = env("RISK", RISK)
        self.kill_pct = env("KILL_PCT", KILL_PCT)
        self.min_move = env("MIN_MOVE", MIN_MOVE)
        self.offset = env("OFFSET", OFFSET, int)
        self.width = env("WIDTH", WIDTH, int)
        self.trading = TradingClient(key, secret, paper=not self.live)
        self.stocks = StockHistoricalDataClient(key, secret)
        self.options = OptionHistoricalDataClient(key, secret)
        self.log_lines = []
        self.first30 = {}          # sym -> (open, price at 9:00)
        self.done = set()          # symbols handled today (traded or skipped)
        self.pending = {}          # sym -> dict(order_id, tries, spread info)
        self.spreads = {}          # sym -> spread info (filled)
        self.safety_done = False

    # ---------------------------------------------------------------- utils
    def log(self, msg):
        line = f"{datetime.now(CT):%I:%M %p} {msg}"
        print(line, flush=True)
        self.log_lines.append(line)

    def stop_title(self):
        return f"{self.name}: guardrail stop"

    def stopped_for_good(self):
        try:
            for i in github_api("GET", "/issues?state=open&per_page=100") or []:
                if i["title"] == self.stop_title():
                    return True
        except Exception as e:
            self.log(f"Couldn't check for a guardrail stop: {e}")
        return False

    def raise_stop(self, equity):
        self.log(f"GUARDRAIL STOP: account {equity:,.2f} is below {self.kill_pct:.0%} of the start. "
                 "No new trades until you close the GitHub issue.")
        try:
            github_api("POST", "/issues", {"title": self.stop_title(), "body":
                       f"The account fell to **{equity:,.2f} USD** (start {self.start_eq:,.0f}). The bot will not open new "
                       "trades while this issue is open. Close this issue to let it trade again."})
        except Exception as e:
            self.log(f"Couldn't open the guardrail issue: {e}")

    def bars(self, sym, start, end, minutes=5):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
        df = self.stocks.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                                                         start=start, end=end, feed=DataFeed.IEX)).df
        if df.empty:
            return df
        g = df.xs(sym, level="symbol")
        g.index = g.index.tz_convert(CT)
        return g

    def price_now(self, sym):
        from alpaca.data.requests import StockLatestTradeRequest
        from alpaca.data.enums import DataFeed
        t = self.stocks.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym, feed=DataFeed.IEX))[sym]
        return float(t.price)

    def quote(self, sym_occ):
        from alpaca.data.requests import OptionLatestQuoteRequest
        from alpaca.data.enums import OptionsFeed
        q = self.options.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=sym_occ,
                                                                          feed=OptionsFeed.INDICATIVE)).get(sym_occ)
        if q is None:
            return None
        return float(q.bid_price or 0), float(q.ask_price or 0)

    # ---------------------------------------------------------------- steps
    def read_first30(self, today):
        for sym in SYMBOLS:
            if sym in self.first30:
                continue
            g = self.bars(sym, datetime.combine(today, OPEN, CT), datetime.combine(today, T30, CT) + timedelta(minutes=1))
            g = g[(g.index.time >= OPEN) & (g.index.time < T30)] if len(g) else g
            if len(g) >= 5 and g.index[0].time() == OPEN:
                o, c = float(g["open"].iloc[0]), float(g["close"].iloc[-1])
                self.first30[sym] = (o, c)
                self.log(f"{sym} first 30 minutes: {c / o * 100 - 100:+.2f}% "
                         f"({'up' if c > o else 'down' if c < o else 'flat'}).")

    def try_enter(self, sym, today, equity):
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderClass, TimeInForce
        if sym not in self.first30:
            self.log(f"{sym}: no first-30-minute reading today. Skipped.")
            self.done.add(sym)
            return
        o, p930 = self.first30[sym]
        S = self.price_now(sym)
        d = signal(o, p930, S, self.min_move)
        self.done.add(sym)
        if d == 0:
            self.log(f"{sym}: no clear direction at 1:30pm (first 30 min {p930 / o * 100 - 100:+.2f}%, "
                     f"day {S / o * 100 - 100:+.2f}%). No trade.")
            return
        cp, ks, kl = plan_spread(S, d, self.offset, self.width)
        s_occ, l_occ = occ(sym, today, cp, ks), occ(sym, today, cp, kl)
        qs, ql = self.quote(s_occ), self.quote(l_occ)
        if not qs or not ql or qs[1] <= 0:
            self.log(f"{sym}: no quotes for {s_occ} / {l_occ}. Skipped.")
            return
        credit = round((qs[0] + qs[1]) / 2 - (ql[0] + ql[1]) / 2, 2)
        if credit < MIN_CREDIT:
            self.log(f"{sym}: the spread only pays {credit:.2f}. Skipped.")
            return
        max_loss = self.width - credit
        n = contracts_for(max_loss, equity, self.risk)
        if n < 1:
            self.log(f"{sym}: one spread can lose {max_loss * 100:,.0f} USD — too much for this account. Skipped.")
            return
        side = "PUT" if cp == "P" else "CALL"
        legs = open_legs(s_occ, l_occ)
        order = self.trading.submit_order(LimitOrderRequest(qty=n, order_class=OrderClass.MLEG, legs=legs,
                                                            limit_price=-credit, time_in_force=TimeInForce.DAY))
        info = dict(sym=sym, cp=cp, ks=ks, kl=kl, s_occ=s_occ, l_occ=l_occ, qty=n, credit=credit)
        self.pending[sym] = dict(order_id=order.id, tries=1, placed=time.time(), info=info)
        self.log(f"SIGNAL {sym} {'UP' if d > 0 else 'DOWN'} day ({S / o * 100 - 100:+.2f}%): SELL {n} x {sym} {side} spread "
                 f"{ks}/{kl} for about {credit:.2f} ({n * credit * 100:,.0f} USD in, can lose up to "
                 f"{n * max_loss * 100:,.0f} USD). Expires 3pm today.")

    def manage_pending(self):
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderClass, TimeInForce
        for sym, p in list(self.pending.items()):
            o = self.trading.get_order_by_id(p["order_id"])
            status = str(o.status).split(".")[-1].lower()
            info = p["info"]
            if status == "filled":
                px = abs(float(o.filled_avg_price or info["credit"]))
                info["credit"] = px
                self.spreads[sym] = info
                del self.pending[sym]
                self.log(f"Filled: sold {info['qty']} x {sym} {info['ks']}/{info['kl']} spread for {px:.2f} "
                         f"({info['qty'] * px * 100:,.2f} USD collected).")
                continue
            if status in ("canceled", "cancelled", "expired", "rejected"):
                self.log(f"{sym} order {status}. No trade.")
                del self.pending[sym]
                continue
            late = datetime.now(CT).time() >= ENTRY_LAST
            if time.time() - p["placed"] < 90 and not late:
                continue
            self.trading.cancel_order_by_id(p["order_id"])
            time.sleep(2)
            if late or p["tries"] >= 3 or info["credit"] - 0.01 < MIN_CREDIT:
                self.log(f"{sym}: not filled at an acceptable price. No trade.")
                del self.pending[sym]
                continue
            info["credit"] = round(info["credit"] - 0.01, 2)
            legs = open_legs(info["s_occ"], info["l_occ"])
            new = self.trading.submit_order(LimitOrderRequest(qty=info["qty"], order_class=OrderClass.MLEG, legs=legs,
                                                              limit_price=-info["credit"], time_in_force=TimeInForce.DAY))
            self.pending[sym] = dict(order_id=new.id, tries=p["tries"] + 1, placed=time.time(), info=info)
            self.log(f"{sym}: re-offered at {info['credit']:.2f}.")

    def recover_spreads(self, today):
        """After a restart: rebuild today's open spreads from the account's positions."""
        for pos in self.trading.get_all_positions():
            s = pos.symbol
            if len(s) < 16 or s[-15:-9] != f"{today:%y%m%d}":
                continue
            root, cp, k = s[:-15], s[-9], int(s[-8:]) / 1000
            q = int(float(pos.qty))
            sp = self.spreads.setdefault(root, dict(sym=root, cp=cp, qty=abs(q), credit=0.0))
            if q < 0:
                sp.update(ks=k, s_occ=s)
            else:
                sp.update(kl=k, l_occ=s)
        for root in [r for r, sp in self.spreads.items() if "ks" not in sp or "kl" not in sp]:
            self.log(f"{root}: found an unmatched option position; leaving it to you to review.")
            del self.spreads[root]
        for root in self.spreads:
            self.done.add(root)

    def safety_check(self):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderClass, TimeInForce
        self.safety_done = True
        for sym, sp in list(self.spreads.items()):
            S = self.price_now(sym)
            in_money = S < sp["ks"] if sp["cp"] == "P" else S > sp["ks"]
            if not in_money:
                self.log(f"Safety check {sym}: price {S:.2f}, sold strike {sp['ks']} is safe. Letting it expire.")
                continue
            legs = close_legs(sp["s_occ"], sp["l_occ"])
            try:
                self.trading.submit_order(MarketOrderRequest(qty=sp["qty"], order_class=OrderClass.MLEG, legs=legs,
                                                             time_in_force=TimeInForce.DAY))
                self.log(f"Safety check {sym}: price {S:.2f} crossed the sold strike {sp['ks']}. Bought the spread back "
                         "so the account is not handed shares.")
            except Exception as e:
                self.log(f"Safety check {sym}: couldn't buy back ({e}). Please check the Alpaca app before 2:30pm.")

    # ---------------------------------------------------------------- the day
    def run_day(self):
        now = datetime.now(CT)
        today = now.date()
        self.log(f"{self.name} starting ({'LIVE' if self.live else 'paper'}).")
        clock = self.trading.get_clock()
        if not clock.is_open and clock.next_open.astimezone(CT).date() != today:
            self.log("Market closed today.")
            return
        close = clock.next_close.astimezone(CT)
        if close.date() == today and close.time() < dtime(15, 0):
            self.log(f"Early close today ({close:%I:%M %p}). No trades on short days.")
            return
        acct = self.trading.get_account()
        morning = float(acct.equity)
        stopped = self.stopped_for_good()
        if stopped:
            self.log("Guardrail stop is active (GitHub issue open). Watching only, no trades.")
        elif morning < self.kill_pct * self.start_eq:
            self.raise_stop(morning)
            stopped = True
        self.recover_spreads(today)
        while datetime.now(CT).time() < T_STOP:
            t = datetime.now(CT).time()
            try:
                if t >= T30 and len(self.first30) < len(SYMBOLS) and t < ENTRY:
                    self.read_first30(today)
                if not stopped and ENTRY <= t < ENTRY_LAST:
                    if len(self.first30) < len(SYMBOLS):
                        self.read_first30(today)
                    equity = float(self.trading.get_account().equity)
                    for sym in SYMBOLS:
                        if sym not in self.done:
                            self.try_enter(sym, today, equity)
                if self.pending:
                    self.manage_pending()
                if SAFETY <= t < dtime(14, 30) and not self.safety_done:
                    self.safety_check()
            except Exception:
                self.log("Error this pass (will retry next minute):\n" + traceback.format_exc()[-600:])
            time.sleep(30)
        eq = float(self.trading.get_account().equity)
        self.log(f"Day over. Account {eq:,.2f} USD ({eq / self.start_eq * 100 - 100:+.0f}% vs start). "
                 "Expired spreads settle after the close; tomorrow's report shows the final number.")
        path = os.getenv("GITHUB_STEP_SUMMARY")
        if path:
            open(path, "a").write("## Spread bot day log\n```\n" + "\n".join(self.log_lines) + "\n```\n")

    def connection_check(self):
        try:
            acct = self.trading.get_account()
            clock = self.trading.get_clock()
            eq = float(acct.equity)
            self.log(f"Connection check OK: Alpaca {'LIVE' if self.live else 'paper'} account reached, {eq:,.2f} USD "
                     f"({eq / self.start_eq * 100 - 100:+.0f}% vs start), options level {acct.options_trading_level}. "
                     f"Next market open: {clock.next_open.astimezone(CT):%a %b %d %I:%M %p} Central.")
        except Exception as e:
            self.log(f"CONNECTION CHECK FAILED: {e}. Check the two Alpaca key secrets in GitHub.")
            raise SystemExit(1)


# --------------------------------------------------------- always on ---
T_READY = dtime(8, 25)
MAX_WAIT = 5.5 * 3600


def main():
    bot = Bot()
    now = datetime.now(CT)
    clock = bot.trading.get_clock()
    session_today = clock.is_open or clock.next_open.astimezone(CT).date() == now.date()
    if session_today and dtime(8, 0) <= now.time() < T_STOP:
        bot.run_day()
        return
    nxt = clock.next_open.astimezone(CT)
    target = datetime.combine(nxt.date(), T_READY, CT)
    wait = (target - now).total_seconds()
    bot.log(f"{bot.name} on standby ({'LIVE' if bot.live else 'paper'}). Next session: {nxt:%a %b %d}, "
            f"trading run starts {target:%I:%M %p} Central.")
    bot.connection_check()
    if wait > 10 * 60:
        nap = min(wait - 5 * 60, MAX_WAIT)
        bot.log(f"Waiting {nap / 3600:.1f} hours, then handing over to a fresh run.")
        time.sleep(nap)
        return
    time.sleep(max(wait, 0))
    bot.run_day()


if __name__ == "__main__":
    if os.getenv("CHECK_ONLY"):
        Bot().connection_check()
    else:
        main()
