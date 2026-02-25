import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
import feedparser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SYDNEY_TZ = pytz.timezone("Australia/Sydney")
SYMBOL = "NVDA"

SEND_HOUR = int(os.getenv("SEND_HOUR", "7"))
SEND_MINUTE = int(os.getenv("SEND_MINUTE", "30"))
SEND_WINDOW_MINUTES = int(os.getenv("SEND_WINDOW_MINUTES", "10"))

STATE_PATH = os.getenv("STATE_PATH", os.path.join("state", "last_sent.json"))

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.mail.yahoo.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_APP_PASSWORD")
EMAIL_FROM = os.getenv("FROM_EMAIL")
EMAIL_TO = os.getenv("TO_EMAIL")

SUBJECT_PREFIX = os.getenv("SUBJECT_PREFIX", "NVDA 每日简报")


def now_sydney():
    return datetime.now(tz=SYDNEY_TZ)


def within_send_window(now_dt):
    target = now_dt.replace(hour=SEND_HOUR, minute=SEND_MINUTE, second=0, microsecond=0)
    delta = abs((now_dt - target).total_seconds())
    return delta <= SEND_WINDOW_MINUTES * 60


def read_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def write_state(date_str, sent_at_iso):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "sent_at": sent_at_iso}, f, ensure_ascii=False, indent=2)


def already_sent_today(now_dt):
    state = read_state()
    return state.get("date") == now_dt.strftime("%Y-%m-%d")


def fetch_price_history():
    ticker = yf.Ticker(SYMBOL)
    hist = ticker.history(period="6mo", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise RuntimeError("No price history")
    return ticker, hist


def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = gain_ema / (loss_ema + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower


def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def tech_summary(hist):
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    last = close.iloc[-1]
    prev = close.iloc[-2]
    chg = last - prev
    chg_pct = chg / prev * 100

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

    rsi14 = rsi(close).iloc[-1]
    macd_line, signal_line, hist_line = macd(close)
    macd_last = macd_line.iloc[-1]
    signal_last = signal_line.iloc[-1]

    bb_sma, bb_upper, bb_lower = bollinger_bands(close)
    bb_upper_last = bb_upper.iloc[-1]
    bb_lower_last = bb_lower.iloc[-1]
    bb_sma_last = bb_sma.iloc[-1]
    bb_width = (bb_upper_last - bb_lower_last) / bb_sma_last if bb_sma_last else np.nan

    atr14 = atr(high, low, close).iloc[-1]

    high_20 = close.rolling(20).max().iloc[-1]
    low_20 = close.rolling(20).min().iloc[-1]

    return {
        "last": last,
        "chg": chg,
        "chg_pct": chg_pct,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": rsi14,
        "macd": macd_last,
        "signal": signal_last,
        "macd_hist": hist_line.iloc[-1],
        "high_20": high_20,
        "low_20": low_20,
        "bb_upper": bb_upper_last,
        "bb_lower": bb_lower_last,
        "bb_sma": bb_sma_last,
        "bb_width": bb_width,
        "atr14": atr14,
    }


def fundamentals(ticker):
    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    def g(key):
        return info.get(key)

    return {
        "market_cap": g("marketCap"),
        "trailing_pe": g("trailingPE"),
        "forward_pe": g("forwardPE"),
        "ps": g("priceToSalesTrailing12Months"),
        "profit_margin": g("profitMargins"),
        "roe": g("returnOnEquity"),
        "revenue_growth": g("revenueGrowth"),
        "earnings_growth": g("earningsGrowth"),
        "fifty_two_week_high": g("fiftyTwoWeekHigh"),
        "fifty_two_week_low": g("fiftyTwoWeekLow"),
    }


def options_summary(ticker):
    try:
        exps = ticker.options
    except Exception:
        return None
    if not exps:
        return None

    exp = exps[0]
    try:
        chain = ticker.option_chain(exp)
    except Exception:
        return None

    calls = chain.calls
    puts = chain.puts
    if calls.empty or puts.empty:
        return None

    total_call_oi = calls["openInterest"].fillna(0).sum()
    total_put_oi = puts["openInterest"].fillna(0).sum()
    put_call_oi = total_put_oi / total_call_oi if total_call_oi > 0 else None

    total_call_vol = calls["volume"].fillna(0).sum()
    total_put_vol = puts["volume"].fillna(0).sum()
    put_call_vol = total_put_vol / total_call_vol if total_call_vol > 0 else None

    iv_call = calls["impliedVolatility"].dropna().median()
    iv_put = puts["impliedVolatility"].dropna().median()
    iv_mid = np.nanmean([iv_call, iv_put])

    return {
        "expiration": exp,
        "put_call_oi": put_call_oi,
        "put_call_vol": put_call_vol,
        "iv_mid": iv_mid,
    }


def news_items():
    rss_urls = [
        "https://nvidianews.nvidia.com/releases.xml",
        "https://feeds.feedburner.com/nvidiablog",
        "https://developer.nvidia.com/blog/feed",
        "https://blogs.nvidia.cn/blog/category/news/feed/",
        "https://blogs.nvidia.cn/feed/",
        "https://developer.nvidia.cn/zh-cn/blog/feed",
        "https://news.google.com/rss/search?q=" + requests.utils.quote("NVIDIA stock"),
        "https://news.google.com/rss/search?q=" + requests.utils.quote("NVDA earnings"),
        "https://news.google.com/rss/search?q=" + requests.utils.quote("NVIDIA AI chip"),
        "https://news.google.com/rss/search?q=" + requests.utils.quote("英伟达 OR NVIDIA"),
    ]

    items = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for e in feed.entries[:5]:
            items.append({
                "title": e.get("title"),
                "link": e.get("link"),
                "source": e.get("source", {}).get("title") if isinstance(e.get("source"), dict) else None,
            })
    # de-dup by title
    seen = set()
    unique = []
    for it in items:
        if not it["title"]:
            continue
        if it["title"] in seen:
            continue
        seen.add(it["title"])
        unique.append(it)
    return unique[:6]


def format_money(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if abs(value) >= 1e12:
        return f"{value/1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value/1e6:.2f}M"
    return f"{value:.2f}"


def pct(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{value*100:.2f}%"


def compose_report(ticker, hist):
    t = tech_summary(hist)
    f = fundamentals(ticker)
    o = options_summary(ticker)
    n = news_items()

    last_date = hist.index[-1].strftime("%Y-%m-%d")
    now_dt = now_sydney()
    subject = f"{SUBJECT_PREFIX} - {now_dt.strftime('%Y-%m-%d')}"

    tech_lines = [
        f"最新价: ${t['last']:.2f} (日变动 {t['chg']:+.2f}, {t['chg_pct']:+.2f}%)  数据日期: {last_date}",
        f"均线: SMA20 {t['sma20']:.2f}, SMA50 {t['sma50']:.2f}, SMA200 {t['sma200']:.2f}"
        if not np.isnan(t['sma200']) else f"均线: SMA20 {t['sma20']:.2f}, SMA50 {t['sma50']:.2f}",
        f"RSI14: {t['rsi14']:.1f}  MACD: {t['macd']:.2f} / Signal {t['signal']:.2f} / Hist {t['macd_hist']:.2f}",
        f"近20日区间: 高 {t['high_20']:.2f} / 低 {t['low_20']:.2f}",
        f"布林带(20,2): 中轨 {t['bb_sma']:.2f} / 上轨 {t['bb_upper']:.2f} / 下轨 {t['bb_lower']:.2f} / 带宽 {t['bb_width']:.2%}",
        f"ATR14: {t['atr14']:.2f} ({t['atr14'] / t['last']:.2%} of price)",
    ]

    fund_lines = [
        f"市值: {format_money(f['market_cap'])}",
        f"估值: trailing PE {f['trailing_pe'] if f['trailing_pe'] is not None else 'NA'}, forward PE {f['forward_pe'] if f['forward_pe'] is not None else 'NA'}, P/S {f['ps'] if f['ps'] is not None else 'NA'}",
        f"盈利能力: 净利率 {pct(f['profit_margin'])}, ROE {pct(f['roe'])}",
        f"增长: 收入增速 {pct(f['revenue_growth'])}, 利润增速 {pct(f['earnings_growth'])}",
        f"52周区间: {f['fifty_two_week_low']} - {f['fifty_two_week_high']}",
    ]

    options_lines = []
    if o:
        options_lines = [
            f"最近到期: {o['expiration']}",
            f"Put/Call 成交量比: {o['put_call_vol']:.2f}" if o['put_call_vol'] is not None else "Put/Call 成交量比: NA",
            f"Put/Call 持仓量比: {o['put_call_oi']:.2f}" if o['put_call_oi'] is not None else "Put/Call 持仓量比: NA",
            f"隐含波动率中位: {o['iv_mid']:.2%}" if o['iv_mid'] is not None else "隐含波动率中位: NA",
        ]
    else:
        options_lines = ["期权数据: 暂不可用"]

    news_lines = ["- " + (f"{it['title']}" + (f" ({it['source']})" if it.get("source") else "")) for it in n]
    if not news_lines:
        news_lines = ["- 暂无可用新闻"]

    risk_lines = []
    if t["rsi14"] >= 70:
        risk_lines.append("RSI 处于超买区间，短期回撤风险上升。")
    if t["rsi14"] <= 30:
        risk_lines.append("RSI 处于超卖区间，波动加大。")
    if o and o["iv_mid"] is not None and o["iv_mid"] >= 0.6:
        risk_lines.append("期权隐含波动率偏高，事件驱动风险增大。")
    if not risk_lines:
        risk_lines.append("注意宏观与行业情绪变化可能放大波动。")

    body = "\n".join([
        f"NVDA 每日简报 ({now_dt.strftime('%Y-%m-%d')} Sydney)",
        "",
        "[技术面]",
        *tech_lines,
        "",
        "[基本面]",
        *fund_lines,
        "",
        "[新闻催化]",
        *news_lines,
        "",
        "[期权]",
        *options_lines,
        "",
        "[风险提示]",
        *risk_lines,
        "",
        "提示: 本简报仅供信息参考，不构成投资建议。",
        "来源: Yahoo Finance, NVIDIA RSS, Google News RSS",
    ])

    return subject, body


def send_email(subject, body):
    if not all([SMTP_USER, SMTP_PASS, EMAIL_FROM, EMAIL_TO]):
        raise RuntimeError("Missing SMTP_USER/SMTP_APP_PASSWORD/FROM_EMAIL/TO_EMAIL")

    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


def main():
    now_dt = now_sydney()
    if not within_send_window(now_dt):
        print("Not in send window.")
        return 0

    if already_sent_today(now_dt):
        print("Already sent today.")
        return 0

    ticker, hist = fetch_price_history()
    subject, body = compose_report(ticker, hist)

    dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
    if dry_run:
        print(subject)
        print(body)
    else:
        send_email(subject, body)

    write_state(now_dt.strftime("%Y-%m-%d"), now_dt.isoformat())
    print("Sent and state updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
