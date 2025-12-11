#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import time
import logging
from telegram import Bot
from telegram.constants import ParseMode

# ML (опционально)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os

# TA
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, CCIIndicator, ADXIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator

# === CONFIG ===
TELEGRAM_TOKEN = "8440969823:AAHhS-fhgDG9T9K3tA7tadSWuBTdpBxIeL8"
YOUR_CHAT_ID = 5425531321
USE_ML = False
MODEL_CACHE_DIR = "./models_cache"
SYMBOLS_CACHE_TTL = 3600
CONCURRENCY = 6
OHLCV_LIMIT = 300
TIMEFRAME = '15m'
MIN_24H_VOLUME = 10_000_000
PROBA_THRESHOLD = 0.75

BLACKLIST = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TON", "SHIB",
    "TRX", "DOT", "LTC", "BCH", "LINK", "MATIC", "UNI", "AVAX", "ATOM",
    "XLM", "ETC", "FIL", "APT", "NEAR", "VET", "ICP", "HBAR", "MANA",
    "SAND", "AXS", "GRT", "ENJ", "CHZ", "THETA", "FTM", "FLOW", "OP",
    "ARB", "MKR", "AAVE", "SNX", "CRV", "COMP", "YFI", "LDO"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

exchange = None
_markets_cache = {"updated": None, "symbols": []}
recent_signals = {}
semaphore = asyncio.Semaphore(CONCURRENCY)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


async def get_exchange():
    global exchange
    if exchange is None:
        exchange = ccxt_async.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
    return exchange


def get_news_insights(base, max_news=2):
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?currencies={base}&public=true&limit=5"
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if not isinstance(data, dict) or 'results' not in data:
            return []
        insights = []
        for post in data.get('results', [])[:max_news]:
            kind = post.get('kind', 'neutral')
            label = "[Позитив]" if kind == 'positive' else "[Негатив]" if kind == 'negative' else "[Нейтрал]"
            created = post.get('created_at')
            time_ago = ""
            if created:
                try:
                    pub_time = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    delta = now - pub_time
                    hours = int(delta.total_seconds() // 3600)
                    if hours < 1:
                        time_ago = "только что"
                    elif hours < 24:
                        time_ago = f"{hours} ч. назад"
                    else:
                        days = hours // 24
                        time_ago = f"{days} дн. назад"
                except Exception:
                    time_ago = ""
            title = post.get('title', 'Без заголовка')
            if len(title) > 80:
                title = title[:77] + "..."
            insights.append({'label': label, 'title': title, 'time_ago': time_ago})
        return insights
    except Exception as e:
        logging.debug("News fetch failed: %s", e)
        return []


async def get_futures_symbols(limit=50):
    global _markets_cache
    now = time.time()
    if _markets_cache['updated'] is None or (now - _markets_cache['updated']) > SYMBOLS_CACHE_TTL:
        try:
            ex = await get_exchange()
            markets = await ex.load_markets()
            symbols = []
            for s, meta in markets.items():
                info = meta.get('info', {})
                contract_type = info.get('contractType')
                if contract_type == 'LinearPerpetual' and meta.get('active', True):
                    base = s.split('/')[0]
                    clean_symbol = base + "USDT"
                    symbols.append(clean_symbol)
            symbols = sorted(list(set(symbols)))
            filtered = [s for s in symbols if s.replace('USDT', '') not in BLACKLIST]
            _markets_cache['symbols'] = filtered[:limit]
            _markets_cache['updated'] = now
            logging.info("Загружено %d фьючерсных пар", len(_markets_cache['symbols']))
        except Exception as e:
            logging.error("Ошибка загрузки рынков: %s", e)
            _markets_cache['symbols'] = ["METISUSDT", "PENDLEUSDT", "ONDOUSDT"]
            _markets_cache['updated'] = now
    return _markets_cache['symbols']


async def fetch_ohlcv_async(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT):
    ex = await get_exchange()
    try:
        async with semaphore:
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 50:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        logging.debug("fetch_ohlcv failed for %s: %s", symbol, e)
        return None


def detect_support_resistance(df, lookback=200, pdistance=0.01, top_n=10):
    df2 = df.copy().tail(lookback).reset_index(drop=True)
    lows = []
    highs = []
    N = len(df2)
    for i in range(2, N - 2):
        window = df2.loc[i - 2:i + 2]
        if df2.loc[i, "low"] == window["low"].min():
            lows.append(df2.loc[i, "low"])
        if df2.loc[i, "high"] == window["high"].max():
            highs.append(df2.loc[i, "high"])
    def cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current = [levels[0]]
        for lv in levels[1:]:
            if abs(lv - np.mean(current)) / np.mean(current) <= pdistance:
                current.append(lv)
            else:
                clusters.append(np.mean(current))
                current = [lv]
        clusters.append(np.mean(current))
        return clusters[:top_n]
    supports = cluster(lows)
    resistances = cluster(highs)
    return supports, resistances


def is_near_level(price, levels, threshold=0.01):
    for level in levels:
        if level == 0:
            continue
        if abs(price - level) / level <= threshold:
            return True
    return False


def add_features(df):
    df = df.copy()
    try:
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        df['macd'] = MACD(close=df['close']).macd()
        df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
        df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
        df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    except Exception as e:
        logging.debug("TA calc error: %s", e)
    df['roc'] = df['close'].pct_change(periods=10)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['hour'] = df['timestamp'].dt.hour
    return df.dropna().reset_index(drop=True)


def add_target(df, threshold=0.025, future_bars=4):
    df = df.copy()
    df['future_high'] = df['high'].shift(-future_bars)
    df['target'] = (df['future_high'] > df['close'] * (1 + threshold)).astype(int)
    return df.dropna().reset_index(drop=True)


def get_model_path(symbol):
    return os.path.join(MODEL_CACHE_DIR, f"{symbol}_xgb.pkl")


def train_model_for_symbol(symbol, df):
    try:
        dff = add_features(df)
        if len(dff) < 200:
            return None
        dft = add_target(dff)
        feature_cols = [c for c in ['rsi', 'macd', 'ema9', 'ema21', 'ema50', 'cci', 'adx', 'mfi', 'obv',
                                    'bb_high', 'bb_low', 'atr', 'roc', 'vol_ratio', 'hour'] if c in dft.columns]
        X = dft[feature_cols].values
        y = dft['target'].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = XGBClassifier(n_estimators=40, max_depth=4, use_label_encoder=False, eval_metric='logloss', verbosity=0)
        model.fit(Xs, y)
        joblib.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, get_model_path(symbol))
        return {'model': model, 'scaler': scaler, 'features': feature_cols}
    except Exception as e:
        return None


def load_cached_model(symbol):
    p = get_model_path(symbol)
    if os.path.exists(p):
        try:
            return joblib.load(p)
        except:
            pass
    return None


def predict_proba_with_model(modelobj, df_last_row):
    try:
        features = modelobj['features']
        scaler = modelobj['scaler']
        model = modelobj['model']
        Xn = df_last_row[features].values.reshape(1, -1)
        Xs = scaler.transform(Xn)
        proba = model.predict_proba(Xs)[0][1]
        return float(proba)
    except:
        return None


async def analyze_pair(symbol, bot):
    base = symbol.replace("USDT", "")
    if base in BLACKLIST:
        return

    df = await fetch_ohlcv_async(symbol)
    if df is None or df.empty:
        return

    tf_minutes = 15
    candles_24h = (24 * 60) // tf_minutes
    vol_24h = df['volume'].tail(candles_24h).sum()
    if vol_24h < MIN_24H_VOLUME:
        return

    key = f"{base}_Futures"
    if key in recent_signals:
        last = recent_signals[key]
        if (datetime.now(timezone.utc) - last["timestamp"]).total_seconds() < 14400:
            price_diff = abs(df['close'].iloc[-1] - last["last_price"]) / max(1e-8, last["last_price"])
            if price_diff < 0.03:
                return

    current_price = float(df['close'].iloc[-1])
    supports, resistances = detect_support_resistance(df)
    near_support = is_near_level(current_price, supports, threshold=0.01)
    near_resistance = is_near_level(current_price, resistances, threshold=0.01)
    rsi_series = RSIIndicator(close=df['close'], window=14).rsi()
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else 50.0
    vol_ratio = float(df['volume'].iloc[-1] / max(1e-9, df['volume'].rolling(20).mean().iloc[-1]))

    if not ((near_support and rsi < 40) or (near_resistance and rsi > 70)):
        return
    if vol_ratio < 1.2:
        return

    proba = None
    if USE_ML:
        modelobj = load_cached_model(base)
        if modelobj is None:
            modelobj = train_model_for_symbol(base, df.tail(1000))
        if modelobj:
            df_feat = add_features(df)
            if not df_feat.empty:
                proba = predict_proba_with_model(modelobj, df_feat.iloc[-1])
    else:
        proba = 0.9

    if proba is None or proba < PROBA_THRESHOLD:
        return

    is_long = near_support
    tp_percent = 30 if proba > 0.88 else 20 if proba > 0.82 else 10
    tp = round(current_price * (1 + tp_percent / 100), 4) if is_long else round(current_price * (1 - tp_percent / 100), 4)
    sl = round(current_price * 0.90, 4) if is_long else round(current_price * 1.10, 4)
    emoji = "🟢" if is_long else "🔴"

    recent_signals[key] = {
        "last_price": current_price,
        "last_tp": tp,
        "last_sl": sl,
        "timestamp": datetime.now(timezone.utc)
    }

    msg_lines = [
        f"{emoji} *{'LONG' if is_long else 'SHORT'}* | Futures",
        f"Монета: *{base}USDT*",
        f"📍 Цена: ${current_price:.4f}",
        f"📊 RSI: {rsi:.1f} | Объём: x{vol_ratio:.2f}",
        f"🎯 TP: ${tp} (+{tp_percent}%) | SL: ${sl}",
        f"🧠 Уверенность ML: {proba:.1%}"
    ]

    news_insights = get_news_insights(base, max_news=2)
    if news_insights:
        msg_lines.append("\n📰 Новости:")
        for ni in news_insights:
            line = f"• {ni['label']} {ni['title']}"
            if ni['time_ago']:
                line += f" ({ni['time_ago']})"
            msg_lines.append(line)

    msg = "\n".join(msg_lines)
    try:
        await bot.send_message(chat_id=YOUR_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
        logging.info("Сигнал отправлен: %s | %s", base, "LONG" if is_long else "SHORT")
    except Exception as e:
        logging.error("Ошибка отправки: %s", e)


async def analyze_and_send(bot):
    symbols = await get_futures_symbols(limit=50)
    tasks = [asyncio.create_task(analyze_pair(sym, bot)) for sym in symbols]
    await asyncio.gather(*tasks)


async def main():
    if TELEGRAM_TOKEN == "ВАШ_ТОКЕН_ОТ_BOTFATHER" or YOUR_CHAT_ID == 987654321:
        logging.error("Укажите TELEGRAM_TOKEN и YOUR_CHAT_ID!")
        return

    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        await bot.send_message(chat_id=YOUR_CHAT_ID, text="✅ Бот запущен.", parse_mode=ParseMode.MARKDOWN)
    except:
        pass

    logging.info("Бот запущен. Сканирование каждые 15 минут.")
    try:
        while True:
            await analyze_and_send(bot)
            await asyncio.sleep(900)
    except Exception as e:
        logging.exception("Критическая ошибка: %s", e)
    finally:
        if exchange:
            await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
