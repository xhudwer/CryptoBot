import asyncio
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from telegram import Bot

# === КОНФИГУРАЦИЯ ===
TELEGRAM_TOKEN = "ВАШ_ТОКЕН_ОТ_BOTFATHER"  # ← ЗАМЕНИ
YOUR_CHAT_ID = 987654321                   # ← ЗАМЕНИ

recent_signals = {}

SECTOR_MAP = {
    "METIS": "Layer 2", "PENDLE": "DeFi", "ONDO": "RWA", "TAO": "AI",
    "RNDR": "AI", "INJ": "DeFi", "POLYX": "RWA", "AKT": "AI",
    "GALA": "Gaming", "IMX": "Gaming", "STRK": "Layer 2",
    "PYTH": "Oracle", "ALT": "AI", "BOME": "Meme", "WLD": "AI",
    "SUI": "Layer 1", "SEI": "Layer 1", "AR": "Storage", "FET": "AI",
    "PEOPLE": "Meme", "JUP": "DeFi", "RENDER": "AI"
}

DESIRED_BASES = set(SECTOR_MAP.keys())

# === ЗАГРУЗКА НОВОСТЕЙ ===
def get_news_sentiment(base):
    """Возвращает средний сентимент по монете (1.0 = позитив, -1.0 = негатив)"""
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?currencies={base}&public=true&limit=5"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return 0.0
        data = resp.json()
        if 'results' not in data:
            return 0.0
        sentiments = []
        for post in data['results']:
            if 'kind' in post:
                if post['kind'] == 'positive':
                    sentiments.append(1.0)
                elif post['kind'] == 'negative':
                    sentiments.append(-1.0)
        return np.mean(sentiments) if sentiments else 0.0
    except:
        return 0.0

# === ЗАГРУЗКА ДАННЫХ ===
def fetch_ohlcv(symbol, market_type='spot', interval='15m', max_candles=3000):
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': market_type}
        })
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=max_candles)
        if not ohlcv or len(ohlcv) < 100:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"Ошибка {symbol} ({market_type}): {e}")
        return None

def get_active_symbols(market_type='spot'):
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': market_type}
        })
        markets = exchange.load_markets()
        suffix = 'USDT'
        if market_type == 'spot':
            suffix = '/USDT'
        usdt_pairs = [
            symbol for symbol in markets.keys()
            if symbol.endswith(suffix) and markets[symbol]['active']
        ]
        if market_type == 'spot':
            filtered = [sym for sym in usdt_pairs if sym.split('/')[0] in DESIRED_BASES]
        else:
            filtered = [sym for sym in usdt_pairs if sym.replace('USDT', '') in DESIRED_BASES]
        return filtered
    except:
        # Резервный список
        if market_type == 'spot':
            return [f"{base}/USDT" for base in list(DESIRED_BASES)[:15]]
        else:
            return [f"{base}USDT" for base in list(DESIRED_BASES)[:15]]

# === ТЕХАНАЛИЗ ===
def calculate_poc(df, bins=50):
    min_p = df['low'].min()
    max_p = df['high'].max()
    if min_p == max_p:
        return df['close'].iloc[-1]
    price_range = np.linspace(min_p, max_p, bins)
    df['price_bin'] = pd.cut(df['close'], bins=price_range, labels=False, include_lowest=True)
    vol_by_bin = df.groupby('price_bin')['volume'].sum()
    if vol_by_bin.empty:
        return df['close'].iloc[-1]
    poc_bin = vol_by_bin.idxmax()
    return price_range[poc_bin]

def detect_support_resistance(df, window=30):
    lows = df['low'].rolling(window=3, center=True).min()
    highs = df['high'].rolling(window=3, center=True).max()
    supports = df[df['low'] == lows]['low'][-window:].dropna().values
    resistances = df[df['high'] == highs]['high'][-window:].dropna().values
    return supports, resistances

def is_near_level(price, levels, threshold=0.005):
    for level in levels:
        if abs(price - level) / level <= threshold:
            return True
    return False

def add_features(df):
    df = df.copy()
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
    df['roc'] = df['close'].pct_change(periods=10)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['volume_ma']
    df['hour'] = df['timestamp'].dt.hour
    return df.dropna()

def add_target(df, threshold=0.025, future_bars=4):
    df = df.copy()
    df['future_high'] = df['high'].shift(-future_bars)
    df['target'] = (df['future_high'] > df['close'] * (1 + threshold)).astype(int)
    return df.dropna()

def train_model(df):
    df = add_features(df)
    df = add_target(df)
    feature_cols = [
        'rsi', 'macd', 'ema9', 'ema21', 'ema50', 'cci', 'adx', 'mfi', 'obv',
        'bb_high', 'bb_low', 'atr', 'roc', 'vol_ratio', 'hour'
    ]
    X = df[feature_cols].values
    y = df['target'].values
    if len(X) < 100:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    model.fit(X_scaled, y)
    return model, scaler

def is_signal_allowed(base, market_type, current_price, cooldown_hours=4):
    key = f"{base}_{market_type}"
    now = datetime.now()
    last = recent_signals.get(key)
    if not last:
        return True
    if (now - last["timestamp"]) > timedelta(hours=cooldown_hours):
        return True
    if current_price > last["last_tp"] or current_price < last["last_sl"]:
        return True
    price_diff = abs(current_price - last["last_price"]) / last["last_price"]
    if price_diff < 0.03:
        return False
    return True

def mark_signal_sent(base, market_type, price, tp, sl):
    key = f"{base}_{market_type}"
    recent_signals[key] = {
        "last_price": price,
        "last_tp": tp,
        "last_sl": sl,
        "timestamp": datetime.now()
    }

# === ОСНОВНОЙ АНАЛИЗ ===
async def analyze_pair(symbol, market_type, bot):
    df = fetch_ohlcv(symbol, market_type)
    if df is None or len(df) < 200:
        return

    if market_type == 'spot':
        base = symbol.split('/')[0]
    else:
        base = symbol.replace('USDT', '')

    current_price = df['close'].iloc[-1]
    if not is_signal_allowed(base, market_type, current_price):
        return

    sector = SECTOR_MAP.get(base, "Other")
    vol_24h = df['volume'][-96:].sum()
    if vol_24h < 5_000_000:
        return

    # === Анализ ===
    poc = calculate_poc(df)
    supports, resistances = detect_support_resistance(df)
    near_support = is_near_level(current_price, supports)
    near_resistance = is_near_level(current_price, resistances)
    rsi = RSIIndicator(close=df['close'], window=14).rsi().iloc[-1]
    vol_ratio = (df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])

    model, scaler = train_model(df)
    if model is None:
        return

    df_feat = add_features(df)
    if df_feat.empty:
        return

    last_row = df_feat.iloc[-1]
    X = last_row[[
        'rsi', 'macd', 'ema9', 'ema21', 'ema50', 'cci', 'adx', 'mfi', 'obv',
        'bb_high', 'bb_low', 'atr', 'roc', 'vol_ratio', 'hour'
    ]].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0][1]

    # === Новостной сентимент ===
    sentiment = get_news_sentiment(base)
    if sentiment > 0.3:
        proba = min(proba + 0.1, 0.99)  # Boost Long
    elif sentiment < -0.3:
        proba = min(proba + 0.1, 0.99)  # Boost Short

    # === Long сигнал ===
    is_long = (proba > 0.75 and near_support and rsi < 35 and vol_ratio > 1.5)
    # === Short сигнал ===
    is_short = (proba > 0.75 and near_resistance and rsi > 70 and vol_ratio > 1.5)

    if is_long or is_short:
        if proba > 0.88:
            tp_percent = 30
        elif proba > 0.82:
            tp_percent = 20
        else:
            tp_percent = 10

        sl_percent = 10
        if is_long:
            tp = round(current_price * (1 + tp_percent / 100), 4)
            sl = round(current_price * (1 - sl_percent / 100), 4)
            direction_str = "рост"
            emoji = "🟢"
        else:
            tp = round(current_price * (1 - tp_percent / 100), 4)
            sl = round(current_price * (1 + sl_percent / 100), 4)
            direction_str = "падение"
            emoji = "🔴"

        msg = (
            f"{emoji} **{'LONG' if is_long else 'SHORT'}** | {market_type}\n"
            f"Монета: {base}USDT\n"
            f"Сектор: {sector}\n\n"
            f"📍 Цена: ${current_price:.2f}\n"
            f"📊 POC: ${poc:.2f} | RSI: {rsi:.1f}\n"
            f"🧠 Уверенность: {proba:.1%}\n"
            f"🎯 TP: ${tp} ({'+' if is_long else '-'}{tp_percent}%)\n"
            f"🛑 SL: ${sl} ({'-' if is_long else '+'}{sl_percent}%)\n"
            f"🗞️ Новости: {'Позитив' if sentiment > 0.3 else 'Негатив' if sentiment < -0.3 else 'Нейтрал'}\n\n"
            f"⏱️ Прогноз: {direction_str} в течение 1–4 часов"
        )
        await bot.send_message(chat_id=YOUR_CHAT_ID, text=msg, parse_mode="Markdown")
        mark_signal_sent(base, market_type, current_price, tp, sl)
        print(f"✅ Сигнал: {base} | {market_type} | {'LONG' if is_long else 'SHORT'}")

async def analyze_and_send(bot):
    print("🔍 Сканирование Spot и Futures...")
    spot_symbols = get_active_symbols('spot')
    futures_symbols = get_active_symbols('future')
    for sym in spot_symbols:
        await analyze_pair(sym, "Spot", bot)
        time.sleep(0.5)
    for sym in futures_symbols:
        await analyze_pair(sym, "Futures", bot)
        time.sleep(0.5)

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=YOUR_CHAT_ID, text="✅ Бот запущен.\n🔍 Сканирование Spot и Futures каждые 15 минут.")
    print("✅ Бот запущен.")

    while True:
        await analyze_and_send(bot)
        print("💤 Ожидание 15 минут...")
        time.sleep(15 * 60)

if __name__ == "__main__":
    asyncio.run(main())
