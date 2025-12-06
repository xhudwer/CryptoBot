import asyncio
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from telegram import Bot

# === КОНФИГУРАЦИЯ ===
TELEGRAM_TOKEN = "8440969823:AAHhS-fhgDG9T9K3tA7tadSWuBTdpBxIeL8"  # ← ОБЯЗАТЕЛЬНО ЗАМЕНИ
YOUR_CHAT_ID = 5425531321                   # ← ОБЯЗАТЕЛЬНО ЗАМЕНИ

# ❌ ЧЁРНЫЙ СПИСОК: исключаем мегакапы (даже если в топе объёма)
BLACKLIST = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TON", "SHIB",
    "TRX", "DOT", "LTC", "BCH", "LINK", "MATIC", "UNI", "AVAX", "ATOM",
    "XLM", "ETC", "FIL", "APT", "NEAR", "VET", "ICP", "HBAR", "MANA",
    "SAND", "AXS", "GRT", "ENJ", "CHZ", "THETA", "FTM", "FLOW"
}

recent_signals = {}
last_markets_update = None
cached_futures = []

# === ИМПОРТЫ ДЛЯ ТЕХАНАЛИЗА И ML ===
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# === НОВОСТИ: CryptoPanic (бесплатно, без ключа) ===
def get_news_sentiment(base):
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?currencies={base}&public=true&limit=3"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return 0.0
        data = resp.json()
        if 'results' not in 
            return 0.0
        sentiments = []
        for post in data['results']:
            if post.get('kind') == 'positive':
                sentiments.append(1.0)
            elif post.get('kind') == 'negative':
                sentiments.append(-1.0)
        return np.mean(sentiments) if sentiments else 0.0
    except:
        return 0.0

# === ЗАГРУЗКА ФЬЮЧЕРСНЫХ ПАР ===
def get_futures_symbols():
    global last_markets_update, cached_futures
    now = datetime.now()
    if last_markets_update is None or (now - last_markets_update).total_seconds() > 3600:
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            markets = exchange.load_markets()
            futures = [
                s for s in markets
                if s.endswith('USDT')
                and not s.endswith('_USDT')
                and markets[s].get('type') == 'future'
                and markets[s]['active']
            ]
            cached_futures = futures[:60]  # топ-60 по объёму
            last_markets_update = now
            print(f"✅ Загружено {len(cached_futures)} фьючерсных пар")
        except Exception as e:
            print(f"Ошибка загрузки futures: {e}")
            cached_futures = ["METISUSDT", "PENDLEUSDT", "ONDOUSDT"]
    return cached_futures

# === ЗАГРУЗКА СВЕЧЕЙ ===
def fetch_ohlcv(symbol, interval='15m'):
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=300)
        if not ohlcv or len(ohlcv) < 100:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").reset_index(drop=True)
    except:
        return None

# === ТЕХАНАЛИЗ: ПОДДЕРЖКА/СОПРОТИВЛЕНИЕ ===
def detect_support_resistance(df, window=20):
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

# === ML: ГЕНЕРАЦИЯ ПРИЗНАКОВ ===
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

# === ML: ЦЕЛЕВАЯ ПЕРЕМЕННАЯ И ОБУЧЕНИЕ ===
def add_target(df, threshold=0.025, future_bars=4):
    df = df.copy()
    df['future_high'] = df['high'].shift(-future_bars)
    df['target'] = (df['future_high'] > df['close'] * (1 + threshold)).astype(int)
    return df.dropna()

def train_and_predict(df):
    df_feat = add_features(df)
    if df_feat.empty or len(df_feat) < 100:
        return None
    df_target = add_target(df_feat)
    if df_target.empty:
        return None
    feature_cols = [
        'rsi', 'macd', 'ema9', 'ema21', 'ema50', 'cci', 'adx', 'mfi', 'obv',
        'bb_high', 'bb_low', 'atr', 'roc', 'vol_ratio', 'hour'
    ]
    X = df_target[feature_cols].values[-1:].reshape(1, -1)
    y = df_target['target'].values
    if len(y) < 50:
        return None
    scaler = StandardScaler()
    X_full = df_target[feature_cols].values
    X_scaled = scaler.fit_transform(X_full)
    model = XGBClassifier(n_estimators=30, random_state=42, eval_metric='logloss')
    model.fit(X_scaled, y)
    X_scaled_new = scaler.transform(X)
    proba = model.predict_proba(X_scaled_new)[0][1]
    return proba

# === АНАЛИЗ ОДНОЙ МОНЕТЫ ===
async def analyze_pair(symbol, bot):
    base = symbol.replace("USDT", "")
    
    # ⚠️ ПРОПУСКАЕМ МЕГАКАПЫ
    if base in BLACKLIST:
        return

    df = fetch_ohlcv(symbol)
    if df is None:
        return

    # Фильтр по объёму (только ликвидные)
    vol_24h = df['volume'].sum()
    if vol_24h < 10_000_000:
        return

    # Фильтр по дублям
    key = f"{base}_Futures"
    if key in recent_signals:
        last = recent_signals[key]
        if (datetime.now() - last["timestamp"]).total_seconds() < 4 * 3600:
            price_diff = abs(df['close'].iloc[-1] - last["last_price"]) / last["last_price"]
            if price_diff < 0.03:
                return

    # Теханализ
    current_price = df['close'].iloc[-1]
    supports, resistances = detect_support_resistance(df)
    near_support = is_near_level(current_price, supports)
    near_resistance = is_near_level(current_price, resistances)
    rsi = RSIIndicator(close=df['close'], window=14).rsi().iloc[-1]
    vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

    # Быстрая проверка перед ML
    if not ((near_support and rsi < 35) or (near_resistance and rsi > 70)):
        return
    if vol_ratio < 1.5:
        return

    # ML
    proba = train_and_predict(df)
    if proba is None or proba < 0.75:
        return

    # Новостной буст
    sentiment = get_news_sentiment(base)
    if (near_support and sentiment > 0.3) or (near_resistance and sentiment < -0.3):
        proba = min(proba + 0.1, 0.99)

    # Направление
    is_long = near_support

    # TP/SL
    if proba > 0.88:
        tp_percent = 30
    elif proba > 0.82:
        tp_percent = 20
    else:
        tp_percent = 10

    if is_long:
        tp = round(current_price * (1 + tp_percent / 100), 4)
        sl = round(current_price * 0.90, 4)
        emoji = "🟢"
    else:
        tp = round(current_price * (1 - tp_percent / 100), 4)
        sl = round(current_price * 1.10, 4)
        emoji = "🔴"

    # Сохраняем сигнал
    recent_signals[key] = {
        "last_price": current_price,
        "last_tp": tp,
        "last_sl": sl,
        "timestamp": datetime.now()
    }

    # Отправка
    msg = (
        f"{emoji} **{'LONG' if is_long else 'SHORT'}** | Futures\n"
        f"Монета: {base}USDT\n"
        f"📍 Цена: ${current_price:.2f}\n"
        f"📊 RSI: {rsi:.1f} | Объём: x{vol_ratio:.1f}\n"
        f"🎯 TP: ${tp} (+{tp_percent}%) | SL: ${sl}\n"
        f"🧠 Уверенность ML: {proba:.1%}\n"
        f"🗞️ Новости: {'+' if sentiment > 0.3 else '-' if sentiment < -0.3 else '0'}"
    )
    await bot.send_message(chat_id=YOUR_CHAT_ID, text=msg, parse_mode="Markdown")
    print(f"✅ Сигнал: {base} | {'LONG' if is_long else 'SHORT'}")

# === ОСНОВНОЙ ЦИКЛ ===
async def analyze_and_send(bot):
    print("🔍 Сканирование непопулярных фьючерсов на Binance...")
    symbols = get_futures_symbols()
    for sym in symbols:
        await analyze_pair(sym, bot)
        await asyncio.sleep(0.3)  # уважаем лимиты Binance

# === ЗАПУСК ===
async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    start_time = datetime.now()
    await bot.send_message(
        chat_id=YOUR_CHAT_ID,
        text=f"✅ Бот запущен ({start_time.strftime('%Y-%m-%d %H:%M')}).\n🔍 Анализ только непопулярных фьючерсов каждые 15 минут."
    )
    print("✅ Бот запущен.")

    while True:
        try:
            await analyze_and_send(bot)
            await asyncio.sleep(15 * 60)
        except Exception as e:
            print(f"❗️Ошибка: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
