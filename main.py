import asyncio
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Bot

# === КОНФИГУРАЦИЯ ===
TELEGRAM_TOKEN = "8440969823:AAHhS-fhgDG9T9K3tA7tadSWuBTdpBxIeL8"  # ← ЗАМЕНИ НА СВОЙ
YOUR_CHAT_ID = 5425531321                   # ← ЗАМЕНИ НА СВОЙ

# === КЭШ СИГНАЛОВ ===
recent_signals = {}

# === СЕКТОРЫ ===
SECTOR_MAP = {
    "METIS": "Layer 2", "PENDLE": "DeFi", "ONDO": "RWA", "TAO": "AI",
    "RNDR": "AI", "INJ": "DeFi", "POLYX": "RWA", "AKT": "AI",
    "GALA": "Gaming", "IMX": "Gaming", "STRK": "Layer 2",
    "PYTH": "Oracle", "ALT": "AI", "BOME": "Meme", "WLD": "AI",
    "SUI": "Layer 1", "SEI": "Layer 1", "AR": "Storage", "FET": "AI",
    "PEOPLE": "Meme", "JUP": "DeFi", "MNT": "Layer 2", "RENDER": "AI"
}

# === СПИСОК МОНЕТ (BINANCE ФОРМАТ) ===
SYMBOLS = [
    "METIS/USDT", "PENDLE/USDT", "ONDO/USDT", "TAO/USDT", "RNDR/USDT",
    "INJ/USDT", "POLYX/USDT", "AKT/USDT", "GALA/USDT", "IMX/USDT",
    "STRK/USDT", "PYTH/USDT", "ALT/USDT", "BOME/USDT", "WLD/USDT",
    "SUI/USDT", "SEI/USDT", "AR/USDT", "FET/USDT", "PEOPLE/USDT",
    "JUP/USDT", "MNT/USDT", "RENDER/USDT"
]

# === ИМПОРТЫ ===
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# === ЗАГРУЗКА ДАННЫХ ===
def fetch_full_history(symbol, interval='15m', max_candles=3000):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=max_candles)
        if not ohlcv or len(ohlcv) < 100:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"Ошибка {symbol}: {e}")
        return None

# === POC (POINT OF CONTROL) ИЗ VOLUME PROFILE ===
def calculate_poc(df, bins=50):
    min_price = df['low'].min()
    max_price = df['high'].max()
    if min_price == max_price:
        return df['close'].iloc[-1]
    price_range = np.linspace(min_price, max_price, bins)
    df['price_bin'] = pd.cut(df['close'], bins=price_range, labels=False, include_lowest=True)
    vol_by_bin = df.groupby('price_bin')['volume'].sum()
    if vol_by_bin.empty:
        return df['close'].iloc[-1]
    poc_bin = vol_by_bin.idxmax()
    return price_range[poc_bin]

# === MARKET STRUCTURE (HH/HL, LH/LL) ===
def detect_market_structure(df, window=5):
    highs = df['high'].rolling(window, center=True).max()
    lows = df['low'].rolling(window, center=True).min()
    df['is_high'] = (df['high'] == highs)
    df['is_low'] = (df['low'] == lows)
    return df

# === ПОДДЕРЖКА / СОПРОТИВЛЕНИЕ ===
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

# === ПРИЗНАКИ ДЛЯ ML ===
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

# === ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ===
def add_target(df, threshold=0.025, future_bars=4):
    df = df.copy()
    df['future_high'] = df['high'].shift(-future_bars)
    df['target'] = (df['future_high'] > df['close'] * (1 + threshold)).astype(int)
    return df.dropna()

# === ОБУЧЕНИЕ МОДЕЛИ ===
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
    model = XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)
    return model, scaler

# === ГИБКАЯ ФИЛЬТРАЦИЯ ДУБЛЕЙ ===
def is_signal_allowed(base, current_price, cooldown_hours=4, price_diff_threshold=0.03):
    now = datetime.now()
    last = recent_signals.get(base)
    if not last:
        return True
    if (now - last["timestamp"]) > timedelta(hours=cooldown_hours):
        return True
    if current_price > last["last_tp"]:
        return True
    if current_price < last["last_sl"]:
        return True
    price_diff = abs(current_price - last["last_price"]) / last["last_price"]
    if price_diff < price_diff_threshold:
        return False
    return True

def mark_signal_sent(base, price, tp, sl):
    recent_signals[base] = {
        "last_price": price,
        "last_tp": tp,
        "last_sl": sl,
        "timestamp": datetime.now()
    }

# === ОСНОВНОЙ ЦИКЛ ===
async def analyze_and_send():
    bot = Bot(token=TELEGRAM_TOKEN)
    for sym in SYMBOLS:
        try:
            df = fetch_full_history(sym)
            if df is None or len(df) < 200:
                continue

            base = sym.split('/')[0]
            current_price = df['close'].iloc[-1]

            if not is_signal_allowed(base, current_price):
                continue

            sector = SECTOR_MAP.get(base, "Other")
            vol_24h = df['volume'][-96:].sum()
            if vol_24h < 5_000_000:
                continue

            # === ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ===
            poc = calculate_poc(df)
            df = detect_market_structure(df)
            supports, _ = detect_support_resistance(df)
            near_support = is_near_level(current_price, supports)
            above_poc = current_price > poc

            model, scaler = train_model(df)
            if model is None:
                continue

            df_feat = add_features(df)
            if df_feat.empty:
                continue

            last_row = df_feat.iloc[-1]
            X = last_row[[
                'rsi', 'macd', 'ema9', 'ema21', 'ema50', 'cci', 'adx', 'mfi', 'obv',
                'bb_high', 'bb_low', 'atr', 'roc', 'vol_ratio', 'hour'
            ]].values.reshape(1, -1)
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)[0][1]

            # === УСЛОВИЕ ВХОДА: поддержка + выше POC + ML ===
            if proba > 0.75 and near_support and above_poc:
                if proba > 0.88:
                    tp_percent = 30
                elif proba > 0.82:
                    tp_percent = 20
                else:
                    tp_percent = 10

                sl_percent = 10
                tp = round(current_price * (1 + tp_percent / 100), 4)
                sl = round(current_price * (1 - sl_percent / 100), 4)

                msg = (
                    f"💎 **АЛМАЗНЫЙ СИГНАЛ**\n"
                    f"Монета: {base}USDT\n"
                    f"Сектор: {sector}\n\n"
                    f"📍 Цена: ${current_price:.2f}\n"
                    f"📊 POC: ${poc:.2f} | RSI: {last_row['rsi']:.1f}\n"
                    f"🧠 Уверенность ML: {proba:.1%}\n"
                    f"🎯 TP: ${tp} (+{tp_percent}%)\n"
                    f"🛑 SL: ${sl} (-{sl_percent}%)\n\n"
                    f"⏱️ Прогноз: рост в течение 1–4 часов"
                )
                await bot.send_message(chat_id=YOUR_CHAT_ID, text=msg, parse_mode="Markdown")
                mark_signal_sent(base, current_price, tp, sl)
                print(f"✅ Сигнал: {base} | Уверенность: {proba:.1%} | TP: +{tp_percent}%")
        except Exception as e:
            print(f"❌ Ошибка {sym}: {e}")
        time.sleep(1)

# === ЗАПУСК ===
async def main():
    while True:
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        print(f"\n🕒 [{now}] Запуск анализа...")
        await analyze_and_send()
        print("💤 Ожидание 15 минут...")
        time.sleep(15 * 60)

if __name__ == "__main__":
    asyncio.run(main())
