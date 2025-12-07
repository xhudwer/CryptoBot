import asyncio
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from telegram import Bot

# === КОНФИГУРАЦИЯ ===
TELEGRAM_TOKEN = "ВАШ_ТОКЕН_ОТ_BOTFATHER"  # ← ОБЯЗАТЕЛЬНО ЗАМЕНИ
YOUR_CHAT_ID = 987654321                   # ← ОБЯЗАТЕЛЬНО ЗАМЕНИ

# ❌ ЧЁРНЫЙ СПИСОК: исключаем топовые монеты
BLACKLIST = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TON", "SHIB",
    "TRX", "DOT", "LTC", "BCH", "LINK", "MATIC", "UNI", "AVAX", "ATOM",
    "XLM", "ETC", "FIL", "APT", "NEAR", "VET", "ICP", "HBAR", "MANA",
    "SAND", "AXS", "GRT", "ENJ", "CHZ", "THETA", "FTM", "FLOW", "OP",
    "ARB", "MKR", "AAVE", "SNX", "CRV", "COMP", "YFI", "LDO"
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

# === НОВОСТИ: ВОЗВРАЩАЕТ ИНСАЙТЫ (НЕ ОБЯЗАТЕЛЬНЫЕ) ===
def get_news_insights(base, max_news=2):
    """
    Возвращает список инсайтов в формате:
    [
        {'label': '[Позитив]', 'title': 'Заголовок...', 'time_ago': '2 ч. назад'},
        ...
    ]
    """
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?currencies={base}&public=true&limit=5"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if 'results' not in 
            return []
        
        insights = []
        for post in data['results'][:max_news]:
            kind = post.get('kind', 'neutral')
            label = "[Позитив]" if kind == 'positive' else "[Негатив]" if kind == 'negative' else "[Нейтрал]"
            
            # Время публикации
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
                except:
                    time_ago = ""
            
            title = post.get('title', 'Без заголовка')
            if len(title) > 60:
                title = title[:57] + "..."
            insights.append({
                'label': label,
                'title': title,
                'time_ago': time_ago
            })
        return insights
    except Exception as e:
        return []

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
            cached_futures = futures[:50]
            last_markets_update = now
        except:
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

# === ТЕХАНАЛИЗ ===
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

# === ML ===
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
    if base in BLACKLIST:
        return

    df = fetch_ohlcv(symbol)
    if df is None:
        return

    vol_24h = df['volume'].sum()
    if vol_24h < 10_000_000:
        return

    key = f"{base}_Futures"
    if key in recent_signals:
        last = recent_signals[key]
        if (datetime.now() - last["timestamp"]).total_seconds() < 4 * 3600:
            price_diff = abs(df['close'].iloc[-1] - last["last_price"]) / last["last_price"]
            if price_diff < 0.03:
                return

    current_price = df['close'].iloc[-1]
    supports, resistances = detect_support_resistance(df)
    near_support = is_near_level(current_price, supports)
    near_resistance = is_near_level(current_price, resistances)
    rsi = RSIIndicator(close=df['close'], window=14).rsi().iloc[-1]
    vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

    if not ((near_support and rsi < 35) or (near_resistance and rsi > 75)):
        return
    if vol_ratio < 1.5:
        return

    proba = train_and_predict(df)
    if proba is None or proba < 0.75:
        return

    is_long = near_support
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

    recent_signals[key] = {
        "last_price": current_price,
        "last_tp": tp,
        "last_sl": sl,
        "timestamp": datetime.now()
    }

    # === ФОРМИРОВАНИЕ СООБЩЕНИЯ ===
    msg = (
        f"{emoji} **{'LONG' if is_long else 'SHORT'}** | Futures\n"
        f"Монета: {base}USDT\n"
        f"📍 Цена: ${current_price:.2f}\n"
        f"📊 RSI: {rsi:.1f} | Объём: x{vol_ratio:.1f}\n"
        f"🎯 TP: ${tp} (+{tp_percent}%) | SL: ${sl}\n"
        f"🧠 Уверенность ML: {proba:.1%}"
    )

    # === ДОБАВЛЯЕМ НОВОСТИ, ЕСЛИ ЕСТЬ ===
    news_insights = get_news_insights(base, max_news=2)
    if news_insights:
        news_lines = ["\n📰 Новости:"]
        for ni in news_insights:
            line = f"• {ni['label']} {ni['title']}"
            if ni['time_ago']:
                line += f" ({ni['time_ago']})"
            news_lines.append(line)
        msg += "\n" + "\n".join(news_lines)

    await bot.send_message(chat_id=YOUR_CHAT_ID, text=msg, parse_mode="Markdown")
    print(f"✅ Сигнал: {base} | {'LONG' if is_long else 'SHORT'}")

# === ОСНОВНОЙ ЦИКЛ ===
async def analyze_and_send(bot):
    print("🔍 Сканирование непопулярных фьючерсов...")
    symbols = get_futures_symbols()
    for sym in symbols:
        await analyze_pair(sym, bot)
        await asyncio.sleep(0.3)

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    start_time = datetime.now()
    await bot.send_message(
        chat_id=YOUR_CHAT_ID,
        text=f"✅ Бот запущен ({start_time.strftime('%Y-%m-%d %H:%M')}).\n🔍 Анализ непопулярных фьючерсов каждые 15 минут."
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
