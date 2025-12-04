import asyncio
import logging
import os
import pandas as pd
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

def main():
    # Добавь задержку, если запущено в облаке
    if os.getenv("RAILWAY_ENVIRONMENT"):
        logging.info("Ожидание 10 секунд для завершения предыдущего экземпляра...")
        asyncio.run(asyncio.sleep(10))

# ML
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange
from pycoingecko import CoinGeckoAPI
from pybit.unified_trading import HTTP

# === CONFIG ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
YOUR_CHAT_ID = os.getenv("YOUR_CHAT_ID")

if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN не задан в переменных окружения")
if not YOUR_CHAT_ID:
    raise ValueError("❌ YOUR_CHAT_ID не задан в переменных окружения")
YOUR_CHAT_ID = int(YOUR_CHAT_ID)

# Глобальные данные
models = {}
scalers = {}
last_trained = {}
price_history = {}

logging.basicConfig(level=logging.INFO)

# === Топ монет пока вручную ===
def get_top_symbols(limit=15):
    return [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT",
        "DOTUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "SHIBUSDT"
    ][:limit]

# === Генерация признаков для ML ===
def add_features(df):
    df = df.copy()
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['ema_diff'] = df['ema9'] - df['ema21']
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['volume_ma']
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['price_norm'] = (df['close'] - df['close'].rolling(30).mean()) / df['close'].rolling(30).std()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    return df.dropna()

# === Целевая переменная: рост >2% за 3 свечи (45 мин) ===
def add_target(df, threshold=0.02, future_bars=3):
    df = df.copy()
    df['future_high'] = df['high'].shift(-future_bars)
    df['target'] = (df['future_high'] > df['close'] * (1 + threshold)).astype(int)
    return df.dropna()

# === Обучение модели ===
def train_model(df):
    df = add_features(df)
    df = add_target(df)
    feature_cols = ['rsi', 'ema_diff', 'macd', 'vol_ratio', 'atr', 'price_norm', 'hour']
    X = df[feature_cols]
    y = df['target']

    if len(X) < 100:
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

# === Прогноз ===
def predict_signal(model, scaler, last_row):
    feature_cols = ['rsi', 'ema_diff', 'macd', 'vol_ratio', 'atr', 'price_norm', 'hour']
    X = last_row[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X.values)
    proba = model.predict_proba(X_scaled)[0][1]
    return proba > 0.75, proba

# === Анализ одной монеты через Bybit ===
async def analyze_symbol(context: ContextTypes.DEFAULT_TYPE, symbol: str):
    try:
        client = HTTP()
        resp = client.get_kline(
            category="linear",
            symbol=symbol,
            interval=15,  # 15-минутные свечи
            limit=600
        )
        if "result" not in resp or "list" not in resp["result"]:
            logging.warning(f"Bybit: нет данных для {symbol}")
            return

        data = resp["result"]["list"]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")

        price_history[symbol] = df

        now = datetime.now()
        last_train_time = last_trained.get(symbol, datetime(2020, 1, 1))
        if (now - last_train_time).total_seconds() > 6 * 3600:  # Обучаем раз в 6 часов
            model, scaler = train_model(df)
            if model is not None:
                models[symbol] = model
                scalers[symbol] = scaler
                last_trained[symbol] = now
                logging.info(f"✅ Модель для {symbol} обновлена")

        if symbol in models:
            df_feat = add_features(df)
            if not df_feat.empty:
                last_row = df_feat.iloc[-1]
                has_signal, proba = predict_signal(models[symbol], scalers[symbol], last_row)
                if has_signal:
                    entry = last_row['close']
                    tp = round(entry * 1.10, 4)
                    sl = round(entry * 0.90, 4)
                    msg = (
                        f"🧠 **ML-СИГНАЛ** (уверенность: {proba:.1%})\n"
                        f"Монета: `{symbol}`\n"
                        f"Направление: LONG\n"
                        f"Вход: {entry}\n"
                        f"TP: {tp} (+10%)\n"
                        f"SL: {sl} (-10%)\n"
                        f"Время: {now.strftime('%Y-%m-%d %H:%M')}"
                    )
                    await context.bot.send_message(chat_id=YOUR_CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"❌ Ошибка {symbol}: {e}")

# === Команды Telegram ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Привет! Используй /scan для анализа рынка.")

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Запускаю ML-анализ топ-15 монет...")
    symbols = get_top_symbols(15)
    for symbol in symbols:
        await analyze_symbol(context, symbol)
    await update.message.reply_text("✅ Готово!")

# === Автоматический скан каждые 30 минут ===
async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    symbols = get_top_symbols(15)
    for symbol in symbols:
        await analyze_symbol(context, symbol)

# === Запуск бота ===
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan))
    app.job_queue.run_repeating(scheduled_scan, interval=30 * 60, first=10)
    app.run_polling()

if __name__ == "__main__":
    main()
