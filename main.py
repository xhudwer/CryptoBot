import asyncio
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from telegram import Bot

# === НАСТРОЙКИ (Scalingo будет подставлять из переменных окружения) ===
import os
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
YOUR_CHAT_ID = int(os.getenv("YOUR_CHAT_ID"))

# Список монет (KuCoin формат)
SYMBOLS = [
    "METIS-USDT", "PENDLE-USDT", "ONDO-USDT", "TAO-USDT", "RNDR-USDT",
    "INJ-USDT", "POLYX-USDT", "AKT-USDT", "GALA-USDT", "IMX-USDT"
]

# Импорты для теханализа
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

# === Функции (упрощённая версия для Scalingo) ===
def fetch_ohlcv(symbol, limit=200):
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        data = exchange.fetch_ohlcv(symbol, '15m', limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.astype(float)
        return df
    except:
        return None

def is_good_signal(df):
    if df is None or len(df) < 50:
        return False, 0
    close = df['close'].iloc[-1]
    rsi = RSIIndicator(close=df['close'], window=14).rsi().iloc[-1]
    bb = BollingerBands(close=df['close'])
    bb_low = bb.bollinger_lband().iloc[-1]
    near_support = close <= bb_low * 1.01
    volume_up = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
    return (rsi < 35 and near_support and volume_up), rsi

async def analyze_and_send():
    bot = Bot(token=TELEGRAM_TOKEN)
    for symbol in SYMBOLS:
        df = fetch_ohlcv(symbol)
        is_signal, rsi = is_good_signal(df)
        if is_signal:
            price = df['close'].iloc[-1]
            tp = round(price * 1.25, 4)
            sl = round(price * 0.90, 4)
            msg = (
                f"💎 Сигнал: {symbol}\n"
                f"Цена: {price}\n"
                f"RSI: {rsi:.1f}\n"
                f"TP: {tp} (+25%)\n"
                f"SL: {sl} (-10%)"
            )
            await bot.send_message(chat_id=YOUR_CHAT_ID, text=msg)
            print(f"Отправлен сигнал: {symbol}")

# === ОСНОВНОЙ ЦИКЛ ===
async def main():
    while True:
        print(f"[{datetime.now()}] Запуск анализа...")
        await analyze_and_send()
        print("Ожидание 15 минут...")
        time.sleep(15 * 60)

if __name__ == "__main__":
    asyncio.run(main())
