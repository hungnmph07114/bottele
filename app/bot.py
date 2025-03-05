import logging
import requests
import telebot
import pandas as pd
import numpy as np
import sys
import json
import os

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot.log', encoding='utf-8'),
        logging.StreamHandler(stream=open(sys.stdout.fileno(), 'w', encoding='utf-8', errors='replace'))
    ]
)
logger = logging.getLogger('CryptoBot')

# Khởi tạo bot
TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY'
bot = telebot.TeleBot(TOKEN, threaded=False)

# Cấu hình
BINANCE_API = "https://api.binance.com/api/v3"
timeframes = {
    '1m': '1 phút', '5m': '5 phút', '15m': '15 phút',
    '1h': '1 giờ', '4h': '4 giờ', '1d': '1 ngày',
    'w1': '1 tuần', 'M1': '1 tháng'
}

# Lưu trữ lịch sử tín hiệu
HISTORY_FILE = 'signal_history.json'
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        signal_history = json.load(f)
else:
    signal_history = []

# Chỉ báo kỹ thuật
def compute_rsi(close, period=14):
    try:
        if len(close) < period:
            return 50
        delta = close.diff().dropna()
        if delta.empty:
            return 50
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan).ffill().bfill()
        return 100 - (100 / (1 + rs)).iloc[-1]
    except Exception as e:
        logger.error(f"RSI calculation error: {str(e)}")
        return 50

def compute_ma(close, period=20):
    return close.rolling(window=period, min_periods=1).mean().iloc[-1] if not close.empty else 0

def compute_macd(close, fast=12, slow=26, signal_period=9):
    try:
        if len(close) < slow:
            return 0, 0
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd.iloc[-1], signal_line.iloc[-1]
    except Exception as e:
        logger.error(f"MACD calculation error: {str(e)}")
        return 0, 0

def compute_bollinger_bands(close, period=20, std_dev=2):
    try:
        if len(close) < period:
            return 0, 0, 0
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return (
            sma.iloc[-1] + (std.iloc[-1] * std_dev),
            sma.iloc[-1],
            sma.iloc[-1] - (std.iloc[-1] * std_dev)
        )
    except Exception as e:
        logger.error(f"Bollinger Bands error: {str(e)}")
        return 0, 0, 0

def compute_adx(df, period=14):
    try:
        if len(df) < period * 2:
            return 0
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < 0) | (plus_dm > minus_dm)] = 0
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().bfill()
        plus_di = 100 * (plus_dm.rolling(period).mean()) / atr
        minus_di = 100 * (minus_dm.rolling(period).mean()) / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx.rolling(period).mean().iloc[-1]
    except Exception as e:
        logger.error(f"ADX calculation error: {str(e)}")
        return 0

def compute_atr(df, period=14):
    try:
        if len(df) < period:
            return 0
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
    except Exception as e:
        logger.error(f"ATR calculation error: {str(e)}")
        return 0

def compute_fibonacci(df):
    try:
        if df.empty:
            return {}
        high = df['high'].astype(float).max()
        low = df['low'].astype(float).min()
        diff = high - low
        return {
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50.0%': high - diff * 0.5,
            '61.8%': high - diff * 0.618
        }
    except Exception as e:
        logger.error(f"Fibonacci error: {str(e)}")
        return {}

def find_support_resistance(df):
    try:
        if len(df) < 20:
            return 0, 0
        return (
            df['low'].rolling(20).min().iloc[-1],
            df['high'].rolling(20).max().iloc[-1]
        )
    except Exception as e:
        logger.error(f"Support/Resistance error: {str(e)}")
        return 0, 0

# Lấy dữ liệu từ Binance
def fetch_klines(symbol, pair, timeframe, limit=100):
    try:
        symbol = symbol.upper()
        pair = pair.upper()
        response = requests.get(
            f"{BINANCE_API}/klines",
            params={
                'symbol': f"{symbol}{pair}",
                'interval': timeframe,
                'limit': limit
            },
            timeout=10
        )
        response.raise_for_status()
        df = pd.DataFrame(
            response.json(),
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
        )
        if df.empty:
            logger.warning(f"Empty data for {symbol}{pair} {timeframe}")
            return None
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=numeric_cols)
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
    return None

# Tự điều chỉnh ngưỡng dựa trên lịch sử
def adjust_thresholds():
    if not signal_history:
        return 30, 70, 25  # Mặc định: RSI Long < 30, RSI Short > 70, ADX > 25

    df = pd.DataFrame(signal_history)
    if len(df) < 10:  # Cần ít nhất 10 tín hiệu để điều chỉnh
        return 30, 70, 25

    long_signals = df[df['signal'] == 'LONG']
    short_signals = df[df['signal'] == 'SHORT']

    # Tính tỷ lệ thành công theo ngưỡng RSI
    rsi_long_success = long_signals.groupby(pd.cut(long_signals['rsi_short'], bins=[0, 25, 30, 35])).agg({'success': 'mean'}).idxmax()['success']
    rsi_short_success = short_signals.groupby(pd.cut(short_signals['rsi_short'], bins=[65, 70, 75, 100])).agg({'success': 'mean'}).idxmax()['success']
    adx_success = df.groupby(pd.cut(df['adx'], bins=[20, 25, 30, 35])).agg({'success': 'mean'}).idxmax()['success']

    rsi_long = (rsi_long_success.left + rsi_long_success.right) / 2 if rsi_long_success else 30
    rsi_short = (rsi_short_success.left + rsi_short_success.right) / 2 if rsi_short_success else 70
    adx_threshold = (adx_success.left + adx_success.right) / 2 if adx_success else 25

    return rsi_long, rsi_short, adx_threshold

# Phân tích kỹ thuật tối ưu
def get_crypto_analysis(symbol, pair, timeframe, chat_id=None):
    try:
        if timeframe not in timeframes:
            return "⚠️ Khung thời gian không hợp lệ"

        df_short = fetch_klines(symbol, pair, timeframe)
        if df_short is None or df_short.empty:
            return "❗ Không thể lấy dữ liệu hoặc cặp tiền không tồn tại"

        df_long = fetch_klines(symbol, pair, '4h')
        if df_long is None or df_long.empty:
            df_long = df_short.copy()

        if len(df_short) < 50:
            return f"❗ Không đủ dữ liệu (cần ít nhất 50 nến, hiện có {len(df_short)})"

        close = df_short['close']
        current_price = close.iloc[-1]
        volume = df_short['volume']

        # Tính toán các chỉ báo
        rsi_short = compute_rsi(close)
        ma_short = compute_ma(close, 10)
        ma_long = compute_ma(close, 50)
        macd, signal = compute_macd(close)
        upper_bb, sma_bb, lower_bb = compute_bollinger_bands(close)
        adx = compute_adx(df_short)
        atr = compute_atr(df_short)
        support, resistance = find_support_resistance(df_short)
        fib_levels = compute_fibonacci(df_short)

        rsi_long = compute_rsi(df_long['close'])
        macd_long, signal_long = compute_macd(df_long['close'])

        # Điều chỉnh ngưỡng từ lịch sử
        rsi_long_threshold, rsi_short_threshold, adx_threshold = adjust_thresholds()

        # Tạo tín hiệu giao dịch
        signal_text = ""
        entry = tp = sl = None
        details = []

        try:
            volume_avg_5 = volume.rolling(5).mean().iloc[-1]
            volume_avg_20 = volume.rolling(20).mean().iloc[-1]
            volume_trend = volume.iloc[-3:].is_monotonic_increasing
            volume_spike = volume.iloc[-1] > volume_avg_20 * 1.5
            ma_strength = (ma_short - ma_long) / ma_long
            fib_targets = sorted(fib_levels.values())
            sl_factor = 1.0 if adx < 35 else 1.5

            if adx < adx_threshold:
                signal_text = f"🟡 GIỮ - Thị trường sideway (ADX < {adx_threshold})"
                details.append(f"ADX: {adx:.1f}")
            else:
                # Tín hiệu LONG
                if (rsi_short < rsi_long_threshold and rsi_long < 50 and
                        ma_short > ma_long and ma_strength > 0.005 and
                        macd > signal and macd_long > signal_long and
                        current_price > lower_bb + atr * 0.5 and
                        (volume_trend or volume_spike)):

                    entry = current_price
                    tp = next((f for f in fib_targets if f > entry), entry + 2 * atr)
                    sl = support if support > 0 else entry - atr * sl_factor
                    signal_text = "🟢 LONG - Tín hiệu mua mạnh"
                    details.append(f"RSI ngắn: {rsi_short:.1f} (<{rsi_long_threshold:.1f}), RSI 4h: {rsi_long:.1f} (<50)")
                    details.append(f"MA10 cắt lên MA50 (sức mạnh: {ma_strength:.3f})")
                    details.append(f"MACD tăng (khung ngắn + 4h)")
                    details.append(f"Volume: {'tăng đều' if volume_trend else f'đột biến +{((volume.iloc[-1]/volume_avg_20 - 1) * 100):.1f}%'}")
                    details.append(f"Giá trên BB dưới + {atr * 0.5:.3f}")

                # Tín hiệu SHORT
                elif (rsi_short > rsi_short_threshold and rsi_long > 50 and
                      ma_short < ma_long and ma_strength < -0.005 and
                      macd < signal and macd_long < signal_long and
                      current_price < upper_bb - atr * 0.5 and
                      volume_spike):

                    entry = current_price
                    tp = next((f for f in fib_targets[::-1] if f < entry), entry - 2 * atr)
                    sl = resistance if resistance > 0 else entry + atr * sl_factor
                    signal_text = "🔴 SHORT - Tín hiệu bán mạnh"
                    details.append(f"RSI ngắn: {rsi_short:.1f} (>{rsi_short_threshold:.1f}), RSI 4h: {rsi_long:.1f} (>50)")
                    details.append(f"MA10 cắt xuống MA50 (sức mạnh: {ma_strength:.3f})")
                    details.append(f"MACD giảm (khung ngắn + 4h)")
                    details.append(f"Volume đột biến +{((volume.iloc[-1]/volume_avg_20 - 1) * 100):.1f}%")
                    details.append(f"Giá dưới BB trên - {atr * 0.5:.3f}")

                else:
                    signal_text = "⚪️ ĐỢI - Chưa có tín hiệu rõ ràng"
                    details.append(f"ADX: {adx:.1f}")
                    details.append(f"MA strength: {ma_strength:.3f}")
                    details.append(f"RSI ngắn: {rsi_short:.1f}")

            # Điều kiện thoát tín hiệu
            exit_signal = ""
            if signal_text.startswith("🟢") and rsi_short > 70:
                exit_signal = "⚠️ Thoát Long - RSI quá mua"
            elif signal_text.startswith("🔴") and rsi_short < 30:
                exit_signal = "⚠️ Thoát Short - RSI quá bán"

            # Lưu tín hiệu vào lịch sử (chưa có kết quả success)
            if entry and tp and sl and chat_id:
                signal_data = {
                    'chat_id': chat_id,
                    'symbol': symbol,
                    'pair': pair,
                    'timeframe': timeframe,
                    'signal': signal_text.split()[1],  # LONG hoặc SHORT
                    'entry': entry,
                    'tp': tp,
                    'sl': sl,
                    'rsi_short': rsi_short,
                    'adx': adx,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'success': None  # Chưa xác định
                }
                signal_history.append(signal_data)
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(signal_history, f)

        except Exception as e:
            logger.error(f"Signal generation error: {str(e)}")
            signal_text = "⚠️ Lỗi tạo tín hiệu"

        # Tạo báo cáo
        analysis = [
            f"📊 *Phân tích {symbol}/{pair} ({timeframes[timeframe]})*",
            f"💰 Giá hiện tại: {current_price:.4f}",
            f"📈 RSI: {rsi_short:.1f} (4h: {rsi_long:.1f})",
            f"📉 MA10/MA50: {ma_short:.4f}/{ma_long:.4f}",
            f"📊 Bollinger: {lower_bb:.4f} - {upper_bb:.4f}",
            f"📈 ADX: {adx:.1f}",
            f"⚡️ *{signal_text}*"
        ]

        if details:
            analysis.append("\n".join(["✓ " + d for d in details]))

        if entry and tp and sl:
            fib_nearest = min(fib_levels.items(), key=lambda x: abs(current_price - x[1]))[0]
            analysis.extend([
                f"\n📥 *Entry*: {entry:.4f}",
                f"🎯 *Take Profit*: {tp:.4f}",
                f"🛑 *Stop Loss*: {sl:.4f}",
                f"📏 Fibonacci gần nhất: {fib_nearest}",
                f"🏁 Hỗ trợ/Kháng cự: {support:.4f} | {resistance:.4f}"
            ])

        if exit_signal:
            analysis.append(f"\n{exit_signal}")

        analysis.append(f"\n🔧 Ngưỡng: RSI Long < {rsi_long_threshold:.1f}, RSI Short > {rsi_short_threshold:.1f}, ADX > {adx_threshold:.1f}")
        return "\n".join(analysis)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return "❗ Đã xảy ra lỗi trong quá trình phân tích"

# Bot Handlers
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    help_text = (
        "Chào mừng đến với Crypto Trading Bot!\n"
        "Các lệnh hỗ trợ:\n"
        "/analyze [symbol],[pair],[timeframe] - Phân tích kỹ thuật\n"
        "Ví dụ: /analyze btc,usdt,1h hoặc ?ada,usdt,15m\n"
        "/feedback [signal_id],[result] - Cập nhật kết quả tín hiệu\n"
        "Ví dụ: /feedback 0,win\n\n"
        "📊 Chỉ báo: RSI, MA, MACD, Bollinger, ADX, ATR, Fibonacci\n"
        "🔄 Tự học hỏi từ lịch sử tín hiệu"
    )
    bot.reply_to(message, help_text, parse_mode='Markdown')

@bot.message_handler(func=lambda message: message.text.startswith('?'))
def handle_question_mark(message):
    try:
        cleaned = message.text[1:].replace('.', ',')
        parts = cleaned.split(',')
        if len(parts) != 3:
            raise ValueError
        symbol, pair, timeframe = parts
        message.text = f"/analyze {symbol},{pair},{timeframe}"
        analyze(message)
    except:
        bot.reply_to(message, "⚠️ Sai định dạng! Ví dụ: ?ada,usdt,15m hoặc ?btc.usdt.4h")

@bot.message_handler(commands=['analyze'])
def analyze(message):
    try:
        _, params = message.text.split(' ', 1)
        parts = params.split(',')
        if len(parts) != 3:
            raise ValueError
        symbol, pair, timeframe = parts
        symbol = symbol.upper().strip()
        pair = pair.upper().strip()
        timeframe = timeframe.lower().strip()

        if not symbol.isalpha() or not pair.isalpha():
            return bot.reply_to(message, "⚠️ Tên symbol/pair chỉ được chứa chữ cái")
        if timeframe not in timeframes:
            return bot.reply_to(message, f"⚠️ Khung thời gian không hợp lệ. Chọn một trong:\n{', '.join(timeframes.keys())}")

        bot.send_chat_action(message.chat.id, 'typing')
        analysis = get_crypto_analysis(symbol, pair, timeframe, message.chat.id)
        bot.reply_to(message, analysis + f"\n📌 Signal ID: {len(signal_history) - 1}", parse_mode='Markdown')

    except ValueError:
        bot.reply_to(message, "⚠️ Sai định dạng! Ví dụ: /analyze btc,usdt,1h")
    except Exception as e:
        logger.error(f"Analyze command error: {str(e)}")
        bot.reply_to(message, "⚠️ Lỗi hệ thống! Vui lòng thử lại sau")

@bot.message_handler(commands=['feedback'])
def feedback(message):
    try:
        _, params = message.text.split(' ', 1)
        signal_id, result = params.split(',')
        signal_id = int(signal_id)
        result = result.lower().strip()

        if signal_id < 0 or signal_id >= len(signal_history):
            return bot.reply_to(message, "⚠️ Signal ID không hợp lệ!")
        if result not in ['win', 'loss']:
            return bot.reply_to(message, "⚠️ Kết quả phải là 'win' hoặc 'loss'!")

        signal_history[signal_id]['success'] = (result == 'win')
        with open(HISTORY_FILE, 'w') as f:
            json.dump(signal_history, f)

        rsi_long, rsi_short, adx = adjust_thresholds()
        bot.reply_to(message, f"Đã cập nhật tín hiệu #{signal_id}: {result}\nNgưỡng mới: RSI Long < {rsi_long:.1f}, RSI Short > {rsi_short:.1f}, ADX > {adx:.1f}")

    except ValueError:
        bot.reply_to(message, "⚠️ Sai định dạng! Ví dụ: /feedback 0,win")
    except Exception as e:
        logger.error(f"Feedback command error: {str(e)}")
        bot.reply_to(message, "⚠️ Lỗi hệ thống! Vui lòng thử lại sau")

if __name__ == "__main__":
    logger.info("Khởi động bot...")
    bot.polling(none_stop=True)