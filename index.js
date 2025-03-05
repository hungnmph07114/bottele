const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR, PSAR, StochasticRSI } = require('technicalindicators');
const tf = require('@tensorflow/tfjs');

// Configuration
const TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY'; // Thay bằng token thực tế của bạn
const BINANCE_API = 'https://api.binance.com/api/v3';
const bot = new TelegramBot(TOKEN, { polling: true });

const timeframes = { '1m': '1 phút', '5m': '5 phút', '15m': '15 phút', '1h': '1 giờ', '4h': '4 giờ', '1d': '1 ngày' };

// Hàm lấy dữ liệu nến từ Binance
async function fetchKlines(symbol, pair, timeframe, limit = 200) {
    try {
        const response = await axios.get(`${BINANCE_API}/klines`, {
            params: { symbol: `${symbol.toUpperCase()}${pair.toUpperCase()}`, interval: timeframe, limit }
        });
        return response.data.map(candle => ({
            time: candle[0], open: parseFloat(candle[1]), high: parseFloat(candle[2]),
            low: parseFloat(candle[3]), close: parseFloat(candle[4]), volume: parseFloat(candle[5])
        }));
    } catch (error) {
        console.error('❌ Lỗi lấy dữ liệu nến:', error.message);
        return null;
    }
}

// Khởi tạo mô hình AI (LSTM)
let model;
async function initializeModel() {
    model = tf.sequential();
    model.add(tf.layers.lstm({ units: 64, returnSequences: true, inputShape: [10, 7] }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 32 }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    console.log('✅ Mô hình AI (LSTM) đã được khởi tạo');
}

// Hàm phân tích dữ liệu
async function getCryptoAnalysis(symbol, pair, timeframe) {
    const df = await fetchKlines(symbol, pair, timeframe);
    if (!df) return '❗ Không thể lấy dữ liệu';

    const close = df.map(d => d.close);
    const currentPrice = close[close.length - 1];
    const rsi = computeRSI(close);
    const stochasticRsi = StochasticRSI.calculate({ values: close, period: 14 })[0];
    const ma10 = computeMA(close, 10);
    const ma50 = computeMA(close, 50);
    const ma200 = computeMA(close, 200);
    const [, , histogram] = computeMACD(close);
    const [lowerBB, middleBB, upperBB] = computeBollingerBands(close);
    const adx = computeADX(df);
    const atr = computeATR(df);
    const parabolicSAR = PSAR.calculate({ high: df.map(d => d.high), low: df.map(d => d.low) });
    const lastSAR = parabolicSAR[parabolicSAR.length - 1];
    const trendDirection = currentPrice > lastSAR ? 'Xu hướng tăng' : 'Xu hướng giảm';
    const support = Math.min(...close.slice(-10));
    const resistance = Math.max(...close.slice(-10));

    if (!model) return '❗ Mô hình AI chưa được khởi tạo';

    const input = tf.tensor3d([[df.slice(-10).map(d => [rsi, adx, histogram, ma10 - ma50, currentPrice - middleBB, lastSAR])]]);
    const prediction = model.predict(input);
    const [longProb, shortProb, waitProb] = prediction.dataSync();
    input.dispose();
    prediction.dispose();

    let signalText = '⚪️ ĐỢI';
    let signalEmoji = '⚪️';
    if (longProb > shortProb && longProb > waitProb) {
        signalText = '🟢 LONG - Mua';
        signalEmoji = '🟢';
    } else if (shortProb > longProb && shortProb > waitProb) {
        signalText = '🔴 SHORT - Bán';
        signalEmoji = '🔴';
    }

    const confidence = Math.max(longProb, shortProb, waitProb) * 100;
    const entryPrice = currentPrice;
    const stopLoss = signalText.includes('LONG') ? currentPrice - 1.5 * atr : currentPrice + 1.5 * atr;
    const takeProfit = signalText.includes('LONG') ? currentPrice + 2.5 * atr : currentPrice - 2.5 * atr;

    return `📊 *Phân tích ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})*
💰 Giá: ${currentPrice.toFixed(4)}
⚡️ *${signalEmoji} ${signalText}*
📈 RSI: ${rsi.toFixed(1)}
📉 Stoch RSI: ${stochasticRsi.toFixed(1)}
📊 MACD: ${histogram.toFixed(4)}
📉 ADX: ${adx.toFixed(1)}
📦 Volume: ${df[df.length - 1].volume > df[df.length - 2].volume ? 'TĂNG ĐỘT BIẾN' : 'BÌNH THƯỜNG'}
📏 Bollinger: ${lowerBB.toFixed(4)} - ${upperBB.toFixed(4)}
📉 SMA 200: ${ma200.toFixed(4)}
📊 Parabolic SAR: ${lastSAR.toFixed(4)} (${trendDirection})
🛡️ Hỗ trợ: ${support.toFixed(4)}, Kháng cự: ${resistance.toFixed(4)}
🤖 AI Signal: ${signalText} (${confidence.toFixed(0)}%)
✅ Độ tin cậy kết hợp: ${confidence.toFixed(0)}%
🎯 Điểm vào: ${entryPrice.toFixed(4)}
🛑 SL: ${stopLoss.toFixed(4)}
💰 TP: ${takeProfit.toFixed(4)}
${confidence > 80 ? '🚨 CẢNH BÁO: Tín hiệu mạnh!' : ''}`;
}

// Bot Telegram xử lý lệnh người dùng
(async () => {
    await initializeModel();
    bot.onText(/\?(.+)/, async (msg, match) => {
        const parts = match[1].split(',');
        if (parts.length < 3) return bot.sendMessage(msg.chat.id, '⚠️ Sai định dạng! VD: ?ada,usdt,15m');
        const [symbol, pair, timeframe] = parts.map(p => p.trim().toLowerCase());
        if (!timeframes[timeframe]) return bot.sendMessage(msg.chat.id, '⚠️ Khung thời gian không hợp lệ');

        const result = await getCryptoAnalysis(symbol, pair, timeframe);
        bot.sendMessage(msg.chat.id, result, { parse_mode: 'Markdown' });
    });

    console.log('✅ Bot đang chạy...');
})();