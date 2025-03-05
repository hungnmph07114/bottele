const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR } = require('technicalindicators');

// Configuration
const TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY'; // Thay bằng token thực tế của bạn
const BINANCE_API = 'https://api.binance.com/api/v3';
const bot = new TelegramBot(TOKEN, { polling: true });

const timeframes = { '1m': '1 phút', '5m': '5 phút', '15m': '15 phút', '1h': '1 giờ', '4h': '4 giờ', '1d': '1 ngày' };

async function fetchKlines(symbol, pair, timeframe, limit = 100) {
    try {
        const response = await axios.get(`${BINANCE_API}/klines`, {
            params: { symbol: `${symbol.toUpperCase()}${pair.toUpperCase()}`, interval: timeframe, limit },
            timeout: 10000,
        });
        return response.data.map(d => ({
            timestamp: d[0], open: parseFloat(d[1]), high: parseFloat(d[2]),
            low: parseFloat(d[3]), close: parseFloat(d[4]), volume: parseFloat(d[5])
        }));
    } catch (error) {
        console.error(`API Error: ${error.message}`);
        return null;
    }
}

function computeRSI(close, period = 14) { return RSI.calculate({ values: close, period }).slice(-1)[0] || 50; }
function computeMA(close, period = 20) { return SMA.calculate({ values: close, period }).slice(-1)[0] || 0; }
function computeMACD(close) {
    const result = MACD.calculate({ values: close, fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 });
    const last = result.slice(-1)[0];
    return [last?.MACD || 0, last?.signal || 0, last?.histogram || 0];
}
function computeBollingerBands(close, period = 20, stdDev = 2) {
    const result = BollingerBands.calculate({ values: close, period, stdDev });
    const last = result.slice(-1)[0];
    return [last?.upper || 0, last?.middle || 0, last?.lower || 0];
}
function computeADX(data, period = 14) {
    const result = ADX.calculate({ high: data.map(d => d.high), low: data.map(d => d.low), close: data.map(d => d.close), period });
    return result.slice(-1)[0]?.adx || 0;
}
function computeATR(data, period = 14) {
    const result = ATR.calculate({ high: data.map(d => d.high), low: data.map(d => d.low), close: data.map(d => d.close), period });
    return result.slice(-1)[0] || 0;
}

function computeSupportResistance(data) {
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const support = Math.min(...lows); // Mức thấp nhất
    const resistance = Math.max(...highs); // Mức cao nhất
    return { support, resistance };
}

async function getCryptoAnalysis(symbol, pair, timeframe, customThresholds = {}) {
    const df = await fetchKlines(symbol, pair, timeframe);
    if (!df) return '❗ Không thể lấy dữ liệu';

    const close = df.map(d => d.close);
    const volume = df.map(d => d.volume);
    const currentPrice = close[close.length - 1];
    const rsi = computeRSI(close);
    const ma10 = computeMA(close, 10);
    const ma50 = computeMA(close, 50);
    const [macd, signal, histogram] = computeMACD(close);
    const [upperBB, middleBB, lowerBB] = computeBollingerBands(close);
    const adx = computeADX(df);
    const atr = computeATR(df);
    const volumeMA = computeMA(volume, 20);
    const volumeSpike = volume[volume.length - 1] > volumeMA * 1.5;
    const { support, resistance } = computeSupportResistance(df);

    // Ngưỡng mặc định hoặc tùy chỉnh
    const rsiOverbought = customThresholds.rsiOverbought || 70;
    const rsiOversold = customThresholds.rsiOversold || 30;
    const adxStrongTrend = customThresholds.adxStrongTrend || 30;
    const adxWeakTrend = customThresholds.adxWeakTrend || 20;

    let signalText = '⚪️ ĐỢI - Chưa có tín hiệu';
    let confidence = 0;
    let entry = 0, sl = 0, tp = 0;
    let details = [
        `RSI: ${rsi.toFixed(1)} (Overbought: ${rsiOverbought}, Oversold: ${rsiOversold})`,
        `MACD: ${macd.toFixed(4)} / ${signal.toFixed(4)}`,
        `ADX: ${adx.toFixed(1)}`,
        `Volume: ${volumeSpike ? 'TĂNG ĐỘT BIẾN' : 'BÌNH THƯỜNG'}`
    ];

    if (adx > adxStrongTrend) {
        if (rsi < rsiOversold && ma10 > ma50 && histogram > 0 && volumeSpike && currentPrice < middleBB) {
            signalText = '🟢 LONG - Mua mạnh';
            confidence = 90;
            entry = currentPrice;
            sl = Math.max(currentPrice - atr * 2, support); // SL không dưới support
            tp = Math.min(currentPrice + atr * 4, resistance); // TP không vượt resistance
        } else if (rsi > rsiOverbought && ma10 < ma50 && histogram < 0 && volumeSpike && currentPrice > middleBB) {
            signalText = '🔴 SHORT - Bán mạnh';
            confidence = 90;
            entry = currentPrice;
            sl = Math.min(currentPrice + atr * 2, resistance); // SL không vượt resistance
            tp = Math.max(currentPrice - atr * 4, support); // TP không dưới support
        }
        // Tín hiệu yếu hơn nếu thiếu volume hoặc Bollinger Bands
        else if (rsi < rsiOversold && ma10 > ma50 && histogram > 0) {
            signalText = '🟢 LONG - Mua (chưa xác nhận volume)';
            confidence = 60;
            entry = currentPrice;
            sl = Math.max(currentPrice - atr * 2, support);
            tp = Math.min(currentPrice + atr * 3, resistance);
        } else if (rsi > rsiOverbought && ma10 < ma50 && histogram < 0) {
            signalText = '🔴 SHORT - Bán (chưa xác nhận volume)';
            confidence = 60;
            entry = currentPrice;
            sl = Math.min(currentPrice + atr * 2, resistance);
            tp = Math.max(currentPrice - atr * 3, support);
        }
    } else if (adx > adxWeakTrend && adx <= adxStrongTrend) {
        if (rsi < rsiOversold && ma10 > ma50 && histogram > 0) {
            signalText = '🟢 LONG SỚM - Xu hướng tăng tiềm năng';
            confidence = 50;
            entry = currentPrice;
            sl = Math.max(currentPrice - atr * 1.5, support);
            tp = Math.min(currentPrice + atr * 3, resistance);
        } else if (rsi > rsiOverbought && ma10 < ma50 && histogram < 0) {
            signalText = '🔴 SHORT SỚM - Xu hướng giảm tiềm năng';
            confidence = 50;
            entry = currentPrice;
            sl = Math.min(currentPrice + atr * 1.5, resistance);
            tp = Math.max(currentPrice - atr * 3, support);
        }
    } else if (adx < adxWeakTrend) {
        if (currentPrice <= lowerBB && rsi < rsiOversold) {
            signalText = '🟢 LONG NGẮN - Giá chạm đáy Bollinger';
            confidence = 70;
            entry = currentPrice;
            sl = lowerBB - atr * 0.5;
            tp = middleBB;
        } else if (currentPrice >= upperBB && rsi > rsiOverbought) {
            signalText = '🔴 SHORT NGẮN - Giá chạm đỉnh Bollinger';
            confidence = 70;
            entry = currentPrice;
            AscendingDescendingOrder = true;
            sl = upperBB + atr * 0.5;
            tp = middleBB;
        } else {
            signalText = '🟡 GIỮ - Thị trường sideway';
            confidence = 30;
        }
    }

    details.push(`Bollinger: ${lowerBB.toFixed(4)} - ${upperBB.toFixed(4)}`);
    details.push(`Hỗ trợ: ${support.toFixed(4)}, Kháng cự: ${resistance.toFixed(4)}`);
    if (confidence > 0) {
        details.push(`Độ tin cậy: ${confidence}%`);
        details.push(`Điểm vào: ${entry.toFixed(4)}`);
        details.push(`SL: ${sl.toFixed(4)}`);
        details.push(`TP: ${tp.toFixed(4)}`);
    }

    const result = `📊 *Phân tích ${symbol}/${pair} (${timeframes[timeframe]})*
💰 Giá: ${currentPrice.toFixed(4)}
⚡️ *${signalText}*
${details.join('\n')}`;

    return { result, confidence, chatId: null }; // Trả về object để xử lý cảnh báo sau
}

bot.onText(/\?(.+)/, async (msg, match) => {
    const parts = match[1].split(',');
    if (parts.length < 3) return bot.sendMessage(msg.chat.id, '⚠️ Sai định dạng! VD: ?ada,usdt,15m hoặc ?ada,usdt,15m,rsi25-75');

    const [symbol, pair, timeframe, customThreshold] = parts.map(p => p.trim().toLowerCase());
    if (!timeframes[timeframe]) return bot.sendMessage(msg.chat.id, '⚠️ Khung thời gian không hợp lệ');

    let customThresholds = {};
    if (customThreshold && customThreshold.startsWith('rsi')) {
        const [oversold, overbought] = customThreshold.replace('rsi', '').split('-').map(Number);
        if (!isNaN(oversold) && !isNaN(overbought) && oversold < overbought) {
            customThresholds.rsiOversold = oversold;
            customThresholds.rsiOverbought = overbought;
        } else {
            return bot.sendMessage(msg.chat.id, '⚠️ Định dạng RSI không hợp lệ! VD: rsi25-75');
        }
    }

    const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe, customThresholds);
    bot.sendMessage(msg.chat.id, result, { parse_mode: 'Markdown' });

    // Gửi cảnh báo nếu độ tin cậy > 80%
    if (confidence > 80) {
        bot.sendMessage(msg.chat.id, `🚨 *CẢNH BÁO* 🚨\n${result}`, { parse_mode: 'Markdown' });
    }
});

console.log('✅ Bot đang chạy...');