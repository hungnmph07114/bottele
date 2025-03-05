const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR } = require('technicalindicators');
const tf = require('@tensorflow/tfjs');

// Configuration
const TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY'; // Thay bằng token thực tế của bạn
const BINANCE_API = 'https://api.binance.com/api/v3';
const bot = new TelegramBot(TOKEN, { polling: true });

const timeframes = { '1m': '1 phút', '5m': '5 phút', '15m': '15 phút', '1h': '1 giờ', '4h': '4 giờ', '1d': '1 ngày' };

// Khởi tạo mô hình TensorFlow.js
let model;
async function initializeModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 50, activation: 'relu', inputShape: [6] }));
    model.add(tf.layers.dense({ units: 20, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    console.log('✅ Mô hình TensorFlow.js đã được khởi tạo');
}

// Huấn luyện mô hình với nhãn từ logic rule-based
async function trainModel(data) {
    const inputs = [];
    const outputs = [];
    for (let i = 1; i < data.length; i++) {
        const curr = data[i];
        const close = data.slice(0, i).map(d => d.close);
        const volume = data.slice(0, i).map(d => d.volume);
        const rsi = computeRSI(close);
        const ma10 = computeMA(close, 10);
        const ma50 = computeMA(close, 50);
        const [, , histogram] = computeMACD(close);
        const [, middleBB] = computeBollingerBands(close);
        const adx = computeADX(data.slice(0, i));
        const volumeMA = computeMA(volume, 20);
        const volumeSpike = curr.volume > volumeMA * 1.5 ? 1 : 0;

        inputs.push([rsi, adx, histogram, volumeSpike, ma10 - ma50, curr.close - middleBB]);

        // Nhãn từ logic rule-based
        let signal = [0, 0, 1]; // ĐỢI mặc định
        if (adx > 30) {
            if (rsi < 30 && ma10 > ma50 && histogram > 0 && volumeSpike && curr.close < middleBB) signal = [1, 0, 0]; // LONG mạnh
            else if (rsi > 70 && ma10 < ma50 && histogram < 0 && volumeSpike && curr.close > middleBB) signal = [0, 1, 0]; // SHORT mạnh
        }
        outputs.push(signal);
    }

    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(outputs);
    await model.fit(xs, ys, { epochs: 20, batchSize: 32, shuffle: true });
    console.log('✅ Mô hình đã được huấn luyện với logic rule-based');
    xs.dispose();
    ys.dispose();
}

async function fetchKlines(symbol, pair, timeframe, limit = 200) {
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
    const support = Math.min(...lows);
    const resistance = Math.max(...highs);
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
    const volumeSpike = volume[volume.length - 1] > volumeMA * 1.5 ? 1 : 0;
    const { support, resistance } = computeSupportResistance(df);

    // Ngưỡng tùy chỉnh
    const rsiOverbought = customThresholds.rsiOverbought || 70;
    const rsiOversold = customThresholds.rsiOversold || 30;
    const adxStrongTrend = customThresholds.adxStrongTrend || 30;
    const adxWeakTrend = customThresholds.adxWeakTrend || 20;

    // Dự đoán AI
    const input = tf.tensor2d([[rsi, adx, histogram, volumeSpike, ma10 - ma50, currentPrice - middleBB]]);
    const prediction = model.predict(input);
    const [longProb, shortProb, waitProb] = prediction.dataSync();
    input.dispose();
    prediction.dispose();

    let signalText, confidence, entry = 0, sl = 0, tp = 0;
    const maxProb = Math.max(longProb, shortProb, waitProb);
    confidence = Math.round(maxProb * 100);

    // Logic rule-based để xác nhận
    let ruleBasedSignal = '⚪️ ĐỢI - Chưa có tín hiệu';
    let ruleConfidence = 30;
    if (adx > adxStrongTrend) {
        if (rsi < rsiOversold && ma10 > ma50 && histogram > 0 && volumeSpike && currentPrice < middleBB) {
            ruleBasedSignal = '🟢 LONG - Mua mạnh';
            ruleConfidence = 90;
        } else if (rsi > rsiOverbought && ma10 < ma50 && histogram < 0 && volumeSpike && currentPrice > middleBB) {
            ruleBasedSignal = '🔴 SHORT - Bán mạnh';
            ruleConfidence = 90;
        } else if (rsi < rsiOversold && ma10 > ma50 && histogram > 0) {
            ruleBasedSignal = '🟢 LONG - Mua (chưa xác nhận volume)';
            ruleConfidence = 60;
        } else if (rsi > rsiOverbought && ma10 < ma50 && histogram < 0) {
            ruleBasedSignal = '🔴 SHORT - Bán (chưa xác nhận volume)';
            ruleConfidence = 60;
        }
    } else if (adx > adxWeakTrend && adx <= adxStrongTrend) {
        if (rsi < rsiOversold && ma10 > ma50 && histogram > 0) {
            ruleBasedSignal = '🟢 LONG SỚM - Xu hướng tăng tiềm năng';
            ruleConfidence = 50;
        } else if (rsi > rsiOverbought && ma10 < ma50 && histogram < 0) {
            ruleBasedSignal = '🔴 SHORT SỚM - Xu hướng giảm tiềm năng';
            ruleConfidence = 50;
        }
    } else if (adx < adxWeakTrend) {
        if (currentPrice <= lowerBB && rsi < rsiOversold) {
            ruleBasedSignal = '🟢 LONG NGẮN - Giá chạm đáy Bollinger';
            ruleConfidence = 70;
        } else if (currentPrice >= upperBB && rsi > rsiOverbought) {
            ruleBasedSignal = '🔴 SHORT NGẮN - Giá chạm đỉnh Bollinger';
            ruleConfidence = 70;
        }
    }

    // Kết hợp AI và Rule-Based
    if (maxProb === longProb) {
        signalText = '🟢 LONG - Mua';
        entry = currentPrice;
        sl = Math.max(currentPrice - atr * 2, support);
        tp = Math.min(currentPrice + atr * 4, resistance);
        if (ruleBasedSignal.includes('LONG')) confidence = Math.max(confidence, ruleConfidence); // Lấy confidence cao hơn
    } else if (maxProb === shortProb) {
        signalText = '🔴 SHORT - Bán';
        entry = currentPrice;
        sl = Math.min(currentPrice + atr * 2, resistance);
        tp = Math.max(currentPrice - atr * 4, support);
        if (ruleBasedSignal.includes('SHORT')) confidence = Math.max(confidence, ruleConfidence);
    } else {
        signalText = '⚪️ ĐỢI - Chưa có tín hiệu';
        confidence = Math.min(confidence, ruleConfidence); // Confidence thấp nếu ĐỢI
    }

    // Chi tiết kết quả
    let details = [
        `📈 RSI: ${rsi.toFixed(1)}`,
        `📊 MACD: ${macd.toFixed(4)} / ${signal.toFixed(4)}`,
        `📉 ADX: ${adx.toFixed(1)}`,
        `📦 Volume: ${volumeSpike ? 'TĂNG ĐỘT BIẾN' : 'BÌNH THƯỜNG'}`
    ];
    details.push(`📏 Bollinger: ${lowerBB.toFixed(4)} - ${upperBB.toFixed(4)}`);
    details.push(`🛡️ Hỗ trợ: ${support.toFixed(4)}, Kháng cự: ${resistance.toFixed(4)}`);
    details.push(`🤖 AI Signal: ${maxProb === longProb ? 'LONG' : maxProb === shortProb ? 'SHORT' : 'ĐỢI'} (${confidence}%)`);
    details.push(`📏 Rule Signal: ${ruleBasedSignal.split(' - ')[1] || 'Chưa có tín hiệu'} (${ruleConfidence}%)`);
    if (signalText !== '⚪️ ĐỢI - Chưa có tín hiệu') {
        details.push(`✅ Độ tin cậy kết hợp: ${confidence}%`);
        details.push(`🎯 Điểm vào: ${entry.toFixed(4)}`);
        details.push(`🛑 SL: ${sl.toFixed(4)}`);
        details.push(`💰 TP: ${tp.toFixed(4)}`);
    }

    const result = `📊 *Phân tích ${symbol}/${pair} (${timeframes[timeframe]})*\n💰 Giá: ${currentPrice.toFixed(4)}\n⚡️ *${signalText}*\n${details.join('\n')}`;
    return { result, confidence };
}

// Khởi động bot và huấn luyện mô hình
(async () => {
    await initializeModel();
    const initialData = await fetchKlines('BTC', 'USDT', '1h', 200);
    if (initialData) await trainModel(initialData);

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

        if (confidence > 80) {
            bot.sendMessage(msg.chat.id, `🚨 *CẢNH BÁO* 🚨\n${result}`, { parse_mode: 'Markdown' });
        }
    });

    console.log('✅ Bot đang chạy...');
})();