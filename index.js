const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR } = require('technicalindicators');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const sqlite3 = require('sqlite3').verbose();

// Configuration
const TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY';
const BINANCE_API = 'https://api.binance.com/api/v3';
let bot;

// Timeframes (hỗ trợ cả 15m và m15, 1h và h1, v.v.)
const timeframes = {
    '1m': '1 phút', 'm1': '1 phút',
    '5m': '5 phút', 'm5': '5 phút',
    '15m': '15 phút', 'm15': '15 phút',
    '30m': '30 phút', 'm30': '30 phút',
    '1h': '1 giờ', 'h1': '1 giờ',
    '2h': '2 giờ', 'h2': '2 giờ',
    '4h': '4 giờ', 'h4': '4 giờ',
    '6h': '6 giờ', 'h6': '6 giờ',
    '8h': '8 giờ', 'h8': '8 giờ',
    '12h': '12 giờ', 'h12': '12 giờ',
    '1d': '1 ngày', 'd1': '1 ngày',
    '3d': '3 ngày', 'd3': '3 ngày',
    '1w': '1 tuần', 'w1': '1 tuần',
    '1M': '1 tháng', 'M1': '1 tháng'
};

// Chuyển đổi định dạng timeframe về chuẩn Binance API
function normalizeTimeframe(tf) {
    const mapping = {
        'm1': '1m', '1m': '1m',
        'm5': '5m', '5m': '5m',
        'm15': '15m', '15m': '15m',
        'm30': '30m', '30m': '30m',
        'h1': '1h', '1h': '1h',
        'h2': '2h', '2h': '2h',
        'h4': '4h', '4h': '4h',
        'h6': '6h', '6h': '6h',
        'h8': '8h', '8h': '8h',
        'h12': '12h', '12h': '12h',
        'd1': '1d', '1d': '1d',
        'd3': '3d', '3d': '3d',
        'w1': '1w', '1w': '1w',
        'M1': '1M', '1M': '1M'
    };
    return mapping[tf] || null;
}

// Khởi tạo cơ sở dữ liệu SQLite
const db = new sqlite3.Database('bot.db', (err) => {
    if (err) {
        console.error(err.message);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi kết nối SQLite: ${err.message}\n`);
    } else {
        console.log('✅ Đã kết nối với cơ sở dữ liệu SQLite.');
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Đã kết nối với cơ sở dữ liệu SQLite.\n`);
    }
});

db.serialize(() => {
    db.run(`
        CREATE TABLE IF NOT EXISTS watch_configs (
            chatId INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            pair TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            PRIMARY KEY (chatId, symbol, pair, timeframe)
        )
    `, (err) => {
        if (err) {
            console.error("Lỗi tạo bảng watch_configs:", err.message);
            fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi tạo bảng watch_configs: ${err.message}\n`);
        }
    });
});

const insertStmt = db.prepare(`
    INSERT OR IGNORE INTO watch_configs (chatId, symbol, pair, timeframe)
    VALUES (?, ?, ?, ?)
`);
const deleteStmt = db.prepare(`
    DELETE FROM watch_configs
    WHERE chatId = ? AND symbol = ? AND pair = ? AND timeframe = ?
`);

// Khởi tạo mô hình TensorFlow.js
let model;
async function initializeModel() {
    model = tf.sequential();
    model.add(tf.layers.lstm({ units: 64, returnSequences: false, inputShape: [1, 10] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    console.log('✅ Mô hình TensorFlow.js đã được khởi tạo mới (LSTM)');
    fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Mô hình TensorFlow.js đã được khởi tạo mới (LSTM)\n`);
}

// Huấn luyện mô hình ban đầu
async function trainModel(data) {
    const inputs = [];
    const outputs = [];
    const atrPeriod = 14;
    for (let i = atrPeriod + 20; i < data.length; i++) {
        const curr = data[i];
        const close = data.slice(0, i).map(d => d.close);
        const volume = data.slice(0, i).map(d => d.volume);

        const rsi = computeRSI(close);
        const ma10 = computeMA(close, 10);
        const ma50 = computeMA(close, 50);
        const [, , histogram] = computeMACD(close);
        const [, middleBB] = computeBollingerBands(close);
        const adx = computeADX(data.slice(0, i));
        const atr = computeATR(data.slice(i - atrPeriod, i));
        const volumeMA = computeMA(volume, 20);
        const volumeSpike = curr.volume > volumeMA * 1.5 ? 1 : 0;

        const normalizedRsi = rsi / 100;
        const normalizedAdx = adx / 100;
        const normalizedHistogram = histogram / 1000;
        const normalizedMaDiff = (ma10 - ma50) / curr.close;
        const normalizedBbDiff = (curr.close - middleBB) / curr.close;
        const normalizedAtr = atr / curr.close;
        const closeLag1 = data[i - 1].close / curr.close;
        const closeLag2 = data[i - 2].close / curr.close;
        const rsiRollingMean = SMA.calculate({ values: close.slice(-5).map(d => computeRSI([d])), period: 5 })[0] / 100 || normalizedRsi;

        inputs.push([[normalizedRsi, normalizedAdx, normalizedHistogram, volumeSpike, normalizedMaDiff, normalizedBbDiff, closeLag1, closeLag2, normalizedAtr, rsiRollingMean]]);
        let signal = [0, 0, 1]; // WAIT
        if (adx > 30) {
            if (rsi < 30 && ma10 > ma50 && histogram > 0 && volumeSpike && curr.close < middleBB) signal = [1, 0, 0];
            else if (rsi > 70 && ma10 < ma50 && histogram < 0 && volumeSpike && curr.close > middleBB) signal = [0, 1, 0];
        }
        outputs.push(signal);
    }

    const xs = tf.tensor3d(inputs);
    const ys = tf.tensor2d(outputs);
    await model.fit(xs, ys, { epochs: 10, batchSize: 32, shuffle: true });
    console.log('✅ Mô hình đã được huấn luyện với logic rule-based (LSTM)');
    fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Mô hình đã được huấn luyện với logic rule-based (LSTM)\n`);
    xs.dispose();
    ys.dispose();
}

// Kiểm tra cặp giao dịch hợp lệ
async function isValidMarket(symbol, pair) {
    try {
        const response = await axios.get(`${BINANCE_API}/ticker/price`, {
            params: { symbol: `${symbol.toUpperCase()}${pair.toUpperCase()}` },
            timeout: 5000,
        });
        return !!response.data.price;
    } catch (error) {
        console.error(`API Error: ${error.message}`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - API Error: ${error.message}\n`);
        return false;
    }
}

// Lấy dữ liệu klines
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
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - API Error: ${error.message}\n`);
        return null;
    }
}

// Lấy dữ liệu lịch sử cho giả lập (giảm số cây nến xuống 200)
async function fetchHistoricalData(symbol, pair, timeframe, limit = 200) {
    return await fetchKlines(symbol, pair, timeframe, limit);
}

// Các hàm tính chỉ báo kỹ thuật
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

// Phân tích giao dịch
async function getCryptoAnalysis(symbol, pair, timeframe, customThresholds = {}, data = null) {
    const df = data || await fetchKlines(symbol, pair, timeframe);
    if (!df) return { result: '❗ Không thể lấy dữ liệu', confidence: 0 };

    const close = df.map(d => d.close);
    const volume = df.map(d => d.volume);
    const currentPrice = close[close.length - 1];
    const rsi = computeRSI(close);
    const ma10 = computeMA(close, 10);
    const ma50 = computeMA(close, 50);
    const [macd, signal, histogram] = computeMACD(close);
    const [upperBB, middleBB, lowerBB] = computeBollingerBands(close);
    const adx = computeADX(df);
    const atr = computeATR(df.slice(-14));
    const volumeMA = computeMA(volume, 20);
    const volumeSpike = volume[volume.length - 1] > volumeMA * 1.5 ? 1 : 0;
    const { support, resistance } = computeSupportResistance(df.slice(-50));

    const bbWidth = upperBB - lowerBB;
    const avgBBWidth = computeMA(df.map(d => BollingerBands.calculate({ values: df.map(v => v.close), period: 20, stdDev: 2 }).slice(-1)[0].upper - BollingerBands.calculate({ values: df.map(v => v.close), period: 20, stdDev: 2 }).slice(-1)[0].lower), 20);
    const isSideways = adx < 20 && bbWidth < avgBBWidth * 0.8;

    const rsiOverbought = customThresholds.rsiOverbought || 70;
    const rsiOversold = customThresholds.rsiOversold || 30;
    const adxStrongTrend = customThresholds.adxStrongTrend || 30;

    const normalizedRsi = rsi / 100;
    const normalizedAdx = adx / 100;
    const normalizedHistogram = histogram / 1000;
    const normalizedMaDiff = (ma10 - ma50) / currentPrice;
    const normalizedBbDiff = (currentPrice - middleBB) / currentPrice;
    const normalizedAtr = atr / currentPrice;
    const closeLag1 = df[df.length - 2]?.close / currentPrice || 1;
    const closeLag2 = df[df.length - 3]?.close / currentPrice || 1;
    const rsiRollingMean = SMA.calculate({ values: close.slice(-5).map(d => computeRSI([d])), period: 5 })[0] / 100 || normalizedRsi;

    const input = tf.tensor3d([[[normalizedRsi, normalizedAdx, normalizedHistogram, volumeSpike, normalizedMaDiff, normalizedBbDiff, closeLag1, closeLag2, normalizedAtr, rsiRollingMean]]]);
    const prediction = model.predict(input);
    const [longProb, shortProb, waitProb] = prediction.dataSync();
    input.dispose();
    prediction.dispose();

    let signalText, confidence, entry = 0, sl = 0, tp = 0;
    const maxProb = Math.max(longProb, shortProb, waitProb);
    confidence = Math.round(maxProb * 100);

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
    }

    const combinedConfidence = (confidence * 0.7 + ruleConfidence * 0.3);
    confidence = Math.round(combinedConfidence);

    const atrMultiplier = 2;
    const stopLossLong = currentPrice - atr * atrMultiplier;
    const stopLossShort = currentPrice + atr * atrMultiplier;
    const takeProfitLong = currentPrice + atr * atrMultiplier * 2;
    const takeProfitShort = currentPrice - atr * atrMultiplier * 2;

    if (maxProb === longProb || ruleBasedSignal.includes('LONG')) {
        signalText = '🟢 LONG - Mua';
        entry = currentPrice;
        sl = stopLossLong;
        tp = takeProfitLong;
        confidence = Math.max(confidence, ruleConfidence);
    } else if (maxProb === shortProb || ruleBasedSignal.includes('SHORT')) {
        signalText = '🔴 SHORT - Bán';
        entry = currentPrice;
        sl = stopLossShort;
        tp = takeProfitShort;
        confidence = Math.max(confidence, ruleConfidence);
    } else {
        signalText = '⚪️ ĐỢI - Chưa có tín hiệu';
        confidence = Math.min(confidence, ruleConfidence);
    }

    let details = [
        `📈 RSI: ${rsi.toFixed(1)}`,
        `📊 MACD: ${macd.toFixed(4)} / ${signal.toFixed(4)}`,
        `📉 ADX: ${adx.toFixed(1)}`,
        `📦 Volume: ${volumeSpike ? 'TĂNG ĐỘT BIẾN' : 'BÌNH THƯỜNG'}`,
        `⚠️ Lý do: ${ruleBasedSignal}`,
        `📏 Bollinger: ${lowerBB.toFixed(4)} - ${upperBB.toFixed(4)}`,
        `🛡️ Hỗ trợ: ${support.toFixed(4)}, Kháng cự: ${resistance.toFixed(4)}`
    ];
    if (isSideways) details.push(`⚠️ Lưu ý: Thị trường đang đi ngang, tín hiệu có thể không chính xác`);
    if (signalText !== '⚪️ ĐỢI - Chưa có tín hiệu') {
        details.push(`✅ Độ tin cậy: ${confidence}%`);
        details.push(`🎯 Điểm vào: ${entry.toFixed(4)}`);
        details.push(`🛑 SL: ${sl.toFixed(4)}`);
        details.push(`💰 TP: ${tp.toFixed(4)}`);
    }

    const result = `📊 *Phân tích ${symbol}/${pair} (${timeframes[timeframe]})*\n💰 Giá: ${currentPrice.toFixed(4)}\n⚡️ *${signalText}*\n${details.join('\n')}`;
    return { result, confidence };
}

// Tự đánh giá và huấn luyện trong giả lập (giảm tần suất huấn luyện và kiểm tra RAM)
let trainingCounter = 0; // Đếm số lần gọi để giảm tần suất huấn luyện
async function selfEvaluateAndTrain(historicalSlice, currentIndex, fullData) {
    const currentPrice = historicalSlice[historicalSlice.length - 1].close;
    const futureData = fullData.slice(currentIndex + 1, currentIndex + 11);
    if (futureData.length < 10) return;

    trainingCounter++;

    // Kiểm tra RAM trước khi huấn luyện
    const memoryUsage = process.memoryUsage();
    const usedMemoryMB = memoryUsage.heapUsed / 1024 / 1024;
    if (usedMemoryMB > 450) { // Nếu RAM vượt 450 MB (90% của 512 MB)
        console.log(`Bỏ qua huấn luyện tại cây nến ${currentIndex} do RAM cao: ${usedMemoryMB.toFixed(2)} MB (trainingCounter: ${trainingCounter})`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Bỏ qua huấn luyện tại cây nến ${currentIndex} do RAM cao: ${usedMemoryMB.toFixed(2)} MB (trainingCounter: ${trainingCounter})\n`);
        return;
    }

    if (trainingCounter % 10 !== 0) { // Chỉ huấn luyện sau mỗi 10 cây nến
        console.log(`Bỏ qua huấn luyện tại cây nến ${currentIndex} (trainingCounter: ${trainingCounter})`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Bỏ qua huấn luyện tại cây nến ${currentIndex} (trainingCounter: ${trainingCounter})\n`);
        return;
    }

    const futurePrice = futureData[futureData.length - 1].close;
    const priceChange = (futurePrice - currentPrice) / currentPrice * 100;

    let trueSignal;
    if (priceChange > 1) trueSignal = [1, 0, 0]; // LONG
    else if (priceChange < -1) trueSignal = [0, 1, 0]; // SHORT
    else trueSignal = [0, 0, 1]; // WAIT

    const close = historicalSlice.map(d => d.close);
    const volume = historicalSlice.map(d => d.volume);
    const rsi = computeRSI(close);
    const ma10 = computeMA(close, 10);
    const ma50 = computeMA(close, 50);
    const [, , histogram] = computeMACD(close);
    const [, middleBB] = computeBollingerBands(close);
    const adx = computeADX(historicalSlice);
    const atr = computeATR(historicalSlice.slice(-14));
    const volumeMA = computeMA(volume, 20);
    const volumeSpike = volume[volume.length - 1] > volumeMA * 1.5 ? 1 : 0;

    const normalizedRsi = rsi / 100;
    const normalizedAdx = adx / 100;
    const normalizedHistogram = histogram / 1000;
    const normalizedMaDiff = (ma10 - ma50) / currentPrice;
    const normalizedBbDiff = (currentPrice - middleBB) / currentPrice;
    const normalizedAtr = atr / currentPrice;
    const closeLag1 = historicalSlice[historicalSlice.length - 2]?.close / currentPrice || 1;
    const closeLag2 = historicalSlice[historicalSlice.length - 3]?.close / currentPrice || 1;
    const rsiRollingMean = SMA.calculate({ values: close.slice(-5).map(d => computeRSI([d])), period: 5 })[0] / 100 || normalizedRsi;

    const input = [[[normalizedRsi, normalizedAdx, normalizedHistogram, volumeSpike, normalizedMaDiff, normalizedBbDiff, closeLag1, closeLag2, normalizedAtr, rsiRollingMean]]];
    const xs = tf.tensor3d(input);
    const ys = tf.tensor2d([trueSignal]);
    await model.fit(xs, ys, { epochs: 1, batchSize: 1 });
    xs.dispose();
    ys.dispose();

    console.log(`✅ Đã huấn luyện mô hình tại cây nến ${currentIndex} với nhãn thực tế: ${trueSignal} (trainingCounter: ${trainingCounter}, RAM: ${usedMemoryMB.toFixed(2)} MB)`);
    fs.appendFileSync('bot.log', `${new Date().toISOString()} - Đã huấn luyện mô hình tại cây nến ${currentIndex} với nhãn: ${trueSignal} (trainingCounter: ${trainingCounter}, RAM: ${usedMemoryMB.toFixed(2)} MB)\n`);
}

// Giả lập dựa trên watch_configs với retry logic
let isSimulating = false;
let lastIndexMap = new Map(); // Lưu trữ vị trí cây nến cuối cùng của từng cấu hình
async function simulateRealTimeForConfigs(stepInterval = 1000) {
    const getConfigs = () => new Promise((resolve, reject) => {
        db.all("SELECT chatId, symbol, pair, timeframe FROM watch_configs", [], (err, rows) => {
            if (err) reject(err);
            else resolve(rows);
        });
    });

    const simulateConfig = async (config) => {
        const { chatId, symbol, pair, timeframe } = config;
        const configKey = `${chatId}_${symbol}_${pair}_${timeframe}`; // Key để lưu vị trí cây nến
        const historicalData = await fetchHistoricalData(symbol, pair, timeframe);
        if (!historicalData) {
            console.error(`❌ Không thể lấy dữ liệu cho ${symbol}/${pair}`);
            fs.appendFileSync('bot.log', `${new Date().toISOString()} - ❌ Không thể lấy dữ liệu cho ${symbol}/${pair}\n`);
            return;
        }

        let currentIndex = lastIndexMap.has(configKey) ? lastIndexMap.get(configKey) : 50;
        const simulateStep = async () => {
            if (currentIndex >= historicalData.length) {
                console.log(`✅ Hoàn tất giả lập ${symbol}/${pair} (${timeframes[timeframe]})`);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Hoàn tất giả lập ${symbol}/${pair} (${timeframes[timeframe]})\n`);
                lastIndexMap.delete(configKey); // Xóa vị trí sau khi hoàn tất
                return;
            }

            try {
                const historicalSlice = historicalData.slice(0, currentIndex + 1);
                const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe, {}, historicalSlice);

                if (confidence >= 80) {
                    bot.sendMessage(chatId, `🚨 *TÍN HIỆU GIẢ LẬP ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})* 🚨\n${result}`, {
                        parse_mode: 'Markdown'
                    });
                    console.log(`✅ Đã gửi tín hiệu giả lập ${symbol}/${pair} đến chat ${chatId} - Độ tin cậy: ${confidence}%`);
                    fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Đã gửi tín hiệu giả lập ${symbol}/${pair} đến chat ${chatId} - ${confidence}%\n`);
                }

                await selfEvaluateAndTrain(historicalSlice, currentIndex, historicalData);
                lastIndexMap.set(configKey, currentIndex + 1); // Lưu vị trí cây nến hiện tại
                currentIndex++;
                setTimeout(simulateStep, stepInterval);
            } catch (error) {
                console.error(`Lỗi trong giả lập ${symbol}/${pair} tại cây nến ${currentIndex}: ${error.message}`);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi trong giả lập ${symbol}/${pair} tại cây nến ${currentIndex}: ${error.message}\n`);
                setTimeout(simulateStep, 30000); // Thử lại sau 30 giây nếu có lỗi
            }
        };

        console.log(`Bắt đầu giả lập ${symbol}/${pair} (${timeframes[timeframe]}) cho chat ${chatId} từ cây nến ${currentIndex}...`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Bắt đầu giả lập ${symbol}/${pair} (${timeframes[timeframe]}) cho chat ${chatId} từ cây nến ${currentIndex}...\n`);
        await simulateStep();
    };

    try {
        const configs = await getConfigs();
        if (configs.length === 0) {
            console.log('⚠️ Chưa có cấu hình nào để giả lập. Hãy dùng /tinhieu để thêm.');
            fs.appendFileSync('bot.log', `${new Date().toISOString()} - ⚠️ Chưa có cấu hình nào để giả lập.\n`);
            return;
        }

        // Chia configs thành từng nhóm 1 cấu hình để chạy song song
        const batchSize = 1;
        for (let i = 0; i < configs.length; i += batchSize) {
            const batch = configs.slice(i, i + batchSize);
            await Promise.all(batch.map(config => simulateConfig(config)));
        }
    } catch (error) {
        console.error(`Lỗi truy vấn watch_configs: ${error.message}`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi truy vấn watch_configs: ${error.message}\n`);
    }
}

// Kiểm tra tự động trong thời gian thực
function startAutoChecking() {
    const CHECK_INTERVAL = 5 * 60 * 1000;
    function checkAndReschedule() {
        db.all("SELECT chatId, symbol, pair, timeframe FROM watch_configs", [], (err, rows) => {
            if (err) {
                console.error("Lỗi truy vấn database:", err.message);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi truy vấn database: ${err.message}\n`);
                setTimeout(checkAndReschedule, CHECK_INTERVAL);
                return;
            }
            rows.forEach(config => {
                checkAutoSignal(config.chatId, config).catch(err => {
                    console.error(`❌ Lỗi kiểm tra ${config.symbol}/${config.pair}: ${err.message}`);
                    fs.appendFileSync('bot.log', `${new Date().toISOString()} - ❌ Lỗi kiểm tra ${config.symbol}/${config.pair}: ${err.message}\n`);
                });
            });
            setTimeout(checkAndReschedule, CHECK_INTERVAL);
        });
    }
    setTimeout(checkAndReschedule, CHECK_INTERVAL);
}

async function checkAutoSignal(chatId, { symbol, pair, timeframe }, confidenceThreshold = 80) {
    try {
        const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe);
        if (confidence >= confidenceThreshold) {
            bot.sendMessage(chatId, `🚨 *TÍN HIỆU ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})* 🚨\n${result}`, {
                parse_mode: 'Markdown'
            });
            console.log(`✅ Đã gửi tín hiệu ${symbol}/${pair} đến chat ${chatId} - Độ tin cậy: ${confidence}%`);
            fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Đã gửi tín hiệu ${symbol}/${pair} đến chat ${chatId} - Độ tin cậy: ${confidence}%\n`);
        }
    } catch (error) {
        console.error(`❌ Lỗi kiểm tra ${symbol}/${pair}: ${error.message}`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - ❌ Lỗi kiểm tra ${symbol}/${pair}: ${error.message}\n`);
    }
}

// Hàm khởi động bot với xử lý lỗi 409
async function startBot() {
    try {
        bot = new TelegramBot(TOKEN, { polling: true });

        // Xử lý lỗi polling
        bot.on('polling_error', (error) => {
            if (error.code === 'ETELEGRAM' && error.message.includes('409 Conflict')) {
                console.error('Lỗi 409: Nhiều instance bot đang chạy. Thử lại sau 30 giây...');
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi 409: Nhiều instance bot đang chạy. Thử lại sau 30 giây...\n`);
                bot.stopPolling();
                setTimeout(startBot, 30000);
            } else {
                console.error(`Lỗi polling: ${error.message}`);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi polling: ${error.message}\n`);
                bot.stopPolling();
                setTimeout(startBot, 30000);
            }
        });

        // Lệnh phân tích thủ công
        bot.onText(/\?(.+)/, async (msg, match) => {
            const parts = match[1].split(',');
            if (parts.length < 3) {
                return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: ?ada,usdt,5m hoặc ?ada,usdt,m5');
            }
            const [symbol, pair, timeframeInput, customThreshold] = parts.map(p => p.trim().toLowerCase());
            const timeframe = normalizeTimeframe(timeframeInput);
            if (!timeframe) {
                return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ! Hãy dùng một trong các khung sau: ${Object.keys(timeframes).join(', ')}\nVí dụ: ?ada,usdt,5m hoặc ?ada,usdt,m5`);
            }

            const isValid = await isValidMarket(symbol, pair);
            if (!isValid) {
                return bot.sendMessage(msg.chat.id, `⚠️ Cặp giao dịch ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại trên Binance!\nVui lòng kiểm tra lại, ví dụ: ?ada,usdt,5m`);
            }

            let customThresholds = {};
            if (customThreshold && customThreshold.startsWith('rsi')) {
                const [oversold, overbought] = customThreshold.replace('rsi', '').split('-').map(Number);
                if (!isNaN(oversold) && !isNaN(overbought) && oversold < overbought) {
                    customThresholds.rsiOversold = oversold;
                    customThresholds.rsiOverbought = overbought;
                } else {
                    return bot.sendMessage(msg.chat.id, '⚠️ Định dạng RSI không hợp lệ! Hãy nhập theo kiểu: rsi25-75\nVí dụ: ?ada,usdt,5m,rsi25-75');
                }
            }

            const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe, customThresholds);
            bot.sendMessage(msg.chat.id, result, { parse_mode: 'Markdown' });
        });

        // Lệnh theo dõi tự động
        bot.onText(/\/tinhieu (.+)/, async (msg, match) => {
            const parts = match[1].split(',');
            if (parts.length < 3) {
                return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: /tinhieu ada,usdt,5m hoặc /tinhieu ada,usdt,m5');
            }
            const [symbol, pair, timeframeInput] = parts.map(p => p.trim().toLowerCase());
            const timeframe = normalizeTimeframe(timeframeInput);
            if (!timeframe) {
                return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ! Hãy dùng một trong các khung sau: ${Object.keys(timeframes).join(', ')}\nVí dụ: /tinhieu ada,usdt,5m hoặc /tinhieu ada,usdt,m5`);
            }

            const isValid = await isValidMarket(symbol, pair);
            if (!isValid) {
                return bot.sendMessage(msg.chat.id, `⚠️ Cặp giao dịch ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại trên Binance!\nVui lòng kiểm tra lại, ví dụ: /tinhieu ada,usdt,5m`);
            }

            const chatId = msg.chat.id;
            insertStmt.run(chatId, symbol, pair, timeframe, (err) => {
                if (err) {
                    console.error(err.message);
                    bot.sendMessage(msg.chat.id, `❌ Lỗi khi thêm cấu hình theo dõi: ${err.message}`);
                    fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi thêm config: ${err.message}\n`);
                } else {
                    bot.sendMessage(chatId, `✅ Bắt đầu theo dõi tín hiệu ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframeInput]})`);
                    fs.appendFileSync('bot.log', `${new Date().toISOString()} - Thêm config ${symbol}/${pair} cho chat ${chatId} thành công.\n`);
                    if (!isSimulating) {
                        isSimulating = true;
                        simulateRealTimeForConfigs(1000);
                    }
                }
            });
        });

        // Lệnh dừng theo dõi
        bot.onText(/\/dungtinhieu (.+)/, (msg, match) => {
            const parts = match[1].split(',');
            if (parts.length < 3) {
                return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: /dungtinhieu ada,usdt,5m hoặc /dungtinhieu ada,usdt,m5');
            }
            const [symbol, pair, timeframeInput] = parts.map(p => p.trim().toLowerCase());
            const timeframe = normalizeTimeframe(timeframeInput);
            if (!timeframe) {
                return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ! Hãy dùng một trong các khung sau: ${Object.keys(timeframes).join(', ')}\nVí dụ: /dungtinhieu ada,usdt,5m hoặc /dungtinhieu ada,usdt,m5`);
            }

            const chatId = msg.chat.id;
            deleteStmt.run(chatId, symbol, pair, timeframe, (err) => {
                if (err) {
                    console.error(err.message);
                    bot.sendMessage(msg.chat.id, `❌ Lỗi khi xóa cấu hình theo dõi: ${err.message}`);
                    fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi xóa config: ${err.message}\n`);
                } else {
                    bot.sendMessage(chatId, `✅ Đã dừng theo dõi tín hiệu ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframeInput]})`);
                    fs.appendFileSync('bot.log', `${new Date().toISOString()} - Xóa config ${symbol}/${pair} cho chat ${chatId} thành công.\n`);
                }
            });
        });

        // Lệnh trợ giúp
        bot.onText(/\/trogiup/, (msg) => {
            const helpMessage = `
📚 *HƯỚNG DẪN SỬ DỤNG BOT GIAO DỊCH*

1. **?symbol,pair,timeframe[,rsiOversold-rsiOverbought]**
   - *Mô tả*: Phân tích thủ công cặp giao dịch.
   - *Ví dụ*: ?ada,usdt,5m hoặc ?ada,usdt,m5
              ?btc,usdt,1h,rsi25-75 hoặc ?btc,usdt,h1,rsi25-75

2. **/tinhieu symbol,pair,timeframe**
   - *Mô tả*: Kích hoạt theo dõi tự động.
   - *Ví dụ*: /tinhieu ada,usdt,5m hoặc /tinhieu ada,usdt,m5
              /tinhieu btc,usdt,1h hoặc /tinhieu btc,usdt,h1

3. **/dungtinhieu symbol,pair,timeframe**
   - *Mô tả*: Dừng theo dõi tự động.
   - *Ví dụ*: /dungtinhieu ada,usdt,5m hoặc /dungtinhieu ada,usdt,m5

4. **/trogiup**
   - *Mô tả*: Hiển thị hướng dẫn.
   - *Ví dụ*: /trogiup

*Lưu ý*: Khung thời gian có thể viết như 5m hoặc m5, 1h hoặc h1, v.v.
            `;
            bot.sendMessage(msg.chat.id, helpMessage, { parse_mode: 'Markdown' });
        });

        console.log('✅ Bot đã khởi động thành công');
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Bot đã khởi động thành công\n`);
    } catch (error) {
        console.error(`Lỗi khởi động bot: ${error.message}`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi khởi động bot: ${error.message}\n`);
        setTimeout(startBot, 30000);
    }
}

// Khởi động bot
(async () => {
    try {
        await initializeModel();
        const initialData = await fetchKlines('BTC', 'USDT', '1h', 250);
        if (initialData) {
            await trainModel(initialData);
        } else {
            console.error('❌ Không thể lấy dữ liệu ban đầu để huấn luyện mô hình');
            fs.appendFileSync('bot.log', `${new Date().toISOString()} - ❌ Không thể lấy dữ liệu ban đầu để huấn luyện mô hình\n`);
        }

        // Khởi động bot
        await startBot();

        // Chạy giả lập và kiểm tra tự động
        await simulateRealTimeForConfigs(1000);
        startAutoChecking();
        console.log('✅ Bot đang chạy với giả lập tối ưu và kiểm tra tự động...');
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Bot đang chạy với giả lập tối ưu và kiểm tra tự động...\n`);
    } catch (error) {
        console.error("Lỗi khởi tạo:", error);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi khởi tạo: ${error.message}\n`);
    }

    // Đóng database khi tắt bot
    process.on('SIGINT', () => {
        insertStmt.finalize();
        deleteStmt.finalize();
        db.close((err) => {
            if (err) {
                console.error("Lỗi đóng SQLite:", err.message);
            }
            console.log('Đóng kết nối database.');
            process.exit(0);
        });
    });
})();