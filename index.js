/********************************************
 *  BOT PHÂN TÍCH CRYPTO VỚI TÍNH NĂNG LƯU TRỮ SQL VÀ GIẢ LẬP
 ********************************************/

const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR } = require('technicalindicators');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const sqlite3 = require('sqlite3').verbose();

// =====================
//     CẤU HÌNH
// =====================

const TOKEN = process.env.TELEGRAM_BOT_TOKEN || '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY';
const BINANCE_API = 'https://api.binance.com/api/v3';

const timeframes = {
    '1m': '1 phút', 'm1': '1 phút',
    '3m': '3 phút', 'm3': '3 phút',
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

function normalizeTimeframe(tfInput) {
    const mapping = {
        'm1': '1m', '1m': '1m',
        'm3': '3m', '3m': '3m',
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
    return mapping[tfInput] || null;
}

const bot = new TelegramBot(TOKEN, { polling: true });

// =====================
//  SQLITE - LƯU TRỮ DỮ LIỆU
// =====================

const db = new sqlite3.Database('bot.db', (err) => {
    if (err) {
        console.error('SQLite Error:', err.message);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi kết nối SQLite: ${err.message}\n`);
    } else {
        console.log('✅ Kết nối SQLite thành công.');
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Kết nối SQLite thành công.\n`);
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
            console.error('Lỗi tạo bảng watch_configs:', err.message);
            fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi tạo bảng watch_configs: ${err.message}\n`);
        }
    });
});

db.run(`
  CREATE TABLE IF NOT EXISTS signal_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chatId INTEGER,
    symbol TEXT,
    pair TEXT,
    timeframe TEXT,
    signal TEXT,
    confidence INTEGER,
    timestamp INTEGER
  )
`, (err) => {
    if (err) {
        console.error('Lỗi tạo bảng signal_history:', err.message);
    }
});

function addWatchConfig(chatId, symbol, pair, timeframe, callback) {
    db.run(
        `INSERT OR IGNORE INTO watch_configs (chatId, symbol, pair, timeframe) VALUES (?, ?, ?, ?)`,
        [chatId, symbol, pair, timeframe],
        callback
    );
}

function deleteWatchConfig(chatId, symbol, pair, timeframe, callback) {
    db.run(
        `DELETE FROM watch_configs WHERE chatId = ? AND symbol = ? AND pair = ? AND timeframe = ?`,
        [chatId, symbol, pair, timeframe],
        callback
    );
}

function loadWatchConfigs() {
    return new Promise((resolve, reject) => {
        db.all("SELECT chatId, symbol, pair, timeframe FROM watch_configs", [], (err, rows) => {
            if (err) reject(err);
            else resolve(rows);
        });
    });
}

// =====================
// MÔ HÌNH & HUẤN LUYỆN AI
// =====================
let model;
async function initializeModel() {
    model = tf.sequential();
    // Mô hình với inputShape: [6]
    model.add(tf.layers.dense({ units: 50, activation: 'relu', inputShape: [6] }));
    model.add(tf.layers.dense({ units: 20, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    console.log('✅ Mô hình AI đã được khởi tạo.');
}

async function trainModelData(data) {
    try {
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

            inputs.push([
                rsi,
                adx,
                histogram,
                volumeSpike,
                ma10 - ma50,
                curr.close - middleBB
            ]);

            let signal = [0, 0, 1];
            if (adx > 30) {
                if (rsi < 30 && ma10 > ma50 && histogram > 0 && volumeSpike && curr.close < middleBB)
                    signal = [1, 0, 0];
                else if (rsi > 70 && ma10 < ma50 && histogram < 0 && volumeSpike && curr.close > middleBB)
                    signal = [0, 1, 0];
            }
            outputs.push(signal);
        }
        if (inputs.length === 0) return;
        const xs = tf.tensor2d(inputs); // shape [*,6]
        const ys = tf.tensor2d(outputs); // shape [*,3]
        await model.fit(xs, ys, { epochs: 20, batchSize: 32, shuffle: true });
        console.log('✅ Mô hình đã được huấn luyện ban đầu.');
        xs.dispose();
        ys.dispose();
    } catch (error) {
        console.error('Lỗi huấn luyện mô hình:', error.message);
    }
}

// =====================
// HÀM TÍNH CHỈ BÁO (TECHNICAL INDICATORS)
// =====================
function computeRSI(close, period = 14) {
    // Tính RSI, nếu result rỗng thì trả về 50
    const result = RSI.calculate({ values: close, period });
    if (!result || result.length === 0) return 50;
    return result[result.length - 1];
}

function computeMACD(close) {
    // Tính MACD, nếu result rỗng thì trả về [0,0,0]
    const result = MACD.calculate({ values: close, fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 });
    if (!result || result.length === 0) return [0, 0, 0];
    const last = result[result.length - 1];
    return [
        last?.MACD || 0,
        last?.signal || 0,
        last?.histogram || 0
    ];
}

function computeBollingerBands(close, period = 20, stdDev = 2) {
    // Tính Bollinger Bands, nếu result rỗng thì trả về [0,0,0]
    const result = BollingerBands.calculate({ values: close, period, stdDev });
    if (!result || result.length === 0) return [0, 0, 0];
    const last = result[result.length - 1];
    return [
        last?.upper || 0,
        last?.middle || 0,
        last?.lower || 0
    ];
}

function computeADX(data, period = 14) {
    // Tính ADX, nếu result rỗng thì trả về 0
    const result = ADX.calculate({
        high: data.map(d => d.high),
        low: data.map(d => d.low),
        close: data.map(d => d.close),
        period
    });
    if (!result || result.length === 0) return 0;
    return result[result.length - 1]?.adx || 0;
}

function computeATR(data, period = 14) {
    // Tính ATR, nếu result rỗng thì trả về 0
    const result = ATR.calculate({
        high: data.map(d => d.high),
        low: data.map(d => d.low),
        close: data.map(d => d.close),
        period
    });
    if (!result || result.length === 0) return 0;
    return result[result.length - 1] || 0;
}

function computeMA(close, period = 20) {
    return SMA.calculate({ values: close, period }).slice(-1)[0] || 0;
}

function computeSupportResistance(data) {
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    return { support: Math.min(...lows), resistance: Math.max(...highs) };
}

// =====================
// PHÂN TÍCH CRYPTO (Online)
// =====================
async function getCryptoAnalysis(symbol, pair, timeframe, customThresholds = {}) {
    const df = await fetchKlines(symbol, pair, timeframe);
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
    const atr = computeATR(df);
    const volumeMA = computeMA(volume, 20);
    const volumeSpike = volume[volume.length - 1] > volumeMA * 1.5 ? 1 : 0;
    const { support, resistance } = computeSupportResistance(df);

    const bbWidth = upperBB - lowerBB;
    const avgBBWidth = computeMA(
        df.map(d => {
            const tmp = BollingerBands.calculate({ values: df.map(v => v.close), period: 20, stdDev: 2 }).slice(-1)[0];
            return tmp.upper - tmp.lower;
        }),
        20
    );
    const isSideways = adx < 20 && bbWidth < avgBBWidth * 0.8;

    const rsiOverbought = customThresholds.rsiOverbought || 70;
    const rsiOversold = customThresholds.rsiOversold || 30;
    const adxStrongTrend = customThresholds.adxStrongTrend || 30;

    // 6 đặc trưng
    const input = tf.tensor2d([[
        rsi,
        adx,
        histogram,
        volumeSpike,
        ma10 - ma50,
        currentPrice - middleBB
    ]]);
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

    if (maxProb === longProb) {
        signalText = '🟢 LONG - Mua';
        entry = currentPrice;
        sl = Math.max(currentPrice - atr * 2, support);
        tp = Math.min(currentPrice + atr * 4, resistance);
        if (ruleBasedSignal.includes('LONG')) {
            confidence = Math.max(confidence, ruleConfidence);
        }
    } else if (maxProb === shortProb) {
        signalText = '🔴 SHORT - Bán';
        entry = currentPrice;
        sl = Math.min(currentPrice + atr * 2, resistance);
        tp = Math.max(currentPrice - atr * 4, support);
        if (ruleBasedSignal.includes('SHORT')) {
            confidence = Math.max(confidence, ruleConfidence);
        }
    } else {
        signalText = '⚪️ ĐỢI - Chưa có tín hiệu';
        confidence = Math.min(confidence, ruleConfidence);
    }

    const details = [];
    details.push(`📈 RSI: ${rsi.toFixed(1)}`);
    details.push(`📊 MACD: ${macd.toFixed(4)} / ${signal.toFixed(4)}`);
    details.push(`📉 ADX: ${adx.toFixed(1)}`);
    details.push(`📦 Volume: ${volumeSpike ? 'TĂNG ĐỘT BIẾN' : 'BÌNH THƯỜNG'}`);
    details.push(`📏 Bollinger: ${lowerBB.toFixed(4)} - ${upperBB.toFixed(4)}`);
    details.push(`🛡️ Hỗ trợ: ${support.toFixed(4)}, Kháng cự: ${resistance.toFixed(4)}`);
    if (isSideways) {
        details.push(`⚠️ Lưu ý: Thị trường đang đi ngang, tín hiệu có thể không chính xác`);
    }
    // Thêm timestamp và xu hướng
    const timestamp = new Date().toLocaleString();
    details.push(`⏰ Thời gian: ${timestamp}`);
    if (isSideways) {
        details.push(`📊 Xu hướng: Đi ngang`);
    } else if (ruleBasedSignal.includes('LONG')) {
        details.push(`📈 Xu hướng: Tăng`);
    } else if (ruleBasedSignal.includes('SHORT')) {
        details.push(`📉 Xu hướng: Giảm`);
    } else {
        details.push(`📊 Xu hướng: Không rõ`);
    }

    // Tính R:R nếu có tín hiệu giao dịch
    if (signalText !== '⚪️ ĐỢI - Chưa có tín hiệu') {
        let risk, reward, rr;
        if (signalText.includes('LONG')) {
            risk = entry - sl;
            reward = tp - entry;
        } else if (signalText.includes('SHORT')) {
            risk = sl - entry;
            reward = entry - tp;
        }
        if (risk > 0) {
            rr = (reward / risk).toFixed(2);
            details.push(`⚖️ R:R: ${rr}:1`);
        }
    }

    details.push(`ℹ️ Độ tin cậy dựa trên sự kết hợp của các chỉ báo RSI, MACD, ADX và Bollinger Bands.`);

    if (signalText !== '⚪️ ĐỢI - Chưa có tín hiệu') {
        details.push(`✅ Độ tin cậy: ${confidence}%`);
        details.push(`🎯 Điểm vào: ${entry.toFixed(4)}`);
        if (signalText.includes('LONG')) {
            details.push(`🛑 SL: ${sl.toFixed(4)}`);
            details.push(`💰 TP: ${tp.toFixed(4)}`);
        } else if (signalText.includes('SHORT')) {
            details.push(`🛑 SL: ${sl.toFixed(4)}`);
            details.push(`💰 TP: ${tp.toFixed(4)}`);
        }
    }

    const resultText = `📊 *Phân tích ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})*\n`
        + `💰 Giá: ${currentPrice.toFixed(4)}\n`
        + `⚡️ *${signalText}*\n`
        + details.join('\n');

    return { result: resultText, confidence };
}

// =====================
// SELF-EVALUATE & TRAIN TRONG GIẢ LẬP (6 đặc trưng)
// =====================
let enableSimulation = true;
let recentAccuracies = [];
let lastAccuracy = 0;
let shouldStopTraining = false;
let trainingCounter = 0;
async function selfEvaluateAndTrain(historicalSlice, currentIndex, fullData) {
    if (shouldStopTraining) return; // Nếu đã đạt ổn định, bỏ qua huấn luyện

    const currentPrice = historicalSlice[historicalSlice.length - 1].close;
    const futureData = fullData.slice(currentIndex + 1, currentIndex + 11);
    if (futureData.length < 10) return;

    trainingCounter++;

    const memoryUsage = process.memoryUsage();
    const usedMemoryMB = memoryUsage.heapUsed / 1024 / 1024;
    if (usedMemoryMB > 450) {
        console.log(`🚨 RAM cao: ${usedMemoryMB.toFixed(2)}MB - bỏ qua huấn luyện tại nến ${currentIndex}`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Bỏ qua huấn luyện tại nến ${currentIndex} do RAM cao: ${usedMemoryMB.toFixed(2)} MB (trainingCounter: ${trainingCounter})\n`);
        return;
    }
    if (trainingCounter % 10 !== 0) {
        console.log(`Bỏ qua huấn luyện tại nến ${currentIndex} (trainingCounter: ${trainingCounter})`);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Bỏ qua huấn luyện tại nến ${currentIndex} (trainingCounter: ${trainingCounter})\n`);
        return;
    }

    const futurePrice = futureData[futureData.length - 1].close;
    const priceChange = (futurePrice - currentPrice) / currentPrice * 100;
    let trueSignal = [0, 0, 1];
    if (priceChange > 1) trueSignal = [1, 0, 0];
    else if (priceChange < -1) trueSignal = [0, 1, 0];

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

    // Sử dụng 6 đặc trưng tương ứng với inputShape: [6]
    const xs = tf.tensor2d([[
        normalizedRsi,
        normalizedAdx,
        normalizedHistogram,
        volumeSpike,
        normalizedMaDiff,
        normalizedBbDiff
    ]]);
    const ys = tf.tensor2d([trueSignal]); // shape [1,3]

    const history = await model.fit(xs, ys, { epochs: 1, batchSize: 1 });
    xs.dispose();
    ys.dispose();

    lastAccuracy = history.history.accuracy[0] || 0;
    recentAccuracies.push(lastAccuracy);
    if (recentAccuracies.length > 50) recentAccuracies.shift();

    console.log(`✅ Huấn luyện tại nến ${currentIndex} | RAM: ${usedMemoryMB.toFixed(2)} MB | Accuracy: ${(lastAccuracy * 100).toFixed(2)}%`);
    fs.appendFileSync('bot.log', `${new Date().toISOString()} - Huấn luyện tại nến ${currentIndex} | RAM: ${usedMemoryMB.toFixed(2)} MB | Accuracy: ${(lastAccuracy * 100).toFixed(2)}%\n`);

    if (recentAccuracies.length >= 50) {
        const avgAcc = recentAccuracies.reduce((sum, val) => sum + val, 0) / recentAccuracies.length;
        const maxAcc = Math.max(...recentAccuracies);
        const minAcc = Math.min(...recentAccuracies);
        if (avgAcc > 0.85 && (maxAcc - minAcc) < 0.05) {
            shouldStopTraining = true;
            enableSimulation = false;
            console.log('✅ Mô hình đã ổn định, dừng tự huấn luyện và giả lập.');
        }
    }
}

// =====================
// CHẾ ĐỘ GIẢ LẬP TỰ ĐỘNG
// =====================

let lastIndexMap = new Map();

async function simulateConfig(config, stepInterval) {
    const { chatId, symbol, pair, timeframe } = config;
    const configKey = `${chatId}_${symbol}_${pair}_${timeframe}`;
    const historicalData = await fetchKlines(symbol, pair, timeframe);
    if (!historicalData) {
        console.error(`❌ Không thể lấy dữ liệu cho ${symbol}/${pair}`);
        return;
    }
    let currentIndex = lastIndexMap.has(configKey) ? lastIndexMap.get(configKey) : 50;

    async function simulateStep() {
        if (currentIndex >= historicalData.length || !enableSimulation) {
            console.log(`✅ Dừng giả lập ${symbol}/${pair} (${timeframes[timeframe]})`);
            lastIndexMap.delete(configKey);
            return;
        }
        try {
            const historicalSlice = historicalData.slice(0, currentIndex + 1);
            const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe, {}, historicalSlice);
            if (confidence >= 80) {
                bot.sendMessage(chatId, `🚨 *TÍN HIỆU GIẢ LẬP ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})* 🚨\n${result}`, { parse_mode: 'Markdown' });
                console.log(`✅ Gửi tín hiệu ${symbol}/${pair} cho chat ${chatId} (Độ tin: ${confidence}%)`);
            }
            if (!shouldStopTraining) {
                await selfEvaluateAndTrain(historicalSlice, currentIndex, historicalData);
            }
            lastIndexMap.set(configKey, currentIndex + 1);
            currentIndex++;
            setTimeout(simulateStep, stepInterval);
        } catch (error) {
            console.error(`Lỗi giả lập ${symbol}/${pair} tại nến ${currentIndex}: ${error.message}`);
            setTimeout(simulateStep, 30000);
        }
    }

    console.log(`Bắt đầu giả lập ${symbol}/${pair} (${timeframes[timeframe]}) cho chat ${chatId} từ nến ${currentIndex}...`);
    simulateStep();
}

async function simulateRealTimeForConfigs(stepInterval = 1000) {
    try {
        const configs = await loadWatchConfigs();
        if (!configs || configs.length === 0) {
            console.log('⚠️ Không có cấu hình watch nào để giả lập.');
            return;
        }
        for (let config of configs) {
            simulateConfig(config, stepInterval);
        }
    } catch (error) {
        console.error(`Lỗi load watch configs: ${error.message}`);
    }
}

// =====================
// HÀM FETCH DỮ LIỆU
// =====================
async function fetchKlines(symbol, pair, timeframe, limit = 200) {
    try {
        const response = await axios.get(`${BINANCE_API}/klines`, {
            params: { symbol: `${symbol.toUpperCase()}${pair.toUpperCase()}`, interval: timeframe, limit },
            timeout: 10000,
        });
        return response.data.map(d => ({
            timestamp: d[0],
            open: parseFloat(d[1]),
            high: parseFloat(d[2]),
            low: parseFloat(d[3]),
            close: parseFloat(d[4]),
            volume: parseFloat(d[5])
        }));
    } catch (error) {
        console.error(`API Error: ${error.message}`);
        return null;
    }
}

// =====================
// LỆNH BOT
// =====================

const autoWatchList = new Map(); // (chatId -> [{ symbol, pair, timeframe }])

async function isValidMarket(symbol, pair) {
    try {
        const response = await axios.get(`${BINANCE_API}/ticker/price`, {
            params: { symbol: `${symbol.toUpperCase()}${pair.toUpperCase()}` },
            timeout: 5000,
        });
        return !!response.data.price;
    } catch (error) {
        return false;
    }
}

// Phân tích thủ công: ?symbol,pair,timeframe[,rsiOversold-rsiOverbought]
bot.onText(/\?(.+)/, async (msg, match) => {
    try {
        const parts = match[1].split(',').map(p => p.trim());
        if (parts.length < 3) {
            return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Ví dụ: ?ada,usdt,5m hoặc ?ada,usdt,5m,rsi25-75');
        }
        const [symbol, pair, timeframeInput, customThreshold] = parts.map(p => p.toLowerCase());
        const timeframe = normalizeTimeframe(timeframeInput);
        if (!timeframes[timeframe]) {
            return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ! Chọn 1 trong: ${Object.keys(timeframes).join(', ')}`);
        }
        const valid = await isValidMarket(symbol, pair);
        if (!valid) {
            return bot.sendMessage(msg.chat.id, `⚠️ Cặp ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại trên Binance!`);
        }
        const customThresholds = {};
        if (customThreshold && customThreshold.startsWith('rsi')) {
            const [oversold, overbought] = customThreshold.replace('rsi', '').split('-').map(Number);
            if (!isNaN(oversold) && !isNaN(overbought) && oversold < overbought) {
                customThresholds.rsiOversold = oversold;
                customThresholds.rsiOverbought = overbought;
            } else {
                return bot.sendMessage(msg.chat.id, '⚠️ Định dạng RSI không hợp lệ! Ví dụ: rsi25-75');
            }
        }
        const { result } = await getCryptoAnalysis(symbol, pair, timeframe, customThresholds);
        bot.sendMessage(msg.chat.id, result, { parse_mode: 'Markdown' });
    } catch (error) {
        bot.sendMessage(msg.chat.id, `❌ Lỗi phân tích: ${error.message}`);
    }
});

// Bật theo dõi tự động: /tinhieu symbol,pair,timeframe
bot.onText(/\/tinhieu (.+)/, async (msg, match) => {
    try {
        let text = match[1].trim();
        let parts = text.split(',').map(p => p.trim());
        if (parts.length < 3) {
            // Tự format nếu thiếu dấu phẩy
            const alt = text.split(/\s+/).map(p => p.trim());
            if (alt.length === 3) parts = alt;
            else return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Ví dụ: /tinhieu ada,usdt,5m');
        }
        const [symbol, pair, timeframeInput] = parts.map(p => p.toLowerCase());
        const timeframe = normalizeTimeframe(timeframeInput);
        if (!timeframes[timeframe]) {
            return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ! Chọn 1 trong: ${Object.keys(timeframes).join(', ')}`);
        }
        const valid = await isValidMarket(symbol, pair);
        if (!valid) {
            return bot.sendMessage(msg.chat.id, `⚠️ Cặp ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại trên Binance!`);
        }
        const chatId = msg.chat.id;
        if (!autoWatchList.has(chatId)) {
            autoWatchList.set(chatId, []);
        }
        const watchList = autoWatchList.get(chatId);
        if (!watchList.some(w => w.symbol === symbol && w.pair === pair && w.timeframe === timeframe)) {
            watchList.push({ symbol, pair, timeframe });
            addWatchConfig(chatId, symbol, pair, timeframe, (err) => {
                if (err) console.error('Lỗi lưu cấu hình:', err.message);
            });
            bot.sendMessage(msg.chat.id, `✅ Đã bật theo dõi ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})`);
            // Chỉ chạy giả lập cho cặp mới nếu chưa có trong lastIndexMap
            const configKey = `${chatId}_${symbol}_${pair}_${timeframe}`;
            if (!lastIndexMap.has(configKey)) {
                simulateConfig({ chatId, symbol, pair, timeframe }, 1000);
            }
        } else {
            bot.sendMessage(msg.chat.id, 'ℹ️ Bạn đã theo dõi cặp này rồi!');
        }
    } catch (error) {
        bot.sendMessage(msg.chat.id, `❌ Lỗi /tinhieu: ${error.message}`);
    }
});

// Dừng theo dõi tự động: /dungtinhieu symbol,pair,timeframe
bot.onText(/\/dungtinhieu (.+)/, (msg, match) => {
    try {
        const parts = match[1].split(',').map(p => p.trim());
        if (parts.length < 3) {
            return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Ví dụ: /dungtinhieu ada,usdt,5m');
        }
        const [symbol, pair, timeframeInput] = parts.map(p => p.toLowerCase());
        const timeframe = normalizeTimeframe(timeframeInput);
        if (!timeframes[timeframe]) {
            return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ! Chọn 1 trong: ${Object.keys(timeframes).join(', ')}`);
        }
        const chatId = msg.chat.id;
        if (!autoWatchList.has(chatId)) {
            return bot.sendMessage(chatId, 'ℹ️ Bạn chưa theo dõi cặp nào.');
        }
        const watchList = autoWatchList.get(chatId);
        const idx = watchList.findIndex(w => w.symbol === symbol && w.pair === pair && w.timeframe === timeframe);
        if (idx !== -1) {
            watchList.splice(idx, 1);
            deleteWatchConfig(chatId, symbol, pair, timeframe, (err) => {
                if (err) console.error('Lỗi xóa cấu hình:', err.message);
            });
            bot.sendMessage(msg.chat.id, `✅ Đã dừng theo dõi ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})`);
        } else {
            bot.sendMessage(msg.chat.id, 'ℹ️ Bạn chưa theo dõi cặp này!');
        }
    } catch (error) {
        bot.sendMessage(msg.chat.id, `❌ Lỗi /dungtinhieu: ${error.message}`);
    }
});

// Lệnh trợ giúp: /trogiup
bot.onText(/\/trogiup/, (msg) => {
    const helpMessage = `
📚 *HƯỚNG DẪN SỬ DỤNG BOT GIAO DỊCH*

1. **?symbol,pair,timeframe[,rsiOversold-rsiOverbought]**
   - Phân tích thủ công.
   - Ví dụ: ?ada,usdt,5m hoặc ?ada,usdt,5m,rsi25-75

2. **/tinhieu symbol,pair,timeframe**
   - Bật theo dõi tự động.
   - Ví dụ: /tinhieu ada,usdt,5m

3. **/dungtinhieu symbol,pair,timeframe**
   - Dừng theo dõi tự động.
   - Ví dụ: /dungtinhieu ada,usdt,5m

4. **/trogiup**
   - Hiển thị hướng dẫn này.
`;
    bot.sendMessage(msg.chat.id, helpMessage, { parse_mode: 'Markdown' });
});

// Kiểm tra tự động (mỗi 5 phút)
function startAutoChecking() {
    const CHECK_INTERVAL = 1 * 60 * 1000;
    setInterval(() => {
        for (const [chatId, watchList] of autoWatchList) {
            watchList.forEach(config => {
                checkAutoSignal(chatId, config).catch(err => console.error(`❌ Lỗi checkAutoSignal: ${err.message}`));
            });
        }
    }, CHECK_INTERVAL);
}

// Hàm kiểm tra và gửi tín hiệu nếu đạt ngưỡng (confidence ≥ 80%)
async function checkAutoSignal(chatId, { symbol, pair, timeframe }, confidenceThreshold = 40) {
    try {
        const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe);
        if (confidence >= confidenceThreshold) {
            bot.sendMessage(chatId, `🚨 *TÍN HIỆU ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})* 🚨\n${result}`, { parse_mode: 'Markdown' });
            console.log(`✅ Gửi tín hiệu ${symbol}/${pair} cho chat ${chatId} (Độ tin: ${confidence}%)`);
        }
    } catch (error) {
        console.error(`❌ Lỗi checkAutoSignal ${symbol}/${pair}: ${error.message}`);
    }
}

// ------------------------------
// KHỞI ĐỘNG BOT VÀ CHƯƠNG TRÌNH CHÍNH
// ------------------------------
(async () => {
    // 1) Khởi tạo mô hình AI
    await initializeModel();

    // 2) Huấn luyện ban đầu với dữ liệu lịch sử (ví dụ BTC/USDT khung 1h)
    const initialData = await fetchKlines('BTC', 'USDT', '1h', 200);
    if (initialData) {
        await trainModelData(initialData);
    } else {
        console.error('❌ Không thể lấy dữ liệu ban đầu để huấn luyện mô hình');
    }

    console.log('✅ Bot đã khởi động và sẵn sàng nhận lệnh.');

    // 3) Kiểm tra tự động các cấu hình watch từ SQLite (mỗi 5 phút)
    startAutoChecking();

    // 4) (Tùy chọn) Chạy chế độ giả lập tự động
    simulateRealTimeForConfigs(1000);
})();

// ------------------------------
// HÀM FETCH DỮ LIỆU
// ------------------------------
async function fetchKlines(symbol, pair, timeframe, limit = 200) {
    try {
        const response = await axios.get(`${BINANCE_API}/klines`, {
            params: { symbol: `${symbol.toUpperCase()}${pair.toUpperCase()}`, interval: timeframe, limit },
            timeout: 10000,
        });
        return response.data.map(d => ({
            timestamp: d[0],
            open: parseFloat(d[1]),
            high: parseFloat(d[2]),
            low: parseFloat(d[3]),
            close: parseFloat(d[4]),
            volume: parseFloat(d[5])
        }));
    } catch (error) {
        console.error(`API Error: ${error.message}`);
        return null;
    }
}
