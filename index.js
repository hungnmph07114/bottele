/********************************************
 *  BOT PHÂN TÍCH CRYPTO VỚI TÍNH NĂNG LƯU TRỮ SQL VÀ GIẢ LẬP
 *  (Sử dụng LSTM với WINDOW_SIZE, dynamic training control và lời khuyên đòn bẩy)
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
const FEAR_GREED_API = 'https://api.alternative.me/fng/';

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
        if (err) console.error('Lỗi tạo bảng watch_configs:', err.message);
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
        if (err) console.error('Lỗi tạo bảng signal_history:', err.message);
    });
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
// HÀM LẤY FEAR & GREED INDEX VÀ QUẢN LÝ BIẾN TOÀN CỤC
// =====================

const FEAR_GREED_UPDATE_INTERVAL = 6 * 60 * 60 * 1000; // 6 giờ
let currentFearGreedValue = 50;

async function updateFearGreedData() {
    try {
        const response = await axios.get(`${FEAR_GREED_API}?date_format=kraken&limit=1`, { timeout: 10000 });
        const data = response.data.data;
        if (data.length > 0) {
            currentFearGreedValue = parseFloat(data[0].value) || 50;
            console.log(`✅ Cập nhật Fear & Greed Index: ${currentFearGreedValue}`);
        }
    } catch (error) {
        console.error('Lỗi cập nhật Fear & Greed Index:', error.message);
    }
}

// Cập nhật Fear & Greed Index lần đầu khi khởi động và sau đó mỗi 6 giờ
(async () => {
    await updateFearGreedData();
    setInterval(updateFearGreedData, FEAR_GREED_UPDATE_INTERVAL);
})();

// =====================
// CẤU HÌNH LSTM
// =====================
const WINDOW_SIZE = 10;

function computeFeature(data, j) {
    const subData = data.slice(0, j + 1);
    const close = subData.map(d => d.close);
    const volume = subData.map(d => d.volume);
    const rsi = computeRSI(close);
    const ma10 = computeMA(close, 10);
    const ma50 = computeMA(close, 50);
    const [, , histogram] = computeMACD(close);
    const [, middleBB] = computeBollingerBands(close);
    const adx = computeADX(subData);
    const currentPrice = close[close.length - 1];
    const volumeMA = computeMA(volume, 20);
    const volumeSpike = volume[volume.length - 1] > volumeMA * 1.5 ? 1 : 0;
    return [rsi, adx, histogram, volumeSpike, ma10 - ma50, currentPrice - middleBB, currentFearGreedValue];
}

// =====================
// MÔ HÌNH & HUẤN LUYỆN AI
// =====================
let model;
async function initializeModel() {
    model = tf.sequential();
    model.add(tf.layers.lstm({ units: 64, inputShape: [WINDOW_SIZE, 7], returnSequences: false }));
    model.add(tf.layers.dense({ units: 20, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    console.log('✅ LSTM model đã được khởi tạo.');
}

async function trainModelData(data) {
    try {
        const inputs = [];
        const outputs = [];
        for (let i = WINDOW_SIZE; i < data.length; i++) {
            const windowFeatures = [];
            for (let j = i - WINDOW_SIZE; j < i; j++) {
                windowFeatures.push(computeFeature(data, j));
            }
            inputs.push(windowFeatures);

            const subData = data.slice(0, i + 1);
            const currentPrice = subData[subData.length - 1].close;
            const futureData = data.slice(i + 1, i + 11);
            let trueSignal = [0, 0, 1];
            if (futureData.length >= 10) {
                const futurePrice = futureData[futureData.length - 1].close;
                const priceChange = (futurePrice - currentPrice) / currentPrice * 100;
                if (priceChange > 1) trueSignal = [1, 0, 0];
                else if (priceChange < -1) trueSignal = [0, 1, 0];
            }
            outputs.push(trueSignal);
        }
        if (inputs.length === 0) return;
        const xs = tf.tensor3d(inputs);
        const ys = tf.tensor2d(outputs);
        await model.fit(xs, ys, { epochs: 20, batchSize: 32, shuffle: true });
        console.log('✅ Mô hình đã được huấn luyện ban đầu.');
        xs.dispose();
        ys.dispose();
    } catch (error) {
        console.error('Lỗi huấn luyện mô hình:', error.message);
    }
}

// =====================
// HÀM TÍNH CHỈ BÁO
// =====================
function computeRSI(close, period = 14) {
    const result = RSI.calculate({ values: close, period });
    return result && result.length > 0 ? result[result.length - 1] : 50;
}

function computeMA(close, period = 20) {
    const ma = SMA.calculate({ values: close, period });
    return ma && ma.length > 0 ? ma[ma.length - 1] : 0;
}

function computeMACD(close) {
    const result = MACD.calculate({ values: close, fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 });
    if (!result || result.length === 0) return [0, 0, 0];
    const last = result[result.length - 1];
    return [last?.MACD || 0, last?.signal || 0, last?.histogram || 0];
}

function computeBollingerBands(close, period = 20, stdDev = 2) {
    const result = BollingerBands.calculate({ values: close, period, stdDev });
    if (!result || result.length === 0) return [0, 0, 0];
    const last = result[result.length - 1];
    return [last?.upper || 0, last?.middle || 0, last?.lower || 0];
}

function computeADX(data, period = 14) {
    const result = ADX.calculate({ high: data.map(d => d.high), low: data.map(d => d.low), close: data.map(d => d.close), period });
    return result && result.length > 0 ? result[result.length - 1]?.adx || 0 : 0;
}

function computeATR(data, period = 14) {
    const result = ATR.calculate({ high: data.map(d => d.high), low: data.map(d => d.low), close: data.map(d => d.close), period });
    return result && result.length > 0 ? result[result.length - 1] || 0 : 0;
}

function computeSupportResistance(data) {
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    return { support: Math.min(...lows), resistance: Math.max(...highs) };
}

// =====================
// PHÂN TÍCH CRYPTO
// =====================
async function getCryptoAnalysis(symbol, pair, timeframe, customThresholds = {}) {
    const df = await fetchKlines(symbol, pair, timeframe);
    if (!df || df.length < WINDOW_SIZE) return { result: '❗ Không thể lấy dữ liệu', confidence: 0 };

    const windowFeatures = [];
    for (let i = df.length - WINDOW_SIZE; i < df.length; i++) {
        windowFeatures.push(computeFeature(df, i));
    }

    const currentPrice = df[df.length - 1].close;
    const atr = computeATR(df);
    const { support, resistance } = computeSupportResistance(df);

    const input = tf.tensor3d([windowFeatures]);
    const prediction = model.predict(input);
    const [longProb, shortProb, waitProb] = prediction.dataSync();
    input.dispose();
    prediction.dispose();

    let signalText, confidence, entry = currentPrice, sl = 0, tp = 0;
    const maxProb = Math.max(longProb, shortProb, waitProb);
    confidence = Math.round(maxProb * 100);

    if (maxProb === longProb) {
        signalText = '🟢 LONG - Mua';
        sl = Math.max(currentPrice - atr * 2, support);
        tp = Math.min(currentPrice + atr * 4, resistance);
    } else if (maxProb === shortProb) {
        signalText = '🔴 SHORT - Bán';
        sl = Math.min(currentPrice + atr * 2, resistance);
        tp = Math.max(currentPrice - atr * 4, support);
    } else {
        signalText = '⚪️ ĐỢI - Chưa có tín hiệu';
    }

    const details = [];
    details.push(`🛡️ Hỗ trợ: ${support.toFixed(4)}, Kháng cự: ${resistance.toFixed(4)}`);
    const timestamp = new Date().toLocaleString('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh' });
    details.push(`⏰ Thời gian: ${timestamp}`);
    details.push(`😨 Fear & Greed Index: ${currentFearGreedValue}`);

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
        details.push(`✅ Độ tin cậy: ${confidence}%`);
        details.push(`🎯 Điểm vào: ${entry.toFixed(4)}`);
        details.push(`🛑 SL: ${sl.toFixed(4)}`);
        details.push(`💰 TP: ${tp.toFixed(4)}`);
        let leverageAdvice = confidence >= 95 ? 'x125' : confidence >= 90 ? 'x50' : confidence >= 85 ? 'x20' : 'x10';
        details.push(`💡 Khuyến nghị đòn bẩy: ${leverageAdvice}`);
    }

    const resultText = `📊 *Phân tích ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})*\n`
        + `💰 Giá: ${currentPrice.toFixed(4)}\n`
        + `⚡️ *${signalText}*\n`
        + details.join('\n');

    return { result: resultText, confidence };
}
// =====================
// SELF-EVALUATE & TRAIN TRONG GIẢ LẬP (LSTM với WINDOW_SIZE)
// =====================
let enableSimulation = true;
let recentAccuracies = [];
let lastAccuracy = 0;
let shouldStopTraining = false;
let trainingCounter = 0;
async function selfEvaluateAndTrain(historicalSlice, currentIndex, fullData) {
    if (shouldStopTraining) return;
    if (historicalSlice.length < WINDOW_SIZE) return;

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

    if (historicalSlice.length < WINDOW_SIZE) return;
    const windowFeatures = [];
    for (let i = historicalSlice.length - WINDOW_SIZE; i < historicalSlice.length; i++) {
        windowFeatures.push(computeFeature(historicalSlice, i));
    }
    const xs = tf.tensor3d([windowFeatures]); // shape [1, WINDOW_SIZE, 6]
    const ys = tf.tensor2d([trueSignal]); // shape [1, 3]
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
            enableSimulation = false; // dừng giả lập

            console.log('✅ Mô hình đã ổn định, dừng giả lập.');
        }
    }
}
// =====================
// CHẾ ĐỘ GIẢ LẬP
// =====================
let lastIndexMap = new Map();
let lastSignalTimestamps = {};
const SIGNAL_COOLDOWN = 10 * 60 * 1000;

async function simulateConfig(config, stepInterval) {
    const { chatId, symbol, pair, timeframe } = config;
    const configKey = `${chatId}_${symbol}_${pair}_${timeframe}`;
    const historicalData = await fetchKlines(symbol, pair, timeframe);
    if (!historicalData) return;

    let currentIndex = lastIndexMap.has(configKey) ? lastIndexMap.get(configKey) : WINDOW_SIZE;

    async function simulateStep() {
        if (currentIndex >= historicalData.length || !enableSimulation) {
            lastIndexMap.delete(configKey);
            return;
        }
        const historicalSlice = historicalData.slice(0, currentIndex);
        if (historicalSlice.length < WINDOW_SIZE) {
            currentIndex++;
            setTimeout(simulateStep, stepInterval);
            return;
        }
        const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe);
        const now = Date.now();
        if (confidence >= 75 && (!lastSignalTimestamps[configKey] || (now - lastSignalTimestamps[configKey] > SIGNAL_COOLDOWN))) {
            bot.sendMessage(chatId, `🚨 *TÍN HIỆU GIẢ LẬP ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})* 🚨\n${result}`, { parse_mode: 'Markdown' });
            lastSignalTimestamps[configKey] = now;
        }
        if (!shouldStopTraining) await selfEvaluateAndTrain(historicalSlice, currentIndex, historicalData);
        lastIndexMap.set(configKey, currentIndex + 1);
        currentIndex++;
        setTimeout(simulateStep, stepInterval);
    }
    console.log(`Bắt đầu giả lập ${symbol}/${pair} (${timeframes[timeframe]}) từ nến ${currentIndex}...`);
    simulateStep();
}

async function simulateRealTimeForConfigs(stepInterval = 1000) {
    const configs = await loadWatchConfigs();
    if (!configs || configs.length === 0) return;
    for (let config of configs) {
        simulateConfig(config, stepInterval);
    }
}

// =====================
// LỆNH BOT
// =====================
const autoWatchList = new Map();

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

bot.onText(/\?(.+)/, async (msg, match) => {
    try {
        const parts = match[1].split(',').map(p => p.trim());
        if (parts.length < 3) return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Ví dụ: ?ada,usdt,5m');
        const [symbol, pair, timeframeInput] = parts.map(p => p.toLowerCase());
        const timeframe = normalizeTimeframe(timeframeInput);
        if (!timeframes[timeframe]) return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ!`);
        const valid = await isValidMarket(symbol, pair);
        if (!valid) return bot.sendMessage(msg.chat.id, `⚠️ Cặp ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại!`);
        const { result } = await getCryptoAnalysis(symbol, pair, timeframe);
        bot.sendMessage(msg.chat.id, result, { parse_mode: 'Markdown' });
    } catch (error) {
        bot.sendMessage(msg.chat.id, `❌ Lỗi phân tích: ${error.message}`);
    }
});

bot.onText(/\/tinhieu (.+)/, async (msg, match) => {
    try {
        const parts = match[1].split(',').map(p => p.trim());
        if (parts.length < 3) return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Ví dụ: /tinhieu ada,usdt,5m');
        const [symbol, pair, timeframeInput] = parts.map(p => p.toLowerCase());
        const timeframe = normalizeTimeframe(timeframeInput);
        if (!timeframes[timeframe]) return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ!`);
        const valid = await isValidMarket(symbol, pair);
        if (!valid) return bot.sendMessage(msg.chat.id, `⚠️ Cặp ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại!`);
        const chatId = msg.chat.id;
        if (!autoWatchList.has(chatId)) autoWatchList.set(chatId, []);
        const watchList = autoWatchList.get(chatId);
        if (!watchList.some(w => w.symbol === symbol && w.pair === pair && w.timeframe === timeframe)) {
            watchList.push({ symbol, pair, timeframe });
            addWatchConfig(chatId, symbol, pair, timeframe, (err) => {
                if (err) console.error('Lỗi lưu cấu hình:', err.message);
            });
            bot.sendMessage(msg.chat.id, `✅ Đã bật theo dõi ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})`);
            simulateConfig({ chatId, symbol, pair, timeframe }, 1000);
        } else {
            bot.sendMessage(msg.chat.id, 'ℹ️ Bạn đã theo dõi cặp này rồi!');
        }
    } catch (error) {
        bot.sendMessage(msg.chat.id, `❌ Lỗi /tinhieu: ${error.message}`);
    }
});

bot.onText(/\/dungtinhieu (.+)/, (msg, match) => {
    try {
        const parts = match[1].split(',').map(p => p.trim());
        if (parts.length < 3) return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Ví dụ: /dungtinhieu ada,usdt,5m');
        const [symbol, pair, timeframeInput] = parts.map(p => p.toLowerCase());
        const timeframe = normalizeTimeframe(timeframeInput);
        if (!timeframes[timeframe]) return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ!`);
        const chatId = msg.chat.id;
        if (!autoWatchList.has(chatId)) return bot.sendMessage(chatId, 'ℹ️ Bạn chưa theo dõi cặp nào.');
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

bot.onText(/\/trogiup/, (msg) => {
    const helpMessage = `
📚 *HƯỚNG DẪN SỬ DỤNG BOT GIAO DỊCH*
1. **?symbol,pair,timeframe** - Phân tích thủ công. Ví dụ: ?ada,usdt,5m
2. **/tinhieu symbol,pair,timeframe** - Bật theo dõi tự động. Ví dụ: /tinhieu ada,usdt,5m
3. **/dungtinhieu symbol,pair,timeframe** - Dừng theo dõi. Ví dụ: /dungtinhieu ada,usdt,5m
4. **/trogiup** - Hiển thị hướng dẫn này.
`;
    bot.sendMessage(msg.chat.id, helpMessage, { parse_mode: 'Markdown' });
});

// Kiểm tra tự động
function startAutoChecking() {
    const CHECK_INTERVAL = 1 * 60 * 1000;
    setInterval(() => {
        for (const [chatId, watchList] of autoWatchList) {
            watchList.forEach(config => checkAutoSignal(chatId, config));
        }
    }, CHECK_INTERVAL);
}

async function checkAutoSignal(chatId, { symbol, pair, timeframe }, confidenceThreshold = 75) {
    try {
        const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe);
        if (confidence >= confidenceThreshold) {
            bot.sendMessage(chatId, `🚨 *TÍN HIỆU ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})* 🚨\n${result}`, { parse_mode: 'Markdown' });
        }
    } catch (error) {
        console.error(`❌ Lỗi checkAutoSignal ${symbol}/${pair}: ${error.message}`);
    }
}

// =====================
// DYNAMIC TRAINING CONTROL
// =====================
function dynamicTrainingControl() {
    if (recentAccuracies.length < 50) return;
    const avgAcc = recentAccuracies.reduce((sum, val) => sum + val, 0) / recentAccuracies.length;
    const maxAcc = Math.max(...recentAccuracies);
    const minAcc = Math.min(...recentAccuracies);
    if (avgAcc > 0.85 && (maxAcc - minAcc) < 0.05) {
        enableSimulation = false;
        console.log("Dynamic Training Control: Mô hình ổn định, dừng giả lập.");
    } else {
        enableSimulation = true;
        console.log("Dynamic Training Control: Hiệu suất chưa ổn định, tiếp tục giả lập.");
    }
}
setInterval(dynamicTrainingControl, 10 * 60 * 1000);

// =====================
// KHỞI ĐỘNG BOT
// =====================
(async () => {
    await initializeModel();
    const initialData = await fetchKlines('BTC', 'USDT', '1h', 200);
    if (initialData) await trainModelData(initialData);
    else console.error('❌ Không thể lấy dữ liệu ban đầu để huấn luyện mô hình');
    console.log('✅ Bot đã khởi động và sẵn sàng nhận lệnh.');
    startAutoChecking();
    simulateRealTimeForConfigs(1000);
})();

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