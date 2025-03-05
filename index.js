const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR } = require('technicalindicators');
const tf = require('@tensorflow/tfjs');
const fs = require('fs').promises;

// Configuration
const TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY';
const BINANCE_API = 'https://api.binance.com/api/v3';
const bot = new TelegramBot(TOKEN, { polling: true });

// Danh sách khung thời gian Binance hỗ trợ
const validTimeframes = {
    '1m': '1 phút', '3m': '3 phút', '5m': '5 phút', '15m': '15 phút', '30m': '30 phút',
    '1h': '1 giờ', '2h': '2 giờ', '4h': '4 giờ', '6h': '6 giờ', '8h': '8 giờ', '12h': '12 giờ',
    '1d': '1 ngày', '3d': '3 ngày', '1w': '1 tuần', '1M': '1 tháng'
};

// Danh sách theo dõi tự động và lịch sử tín hiệu
const autoWatchList = new Map();
const signalHistory = new Map(); // chatId -> [{symbol, pair, timeframe, signal, confidence, timestamp}]
const WATCHLIST_FILE = './watchlist.json';
const HISTORY_FILE = './signal_history.json';

// Khởi tạo mô hình LSTM
let model;
async function initializeModel() {
    model = tf.sequential();
    model.add(tf.layers.lstm({ units: 50, inputShape: [10, 6], returnSequences: true }));
    model.add(tf.layers.lstm({ units: 20 }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    console.log('✅ Mô hình LSTM đã được khởi tạo');
}

// Huấn luyện mô hình với dữ liệu đa dạng
async function trainModel() {
    const pairs = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'];
    const timeframes = ['1h', '4h', '1d'];
    const inputs = [];
    const outputs = [];

    for (const pair of pairs) {
        for (const timeframe of timeframes) {
            const data = await fetchKlines(pair.slice(0, -4), pair.slice(-4), timeframe, 500);
            if (!data) continue;

            for (let i = 10; i < data.length; i++) {
                const window = data.slice(i - 10, i);
                const curr = data[i];
                const close = window.map(d => d.close);
                const volume = window.map(d => d.volume);
                const rsi = computeRSI(close);
                const ma10 = computeMA(close, 10);
                const ma50 = computeMA(close, 50);
                const [, , histogram] = computeMACD(close);
                const [, middleBB] = computeBollingerBands(close);
                const adx = computeADX(window);
                const volumeMA = computeMA(volume, 20);
                const volumeSpike = curr.volume > volumeMA * 1.5 ? 1 : 0;

                const input = window.map(d => [
                    RSI.calculate({ values: window.map(w => w.close), period: 14 }).slice(-1)[0] || 50,
                    computeADX(window),
                    MACD.calculate({ values: window.map(w => w.close), fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 }).slice(-1)[0]?.histogram || 0,
                    d.volume > computeMA(window.map(w => w.volume), 20) * 1.5 ? 1 : 0,
                    computeMA(window.map(w => w.close), 10) - computeMA(window.map(w => w.close), 50),
                    d.close - computeBollingerBands(window.map(w => w.close))[1]
                ]);
                inputs.push(input);

                let signal = [0, 0, 1];
                if (adx > 30) {
                    if (rsi < 30 && ma10 > ma50 && histogram > 0 && volumeSpike && curr.close < middleBB) signal = [1, 0, 0];
                    else if (rsi > 70 && ma10 < ma50 && histogram < 0 && volumeSpike && curr.close > middleBB) signal = [0, 1, 0];
                }
                outputs.push(signal);
            }
        }
    }

    const trainSize = Math.floor(inputs.length * 0.8);
    const xsTrain = tf.tensor3d(inputs.slice(0, trainSize));
    const ysTrain = tf.tensor2d(outputs.slice(0, trainSize));
    const xsTest = tf.tensor3d(inputs.slice(trainSize));
    const ysTest = tf.tensor2d(outputs.slice(trainSize));

    await model.fit(xsTrain, ysTrain, {
        epochs: 20,
        batchSize: 32,
        shuffle: true,
        validationData: [xsTest, ysTest],
        callbacks: {
            onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: Loss = ${logs.loss}, Accuracy = ${logs.acc}, Val Accuracy = ${logs.val_acc}`)
        }
    });

    console.log('✅ Mô hình đã được huấn luyện với dữ liệu đa dạng');
    xsTrain.dispose();
    ysTrain.dispose();
    xsTest.dispose();
    ysTest.dispose();
}

// Lưu và tải watchlist
async function saveWatchList() {
    const data = Array.from(autoWatchList.entries()).map(([chatId, list]) => ({ chatId, list }));
    await fs.writeFile(WATCHLIST_FILE, JSON.stringify(data));
}

async function loadWatchList() {
    try {
        const data = await fs.readFile(WATCHLIST_FILE, 'utf8');
        const parsed = JSON.parse(data);
        parsed.forEach(({ chatId, list }) => autoWatchList.set(chatId, list));
    } catch (error) {
        console.log('ℹ️ Không có watchlist cũ để tải');
    }
}

// Lưu và tải lịch sử tín hiệu
async function saveSignalHistory() {
    const data = Array.from(signalHistory.entries()).map(([chatId, list]) => ({ chatId, list }));
    await fs.writeFile(HISTORY_FILE, JSON.stringify(data));
}

async function loadSignalHistory() {
    try {
        const data = await fs.readFile(HISTORY_FILE, 'utf8');
        const parsed = JSON.parse(data);
        parsed.forEach(({ chatId, list }) => signalHistory.set(chatId, list));
    } catch (error) {
        console.log('ℹ️ Không có lịch sử tín hiệu cũ để tải');
    }
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
        return false;
    }
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

async function getCryptoAnalysis(chatId, symbol, pair, timeframe, customThresholds = {}) {
    bot.sendMessage(chatId, '⏳ Đang phân tích...');
    const df = await fetchKlines(symbol, pair, timeframe, 210);
    if (!df) return { result: '❗ Không thể lấy dữ liệu', confidence: 0 };

    const close = df.map(d => d.close);
    const volume = df.map(d => d.volume);
    const currentPrice = close[close.length - 1];
    const rsi = computeRSI(close);
    const ma10 = computeMA(close, 10);
    const ma50 = computeMA(close, 50);
    const ma200 = computeMA(close, 200);
    const [macd, signalLine, histogram] = computeMACD(close);
    const [upperBB, middleBB, lowerBB] = computeBollingerBands(close);
    const adx = computeADX(df);
    const atr = computeATR(df);
    const volumeMA = computeMA(volume, 20);
    const volumeSpike = volume[volume.length - 1] > volumeMA * 1.5 ? 1 : 0;
    const { support, resistance } = computeSupportResistance(df);

    const bbWidth = upperBB - lowerBB;
    const avgBBWidth = computeMA(df.map(d => {
        const bb = BollingerBands.calculate({ values: df.map(v => v.close), period: 20, stdDev: 2 });
        return bb[bb.length - 1].upper - bb[bb.length - 1].lower;
    }), 20);
    const isSideways = adx < 20 && bbWidth < avgBBWidth * 0.8;

    const rsiOverbought = customThresholds.rsiOverbought || 70;
    const rsiOversold = customThresholds.rsiOversold || 30;
    const adxStrongTrend = customThresholds.adxStrongTrend || 30;

    const input = tf.tensor3d([df.slice(-10).map(d => [
        RSI.calculate({ values: df.slice(-14).map(w => w.close), period: 14 }).slice(-1)[0] || 50,
        computeADX(df.slice(-14)),
        MACD.calculate({ values: df.slice(-26).map(w => w.close), fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 }).slice(-1)[0]?.histogram || 0,
        d.volume > computeMA(df.slice(-20).map(w => w.volume), 20) * 1.5 ? 1 : 0,
        computeMA(df.slice(-10).map(w => w.close), 10) - computeMA(df.slice(-50).map(w => w.close), 50),
        d.close - computeBollingerBands(df.slice(-20).map(w => w.close))[1]
    ])]);
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
        if (ruleBasedSignal.includes('LONG')) confidence = Math.max(confidence, ruleConfidence);
    } else if (maxProb === shortProb) {
        signalText = '🔴 SHORT - Bán';
        entry = currentPrice;
        sl = Math.min(currentPrice + atr * 2, resistance);
        tp = Math.max(currentPrice - atr * 4, support);
        if (ruleBasedSignal.includes('SHORT')) confidence = Math.max(confidence, ruleConfidence);
    } else {
        signalText = '⚪️ ĐỢI - Chưa có tín hiệu';
        confidence = Math.min(confidence, ruleConfidence);
    }

    let details = [
        `📈 RSI: ${rsi.toFixed(1)}`,
        `📊 MACD: ${macd.toFixed(4)} / ${signalLine.toFixed(4)}`,
        `📉 ADX: ${adx.toFixed(1)}`,
        `📦 Volume: ${volumeSpike ? 'TĂNG ĐỘT BIẾN' : 'BÌNH THƯỜNG'}`,
        `📈 MA200: ${ma200.toFixed(4)} (Xu hướng dài hạn: ${currentPrice > ma200 ? 'TĂNG' : 'GIẢM'})`
    ];
    details.push(`📏 Bollinger: ${lowerBB.toFixed(4)} - ${upperBB.toFixed(4)}`);
    details.push(`🛡️ Hỗ trợ: ${support.toFixed(4)}, Kháng cự: ${resistance.toFixed(4)}`);
    if (isSideways) {
        details.push(`⚠️ Lưu ý: Thị trường đang đi ngang, tín hiệu có thể không chính xác`);
    }
    if (signalText !== '⚪️ ĐỢI - Chưa có tín hiệu') {
        details.push(`✅ Độ tin cậy: ${confidence}%`);
        details.push(`🎯 Điểm vào: ${entry.toFixed(4)}`);
        details.push(`🛑 SL: ${sl.toFixed(4)}`);
        details.push(`💰 TP: ${tp.toFixed(4)}`);
    }

    const result = `📊 *Phân tích ${symbol}/${pair} (${validTimeframes[timeframe] || timeframe})*\n💰 Giá: ${currentPrice.toFixed(4)}\n⚡️ *${signalText}*\n${details.join('\n')}`;
    return { result, confidence, signalText };
}

// Hàm kiểm tra tự động và gửi tín hiệu
async function checkAutoSignal(chatId, { symbol, pair, timeframe, confidenceThreshold = 80 }) {
    const { result, confidence, signalText } = await getCryptoAnalysis(chatId, symbol, pair, timeframe);
    if (confidence >= confidenceThreshold) {
        bot.sendMessage(chatId, `🚨 *TÍN HIỆU ${symbol.toUpperCase()}/${pair.toUpperCase()} (${validTimeframes[timeframe] || timeframe})* 🚨\n${result}`, {
            parse_mode: 'Markdown'
        });
        console.log(`✅ Đã gửi tín hiệu ${symbol}/${pair} đến chat ${chatId} - Độ tin cậy: ${confidence}%`);

        if (!signalHistory.has(chatId)) signalHistory.set(chatId, []);
        signalHistory.get(chatId).push({ symbol, pair, timeframe, signal: signalText, confidence, timestamp: Date.now() });
        await saveSignalHistory();
    }
}

// Hàm chạy kiểm tra định kỳ
function startAutoChecking() {
    const CHECK_INTERVAL = 5 * 60 * 1000; // 5 phút
    setInterval(async () => {
        for (const [chatId, watchList] of autoWatchList) {
            for (const config of watchList) {
                await checkAutoSignal(chatId, config).catch(err => console.error(`❌ Lỗi kiểm tra ${config.symbol}/${config.pair}: ${err.message}`));
            }
        }
        await saveWatchList();
    }, CHECK_INTERVAL);
}

// Huấn luyện định kỳ
function startPeriodicTraining() {
    const TRAIN_INTERVAL = 24 * 60 * 60 * 1000; // 24 giờ
    setInterval(async () => {
        console.log('⏳ Bắt đầu huấn luyện lại mô hình...');
        await trainModel();
    }, TRAIN_INTERVAL);
}

// Khởi động bot
(async () => {
    await initializeModel();
    await trainModel();
    await loadWatchList();
    await loadSignalHistory();

    // Lệnh phân tích thủ công
    bot.onText(/\?(.+)/, async (msg, match) => {
        const chatId = msg.chat.id;
        const parts = match[1].split(',');
        if (parts.length < 3) {
            return bot.sendMessage(chatId, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: ?ada,usdt,5m\nHoặc: ?ada,usdt,5m,rsi25-75 (tùy chọn RSI)');
        }
        const [symbol, pair, timeframe, customThreshold] = parts.map(p => p.trim().toLowerCase());
        if (!validTimeframes[timeframe]) {
            return bot.sendMessage(chatId, `⚠️ Khung thời gian không hợp lệ! Hãy dùng một trong các khung sau: ${Object.keys(validTimeframes).join(', ')}\nVí dụ: ?ada,usdt,5m`);
        }

        const isValid = await isValidMarket(symbol, pair);
        if (!isValid) {
            return bot.sendMessage(chatId, `⚠️ Cặp giao dịch ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại trên Binance!\nVui lòng kiểm tra lại, ví dụ: ?ada,usdt,5m`);
        }

        let customThresholds = {};
        if (customThreshold && customThreshold.startsWith('rsi')) {
            const [oversold, overbought] = customThreshold.replace('rsi', '').split('-').map(Number);
            if (!isNaN(oversold) && !isNaN(overbought) && oversold < overbought) {
                customThresholds.rsiOversold = oversold;
                customThresholds.rsiOverbought = overbought;
            } else {
                return bot.sendMessage(chatId, '⚠️ Định dạng RSI không hợp lệ! Hãy nhập theo kiểu: rsi25-75\nVí dụ: ?ada,usdt,5m,rsi25-75');
            }
        }

        const { result, confidence } = await getCryptoAnalysis(chatId, symbol, pair, timeframe, customThresholds);
        bot.sendMessage(chatId, result, { parse_mode: 'Markdown' });
    });

    // Lệnh yêu cầu theo dõi tự động tín hiệu
    bot.onText(/\/tinhieu (.+)/, async (msg, match) => {
        const chatId = msg.chat.id;
        const parts = match[1].split(',');
        if (parts.length < 3) {
            return bot.sendMessage(chatId, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: /tinhieu ada,usdt,5m[,90] (tùy chọn ngưỡng confidence)');
        }
        const [symbol, pair, timeframe, confidenceThreshold] = parts.map(p => p.trim().toLowerCase());
        if (!validTimeframes[timeframe]) {
            return bot.sendMessage(chatId, `⚠️ Khung thời gian không hợp lệ! Hãy dùng một trong các khung sau: ${Object.keys(validTimeframes).join(', ')}\nVí dụ: /tinhieu ada,usdt,5m`);
        }

        const isValid = await isValidMarket(symbol, pair);
        if (!isValid) {
            return bot.sendMessage(chatId, `⚠️ Cặp giao dịch ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại trên Binance!\nVui lòng kiểm tra lại, ví dụ: /tinhieu ada,usdt,5m`);
        }

        const config = {
            symbol,
            pair,
            timeframe,
            confidenceThreshold: confidenceThreshold && !isNaN(parseInt(confidenceThreshold)) ? parseInt(confidenceThreshold) : 80
        };

        if (!autoWatchList.has(chatId)) autoWatchList.set(chatId, []);
        const watchList = autoWatchList.get(chatId);

        if (!watchList.some(w => w.symbol === symbol && w.pair === pair && w.timeframe === timeframe)) {
            watchList.push(config);
            await saveWatchList();
            bot.sendMessage(chatId, `✅ Bắt đầu theo dõi tín hiệu ${symbol.toUpperCase()}/${pair.toUpperCase()} (${validTimeframes[timeframe] || timeframe}) với ngưỡng confidence ${config.confidenceThreshold}%.`);
        } else {
            bot.sendMessage(chatId, `ℹ️ ${symbol.toUpperCase()}/${pair.toUpperCase()} (${validTimeframes[timeframe] || timeframe}) đã được theo dõi rồi.`);
        }
    });

    // Lệnh xem danh sách theo dõi
    bot.onText(/\/danhsach/, (msg) => {
        const chatId = msg.chat.id;
        if (!autoWatchList.has(chatId) || autoWatchList.get(chatId).length === 0) {
            return bot.sendMessage(chatId, 'ℹ️ Bạn chưa theo dõi cặp giao dịch nào.');
        }
        const watchList = autoWatchList.get(chatId);
        const listText = watchList.map(w => `${w.symbol.toUpperCase()}/${w.pair.toUpperCase()} (${validTimeframes[w.timeframe] || w.timeframe}, ngưỡng: ${w.confidenceThreshold}%)`).join('\n');
        bot.sendMessage(chatId, `📋 *Danh sách theo dõi tự động:*\n${listText}`, { parse_mode: 'Markdown' });
    });

    // Lệnh dừng theo dõi
    bot.onText(/\/dungtinhieu (.+)/, async (msg, match) => {
        const chatId = msg.chat.id;
        const parts = match[1].split(',');
        if (parts.length < 3) {
            return bot.sendMessage(chatId, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: /dungtinhieu ada,usdt,5m');
        }
        const [symbol, pair, timeframe] = parts.map(p => p.trim().toLowerCase());
        if (!validTimeframes[timeframe]) {
            return bot.sendMessage(chatId, `⚠️ Khung thời gian không hợp lệ! Hãy dùng một trong các khung sau: ${Object.keys(validTimeframes).join(', ')}\nVí dụ: /dungtinhieu ada,usdt,5m`);
        }

        if (!autoWatchList.has(chatId)) {
            return bot.sendMessage(chatId, 'ℹ️ Bạn chưa theo dõi cặp nào.');
        }

        const watchList = autoWatchList.get(chatId);
        const index = watchList.findIndex(w => w.symbol === symbol && w.pair === pair && w.timeframe === timeframe);
        if (index !== -1) {
            watchList.splice(index, 1);
            if (watchList.length === 0) autoWatchList.delete(chatId);
            await saveWatchList();
            bot.sendMessage(chatId, `✅ Đã dừng theo dõi ${symbol.toUpperCase()}/${pair.toUpperCase()} (${validTimeframes[timeframe] || timeframe}).`);
        } else {
            bot.sendMessage(chatId, `ℹ️ Không tìm thấy ${symbol.toUpperCase()}/${pair.toUpperCase()} (${validTimeframes[timeframe] || timeframe}) trong danh sách theo dõi.`);
        }
    });

    // Lệnh xem lịch sử tín hiệu
    bot.onText(/\/lichsu/, (msg) => {
        const chatId = msg.chat.id;
        if (!signalHistory.has(chatId) || signalHistory.get(chatId).length === 0) {
            return bot.sendMessage(chatId, 'ℹ️ Chưa có tín hiệu nào được ghi nhận trong 24 giờ qua.');
        }

        const history = signalHistory.get(chatId).filter(s => Date.now() - s.timestamp < 24 * 60 * 60 * 1000);
        if (history.length === 0) {
            return bot.sendMessage(chatId, 'ℹ️ Không có tín hiệu nào trong 24 giờ qua.');
        }

        const historyText = history.map(s => `${s.signal} - ${s.symbol.toUpperCase()}/${s.pair.toUpperCase()} (${validTimeframes[s.timeframe] || s.timeframe}) - ${s.confidence}% (${new Date(s.timestamp).toLocaleString()})`).join('\n');
        bot.sendMessage(chatId, `📜 *Lịch sử tín hiệu (24h qua):*\n${historyText}`, { parse_mode: 'Markdown' });
    });

    // Lệnh trợ giúp
    bot.onText(/\/trogiup/, (msg) => {
        const chatId = msg.chat.id;
        const helpMessage = `
📚 *HƯỚNG DẪN SỬ DỤNG BOT GIAO DỊCH*

Dưới đây là các lệnh hiện có và cách sử dụng:

1. **?symbol,pair,timeframe[,rsiOversold-rsiOverbought]**  
   - *Mô tả*: Phân tích thủ công cặp giao dịch, trả về tín hiệu và các mức giá (entry, SL, TP).  
   - *Cú pháp*: ?<coin>,<đồng giao dịch>,<khung thời gian>[,rsi<giá trị thấp>-<giá trị cao>]  
   - *Ví dụ*:  
     - ?ada,usdt,5m  
     - ?btc,usdt,1h,rsi25-75  

2. **/tinhieu symbol,pair,timeframe[,confidenceThreshold]**  
   - *Mô tả*: Kích hoạt theo dõi tự động, gửi tín hiệu khi độ tin cậy ≥ ngưỡng (mặc định 80%).  
   - *Cú pháp*: /tinhieu <coin>,<đồng giao dịch>,<khung thời gian>[,<ngưỡng confidence>]  
   - *Ví dụ*:  
     - /tinhieu ada,usdt,5m  
     - /tinhieu btc,usdt,1h,90  

3. **/danhsach**  
   - *Mô tả*: Xem danh sách các cặp đang theo dõi tự động.  
   - *Cú pháp*: /danhsach  
   - *Ví dụ*: /danhsach  

4. **/dungtinhieu symbol,pair,timeframe**  
   - *Mô tả*: Dừng theo dõi tự động một cặp giao dịch.  
   - *Cú pháp*: /dungtinhieu <coin>,<đồng giao dịch>,<khung thời gian>  
   - *Ví dụ*: /dungtinhieu ada,usdt,5m  

5. **/lichsu**  
   - *Mô tả*: Xem lịch sử tín hiệu trong 24 giờ qua.  
   - *Cú pháp*: /lichsu  
   - *Ví dụ*: /lichsu  

6. **/trogiup**  
   - *Mô tả*: Hiển thị hướng dẫn này.  
   - *Cú pháp*: /trogiup  
   - *Ví dụ*: /trogiup  

*Khung thời gian hợp lệ*: ${Object.keys(validTimeframes).join(', ')}  
*Lưu ý*:  
- Bot sử dụng AI (LSTM) và chỉ báo kỹ thuật để phân tích.  
- Mô hình được huấn luyện lại mỗi 24 giờ với dữ liệu mới.  
- Đảm bảo nhập đúng cặp giao dịch trên Binance (ví dụ: ADA/USDT).  
        `;
        bot.sendMessage(chatId, helpMessage, { parse_mode: 'Markdown' });
    });

    // Bắt đầu kiểm tra và huấn luyện
    startAutoChecking();
    startPeriodicTraining();
    console.log('✅ Bot đang chạy với tính năng nâng cao...');
})();