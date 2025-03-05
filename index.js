const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR } = require('technicalindicators');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const sqlite3 = require('sqlite3').verbose();

// Configuration
const TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY';
const BINANCE_API = 'https://api.binance.com/api/v3';
const bot = new TelegramBot(TOKEN, { polling: true });

const timeframes = {
    '1m': '1 phút',
    '5m': '5 phút',
    '15m': '15 phút',
    '30m': '30 phút',
    '1h': '1 giờ',
    '2h': '2 giờ',
    '4h': '4 giờ',
    '6h': '6 giờ',
    '8h': '8 giờ',
    '12h': '12 giờ',
    '1d': '1 ngày',
    '3d': '3 ngày',
    '1w': '1 tuần',
    '1M': '1 tháng'
};

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

// Tạo bảng để lưu trữ cấu hình theo dõi
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

// Prepare statements
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
    console.log('✅ Mô hình TensorFlow.js đã được khởi tạo (LSTM)');
    fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Mô hình TensorFlow.js đã được khởi tạo (LSTM)\n`);
}

// Huấn luyện mô hình với nhãn từ logic rule-based
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

        // Chuẩn hóa dữ liệu
        const normalizedRsi = rsi / 100;
        const normalizedAdx = adx / 100;
        const normalizedHistogram = histogram / 1000;
        const normalizedMaDiff = (ma10 - ma50) / curr.close;
        const normalizedBbDiff = (curr.close - middleBB) / curr.close;
        const normalizedAtr = atr / curr.close;

        // Lagged features
        const closeLag1 = data[i - 1].close / curr.close;
        const closeLag2 = data[i - 2].close / curr.close;

        // Rolling statistics
        const rsiRollingMean = SMA.calculate({ values: close.slice(-5).map(d => computeRSI([d])), period: 5 })[0] / 100 || normalizedRsi;

        // Tạo input array
        inputs.push([[normalizedRsi, normalizedAdx, normalizedHistogram, volumeSpike, normalizedMaDiff, normalizedBbDiff, closeLag1, closeLag2, normalizedAtr, rsiRollingMean]]);

        let signal = [0, 0, 1]; // ĐỢI mặc định
        if (adx > 30) {
            if (rsi < 30 && ma10 > ma50 && histogram > 0 && volumeSpike && curr.close < middleBB) signal = [1, 0, 0];
            else if (rsi > 70 && ma10 < ma50 && histogram < 0 && volumeSpike && curr.close > middleBB) signal = [0, 1, 0];
        }
        outputs.push(signal);
    }

    const xs = tf.tensor3d(inputs);
    const ys = tf.tensor2d(outputs);
    await model.fit(xs, ys, { epochs: 30, batchSize: 32, shuffle: true });
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

async function fetchKlines(symbol, pair, timeframe, limit = 1000) {
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
    if (!df) return { result: '❗ Không thể lấy dữ liệu', confidence: 0 };

    const close = df.map(d => d.close);
    const volume = df.map(d => d.volume);
    const currentPrice = close[close.length - 1];
    const rsi = computeRSI(close);
    const ma10 = computeMA(close, 10);
    const ma50 = computeMA(close, 50);
    const [macd, signal, histogram] = computeMACD(close);
    const bbResult = BollingerBands.calculate({ values: close, period: 20, stdDev: 2 });
    const upperBB = bbResult[bbResult.length - 1]?.upper || 0;
    const middleBB = bbResult[bbResult.length - 1]?.middle || 0;
    const lowerBB = bbResult[bbResult.length - 1]?.lower || 0;
    const adx = computeADX(df);
    const atrPeriod = 14;
    const atr = computeATR(df.slice(-atrPeriod));
    const volumeMA = computeMA(volume, 20);
    const volumeSpike = volume[volume.length - 1] > volumeMA * 1.5 ? 1 : 0;
    const { support, resistance } = computeSupportResistance(df.slice(-50));

    const bbWidth = upperBB - lowerBB;
    const avgBBWidth = computeMA(df.map(d => BollingerBands.calculate({ values: df.map(v => v.close), period: 20, stdDev: 2 }).slice(-1)[0].upper - BollingerBands.calculate({ values: df.map(v => v.close), period: 20, stdDev: 2 }).slice(-1)[0].lower), 20);
    const isSideways = adx < 20 && bbWidth < avgBBWidth * 0.8;

    const rsiOverbought = customThresholds.rsiOverbought || 70;
    const rsiOversold = customThresholds.rsiOversold || 30;
    const adxStrongTrend = customThresholds.adxStrongTrend || 30;

    // Chuẩn hóa dữ liệu đầu vào cho mô hình
    const normalizedRsi = rsi / 100;
    const normalizedAdx = adx / 100;
    const normalizedHistogram = histogram / 1000;
    const normalizedMaDiff = (ma10 - ma50) / currentPrice;
    const normalizedBbDiff = (currentPrice - middleBB) / currentPrice;
    const normalizedAtr = atr / currentPrice;

    // Lagged features
    const closeLag1 = df[df.length - 2]?.close / currentPrice || 1;
    const closeLag2 = df[df.length - 3]?.close / currentPrice || 1;

    // Rolling statistics
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

    // Kết hợp AI và quy tắc
    const combinedConfidence = (confidence * 0.7 + ruleConfidence * 0.3);
    confidence = Math.round(combinedConfidence);

    // Dynamic Stop Loss and Take Profit based on ATR
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

    const result = `📊 *Phân tích ${symbol}/${pair} (${timeframes[timeframe]})*\n💰 Giá: ${currentPrice.toFixed(4)}\n⚡️ *${signalText}*\n${details.join('\n')}`;
    return { result, confidence };
}


// Hàm kiểm tra tự động và gửi tín hiệu
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

// Hàm chạy kiểm tra định kỳ
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
                checkAutoSignal(config.chatId, config)
                    .catch(err => {
                        console.error(`❌ Lỗi kiểm tra ${config.symbol}/${config.pair}: ${err.message}`);
                        fs.appendFileSync('bot.log', `${new Date().toISOString()} - ❌ Lỗi kiểm tra ${config.symbol}/${config.pair}: ${err.message}\n`);
                    });
            });
            setTimeout(checkAndReschedule, CHECK_INTERVAL);
        });
    }
    setTimeout(checkAndReschedule, CHECK_INTERVAL);
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
    } catch (error) {
        console.error("Lỗi khởi tạo hoặc huấn luyện mô hình:", error);
        fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi khởi tạo hoặc huấn luyện mô hình: ${error.message}\n`);
    }


    // Lệnh phân tích thủ công
    bot.onText(/\?(.+)/, async (msg, match) => {
        const parts = match[1].split(',');
        if (parts.length < 3) {
            return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: ?ada,usdt,5m\nHoặc: ?ada,usdt,5m,rsi25-75 (tùy chọn RSI)');
        }
        const [symbol, pair, timeframe, customThreshold] = parts.map(p => p.trim().toLowerCase());
        if (!timeframes[timeframe]) {
            return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ! Hãy dùng một trong các khung sau: ${Object.keys(timeframes).join(', ')}\nVí dụ: ?ada,usdt,5m`);
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

    // Lệnh yêu cầu theo dõi tự động tín hiệu
    bot.onText(/\/tinhieu (.+)/, async (msg, match) => {
        const parts = match[1].split(',');
        if (parts.length < 3) {
            return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: /tinhieu ada,usdt,5m');
        }
        const [symbol, pair, timeframe] = parts.map(p => p.trim().toLowerCase());
        if (!timeframes[timeframe]) {
            return bot.sendMessage(msg.chat.id, `⚠️ Khung thời gian không hợp lệ! Hãy dùng một trong các khung sau: ${Object.keys(timeframes).join(', ')}\nVí dụ: /tinhieu ada,usdt,5m`);
        }

        const isValid = await isValidMarket(symbol, pair);
        if (!isValid) {
            return bot.sendMessage(msg.chat.id, `⚠️ Cặp giao dịch ${symbol.toUpperCase()}/${pair.toUpperCase()} không tồn tại trên Binance!\nVui lòng kiểm tra lại, ví dụ: /tinhieu ada,usdt,5m`);
        }

        const chatId = msg.chat.id;
        const config = { symbol, pair, timeframe };

        insertStmt.run(chatId, symbol, pair, timeframe, (err) => {
            if (err) {
                console.error(err.message);
                bot.sendMessage(msg.chat.id, `❌ Lỗi khi thêm cấu hình theo dõi: ${err.message}`);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi thêm config vào SQLite: ${err.message}\n`);
            } else {
                bot.sendMessage(chatId, `✅ Bắt đầu theo dõi tín hiệu ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]}). Bot sẽ gửi cảnh báo khi có tín hiệu mạnh.`);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Thêm config ${symbol}/${pair} cho chat ${chatId} thành công.\n`);
            }
        });
    });

    bot.onText(/\/dungtinhieu (.+)/, (msg, match) => {
        const parts = match[1].split(',');
        if (parts.length < 3) {
            return bot.sendMessage(msg.chat.id, '⚠️ Cú pháp sai! Hãy nhập đúng định dạng:\nVí dụ: /dungtinhieu ada,usdt,5m');
        }
        const [symbol, pair, timeframe] = parts.map(p => p.trim().toLowerCase());
        const chatId = msg.chat.id;

        deleteStmt.run(chatId, symbol, pair, timeframe, (err) => {
            if (err) {
                console.error(err.message);
                bot.sendMessage(msg.chat.id, `❌ Lỗi khi xóa cấu hình theo dõi: ${err.message}`);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi xóa config khỏi SQLite: ${err.message}\n`);
            } else {
                bot.sendMessage(chatId, `✅ Đã dừng theo dõi tín hiệu ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})`);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Xóa config ${symbol}/${pair} cho chat ${chatId} thành công.\n`);
            }
        });
    });

    // Lệnh trợ giúp
    bot.onText(/\/trogiup/, (msg) => {
        const chatId = msg.chat.id;
        const helpMessage = `
📚 *HƯỚNG DẪN SỬ DỤNG BOT GIAO DỊCH*

Dưới đây là các lệnh hiện có và cách sử dụng:

1. **?symbol,pair,timeframe[,rsiOversold-rsiOverbought]**
   - *Mô tả*: Phân tích thủ công...

(Phần còn lại của help message của bạn)
        `;

        const keyboard = {
            reply_markup: {
                inline_keyboard: [
                    [
                        { text: 'Liên hệ hỗ trợ', url: 'https://t.me/your_support_channel' },
                        { text: 'Đánh giá bot', callback_data: 'rate_bot' }
                    ]
                ]
            }
        };

        bot.sendMessage(chatId, helpMessage, { parse_mode: 'Markdown', ...keyboard });
    });

    bot.on('callback_query', (query) => {
        const chatId = query.message.chat.id;

        if (query.data === 'rate_bot') {
            bot.sendMessage(chatId, 'Vui lòng đánh giá bot bằng cách cho sao nhé! (Tính năng này chưa hoạt động đầy đủ)');
        }
        bot.answerCallbackQuery(query.id);
    });

    // Bắt đầu kiểm tra tự động
    startAutoChecking();
    console.log('✅ Bot đang chạy với tính năng theo dõi tín hiệu tự động...');
    fs.appendFileSync('bot.log', `${new Date().toISOString()} - ✅ Bot đang chạy với tính năng theo dõi tín hiệu tự động...\n`);

    // Đóng database khi tắt bot
    process.on('SIGINT', () => {
        insertStmt.finalize((err) => {
            if (err) {
                console.error("Lỗi đóng insertStmt:", err.message);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi đóng insertStmt: ${err.message}\n`);
            }
        });
        deleteStmt.finalize((err) => {
            if (err) {
                console.error("Lỗi đóng deleteStmt:", err.message);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi đóng deleteStmt: ${err.message}\n`);
            }
        });
        db.close((err) => {
            if (err) {
                console.error("Lỗi đóng SQLite:", err.message);
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - Lỗi đóng SQLite: ${err.message}\n`);
            } else {
                fs.appendFileSync('bot.log', `${new Date().toISOString()} - SQLite connection closed\n`);
            }
            console.log('Đóng kết nối database.');
            fs.appendFileSync('bot.log', `${new Date().toISOString()} - Đóng kết nối database.\n`);
            process.exit(0);
        });
    });
})();