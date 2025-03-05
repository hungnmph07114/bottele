const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR } = require('technicalindicators');
const tf = require('@tensorflow/tfjs');

// Configuration
const TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY'; // Thay thế bằng token bot của bạn
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


// Danh sách theo dõi tự động (chatId -> [{symbol, pair, timeframe}])
const autoWatchList = new Map();

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
        }); // ĐÃ CHUYỂN DẤU NGOẶC NHỌN VÀO ĐÂY
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
    const avgBBWidth = computeMA(df.map(d => BollingerBands.calculate({ values: df.map(v => v.close), period: 20, stdDev: 2 }).slice(-1)[0].upper - BollingerBands.calculate({ values: df.map(v => v.close), period: 20, stdDev: 2 }).slice(-1)[0].lower), 20);
    const isSideways = adx < 20 && bbWidth < avgBBWidth * 0.8;

    const rsiOverbought = customThresholds.rsiOverbought || 70;
    const rsiOversold = customThresholds.rsiOversold || 30;
    const adxStrongTrend = customThresholds.adxStrongTrend || 30;

    const input = tf.tensor2d([[rsi, adx, histogram, volumeSpike, ma10 - ma50, currentPrice - middleBB]]);
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
        `📊 MACD: ${macd.toFixed(4)} / ${signal.toFixed(4)}`,
        `📉 ADX: ${adx.toFixed(1)}`,
        `📦 Volume: ${volumeSpike ? 'TĂNG ĐỘT BIẾN' : 'BÌNH THƯỜNG'}`
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
    const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe);
    if (confidence >= confidenceThreshold) {
        bot.sendMessage(chatId, `🚨 *TÍN HIỆU ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]})* 🚨\n${result}`, {
            parse_mode: 'Markdown'
        });
        console.log(`✅ Đã gửi tín hiệu ${symbol}/${pair} đến chat ${chatId} - Độ tin cậy: ${confidence}%`);
    }
}

// Hàm chạy kiểm tra định kỳ
function startAutoChecking() {
    const CHECK_INTERVAL = 5 * 60 * 1000; // 5 phút
    setInterval(() => {
        for (const [chatId, watchList] of autoWatchList) {
            watchList.forEach(config => {
                checkAutoSignal(chatId, config).catch(err => console.error(`❌ Lỗi kiểm tra ${config.symbol}/${config.pair}: ${err.message}`));
            });
        }
    }, CHECK_INTERVAL);
}

// Khởi động bot
(async () => {
    await initializeModel();
    const initialData = await fetchKlines('BTC', 'USDT', '1h', 200);
    if (initialData) await trainModel(initialData);
    else console.error('❌ Không thể lấy dữ liệu ban đầu để huấn luyện mô hình');

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

        // Bỏ qua kiểm tra ngưỡng tin cậy cho lệnh phân tích thủ công
        try {
            const { result, confidence } = await getCryptoAnalysis(symbol, pair, timeframe, customThresholds);
            bot.sendMessage(msg.chat.id, result, { parse_mode: 'Markdown' });
        } catch (error) {
            console.error(`Lỗi khi lấy giá và tín hiệu: ${error}`);
            bot.sendMessage(msg.chat.id, `❌ Lỗi khi lấy giá và tín hiệu. Vui lòng thử lại sau.`);
        }
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

        if (!autoWatchList.has(chatId)) autoWatchList.set(chatId, []);
        const watchList = autoWatchList.get(chatId);

        if (!watchList.some(w => w.symbol === symbol && w.pair === pair && w.timeframe === timeframe)) {
            watchList.push(config);
            bot.sendMessage(chatId, `✅ Bắt đầu theo dõi tín hiệu ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]}). Bot sẽ gửi cảnh báo khi có tín hiệu mạnh.`);
        } else {
            bot.sendMessage(chatId, `ℹ️ ${symbol.toUpperCase()}/${pair.toUpperCase()} (${timeframes[timeframe]}) đã được theo dõi rồi.`);
        }
    });

    // Lệnh trợ giúp
    bot.onText(/\/trogiup/, (msg) => {
        const helpMessage = `
📚 *HƯỚNG DẪN SỬ DỤNG BOT GIAO DỊCH*

Dưới đây là các lệnh hiện có và cách sử dụng:

1. **?symbol,pair,timeframe[,rsiOversold-rsiOverbought]**  
   - *Mô tả*: Phân tích thủ công cặp giao dịch, trả về tín hiệu và các mức giá (entry, SL, TP).  
   - *Cú pháp*: ?<coin>,<đồng giao dịch>,<khung thời gian>[,rsi<giá trị thấp>-<giá trị cao>]  
   - *Ví dụ*:  
     - ?ada,usdt,5m (phân tích ADA/USDT khung 5 phút)  
     - ?btc,usdt,1h,rsi25-75 (phân tích BTC/USDT khung 1 giờ, tùy chỉnh RSI 25-75)  
   - *Khung thời gian hợp lệ*: ${Object.keys(timeframes).join(', ')}

2. **/tinhieu symbol,pair,timeframe**  
   - *Mô tả*: Kích hoạt theo dõi tự động, gửi tín hiệu khi độ tin cậy ≥ 80%.  
   - *Cú pháp*: /tinhieu <coin>,<đồng giao dịch>,<khung thời gian>  
   - *Ví dụ*:  
     - /tinhieu ada,usdt,5m (theo dõi ADA/USDT khung 5 phút)  
     - /tinhieu btc,usdt,1h (theo dõi BTC/USDT khung 1 giờ)  
   - *Khung thời gian hợp lệ*: ${Object.keys(timeframes).join(', ')}

3. **/trogiup**  
   - *Mô tả*: Hiển thị danh sách lệnh và hướng dẫn sử dụng (bạn đang xem).  
   - *Cú pháp*: /trogiup  
   - *Ví dụ*: /trogiup

*Lưu ý*:  
- Bot sử dụng AI và chỉ báo kỹ thuật (RSI, MACD, ADX, Bollinger Bands) để phân tích.  
- Nếu thị trường đi ngang, tín hiệu vẫn được đưa ra nhưng kèm cảnh báo độ chính xác thấp.  
- Đảm bảo nhập đúng cặp giao dịch tồn tại trên Binance (ví dụ: ADA/USDT, BTC/USDT).  
        `;
        bot.sendMessage(msg.chat.id, helpMessage, { parse_mode: 'Markdown' });
    });

    // Bắt đầu kiểm tra tự động
    startAutoChecking();
    console.log('✅ Bot đang chạy với tính năng theo dõi tín hiệu tự động...');

})();