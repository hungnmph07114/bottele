const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const fs = require('fs-extra');
const { RSI, SMA, MACD, BollingerBands, ADX, ATR } = require('technicalindicators');

// Configuration
const TOKEN = '7605131321:AAGCW_FWEqBC7xMOt8RwL4nek4vqxPBVluY';
const BINANCE_API = 'https://api.binance.com/api/v3';
const bot = new TelegramBot(TOKEN, { polling: true });

const timeframes = {
    '1m': '1 phút', '5m': '5 phút', '15m': '15 phút',
    '1h': '1 giờ', '4h': '4 giờ', '1d': '1 ngày',
    'w1': '1 tuần', 'M1': '1 tháng'
};

// History file
const HISTORY_FILE = 'signal_history.json';
let signal_history = fs.existsSync(HISTORY_FILE) ? fs.readJsonSync(HISTORY_FILE) : [];

// Utility functions
const logger = {
    info: (msg) => console.log(`[${new Date().toISOString()}] INFO: ${msg}`),
    error: (msg) => console.error(`[${new Date().toISOString()}] ERROR: ${msg}`),
};

// Fetch klines from Binance
async function fetchKlines(symbol, pair, timeframe, limit = 100) {
    try {
        const response = await axios.get(`${BINANCE_API}/klines`, {
            params: {
                symbol: `${symbol.toUpperCase()}${pair.toUpperCase()}`,
                interval: timeframe,
                limit,
            },
            timeout: 10000,
        });
        const data = response.data;
        if (!data.length) {
            logger.warning(`Empty data for ${symbol}${pair} ${timeframe}`);
            return null;
        }
        return data.map(d => ({
            timestamp: d[0],
            open: parseFloat(d[1]),
            high: parseFloat(d[2]),
            low: parseFloat(d[3]),
            close: parseFloat(d[4]),
            volume: parseFloat(d[5]),
        }));
    } catch (error) {
        logger.error(`API request failed: ${error.message}`);
        return null;
    }
}

// Technical indicators (simplified using technicalindicators library)
function computeRSI(close, period = 14) {
    return RSI.calculate({ values: close, period }).slice(-1)[0] || 50;
}

function computeMA(close, period = 20) {
    return SMA.calculate({ values: close, period }).slice(-1)[0] || 0;
}

function computeMACD(close) {
    const result = MACD.calculate({ values: close, fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 });
    const last = result.slice(-1)[0];
    return [last.MACD || 0, last.signal || 0];
}

function computeBollingerBands(close, period = 20, stdDev = 2) {
    const result = BollingerBands.calculate({ values: close, period, stdDev });
    const last = result.slice(-1)[0];
    return [last.upper || 0, last.middle || 0, last.lower || 0];
}

function computeADX(data, period = 14) {
    const result = ADX.calculate({
        high: data.map(d => d.high),
        low: data.map(d => d.low),
        close: data.map(d => d.close),
        period,
    });
    return result.slice(-1)[0].adx || 0;
}

function computeATR(data, period = 14) {
    const result = ATR.calculate({
        high: data.map(d => d.high),
        low: data.map(d => d.low),
        close: data.map(d => d.close),
        period,
    });
    return result.slice(-1)[0] || 0;
}

// Simplified analysis function
async function getCryptoAnalysis(symbol, pair, timeframe, chatId) {
    try {
        if (!timeframes[timeframe]) return '⚠️ Khung thời gian không hợp lệ';

        const dfShort = await fetchKlines(symbol, pair, timeframe);
        if (!dfShort) return '❗ Không thể lấy dữ liệu hoặc cặp tiền không tồn tại';

        const dfLong = await fetchKlines(symbol, pair, '4h') || dfShort;

        const close = dfShort.map(d => d.close);
        const currentPrice = close[close.length - 1];
        const volume = dfShort.map(d => d.volume);

        // Indicators
        const rsiShort = computeRSI(close);
        const maShort = computeMA(close, 10);
        const maLong = computeMA(close, 50);
        const [macd, signal] = computeMACD(close);
        const [upperBB, smaBB, lowerBB] = computeBollingerBands(close);
        const adx = computeADX(dfShort);
        const atr = computeATR(dfShort);

        const rsiLong = computeRSI(dfLong.map(d => d.close));
        const [macdLong, signalLong] = computeMACD(dfLong.map(d => d.close));

        // Signal logic (simplified)
        let signalText = '';
        let details = [];
        const adxThreshold = 25;

        if (adx < adxThreshold) {
            signalText = `🟡 GIỮ - Thị trường sideway (ADX < ${adxThreshold})`;
            details.push(`ADX: ${adx.toFixed(1)}`);
        } else if (rsiShort < 30 && rsiLong < 50 && maShort > maLong && macd > signal) {
            signalText = '🟢 LONG - Tín hiệu mua mạnh';
            details.push(`RSI: ${rsiShort.toFixed(1)} (<30)`);
        } else if (rsiShort > 70 && rsiLong > 50 && maShort < maLong && macd < signal) {
            signalText = '🔴 SHORT - Tín hiệu bán mạnh';
            details.push(`RSI: ${rsiShort.toFixed(1)} (>70)`);
        } else {
            signalText = '⚪️ ĐỢI - Chưa có tín hiệu rõ ràng';
        }

        // Report
        const analysis = [
            `📊 *Phân tích ${symbol}/${pair} (${timeframes[timeframe]})*`,
            `💰 Giá hiện tại: ${currentPrice.toFixed(4)}`,
            `📈 RSI: ${rsiShort.toFixed(1)} (4h: ${rsiLong.toFixed(1)})`,
            `📉 MA10/MA50: ${maShort.toFixed(4)}/${maLong.toFixed(4)}`,
            `📊 Bollinger: ${lowerBB.toFixed(4)} - ${upperBB.toFixed(4)}`,
            `📈 ADX: ${adx.toFixed(1)}`,
            `⚡️ *${signalText}*`,
            ...details.map(d => `✓ ${d}`),
        ];

        return analysis.join('\n');
    } catch (error) {
        logger.error(`Analysis error: ${error.message}`);
        return '❗ Đã xảy ra lỗi trong quá trình phân tích';
    }
}

// Bot handlers
bot.onText(/\/start|\/help/, (msg) => {
    const helpText = `Chào mừng đến với Crypto Trading Bot!\n
Các lệnh hỗ trợ:\n
/analyze [symbol],[pair],[timeframe] - Phân tích kỹ thuật\n
Ví dụ: /analyze btc,usdt,1h hoặc ?ada,usdt,15m\n
📊 Chỉ báo: RSI, MA, MACD, Bollinger, ADX, ATR`;
    bot.sendMessage(msg.chat.id, helpText, { parse_mode: 'Markdown' });
});

bot.onText(/^\?(.+)/, async (msg, match) => {
    const parts = match[1].replace('.', ',').split(',');
    if (parts.length !== 3) return bot.sendMessage(msg.chat.id, '⚠️ Sai định dạng! Ví dụ: ?ada,usdt,15m');
    const [symbol, pair, timeframe] = parts;
    const analysis = await getCryptoAnalysis(symbol, pair, timeframe, msg.chat.id);
    bot.sendMessage(msg.chat.id, analysis, { parse_mode: 'Markdown' });
});

bot.onText(/\/analyze (.+)/, async (msg, match) => {
    const parts = match[1].split(',');
    if (parts.length !== 3) return bot.sendMessage(msg.chat.id, '⚠️ Sai định dạng! Ví dụ: /analyze btc,usdt,1h');
    const [symbol, pair, timeframe] = parts;
    if (!timeframes[timeframe]) return bot.sendMessage(msg.chat.id, '⚠️ Khung thời gian không hợp lệ');
    bot.sendChatAction(msg.chat.id, 'typing');
    const analysis = await getCryptoAnalysis(symbol, pair, timeframe, msg.chat.id);
    bot.sendMessage(msg.chat.id, analysis, { parse_mode: 'Markdown' });
});

// Start bot
logger.info('Khởi động bot...');