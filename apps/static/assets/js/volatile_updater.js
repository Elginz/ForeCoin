// apps/static/assets/js/volatile_updater.js

document.addEventListener('DOMContentLoaded', function () {
    const socket = io();
    const charts = {};
    const assets = ['DOGEUSDT', 'SHIBUSDT'];
    const precision = 8;

    function formatTimestamp(isoString) {
        if (!isoString || !isoString.includes('T')) return 'N/A';
        try {
            const date = new Date(isoString);
            const year = date.getUTCFullYear();
            const month = String(date.getUTCMonth() + 1).padStart(2, '0');
            const day = String(date.getUTCDate()).padStart(2, '0');
            const hours = String(date.getUTCHours()).padStart(2, '0');
            const minutes = String(date.getUTCMinutes()).padStart(2, '0');
            const seconds = String(date.getUTCSeconds()).padStart(2, '0');
            return `${year}-${month}-${day} (${hours}:${minutes}:${seconds} UTC)`;
        } catch (e) {
            console.error("Error formatting timestamp:", e);
            return 'N/A';
        }
    }

    function initializeChart(symbol) {
        const chartContainer = document.getElementById(`chart-container-${symbol}`);
        if (!chartContainer) return;
        chartContainer.innerHTML = '';

        const chart = LightweightCharts.createChart(chartContainer, { width: chartContainer.clientWidth, height: 300, layout: { background: { color: '#ffffff' }, textColor: '#191919' }, grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } } });
        const candlestickSeries = chart.addCandlestickSeries({ upColor: '#28a745', downColor: '#dc3545', borderVisible: false, wickUpColor: '#28a745', wickDownColor: '#dc3545' });
        
        if (window.initialChartData && window.initialChartData[symbol]) {
            candlestickSeries.setData(window.initialChartData[symbol]);
            chart.timeScale().fitContent();
        }
        charts[symbol] = { series: candlestickSeries };
    }

    function updateDashboardUI(data) {
        const symbol = data.symbol;
        if (!assets.includes(symbol)) return;

        let lastClose = null;

        if (data.kline) {
            lastClose = data.kline.close;
            updateText(`lastClose-${symbol}`, `$${lastClose.toFixed(precision)}`);
            if (charts[symbol] && charts[symbol].series) {
                charts[symbol].series.update(data.kline);
            }
        }

        if (data.prediction && lastClose !== null) {
            if (data.prediction.lgbm_quantiles) {
                const quantiles = data.prediction.lgbm_quantiles;
                updateText(`lgbmLow-${symbol}`, quantiles.low ? `$${quantiles.low.toFixed(precision)}` : 'N/A');
                updateText(`lgbmMedian-${symbol}`, quantiles.median ? `$${quantiles.median.toFixed(precision)}` : 'N/A');
                updateText(`lgbmHigh-${symbol}`, quantiles.high ? `$${quantiles.high.toFixed(precision)}` : 'N/A');
                updateSignal(`lgbmSignal-${symbol}`, lastClose, quantiles.median);
            }
            
            updateText(`chronosPredictedNextClose-${symbol}`, data.prediction.chronos_next_hour_close ? `$${data.prediction.chronos_next_hour_close.toFixed(precision)}` : 'N/A');
            updateSignal(`chronosSignal-${symbol}`, lastClose, data.prediction.chronos_next_hour_close);
        }
        
        if (data.event_time_iso) {
            updateText(`eventTime-${symbol}`, formatTimestamp(data.event_time_iso));
        }
    }

    function updateText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) element.textContent = text;
    }

    function updateSignal(signalElementId, lastClose, predictedClose) {
        const element = document.getElementById(signalElementId);
        if (!element) return;
    
        if (predictedClose === null || lastClose === null) {
            element.textContent = 'N/A';
            element.className = 'badge bg-secondary';
            return;
        }
        
        if (predictedClose > lastClose) {
            element.textContent = 'UP';
            element.className = 'badge bg-success';
        } else if (predictedClose < lastClose) {
            element.textContent = 'DOWN';
            element.className = 'badge bg-danger';
        } else {
            element.textContent = 'HOLD';
            element.className = 'badge bg-warning text-dark';
        }
    }

    function updateSentimentBadge(elementId, sentimentLabel) {
        const element = document.getElementById(elementId);
        if (!element) return;
        let text = 'N/A';
        let cssClass = 'badge bg-secondary';
        if (sentimentLabel) {
            const label = sentimentLabel.toLowerCase();
            if (label === 'positive') {
                text = 'Positive';
                cssClass = 'badge bg-success';
            } else if (label === 'negative') {
                text = 'Negative';
                cssClass = 'badge bg-danger';
            } else if (label === 'neutral') {
                text = 'Neutral';
                cssClass = 'badge bg-warning text-dark';
            } else {
                text = sentimentLabel;
            }
        }
        element.className = cssClass;
        element.textContent = text;
    }
    
    // --- Initialize Page ---
    assets.forEach(symbol => {
        initializeChart(symbol);
    });
    
    socket.on('connect', () => console.log('Live connection established for volatile assets.'));
    socket.on('new_kline_data', updateDashboardUI);

    // --- FIX: Add the missing event listener for sentiment updates ---
    socket.on('sentiment_update', (data) => {
        if (data && data.symbol && data.sentiment && assets.includes(data.symbol)) {
            console.log(`Received sentiment update for ${data.symbol}`);
            const symbol = data.symbol;
            const sentiment = data.sentiment;
            
            updateSentimentBadge(`sentimentLabel-${symbol}`, sentiment.label);
            updateText(`sentimentScore-${symbol}`, sentiment.score !== null ? `${(sentiment.score * 100).toFixed(2)} %` : 'N/A');
            updateText(`sentimentConfidence-${symbol}`, sentiment.confidence !== null ? `${(sentiment.confidence * 100).toFixed(2)} %` : 'N/A');
        }
    });
});
