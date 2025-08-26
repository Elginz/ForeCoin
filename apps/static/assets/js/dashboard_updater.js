// apps/static/assets/js/dashboard_updater.js
document.addEventListener('DOMContentLoaded', function () {
    const socket = io();

    // Specific assets to watch for price updates on the dashboard
    const priceAssets = ['BTCUSDT', 'DOGEUSDT'];
    // All assets to watch for sentiment updates
    const sentimentAssets = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'SHIBUSDT'];

    socket.on('connect', () => {
        console.log('Live connection established for dashboard overview.');
    });
    
    socket.on('new_kline_data', (data) => {
        if (!data || !data.symbol) return;
        
        // Update price if the asset is one of the key assets
        if (priceAssets.includes(data.symbol) && data.kline) {
            const price = data.symbol === 'DOGEUSDT' ? data.kline.close.toFixed(5) : data.kline.close.toFixed(2);
            updateText(`current-price-${data.symbol}`, `$${price}`);
        }

        // Update sentiment for any asset that reports it
        if (sentimentAssets.includes(data.symbol) && data.sentiment) {
            updateSentimentBadge(`sentiment-label-${data.symbol}`, data.sentiment.label);
        }
    });

    // --- Helper Functions ---
    function updateText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) element.textContent = text;
    }
    
    function updateSentimentBadge(elementId, sentimentLabel) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        let text = 'N/A';
        let cssClass = 'badge bg-secondary';
        if (sentimentLabel) {
            const label = sentimentLabel.toLowerCase();
            if (label === 'positive') { text = 'Positive'; cssClass = 'badge bg-success'; } 
            else if (label === 'negative') { text = 'Negative'; cssClass = 'badge bg-danger'; } 
            else if (label === 'neutral') { text = 'Neutral'; cssClass = 'badge bg-warning text-dark'; } 
            else { text = sentimentLabel; }
        }
        element.className = cssClass;
        element.textContent = text;
    }

    // Initially populate sentiment badges from the latest data cache passed by Flask
    // This is an example of how you could do it if you passed the data
    sentimentAssets.forEach(symbol => {
        // Since we don't pass the full initial data to the new dashboard JS,
        // we'll just wait for the first socket event to populate the sentiment.
        // The HTML will show "Loading..." until then.
    });
});
