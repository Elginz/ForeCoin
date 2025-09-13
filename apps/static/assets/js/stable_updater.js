// apps/static/assets/js/stable_updater.js

document.addEventListener('DOMContentLoaded', function () {
    // Configuration
    const socket = io();
    const charts = {};
    const assets = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'];
    const precision = 2;

    // Format the time 
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

    // Create and initialise chart for symbol
    function initializeChart(symbol) {
        const chartContainer = document.getElementById(`chart-container-${symbol}`);
        if (!chartContainer) return;
        chartContainer.innerHTML = '';
        // Create chart
        const chart = LightweightCharts.createChart(chartContainer, {
            height: 300,
            layout: { background: { color: '#ffffff' }, textColor: '#191919' },
            grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            }
        });
        // Add candlestick series to chart
        const candlestickSeries = chart.addCandlestickSeries({ upColor: '#28a745', downColor: '#dc3545', borderVisible: false, wickUpColor: '#28a745', wickDownColor: '#dc3545' });
        // Load initial data passed from server
        if (window.initialChartData && window.initialChartData[symbol]) {
            candlestickSeries.setData(window.initialChartData[symbol]);
            chart.timeScale().fitContent();
        }
        // Store chart instance and container for resizing
        charts[symbol] = { chart: chart, series: candlestickSeries, container: chartContainer };
    }

    // Main function to update entire UI 
    function updateDashboardUI(data) {
        const symbol = data.symbol;
        if (!assets.includes(symbol)) return;

        let lastClose = null;
        // Update the candlestick chart and last close price.
        if (data.kline) {
            lastClose = data.kline.close;
            updateText(`lastClose-${symbol}`, `$${lastClose.toFixed(precision)}`);
            if (charts[symbol] && charts[symbol].series) {
                charts[symbol].series.update(data.kline);
            }
        }
        // Update model prediction values and signals
        if (data.prediction && lastClose !== null) {
            updateText(`knnPredictedNextClose-${symbol}`, data.prediction.knn_supertrend_close ? `$${data.prediction.knn_supertrend_close.toFixed(precision)}` : 'N/A');
            updateSignal(`knnSignal-${symbol}`, lastClose, data.prediction.knn_supertrend_close);
            
            updateText(`chronosPredictedNextClose-${symbol}`, data.prediction.chronos_next_hour_close ? `$${data.prediction.chronos_next_hour_close.toFixed(precision)}` : 'N/A');
            updateSignal(`chronosSignal-${symbol}`, lastClose, data.prediction.chronos_next_hour_close);
        }
        // Update news sentiment
        if (data.sentiment && data.sentiment.label) {
            const sentiment = data.sentiment;
            updateSentimentBadge(`sentimentLabel-${symbol}`, sentiment.label);
            updateText(`sentimentScore-${symbol}`, sentiment.score !== null ? `${(sentiment.score * 100).toFixed(2)} %` : 'N/A');
            updateText(`sentimentConfidence-${symbol}`, sentiment.confidence !== null ? `${(sentiment.confidence * 100).toFixed(2)} %` : 'N/A');
        }
        // Update last update timestamp
        if (data.event_time_iso) {
            updateText(`eventTime-${symbol}`, formatTimestamp(data.event_time_iso));
        }
    }

    // Helper function to update text content of an element using ID
    function updateText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) element.textContent = text;
    }
    
    // Update signal badge based on price change
    function updateSignal(signalElementId, lastClose, predictedClose) {
        const element = document.getElementById(signalElementId);
        if (!element) return;

        if (predictedClose === null || lastClose === null || lastClose === 0) {
            element.textContent = 'N/A';
            element.className = 'badge bg-secondary';
            return;
        }

        const change = ((predictedClose - lastClose) / lastClose) * 100;
        let text = `WAIT (${change.toFixed(2)}%)`;
        let style = 'badge bg-warning text-dark';

        if (change > 0.1) {
            text = `UP (${change.toFixed(2)}%)`;
            style = 'badge bg-success';
        } else if (change < -0.1) {
            text = `DOWN (${change.toFixed(2)}%)`;
            style = 'badge bg-danger';
        }
        
        element.textContent = text;
        element.className = style;
    }

    // Update sentiment badge style and text based
    function updateSentimentBadge(elementId, sentimentLabel) {
        const element = document.getElementById(elementId);
        if (!element) return;
    
        let text = 'N/A';
        let label = sentimentLabel ? String(sentimentLabel).trim().toLowerCase() : null;
    
        element.className = 'badge'; 
    
        if (label === "positive") {
            text = "Positive";
            element.classList.add("bg-success");
        } else if (label === "negative") {
            text = "Negative";
            element.classList.add("bg-danger");
        } else if (label === "neutral") {
            text = "Neutral";
            element.classList.add("bg-warning", "text-dark");
        } else {
            text = "N/A";
            element.classList.add("bg-secondary");
        }
        element.textContent = text;
    }

    // --- Initialize Page ---
    assets.forEach(symbol => {
        initializeChart(symbol);
    });
    
    // WebSocket event listeners
    socket.on('connect', () => console.log('Live connection established for stable assets.'));
    socket.on('new_kline_data', updateDashboardUI);
    socket.on('sentiment_update', (data) => {
        if (data && data.symbol && data.sentiment && assets.includes(data.symbol)) {
            const symbol = data.symbol;
            const sentiment = data.sentiment;
            
            updateSentimentBadge(`sentimentLabel-${symbol}`, sentiment.label);
            updateText(`sentimentScore-${symbol}`, sentiment.score !== null ? `${(sentiment.score * 100).toFixed(2)} %` : 'N/A');
            updateText(`sentimentConfidence-${symbol}`, sentiment.confidence !== null ? `${(sentiment.confidence * 100).toFixed(2)} %` : 'N/A');
        }
    });

    // For responsive charts
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            for (const symbol in charts) {
                if (charts.hasOwnProperty(symbol)) {
                    const chartInfo = charts[symbol];
                    chartInfo.chart.applyOptions({
                        width: chartInfo.container.clientWidth
                    });
                }
            }
            // Debounce resize event for performance
        }, 200);
    });

    // --- Advanced Search Functionality ---
    const searchForm = document.getElementById('navbar-search-main');
    const searchInput = document.getElementById('topbarInputIconLeft');
    const contentCards = document.querySelectorAll('.card-title');
    const notyf = new Notyf({ duration: 3000, position: { x: 'right', y: 'top' } });

    if (searchForm && searchInput && contentCards.length > 0) {
        searchForm.addEventListener('submit', function(e) {
            // Stop the page from reloading when Enter is pressed
            e.preventDefault(); 
            const searchTerm = searchInput.value.trim().toLowerCase();
            let foundCard = null;
            let matchInfo = null;

            if (searchTerm === '') {
                notyf.error('Please enter a search term.');
                return;
            }

            // Search through all cards for actual content matches
            for (let i = 0; i < contentCards.length; i++) {
                const card = contentCards[i];
                const cardText = card.textContent.toLowerCase();
                const cardTitle = card.querySelector('h1, h2, h3, h4, h5, h6');
                
                // Check if search term exists
                if (cardText.includes(searchTerm)) {
                    foundCard = card;
                    
                    // Try to get a title for the found section
                    if (cardTitle) {
                        matchInfo = {
                            title: cardTitle.textContent.trim(),
                            cardIndex: i + 1
                        };
                    } else {
                        matchInfo = {
                            title: `Section ${i + 1}`,
                            cardIndex: i + 1
                        };
                    }
                    // Stop after finding the first match
                    break; 
                }
            }

            if (foundCard && matchInfo) {
                // If content is found, scroll
                foundCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                // Temporary highlight effect to the found card
                foundCard.style.transition = 'all 0.5s ease';
                foundCard.style.transform = 'scale(1.02)';
                foundCard.style.boxShadow = '0 0 20px rgba(0, 123, 255, 0.3)';
                foundCard.style.backgroundColor = 'rgba(0, 123, 255, 0.05)';
                
                // Show success notification with the actual section found
                notyf.success(`Found in: ${matchInfo.title}`);
                
                // Remove highlight after 2 seconds
                setTimeout(() => {
                    foundCard.style.transform = '';
                    foundCard.style.boxShadow = '';
                    foundCard.style.backgroundColor = '';
                }, 2000);

            } else {
                // Error handler
                notyf.error(`"${searchInput.value}" not found on this page.`);
            }

            // Clear the search input after search
            searchInput.value = '';
        });

        // Add visual feedback while typing
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.trim().toLowerCase();
            
            if (searchTerm.length > 0) {
                // Add visual feedback that search is ready
                searchInput.style.borderColor = '#007bff';
                searchInput.style.boxShadow = '0 0 0 0.2rem rgba(0, 123, 255, 0.25)';
            } else {
                searchInput.style.borderColor = '';
                searchInput.style.boxShadow = '';
            }
        });

        // Remove visual feedback when input loses focus
        searchInput.addEventListener('blur', function() {
            searchInput.style.borderColor = '';
            searchInput.style.boxShadow = '';
        });
    }

});
