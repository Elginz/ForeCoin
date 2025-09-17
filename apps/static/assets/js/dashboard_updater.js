// apps/static/assets/js/dashboard_updater.js
document.addEventListener('DOMContentLoaded', function () {
    const socket = io();

    // All assets to watch for sentiment updates
    const allAssets = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'SHIBUSDT'];

    socket.on('connect', () => {
        console.log('Live connection established for dashboard overview.');
    });
    
    socket.on('new_kline_data', (data) => {
        if (!data || !data.symbol) return;
        
        // --- Price Update Logic ---
        if (allAssets.includes(data.symbol) && data.kline) {
            let price;
            if (data.symbol === 'DOGEUSDT') {
                price = data.kline.close.toFixed(5);
            } else if (data.symbol === 'SHIBUSDT') {
                price = data.kline.close.toFixed(8);
            } else {
                price = data.kline.close.toFixed(2);
            }
            updateText(`current-price-${data.symbol}`, `$${price}`);
        }

        // Update sentiment for any asset that reports it
        if (allAssets.includes(data.symbol) && data.sentiment) {
            updateSentimentBadge(`sentiment-label-${data.symbol}`, data.sentiment.label);
        }

    // --- Chronos Prediction Updates ---
        if (allAssets.includes(data.symbol) && data.prediction && data.prediction.chronos_next_hour_close) {
            const isStable = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'].includes(data.symbol);
            const precision = isStable ? 2 : 8;
            const chronosPrice = data.prediction.chronos_next_hour_close.toFixed(precision);
            updateText(`chronos-prediction-${data.symbol}`, `$${chronosPrice}`);
        }
    });

    // --- Helper Functions ---
    function updateText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) element.textContent = text;
    }
    
    // --- Sentiment Badge Function ---
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
    // --- Searchbar Function ---
        const searchForm = document.getElementById('navbar-search-main');
        const searchInput = document.getElementById('topbarInputIconLeft');
        const contentCards = document.querySelectorAll('.card-header');
        const notyf = new Notyf({ duration: 3000, position: { x: 'right', y: 'top' } });
    
        if (searchForm && searchInput && contentCards.length > 0) {
            searchForm.addEventListener('submit', function(e) {
                // Stop page from reloading when Enter is pressed
                e.preventDefault(); 
                const searchTerm = searchInput.value.trim().toLowerCase();
                let foundCard = null;
                let matchInfo = null;
    
                if (searchTerm === '') {
                    notyf.error('Please enter a search term.');
                    return;
                }
    
                // Search through all cards for match
                for (let i = 0; i < contentCards.length; i++) {
                    const card = contentCards[i];
                    const cardText = card.textContent.toLowerCase();
                    const cardTitle = card.querySelector('h1, h2, h3, h4, h5, h6');
                    
                    // Check if search exists
                    if (cardText.includes(searchTerm)) {
                        foundCard = card;
                        
                        // Get meaningful title for the found section
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
                    
                    // Temporary highlight effect
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
    
            // Add visual feedback
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

