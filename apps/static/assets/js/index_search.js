// apps/static/assets/js/index_search.js
document.addEventListener('DOMContentLoaded', function () {
    // Search functionality for index page
    const searchForm = document.getElementById('navbar-search-main');
    const searchInput = document.getElementById('topbarInputIconLeft');
    const contentCards = document.querySelectorAll('.card');
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

            // Search through all cards for content 
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
                foundCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
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
