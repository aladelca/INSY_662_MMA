//handle user query
// JavaScript to handle the search action when the button is clicked
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const searchResults = document.getElementById('search-results');
const resultsContainer = document.getElementById('search-results-container');

searchButton.addEventListener('click', () => {
    const query = searchInput.value;
    // Make an AJAX request to the backend with the query
    fetch(`/search?query=${query}`)
        .then(response => response.json())
        .then(data => {
            // Clear previous results
            searchResults.innerHTML = '';

            if (data.length === 0) {
                searchResults.innerHTML = 'No results found.';
            } else {
                // Create and append result items
                data.forEach(result => {
                    const li = document.createElement('li');
                    li.textContent = result;
                    searchResults.appendChild(li);
                });
            }

            // Show the results container
            resultsContainer.style.display = 'block';
        });
});

