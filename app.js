// Global state
let currentNprobe = 32;
let currentK = 10;
let currentExactRerank = false;
let queryHistory = [];
let currentResults = null;
let showWords = null; // Optional parameter for word context

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadQueryHistory();
    setupEventListeners();
});

// Event listeners setup
function setupEventListeners() {
    // Close dropdowns and menus when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.dropdown')) {
            document.querySelectorAll('.dropdown-content').forEach(dropdown => {
                dropdown.style.display = 'none';
            });
        }
        
        if (!event.target.closest('.three-dot-container')) {
            document.querySelectorAll('.three-dot-menu').forEach(menu => {
                menu.classList.remove('show');
            });
        }
    });

    // Make dropdowns work on click instead of hover for mobile
    document.getElementById('nprobeButton').addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        const dropdown = this.nextElementSibling;
        dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
    });

    document.getElementById('kButton').addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        const dropdown = this.nextElementSibling;
        dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
    });
}

// Handle keyboard input
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        submitQuery();
    }
}

// Dropdown selection functions
function selectNprobe(value) {
    currentNprobe = value;
    document.getElementById('nprobeValue').textContent = value;
    // Hide dropdown
    document.querySelector('#nprobeButton + .dropdown-content').style.display = 'none';
}

function selectK(value) {
    currentK = value;
    document.getElementById('kValue').textContent = value;
    // Hide dropdown
    document.querySelector('#kButton + .dropdown-content').style.display = 'none';
}

// Submit query function
async function submitQuery() {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();
    
    if (!query) return;
    
    // Update exact rerank value
    currentExactRerank = document.getElementById('exactRerank').checked;
    
    // Show loading state
    showLoading(true);
    
    try {
        const response = await fetch('http://localhost:30888/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                n_docs: currentK,
                nprobe: currentNprobe,
                exact_rerank: currentExactRerank,
                ...(showWords && { show_words: showWords })
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        currentResults = data;
        
        // Add to query history
        addToQueryHistory(query, data, {
            nprobe: currentNprobe,
            k: currentK,
            exact_rerank: currentExactRerank
        });
        
        // Display results
        displaySearchResults(data, query);
        
        // Clear input
        queryInput.value = '';
        
    } catch (error) {
        console.error('Search error:', error);
        showError('Failed to perform search. Please check if the server is running.');
    } finally {
        showLoading(false);
    }
}

// Display search results
function displaySearchResults(data, query) {
    const initialState = document.getElementById('initialState');
    const resultsContainer = document.getElementById('resultsContainer');
    
    // Hide initial state and show results
    initialState.classList.add('hidden');
    resultsContainer.classList.remove('hidden');
    
    // Clear previous results
    resultsContainer.innerHTML = '';
    
    if (!data.results || !data.results.passages || data.results.passages.length === 0) {
        resultsContainer.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <p>No results found for "${query}"</p>
            </div>
        `;
        return;
    }
    
    // Render each passage
    data.results.passages.forEach((passage, index) => {
        const passageElement = createPassageElement(passage, index, query);
        resultsContainer.appendChild(passageElement);
    });
}

// Create passage element
function createPassageElement(passage, index, query) {
    const div = document.createElement('div');
    div.className = 'bg-white rounded-lg shadow-sm border border-gray-200 p-6';
    div.id = `passage-${index}`;
    
    // Highlight query terms in the passage text
    const highlightedText = highlightQueryTerms(passage.text || passage.content || '', query);
    
    div.innerHTML = `
        <div class="prose max-w-none">
            <p class="text-gray-900 leading-relaxed">${highlightedText}</p>
        </div>
        <div class="mt-4 flex items-center justify-between">
            <div class="text-sm text-gray-500">
                #${index + 1} • Source: ${passage.filename || passage.source || 'Unknown'}
            </div>
            ${passage.index_id !== undefined ? `
                <button 
                    onclick="expandPassage(${passage.index_id}, 1, ${index})"
                    class="text-blue-600 hover:text-blue-800 text-sm font-medium transition-colors"
                >
                    Show More
                </button>
            ` : ''}
        </div>
        <div id="expanded-${index}" class="mt-4 hidden">
            <!-- Expanded content will be inserted here -->
        </div>
    `;
    
    return div;
}

// Highlight query terms
function highlightQueryTerms(text, query) {
    if (!query || !text) return text;
    
    // Split query into individual terms
    const terms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2);
    
    let highlightedText = text;
    
    terms.forEach(term => {
        const regex = new RegExp(`\\b(${escapeRegExp(term)})\\b`, 'gi');
        highlightedText = highlightedText.replace(regex, '<span class="highlight">$1</span>');
    });
    
    return highlightedText;
}

// Escape special regex characters
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Expand passage function
async function expandPassage(indexId, offset, passageIndex) {
    const expandedContainer = document.getElementById(`expanded-${passageIndex}`);
    
    if (!expandedContainer.classList.contains('hidden')) {
        // Collapse the expanded section
        expandedContainer.classList.add('hidden');
        
        // Scroll with buffer
        const passageElement = document.getElementById(`passage-${passageIndex}`);
        if (passageElement) {
            const rect = passageElement.getBoundingClientRect();
            const scrollTop = window.pageYOffset + rect.top - 85; // 85px buffer
            window.scrollTo({ top: scrollTop, behavior: 'smooth' });
        }
        return;
    }
    
    try {
        const response = await fetch('http://localhost:30888/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                expand_index_id: indexId,
                expand_offset: offset
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.results && data.results.passages && data.results.passages.length > 0) {
            const expandedPassages = data.results.passages.map((passage, idx) => `
                <div class="border-l-4 border-blue-200 pl-4 py-2 ${idx > 0 ? 'mt-3' : ''}">
                    <p class="text-gray-800 text-sm leading-relaxed">${passage.text || passage.content || ''}</p>
                    <div class="text-xs text-gray-500 mt-1">
                        Expanded context from: ${passage.filename || passage.source || 'Unknown'}
                    </div>
                </div>
            `).join('');
            
            expandedContainer.innerHTML = expandedPassages;
            expandedContainer.classList.remove('hidden');
        }
        
    } catch (error) {
        console.error('Expand passage error:', error);
        showError('Failed to expand passage.');
    }
}

// Query history management
function addToQueryHistory(query, results, parameters) {
    const historyItem = {
        id: Date.now(),
        query: query,
        results: results,
        parameters: parameters,
        timestamp: new Date().toISOString(),
        displayName: query.length > 50 ? query.substring(0, 50) + '...' : query
    };
    
    // Remove existing entry with same query
    queryHistory = queryHistory.filter(item => item.query !== query);
    
    // Add to beginning of history
    queryHistory.unshift(historyItem);
    
    // Limit history to 50 items
    if (queryHistory.length > 50) {
        queryHistory = queryHistory.slice(0, 50);
    }
    
    saveQueryHistory();
    renderQueryHistory();
}

function renderQueryHistory() {
    const historyContainer = document.getElementById('queryHistory');
    
    if (queryHistory.length === 0) {
        historyContainer.innerHTML = `
            <div class="p-4 text-gray-500 text-sm text-center">
                No search history yet
            </div>
        `;
        return;
    }
    
    historyContainer.innerHTML = queryHistory.map(item => `
        <div class="border-b border-gray-100 hover:bg-gray-50 cursor-pointer group relative" onclick="selectHistoryItem('${item.id}')">
            <div class="p-4">
                <div class="text-sm font-medium text-gray-900 mb-1 pr-8">${item.displayName}</div>
                <div class="text-xs text-gray-500">${formatTimestamp(item.timestamp)}</div>
            </div>
            <div class="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity three-dot-container">
                <button onclick="event.stopPropagation(); toggleThreeDotMenu('${item.id}')" class="p-2 hover:bg-gray-200 rounded">
                    <svg class="h-4 w-4 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z"/>
                    </svg>
                </button>
                <div id="menu-${item.id}" class="three-dot-menu">
                    <a href="#" onclick="event.preventDefault(); event.stopPropagation(); renameHistoryItem('${item.id}')" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Rename</a>
                    <a href="#" onclick="event.preventDefault(); event.stopPropagation(); deleteHistoryItem('${item.id}')" class="block px-4 py-2 text-sm text-red-600 hover:bg-gray-100">Delete</a>
                </div>
            </div>
        </div>
    `).join('');
}

function selectHistoryItem(id) {
    const item = queryHistory.find(h => h.id.toString() === id);
    if (item) {
        // Restore search parameters
        currentNprobe = item.parameters.nprobe;
        currentK = item.parameters.k;
        currentExactRerank = item.parameters.exact_rerank;
        
        // Update UI
        document.getElementById('nprobeValue').textContent = currentNprobe;
        document.getElementById('kValue').textContent = currentK;
        document.getElementById('exactRerank').checked = currentExactRerank;
        
        // Display cached results
        currentResults = item.results;
        displaySearchResults(item.results, item.query);
    }
}

function toggleThreeDotMenu(id) {
    const menu = document.getElementById(`menu-${id}`);
    // Close all other menus
    document.querySelectorAll('.three-dot-menu').forEach(m => {
        if (m !== menu) m.classList.remove('show');
    });
    menu.classList.toggle('show');
}

function renameHistoryItem(id) {
    const item = queryHistory.find(h => h.id.toString() === id);
    if (item) {
        const newName = prompt('Enter new name:', item.displayName);
        if (newName && newName.trim()) {
            item.displayName = newName.trim();
            saveQueryHistory();
            renderQueryHistory();
        }
    }
    // Close menu
    document.getElementById(`menu-${id}`).classList.remove('show');
}

function deleteHistoryItem(id) {
    if (confirm('Are you sure you want to delete this query from history?')) {
        queryHistory = queryHistory.filter(h => h.id.toString() !== id);
        saveQueryHistory();
        renderQueryHistory();
    }
    // Close menu
    const menu = document.getElementById(`menu-${id}`);
    if (menu) menu.classList.remove('show');
}

// Local storage functions
function saveQueryHistory() {
    try {
        localStorage.setItem('ragSearchHistory', JSON.stringify(queryHistory));
    } catch (error) {
        console.error('Failed to save query history:', error);
    }
}

function loadQueryHistory() {
    try {
        const saved = localStorage.getItem('ragSearchHistory');
        if (saved) {
            queryHistory = JSON.parse(saved);
            renderQueryHistory();
        }
    } catch (error) {
        console.error('Failed to load query history:', error);
        queryHistory = [];
    }
}

// Utility functions
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    const sendButton = document.getElementById('sendButton');
    
    if (show) {
        overlay.classList.remove('hidden');
        sendButton.disabled = true;
    } else {
        overlay.classList.add('hidden');
        sendButton.disabled = false;
    }
}

function showError(message) {
    // Create a more elegant error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'fixed top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded shadow-lg z-50';
    errorDiv.innerHTML = `
        <div class="flex items-center">
            <svg class="h-5 w-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
            </svg>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-red-500 hover:text-red-700">
                <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                </svg>
            </button>
        </div>
    `;
    
    document.body.appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

// Mobile menu functionality
function toggleMobileMenu() {
    const sidebar = document.getElementById('historySidebar');
    const overlay = document.getElementById('mobileOverlay');
    
    if (sidebar.classList.contains('-translate-x-full')) {
        // Show sidebar
        sidebar.classList.remove('-translate-x-full');
        overlay.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    } else {
        // Hide sidebar
        sidebar.classList.add('-translate-x-full');
        overlay.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

// Responsive behavior
function handleResize() {
    // Auto-hide mobile menu on desktop
    if (window.innerWidth >= 768) {
        const sidebar = document.getElementById('historySidebar');
        const overlay = document.getElementById('mobileOverlay');
        
        sidebar.classList.remove('-translate-x-full');
        overlay.classList.add('hidden');
        document.body.style.overflow = '';
    } else {
        // Ensure sidebar is hidden on mobile by default
        const sidebar = document.getElementById('historySidebar');
        if (!sidebar.classList.contains('-translate-x-full')) {
            sidebar.classList.add('-translate-x-full');
        }
    }
}

window.addEventListener('resize', handleResize);

// Initialize responsive behavior
handleResize();