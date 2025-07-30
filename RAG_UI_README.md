# RAG Search UI

A modern, responsive web interface for performing retrieval-augmented generation (RAG) searches. Built with vanilla JavaScript, HTML, and Tailwind CSS for a clean, ChatGPT-inspired user experience.

## Features

### 🔍 Advanced Search
- **Query Input**: ChatGPT-style input bar with Enter key support
- **Search Parameters**: 
  - `nprobe`: Configurable values [1, 8, 32, 64, 128] (default: 32)
  - `k`: Number of results [5, 10, 20, 50, 100] (default: 10)
  - `exact_rerank`: Boolean toggle for exact reranking (default: false)

### 📚 Smart Results Display
- **Highlighted Text**: Query terms are automatically highlighted in search results
- **Metadata Display**: Shows passage index and source filename
- **Expandable Content**: "Show More" button to retrieve neighboring passages
- **Scroll Management**: 85px buffer when collapsing expanded content

### 📝 Query History
- **Persistent History**: Automatically saves search queries locally
- **Clickable Results**: Re-display previous search results instantly
- **Management Options**: 
  - Rename queries with custom display names
  - Delete individual queries from history
  - Automatic timestamp tracking

### 📱 Responsive Design
- **Mobile-First**: Fully responsive layout for all screen sizes
- **Collapsible Sidebar**: Mobile hamburger menu for query history
- **Touch-Friendly**: Optimized for mobile interactions
- **Desktop Experience**: Fixed sidebar and optimal layout for larger screens

## Getting Started

### Prerequisites
- A running RAG search backend at `http://localhost:30888/search`
- Modern web browser with JavaScript enabled

### Installation
1. Clone or download the application files:
   ```
   index.html
   app.js
   ```

2. Open `index.html` in your web browser or serve it using a local web server:
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Using Node.js (if you have http-server installed)
   npx http-server
   ```

3. Navigate to the served URL (e.g., `http://localhost:8000`)

### Backend API Requirements

The application expects a backend server running at `http://localhost:30888/search` that accepts:

#### Search Request
```json
POST /search
{
  "query": "example search query",
  "n_docs": 10,
  "nprobe": 32,
  "exact_rerank": false
}
```

#### Expansion Request
```json
POST /search
{
  "expand_index_id": 123,
  "expand_offset": 1
}
```

#### Expected Response Format
```json
{
  "results": {
    "passages": [
      {
        "text": "passage content",
        "filename": "source.txt",
        "index_id": 123
      }
    ]
  }
}
```

## Usage

### Basic Search
1. Enter your search query in the input field at the bottom
2. Adjust search parameters using the dropdown menus:
   - **nprobe**: Controls search accuracy vs speed
   - **k**: Number of results to return
   - **exact rerank**: Enable for more precise ranking
3. Press Enter or click the send button
4. Review highlighted results with source information

### Expanding Results
- Click "Show More" on any result to view neighboring passages
- Click again to collapse the expanded content
- Automatic scroll positioning maintains context

### Managing History
- Previous searches appear in the left sidebar
- Click any history item to reload those results
- Use the three-dot menu (⋮) to:
  - Rename queries for better organization
  - Delete unwanted history entries

### Mobile Usage
- Tap the hamburger menu (☰) to access query history
- All desktop features available with touch-optimized interface
- Swipe or tap overlay to close mobile menu

## Customization

### Styling
The application uses Tailwind CSS classes. Key customization points:

- **Color Scheme**: Modify color classes in `index.html`
- **Layout**: Adjust responsive breakpoints and spacing
- **Typography**: Uses Inter font family (weights: 400, 600)

### Search Parameters
Default values can be modified in `app.js`:
```javascript
let currentNprobe = 32;    // Default nprobe value
let currentK = 10;         // Default k value
let currentExactRerank = false; // Default rerank setting
```

### Word Context (Optional)
Enable word context by setting the `showWords` parameter:
```javascript
let showWords = 50; // Show 50 words before/after matches
```

## Browser Compatibility

- **Modern Browsers**: Chrome 60+, Firefox 55+, Safari 12+, Edge 79+
- **Features Used**: 
  - ES6+ JavaScript (async/await, arrow functions)
  - CSS Grid and Flexbox
  - Local Storage API
  - Fetch API

## Troubleshooting

### Common Issues

**No search results appearing:**
- Verify backend server is running on `http://localhost:30888`
- Check browser console for network errors
- Ensure CORS is properly configured on the backend

**Mobile menu not working:**
- Confirm JavaScript is enabled
- Try refreshing the page
- Check for console errors

**History not saving:**
- Verify browser supports Local Storage
- Check if private/incognito mode is affecting storage
- Clear browser cache and try again

**Styling issues:**
- Ensure internet connection for Tailwind CSS CDN
- Check if content blockers are interfering with external resources

### Performance Tips

- **Large Result Sets**: Consider pagination for queries returning many results
- **History Management**: Automatic cleanup limits history to 50 items
- **Mobile Performance**: Minimize expanded passages on slower devices

## Architecture

### File Structure
```
├── index.html          # Main HTML structure and styling
├── app.js             # JavaScript functionality
└── RAG_UI_README.md   # This documentation
```

### Key Components
- **State Management**: Global variables for current search parameters
- **API Integration**: Fetch-based communication with backend
- **History System**: Local Storage-based persistence
- **Responsive Layout**: CSS Grid and Flexbox with Tailwind utilities

### Data Flow
1. User input → Parameter collection → API request
2. API response → Result processing → DOM rendering
3. History management → Local storage → Sidebar updates

## Contributing

When modifying the application:

1. **Maintain Responsiveness**: Test on both desktop and mobile
2. **Preserve Accessibility**: Keep keyboard navigation and screen reader support
3. **Follow Patterns**: Use existing code patterns for consistency
4. **Test Thoroughly**: Verify all features work with your changes

## License

This application is provided as-is for educational and development purposes.