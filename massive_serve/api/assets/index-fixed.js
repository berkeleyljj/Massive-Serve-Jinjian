(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))s(a);new MutationObserver(a=>{for(const r of a)if(r.type==="childList")for(const i of r.addedNodes)i.tagName==="LINK"&&i.rel==="modulepreload"&&s(i)}).observe(document,{childList:!0,subtree:!0});function t(a){const r={};return a.integrity&&(r.integrity=a.integrity),a.referrerPolicy&&(r.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?r.credentials="include":a.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function s(a){if(a.ep)return;a.ep=!0;const r=t(a);fetch(a.href,r)}})();class p{constructor(e,t,s,a){this.container=e,this.onSelect=t,this.onRename=s,this.onDelete=a,this.history=[],this.editingId=null,this.render()}updateHistory(e){this.history=e,this.render()}render(){if(this.history.length===0){this.container.innerHTML=`
        <div class="p-4 text-center text-chat-text-secondary">
          <p>No search history yet</p>
          <p class="text-sm mt-2">Your searches will appear here</p>
        </div>
      `;return}this.container.innerHTML=this.history.map(e=>this.renderHistoryItem(e)).join(""),this.attachEventListeners()}renderHistoryItem(e){const t=this.editingId===e.id,s=new Date(e.timestamp).toLocaleDateString(),a=new Date(e.timestamp).toLocaleTimeString();return t?`
        <div class="sidebar-item" data-id="${e.id}">
          <div class="flex items-center space-x-2">
            <input 
              type="text" 
              class="flex-1 px-2 py-1 border border-chat-border rounded text-sm"
              value="${e.query}"
              data-edit-input
            >
            <button class="text-green-600 hover:text-green-700" data-save-edit>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
              </svg>
            </button>
            <button class="text-gray-600 hover:text-gray-700" data-cancel-edit>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>
      `:`
      <div class="sidebar-item group" data-id="${e.id}">
        <div class="flex items-start justify-between">
          <div class="flex-1 min-w-0 cursor-pointer" data-select-query>
            <p class="text-sm font-medium text-chat-text truncate">${e.query}</p>
            <p class="text-xs text-chat-text-secondary mt-1">${s} at ${a}</p>
          </div>
          <div class="opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center space-x-1">
            <button class="text-chat-text-secondary hover:text-chat-text p-1" data-edit-query title="Rename">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path>
              </svg>
            </button>
            <button class="text-chat-text-secondary hover:text-red-600 p-1" data-delete-query title="Delete">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
              </svg>
            </button>
          </div>
        </div>
      </div>
    `}attachEventListeners(){this.container.querySelectorAll("[data-select-query]").forEach(e=>{e.addEventListener("click",t=>{const s=parseInt(t.target.closest("[data-id]").dataset.id),a=this.history.find(r=>r.id===s);a&&this.onSelect(a)})}),this.container.querySelectorAll("[data-edit-query]").forEach(e=>{e.addEventListener("click",t=>{t.stopPropagation();const s=parseInt(t.target.closest("[data-id]").dataset.id);this.editingId=s,this.render();const a=this.container.querySelector("[data-edit-input]");a&&(a.focus(),a.select())})}),this.container.querySelectorAll("[data-save-edit]").forEach(e=>{e.addEventListener("click",t=>{t.stopPropagation();const s=parseInt(t.target.closest("[data-id]").dataset.id),r=t.target.closest("[data-id]").querySelector("[data-edit-input]").value.trim();if(r){const i=this.history.find(n=>n.id===s);i&&this.onRename(i,r)}this.editingId=null,this.render()})}),this.container.querySelectorAll("[data-cancel-edit]").forEach(e=>{e.addEventListener("click",t=>{t.stopPropagation(),this.editingId=null,this.render()})}),this.container.querySelectorAll("[data-delete-query]").forEach(e=>{e.addEventListener("click",t=>{t.stopPropagation();const s=parseInt(t.target.closest("[data-id]").dataset.id),a=this.history.find(r=>r.id===s);a&&this.onDelete(a)})}),this.container.querySelectorAll("[data-edit-input]").forEach(e=>{e.addEventListener("keydown",t=>{if(t.key==="Enter"){t.preventDefault();const s=parseInt(t.target.closest("[data-id]").dataset.id),a=t.target.value.trim();if(a){const r=this.history.find(i=>i.id===s);r&&this.onRename(r,a)}this.editingId=null,this.render()}else t.key==="Escape"&&(t.preventDefault(),this.editingId=null,this.render())})})}}class v{constructor(e,t){this.container=e,this.onChange=t,this.searchParams={n_docs:10,nprobe:32,exact_rerank:!1,use_diverse:!1,lambda:.5},this.render()}render(){this.container.innerHTML=`
      <div class="flex flex-wrap gap-8 px-4">
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-gray-700">nprobe:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1">?</button>
            <div class="absolute bottom-full left-0 mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Number of clusters to search in FAISS index. Higher values = more thorough but slower search.
            </div>
          </div>
          <select class="border border-gray-300 rounded px-3 py-2 text-sm" data-nprobe>
            <option value="1">1</option>
            <option value="8">8</option>
            <option value="32" selected>32</option>
            <option value="64">64</option>
            <option value="128">128</option>
          </select>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-gray-700">k:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1">?</button>
            <div class="absolute bottom-full left-0 mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Number of documents to retrieve. Higher values = more results but may be less relevant.
            </div>
          </div>
          <select class="border border-gray-300 rounded px-3 py-2 text-sm" data-k>
            <option value="5">5</option>
            <option value="10" selected>10</option>
            <option value="20">20</option>
            <option value="50">50</option>
            <option value="100">100</option>
          </select>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-gray-700">Exact Rerank:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1">?</button>
            <div class="absolute bottom-full left-0 mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Enable exact reranking for more precise results. Slower but more accurate.
            </div>
          </div>
          <button class="toggle-switch ${this.searchParams.exact_rerank ? 'enabled' : 'disabled'}" data-exact-rerank>
            <span class="toggle-switch-thumb"></span>
          </button>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-gray-700">Diverse Search:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1">?</button>
            <div class="absolute bottom-full left-0 mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Enable diverse search to get more varied results using MMR algorithm.
            </div>
          </div>
          <button class="toggle-switch ${this.searchParams.use_diverse ? 'enabled' : 'disabled'}" data-diverse-search>
            <span class="toggle-switch-thumb"></span>
          </button>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-gray-700">λ:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1">?</button>
            <div class="absolute bottom-full left-0 mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Diversity parameter for MMR algorithm. Higher values = more diverse results.
            </div>
          </div>
          <select class="border border-gray-300 rounded px-3 py-2 text-sm ${this.searchParams.use_diverse ? 'bg-white' : 'bg-gray-100'}" data-lambda ${this.searchParams.use_diverse ? '' : 'disabled'}>
            <option value="0.1">0.1</option>
            <option value="0.3">0.3</option>
            <option value="0.5" selected>0.5</option>
            <option value="0.7">0.7</option>
            <option value="0.9">0.9</option>
          </select>
        </div>
      </div>
    `,this.attachEventListeners()}attachEventListeners(){this.container.querySelector("[data-nprobe]").addEventListener("change",e=>{this.searchParams.nprobe=parseInt(e.target.value),this.onChange(this.searchParams)}),this.container.querySelector("[data-k]").addEventListener("change",e=>{this.searchParams.n_docs=parseInt(e.target.value),this.onChange(this.searchParams)}),this.container.querySelector("[data-exact-rerank]").addEventListener("click",e=>{const t=e.target.closest("[data-exact-rerank]");this.searchParams.exact_rerank=!this.searchParams.exact_rerank,this.searchParams.exact_rerank?(t.classList.remove("disabled"),t.classList.add("enabled")):(t.classList.remove("enabled"),t.classList.add("disabled")),this.onChange(this.searchParams)}),this.container.querySelector("[data-diverse-search]").addEventListener("click",e=>{const t=e.target.closest("[data-diverse-search]");this.searchParams.use_diverse=!this.searchParams.use_diverse,this.searchParams.use_diverse?(t.classList.remove("disabled"),t.classList.add("enabled")):(t.classList.remove("enabled"),t.classList.add("disabled"));const s=this.container.querySelector("[data-lambda]");s.disabled=!this.searchParams.use_diverse,s.classList.toggle("bg-gray-100",!this.searchParams.use_diverse),this.onChange(this.searchParams)}),this.container.querySelector("[data-lambda]").addEventListener("change",e=>{this.searchParams.lambda=parseFloat(e.target.value),this.onChange(this.searchParams)})}setSearchParams(e){this.searchParams={...this.searchParams,...e};const t=this.container.querySelector("[data-nprobe]"),s=this.container.querySelector("[data-k]"),a=this.container.querySelector("[data-exact-rerank]"),r=this.container.querySelector("[data-diverse-search]"),i=this.container.querySelector("[data-lambda]");t&&(t.value=this.searchParams.nprobe),s&&(s.value=this.searchParams.n_docs),i&&(i.value=this.searchParams.lambda),a&&(this.searchParams.exact_rerank?(a.classList.remove("disabled"),a.classList.add("enabled")):(a.classList.remove("enabled"),a.classList.add("disabled"))),r&&(this.searchParams.use_diverse?(r.classList.remove("disabled"),r.classList.add("enabled")):(r.classList.remove("enabled"),r.classList.add("disabled"))),i&&(i.disabled=!this.searchParams.use_diverse,i.classList.toggle("bg-gray-100",!this.searchParams.use_diverse))}getSearchParams(){return{...this.searchParams}}}class m{constructor(e,t){this.container=e,this.onShowMore=t,this.results=null,this.expandedPassages=new Set,this.isLoading=!1,this.render()}render(){if(!this.results){this.container.innerHTML=`
        <div class="flex items-center justify-center h-full text-chat-text-secondary">
          <div class="text-center">
            <img src="./pod.jpg" alt="Icon" class="w-24 h-24 mx-auto mb-4" />
            <h2 class="text-4xl font-bold mb-2 text-gray-900">Dive into Compact-DS!</h2>
            <p class="text-base text-gray-700">Query and cast your valuable vote for relevance!</p>
          </div>
        </div>
      `;return}if(this.isLoading){this.container.innerHTML=`
        <div class="flex items-center justify-center h-full">
          <div class="text-center">
            <svg class="w-8 h-8 mx-auto mb-4 animate-spin text-chat-highlight" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            <p class="text-chat-text-secondary">Searching...</p>
          </div>
        </div>
      `;return}let e=[];if(this.results&&Array.isArray(this.results)){e=this.results}else if(this.results&&this.results.results&&this.results.results.passages){e=this.results.results.passages;if(Array.isArray(e)&&Array.isArray(e[0])){e=e[0]}}console.log("Final results array:",e);if(!e||e.length===0){const stickyContainer=document.getElementById("sticky-query-info");if(stickyContainer){stickyContainer.innerHTML=""}this.container.innerHTML=`
        <div class="flex items-center justify-center h-full text-chat-text-secondary">
          <div class="text-center">
            <svg class="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.47-.881-6.08-2.33"></path>
            </svg>
            <p class="text-lg font-medium">No results found</p>
            <p class="text-sm mt-2">Try adjusting your search parameters</p>
          </div>
        </div>
      `;return}const queryDisplay=this.latency&&this.searchParams?`
      <div class="bg-gray-50 py-6 px-8 rounded-lg shadow-sm mx-8 my-6">
        <div class="text-gray-700 text-lg font-medium mb-3">Query: <span class="font-semibold">${this.currentQuery||"Search Results"}</span></div>
        <div class="flex gap-2 flex-wrap">
          <span class="bg-gray-200 text-gray-700 text-sm px-2 py-1 rounded-full" style="background-color: #e5e7eb; color: #374151; font-size: 0.875rem; padding: 0.25rem 0.5rem; border-radius: 9999px;">nprobe: ${this.searchParams.nprobe}</span>
          <span class="bg-gray-200 text-gray-700 text-sm px-2 py-1 rounded-full" style="background-color: #e5e7eb; color: #374151; font-size: 0.875rem; padding: 0.25rem 0.5rem; border-radius: 9999px;">k: ${this.searchParams.n_docs}</span>
          <span class="bg-gray-200 text-gray-700 text-sm px-2 py-1 rounded-full" style="background-color: #e5e7eb; color: #374151; font-size: 0.875rem; padding: 0.25rem 0.5rem; border-radius: 9999px;">search latency: ${this.latency}s</span>
          ${this.searchParams.exact_rerank?'<span class="bg-blue-100 text-blue-800 text-sm px-2 py-1 rounded-full" style="background-color: #dbeafe; color: #1e40af; font-size: 0.875rem; padding: 0.25rem 0.5rem; border-radius: 9999px;">exact rerank</span>':''}
          ${this.searchParams.use_diverse?'<span class="bg-purple-100 text-purple-800 text-sm px-2 py-1 rounded-full" style="background-color: #f3e8ff; color: #6b21a8; font-size: 0.875rem; padding: 0.25rem 0.5rem; border-radius: 9999px;">diverse search</span>':''}
        </div>
      </div>
    `:'';const stickyContainer=document.getElementById("sticky-query-info");if(stickyContainer){stickyContainer.innerHTML=queryDisplay}this.container.innerHTML=`
      <div class="space-y-6">
        ${e.map((t,s)=>this.renderPassage(t,s)).join("")}
      </div>
    `,this.attachEventListeners()}renderPassage(e,t){const s=this.expandedPassages.has(e.index_id),a=this.highlightQueryTerms(e.text,e.matched_spans||[]);return`
      <div class="bg-white rounded-lg shadow-chat p-6 relative" data-passage-id="${e.index_id}">
        <!-- Passage Info at Top -->
        <div class="text-gray-700 text-sm mb-4 italic border-b border-gray-100 pb-3">
          #${t+1} • Source: <span class="font-semibold">${e.filename||e.source||"Unknown"}</span>
        </div>
        
        <div class="prose max-w-none">
          <div class="text-chat-text leading-relaxed mb-4" data-original-text="${this.escapeHtml(e.text)}">
            ${s?this.getExpandedText(e):a}
          </div>
        </div>
        
        <div class="mt-4 pt-4 border-t border-chat-border">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-4">
              <button 
                class="chat-button-secondary"
                data-show-more
                data-index-id="${e.index_id}"
                data-offset="1"
              >
                ${s?"Show Less":"Show More"}
              </button>
            </div>
          </div>
        </div>
        
        <!-- Rating Feedback Section - Clean Blue Text -->
        <div class="absolute bottom-4 right-4 flex items-center space-x-4">
          <span class="text-base font-medium text-blue-600">Is this passage relevant?</span>
          <button 
            onclick="this.parentElement.innerHTML='<span class=\\'text-sm text-green-600 font-medium\\'>✓ Thanks for the feedback!</span>'"
            class="text-blue-600 hover:text-green-600 text-xl font-bold transition-colors px-2"
            title="Yes, this passage is relevant"
          >
            [YES]
          </button>
          <button 
            onclick="this.parentElement.innerHTML='<span class=\\'text-sm text-red-600 font-medium\\'>✗ Thanks for the feedback!</span>'"
            class="text-blue-600 hover:text-red-600 text-xl font-bold transition-colors px-2"
            title="No, this passage is not relevant"
          >
            [NO]
          </button>
        </div>
      </div>
    `}renderExpandedContent(e){const t=this.getExpandedPassages(e.index_id);return!t||!Array.isArray(t)||t.length===0?`
        <div class="mt-4 p-4 bg-gray-50 rounded-lg">
          <p class="text-sm text-chat-text-secondary">Loading expanded content...</p>
        </div>
      `:`
      <div class="mt-4 space-y-4">
        ${t.map(s=>`
          <div class="pl-4 border-l-2 border-chat-border">
            <div class="prose max-w-none">
              <div class="text-chat-text leading-relaxed text-sm">
                ${this.highlightOriginalText(s.text,e.text)}
              </div>
            </div>
            <div class="mt-2 text-xs text-chat-text-secondary">
              #${s.index_id} • Source: ${s.filename||"Unknown"}
            </div>
          </div>
        `).join("")}
      </div>
    `}highlightQueryTerms(e,t){if(!t||t.length===0)return this.escapeHtml(e);const s=[...t].sort((i,n)=>i.start-n.start);let a="",r=0;return s.forEach(i=>{a+=this.escapeHtml(e.substring(r,i.start));const n=e.substring(i.start,i.end);a+=`<span class="passage-highlight">${this.escapeHtml(n)}</span>`,r=i.end}),a+=this.escapeHtml(e.substring(r)),a}highlightOriginalText(e,t){if(!t||!e)return this.escapeHtml(e);const s=t.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");const a=new RegExp(s,"g");return this.escapeHtml(e).replace(a,`<span class="border-b-2 border-green-500">${this.escapeHtml(t)}</span>`)}getExpandedText(e){const t=this.getExpandedPassages(e.index_id);if(!t||!Array.isArray(t)||t.length===0)return this.highlightQueryTerms(e.text,e.matched_spans||[]);const s=t[0];return this.highlightOriginalText(s.text,e.text)}escapeHtml(e){const t=document.createElement("div");return t.textContent=e,t.innerHTML}attachEventListeners(){this.container.querySelectorAll("[data-show-more]").forEach(e=>{e.addEventListener("click",t=>{const s=parseInt(t.target.dataset.indexId),a=parseInt(t.target.dataset.offset);this.expandedPassages.has(s)?(this.expandedPassages.delete(s),this.removeExpandedPassages(s),this.render(),setTimeout(()=>{const r=this.container.querySelector(`[data-passage-id="${s}"]`);if(r){const i=r.getBoundingClientRect(),n=this.container.getBoundingClientRect(),d=this.container.scrollTop;i.top<n.top+95&&(this.container.scrollTop=d+(i.top-n.top)-95)}},0)):(this.expandedPassages.add(s),this.render(),this.onShowMore(s,a))})});this.container.querySelectorAll("[data-collapse]").forEach(e=>{e.addEventListener("click",t=>{const s=parseInt(t.target.dataset.indexId);this.collapsePassage(s)})})}displayResults(e,latency,searchParams,query){console.log("DisplayResults called with:",e);let t=null;if(e&&e.results&&e.results.passages){t=e.results.passages;if(Array.isArray(t)&&Array.isArray(t[0])){t=t[0]}}else if(e&&Array.isArray(e)){t=e}console.log("Processed results:",t);this.results=t,this.latency=latency,this.searchParams=searchParams,this.currentQuery=query,this.isLoading=false,this.expandedPassages.clear(),this.render()}displayError(e){this.container.innerHTML=`
      <div class="flex items-center justify-center h-full">
        <div class="text-center">
          <svg class="w-16 h-16 mx-auto mb-4 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <p class="text-lg font-medium text-red-600">Error</p>
          <p class="text-sm mt-2 text-chat-text-secondary">${e}</p>
        </div>
      </div>
    `}setLoading(e){console.log("SearchResults setLoading called with:",e);this.isLoading=e,this.render()}expandPassage(e,t){console.log("expandPassage called with:",e,t);let s=[];if(t&&t.results&&t.results.passages){s=t.results.passages;if(Array.isArray(s)&&Array.isArray(s[0])){s=s[0]}}this.setExpandedPassages(e,s);console.log("Set expanded passages for",e,":",s);const a=this.container.querySelector(`[data-passage-id="${e}"]`);if(a&&s.length>0){const r=s[0],i=a.querySelector(".text-chat-text");if(i&&r.text){const d=i.dataset.originalText||i.textContent,o=r.text.trim();let c=o;if(d&&d!==o){const h=d.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");c=o.replace(new RegExp(h,"g"),`<span class="border-b-2 border-green-500">${d}</span>`)}i.innerHTML=c;i.dataset.expandedText=c;const n=a.querySelector("[data-show-more]");if(n){const u=(parseInt(n.dataset.offset)||1)+1;n.dataset.offset=u;if(u>3){n.disabled=true;n.classList.add("text-gray-400","cursor-not-allowed");n.classList.remove("text-blue-600","hover:underline");n.textContent="Expansion limit reached"}}}}this.render()}collapsePassage(e){const a=this.container.querySelector(`[data-passage-id="${e}"]`);if(a){const i=a.querySelector(".text-chat-text");const d=i.dataset.originalText;if(d){i.innerHTML=d;this.expandedPassages.delete(e);this.removeExpandedPassages(e);const n=a.querySelector("[data-show-more]");if(n){n.dataset.offset="1";n.disabled=false;n.classList.remove("text-gray-400","cursor-not-allowed");n.classList.add("text-blue-600","hover:underline");n.textContent="Show More"}this.render()}}}setExpandedPassages(e,t){this.expandedPassagesData||(this.expandedPassagesData=new Map),this.expandedPassagesData.set(e,t)}getExpandedPassages(e){return this.expandedPassagesData?this.expandedPassagesData.get(e)||[]:[]}removeExpandedPassages(e){this.expandedPassagesData&&this.expandedPassagesData.delete(e)}}class y{constructor(e,t){this.container=e,this.onSubmit=t,this.isLoading=!1,this.render()}    render(){this.container.innerHTML=`
      <div class="flex items-end gap-4">
        <textarea 
          class="flex-1 p-4 border border-gray-300 rounded-lg resize-none" 
          rows="1"
          placeholder="Enter your search query..."
          data-query-input
        ></textarea>
        <button 
          class="bg-teal-600 hover:bg-teal-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-lg px-6 py-4 text-sm font-medium transition-colors whitespace-nowrap flex-shrink-0"
          data-send-button
          disabled
          title="Send message"
          style="opacity: 1; visibility: visible; display: block;"
        >
          Send
        </button>
      </div>
    `,this.attachEventListeners()}attachEventListeners(){const e=this.container.querySelector("[data-query-input]"),t=this.container.querySelectorAll("[data-send-button]");e.addEventListener("input",()=>{this.adjustTextareaHeight(e),this.updateSendButtonState()}),e.addEventListener("keydown",s=>{s.key==="Enter"&&!s.shiftKey&&(s.preventDefault(),this.handleSubmit())}),t.forEach(s=>{s.addEventListener("click",()=>{this.handleSubmit()})}),this.adjustTextareaHeight(e)}adjustTextareaHeight(e){e.style.height="auto",e.style.height=Math.min(e.scrollHeight,120)+"px"}updateSendButtonState(){const e=this.container.querySelector("[data-query-input]"),t=this.container.querySelectorAll("[data-send-button]"),s=e.value.trim().length>0;t.forEach(a=>{a.disabled=!s||this.isLoading;a.style.opacity=s&&!this.isLoading?"1":"0.5";a.style.display="block";a.style.visibility="visible"})}handleSubmit(){console.log("handleSubmit called, isLoading:",this.isLoading);if(this.isLoading)return;const t=this.container.querySelector("[data-query-input]").value.trim();console.log("Query value:",t);t&&this.onSubmit(t)}setLoading(e){this.isLoading=e;const t=this.container.querySelectorAll("[data-send-button]");t.forEach(r=>{r.disabled=e;if(e){r.classList.add("animate-spin");r.style.opacity="0.5"}else{r.classList.remove("animate-spin");r.style.opacity="1"}}),this.updateSendButtonState()}setQuery(e){const t=this.container.querySelector("[data-query-input]");t.value=e,this.adjustTextareaHeight(t),this.updateSendButtonState()}clear(){const e=this.container.querySelector("[data-query-input]");e.value="",this.adjustTextareaHeight(e),this.updateSendButtonState()}focus(){this.container.querySelector("[data-query-input]").focus()}}class g{constructor(){this.baseUrl="http://192.222.59.156:30888"}async search(e,t={}){const{n_docs:s=10,nprobe:a=32,exact_rerank:r=!1,use_diverse:i=!1,lambda:n=.5}=t,d={query:e,nprobe:parseInt(a),n_docs:parseInt(s),use_rerank:r,use_diverse:i,lambda:n};console.log("Search payload:",d);try{const o=await fetch(`${this.baseUrl}/search`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});if(!o.ok)throw new Error(`HTTP error! status: ${o.status}`);const c=await o.json();console.log("Search response:",c);return c}catch(o){throw console.error("Search request failed:",o),o}}async expand(e,t){const s={query:"",expand_index_id:e,expand_offset:t};console.log("Expand payload:",s);try{const a=await fetch(`${this.baseUrl}/search`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(s)});if(!a.ok)throw new Error(`HTTP error! status: ${a.status}`);const r=await a.json();console.log("Expand response:",r);return r}catch(a){throw console.error("Expand request failed:",a),a}}}class x{constructor(e){this.root=e,this.searchService=new g,this.queryHistory=[],this.currentResults=null,this.currentSearchParams=null,this.isLoading=!1,this.init()}init(){this.render(),this.loadQueryHistory()}render(){this.root.innerHTML=`
      <div class="flex h-screen bg-chat-bg font-inter">
        <!-- Query History Sidebar -->
        <div class="w-80 bg-white border-r border-chat-border flex flex-col">
          <div class="p-4 border-b border-chat-border">
            <h1 class="text-xl font-semibold text-chat-text">Query History</h1>
          </div>
          <div id="query-history" class="flex-1 overflow-y-auto"></div>
        </div>
        
        <!-- Main Content Area -->
        <div class="flex-1 flex flex-col">
          <!-- Sticky Query Info -->
          <div id="sticky-query-info" class="sticky top-0 z-10 bg-white border-b border-chat-border"></div>
          
          <!-- Search Results -->
          <div id="search-results" class="flex-1 overflow-y-auto p-4"></div>
          
          <!-- Search Controls and Query Input Combined -->
          <div class="bg-white border-t border-chat-border">
            <div id="search-controls" class="p-8 pb-4 px-12"></div>
            <div id="query-input" class="px-12 pb-8"></div>
          </div>
        </div>
      </div>
    `,this.initializeComponents()}initializeComponents(){this.queryHistoryComponent=new p(document.getElementById("query-history"),this.onQueryHistorySelect.bind(this),this.onQueryHistoryRename.bind(this),this.onQueryHistoryDelete.bind(this)),this.searchControlsComponent=new v(document.getElementById("search-controls"),this.onSearchParamsChange.bind(this)),this.searchResultsComponent=new m(document.getElementById("search-results"),this.onShowMore.bind(this)),this.queryInputComponent=new y(document.getElementById("query-input"),this.onQuerySubmit.bind(this))}async onQuerySubmit(e,t){if(e.trim()){this.isLoading=!0,this.updateLoadingState();const startTime=performance.now();try{const searchParams=t||this.currentSearchParams||this.searchControlsComponent.getSearchParams();console.log("Using search params:",searchParams);const s=await this.searchService.search(e,searchParams);const endTime=performance.now();const latencyMs=((endTime-startTime)/1000).toFixed(2);console.log("Search completed, results:",s);this.currentResults=s;const a={id:Date.now(),query:e,timestamp:new Date().toISOString(),results:s,searchParams:searchParams,latency:latencyMs};this.queryHistory.unshift(a),this.saveQueryHistory(),this.queryHistoryComponent.updateHistory(this.queryHistory),this.searchResultsComponent.displayResults(s,latencyMs,searchParams,e)}catch(s){console.error("Search error:",s),this.searchResultsComponent.displayError("Search failed. Please try again.")}finally{this.isLoading=!1,this.updateLoadingState()}}}onQueryHistorySelect(e){this.currentResults=e.results,this.searchResultsComponent.displayResults(e.results,e.latency,e.searchParams,e.query),this.queryInputComponent.setQuery(e.query),this.searchControlsComponent.setSearchParams(e.searchParams)}onQueryHistoryRename(e,t){e.query=t,this.saveQueryHistory(),this.queryHistoryComponent.updateHistory(this.queryHistory)}onQueryHistoryDelete(e){this.queryHistory=this.queryHistory.filter(t=>t.id!==e.id),this.saveQueryHistory(),this.queryHistoryComponent.updateHistory(this.queryHistory)}onSearchParamsChange(e){this.currentSearchParams=e}async onShowMore(e,t){try{const s=await this.searchService.expand(e,t);this.searchResultsComponent.expandPassage(e,s)}catch(s){console.error("Expand error:",s),this.searchResultsComponent.displayError("Failed to expand passage.")}}updateLoadingState(){console.log("updateLoadingState called with isLoading:",this.isLoading);this.queryInputComponent.setLoading(this.isLoading),this.searchResultsComponent.setLoading(this.isLoading)}loadQueryHistory(){try{const e=localStorage.getItem("faiss-rag-query-history");e&&(this.queryHistory=JSON.parse(e),this.queryHistoryComponent.updateHistory(this.queryHistory))}catch(e){console.error("Failed to load query history:",e)}}saveQueryHistory(){try{const e=this.queryHistory.slice(0,50);localStorage.setItem("faiss-rag-query-history",JSON.stringify(e))}catch(e){console.error("Failed to save query history:",e)}}}document.addEventListener("DOMContentLoaded",()=>{const l=document.getElementById("root");new x(l)});
//# sourceMappingURL=index-DBhzJDZb.js.map
