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
      <div class="flex flex-wrap items-center gap-4">
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-chat-text">nprobe:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1" title="Number of clusters to search in FAISS index. Higher values = more thorough but slower search.">?</button>
            <div class="absolute top-full left-0 mt-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Number of clusters to search in FAISS index. Higher values = more thorough but slower search.
            </div>
          </div>
          <select class="dropdown text-sm" data-nprobe>
            <option value="1">1</option>
            <option value="8">8</option>
            <option value="32" selected>32</option>
            <option value="64">64</option>
            <option value="128">128</option>
          </select>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-chat-text">k:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1" title="Number of documents to retrieve. Higher values = more results but may be less relevant.">?</button>
            <div class="absolute top-full left-0 mt-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Number of documents to retrieve. Higher values = more results but may be less relevant.
            </div>
          </div>
          <select class="dropdown text-sm" data-k>
            <option value="5">5</option>
            <option value="10" selected>10</option>
            <option value="20">20</option>
            <option value="50">50</option>
            <option value="100">100</option>
          </select>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-chat-text">Exact Rerank:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1" title="Enable exact reranking for more precise results. Slower but more accurate.">?</button>
            <div class="absolute top-full left-0 mt-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Enable exact reranking for more precise results. Slower but more accurate.
            </div>
          </div>
          <button class="toggle-switch disabled" data-exact-rerank>
            <span class="toggle-switch-thumb"></span>
          </button>
        </div>
        

        
        <!-- Diverse Search Options -->
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-chat-text">Diverse Search:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1" title="Enable diverse search to get more varied results using MMR algorithm.">?</button>
            <div class="absolute top-full left-0 mt-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Enable diverse search to get more varied results using MMR algorithm.
            </div>
          </div>
          <button class="toggle-switch disabled" data-diverse-search>
            <span class="toggle-switch-thumb"></span>
          </button>
        </div>
        
        <div class="flex items-center space-x-2">
          <label class="text-sm font-medium text-chat-text">λ:</label>
          <div class="relative group">
            <button class="text-gray-400 hover:text-gray-600 text-xs ml-1" title="Diversity parameter for MMR algorithm. Higher values = more diverse results.">?</button>
            <div class="absolute top-full left-0 mt-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
              Diversity parameter for MMR algorithm. Higher values = more diverse results.
            </div>
          </div>
          <select class="dropdown text-sm" data-lambda disabled>
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
      `;return}let e=[];if(this.results&&this.results.results&&(e=this.results.results,Array.isArray(e)&&Array.isArray(e[0])&&(e=e[0])),e.length===0){this.container.innerHTML=`
        <div class="flex items-center justify-center h-full text-chat-text-secondary">
          <div class="text-center">
            <svg class="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.47-.881-6.08-2.33"></path>
            </svg>
            <p class="text-lg font-medium">No results found</p>
            <p class="text-sm mt-2">Try adjusting your search parameters</p>
          </div>
        </div>
      `;return}this.container.innerHTML=`
      <div class="space-y-6">
        ${e.map((t,s)=>this.renderPassage(t,s)).join("")}
      </div>
    `,this.attachEventListeners()}renderPassage(e,t){const s=this.expandedPassages.has(e.index_id),a=this.highlightQueryTerms(e.text,e.matched_spans||[]);return`
      <div class="bg-white rounded-lg shadow-chat p-6 relative" data-passage-id="${e.index_id}">
        <div class="prose max-w-none">
          <div class="text-chat-text leading-relaxed mb-4" data-original-text="${this.escapeHtml(e.text)}">
            ${a}
          </div>
        </div>
        
        <div class="mt-4 pt-4 border-t border-chat-border">
          <div class="flex items-center justify-between">
            <div class="text-sm text-chat-text-secondary">
              #${t+1} • Source: ${e.filename||e.source||"Unknown"}
            </div>
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
        
        <!-- Rating Feedback Section -->
        <div class="absolute bottom-4 right-4 flex items-center space-x-3 bg-gray-50 px-3 py-2 rounded-lg border">
          <span class="text-sm font-medium text-gray-700">Is this passage relevant?</span>
          <button 
            onclick="this.parentElement.innerHTML='<span class=\\'text-sm text-green-600 font-medium\\'>✓ Thanks for the feedback!</span>'"
            class="text-gray-400 hover:text-green-600 text-lg font-bold transition-colors"
            title="Yes, this passage is relevant"
          >
            [YES]
          </button>
          <button 
            onclick="this.parentElement.innerHTML='<span class=\\'text-sm text-red-600 font-medium\\'>✗ Thanks for the feedback!</span>'"
            class="text-gray-400 hover:text-red-600 text-lg font-bold transition-colors"
            title="No, this passage is not relevant"
          >
            [NO]
          </button>
        </div>
        
        ${s?this.renderExpandedContent(e):""}
      </div>
    `}renderExpandedContent(e){const t=this.getExpandedPassages(e.index_id);return!t||t.length===0?`
        <div class="mt-4 p-4 bg-gray-50 rounded-lg">
          <p class="text-sm text-chat-text-secondary">Loading expanded content...</p>
        </div>
      `:`
      <div class="mt-4 space-y-4">
        ${t.map(s=>`
          <div class="pl-4 border-l-2 border-chat-border">
            <div class="prose max-w-none">
              <div class="text-chat-text leading-relaxed text-sm">
                ${this.highlightQueryTerms(s.text,s.matched_spans||[])}
              </div>
            </div>
            <div class="mt-2 text-xs text-chat-text-secondary">
              #${s.index_id} • Source: ${s.filename||"Unknown"}
            </div>
          </div>
        `).join("")}
      </div>
    `}highlightQueryTerms(e,t){if(!t||t.length===0)return this.escapeHtml(e);const s=[...t].sort((i,n)=>i.start-n.start);let a="",r=0;return s.forEach(i=>{a+=this.escapeHtml(e.substring(r,i.start));const n=e.substring(i.start,i.end);a+=`<span class="passage-highlight">${this.escapeHtml(n)}</span>`,r=i.end}),a+=this.escapeHtml(e.substring(r)),a}escapeHtml(e){const t=document.createElement("div");return t.textContent=e,t.innerHTML}attachEventListeners(){this.container.querySelectorAll("[data-show-more]").forEach(e=>{e.addEventListener("click",t=>{const s=parseInt(t.target.dataset.indexId),a=parseInt(t.target.dataset.offset);this.expandedPassages.has(s)?(this.expandedPassages.delete(s),this.removeExpandedPassages(s),this.render(),setTimeout(()=>{const r=this.container.querySelector(`[data-passage-id="${s}"]`);if(r){const i=r.getBoundingClientRect(),n=this.container.getBoundingClientRect(),d=this.container.scrollTop;i.top<n.top+95&&(this.container.scrollTop=d+(i.top-n.top)-95)}},0)):(this.expandedPassages.add(s),this.render(),this.onShowMore(s,a))})})}displayResults(e){console.log("DisplayResults called with:",e),this.results=e,this.expandedPassages.clear(),this.render()}displayError(e){this.container.innerHTML=`
      <div class="flex items-center justify-center h-full">
        <div class="text-center">
          <svg class="w-16 h-16 mx-auto mb-4 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <p class="text-lg font-medium text-red-600">Error</p>
          <p class="text-sm mt-2 text-chat-text-secondary">${e}</p>
        </div>
      </div>
    `}setLoading(e){this.isLoading=e,e&&this.render()}expandPassage(e,t){let s=[];t&&t.results&&(s=t.results,Array.isArray(s)&&Array.isArray(s[0])&&(s=s[0])),this.setExpandedPassages(e,s);const a=this.container.querySelector(`[data-passage-id="${e}"]`);if(a&&s.length>0){const r=s[0],i=a.querySelector(".text-chat-text"),n=a.querySelector("[data-show-more]");if(i&&r.text){const d=i.dataset.originalText||i.textContent,o=r.text.trim();let c=o;if(d&&d!==o){const h=d.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");c=o.replace(new RegExp(h,"g"),`<span class="border-b-2 border-green-500">${d}</span>`)}if(i.innerHTML=c,i.dataset.expandedText=c,n){const u=(parseInt(n.dataset.offset)||1)+1;n.dataset.offset=u,u>3&&(n.disabled=!0,n.classList.add("text-gray-400","cursor-not-allowed"),n.classList.remove("text-blue-600","hover:underline"),n.textContent="Expansion limit reached")}}}}setExpandedPassages(e,t){this.expandedPassagesData||(this.expandedPassagesData=new Map),this.expandedPassagesData.set(e,t)}getExpandedPassages(e){return this.expandedPassagesData?this.expandedPassagesData.get(e)||[]:[]}removeExpandedPassages(e){this.expandedPassagesData&&this.expandedPassagesData.delete(e)}}class y{constructor(e,t){this.container=e,this.onSubmit=t,this.isLoading=!1,this.render()}render(){this.container.innerHTML=`
      <div class="flex items-end space-x-3">
        <div class="flex-1 relative">
          <textarea 
            class="chat-input pr-12 resize-none" 
            rows="1"
            placeholder="Enter your search query..."
            data-query-input
          ></textarea>
          <button 
            class="absolute right-3 bottom-3 text-chat-text-secondary hover:text-chat-highlight disabled:opacity-50 disabled:cursor-not-allowed"
            data-send-button
            disabled
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
            </svg>
          </button>
        </div>
        <button 
          class="chat-button disabled:opacity-50 disabled:cursor-not-allowed"
          data-send-button
          disabled
        >
          <span data-button-text>Send</span>
          <svg class="w-4 h-4 ml-2 hidden" data-loading-icon fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
          </svg>
        </button>
      </div>
    `,this.attachEventListeners()}attachEventListeners(){const e=this.container.querySelector("[data-query-input]"),t=this.container.querySelectorAll("[data-send-button]");e.addEventListener("input",()=>{this.adjustTextareaHeight(e),this.updateSendButtonState()}),e.addEventListener("keydown",s=>{s.key==="Enter"&&!s.shiftKey&&(s.preventDefault(),this.handleSubmit())}),t.forEach(s=>{s.addEventListener("click",()=>{this.handleSubmit()})}),this.adjustTextareaHeight(e)}adjustTextareaHeight(e){e.style.height="auto",e.style.height=Math.min(e.scrollHeight,120)+"px"}updateSendButtonState(){const e=this.container.querySelector("[data-query-input]"),t=this.container.querySelectorAll("[data-send-button]"),s=e.value.trim().length>0;t.forEach(a=>{a.disabled=!s||this.isLoading})}handleSubmit(){if(this.isLoading)return;const t=this.container.querySelector("[data-query-input]").value.trim();t&&this.onSubmit(t)}setLoading(e){this.isLoading=e;const t=this.container.querySelectorAll("[data-send-button]"),s=this.container.querySelectorAll("[data-button-text]"),a=this.container.querySelectorAll("[data-loading-icon]");t.forEach(r=>{r.disabled=e}),s.forEach(r=>{r.textContent=e?"Searching...":"Send"}),a.forEach(r=>{e?(r.classList.remove("hidden"),r.classList.add("animate-spin")):(r.classList.add("hidden"),r.classList.remove("animate-spin"))}),this.updateSendButtonState()}setQuery(e){const t=this.container.querySelector("[data-query-input]");t.value=e,this.adjustTextareaHeight(t),this.updateSendButtonState()}clear(){const e=this.container.querySelector("[data-query-input]");e.value="",this.adjustTextareaHeight(e),this.updateSendButtonState()}focus(){this.container.querySelector("[data-query-input]").focus()}}class g{constructor(){this.baseUrl="http://192.222.59.156:30888"}async search(e,t={}){const{n_docs:s=10,nprobe:a=32,exact_rerank:r=!1,use_diverse:i=!1,lambda:n=.5}=t,d={query:e,nprobe:parseInt(a),n_docs:parseInt(s),use_rerank:r,use_diverse:i,lambda:n};console.log("Search payload:",d);try{const o=await fetch(`${this.baseUrl}/search`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});if(!o.ok)throw new Error(`HTTP error! status: ${o.status}`);return await o.json()}catch(o){throw console.error("Search request failed:",o),o}}async expand(e,t){const s={expand_index_id:e,expand_offset:t};console.log("Expand payload:",s);try{const a=await fetch(`${this.baseUrl}/search`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(s)});if(!a.ok)throw new Error(`HTTP error! status: ${a.status}`);return await a.json()}catch(a){throw console.error("Expand request failed:",a),a}}}class x{constructor(e){this.root=e,this.searchService=new g,this.queryHistory=[],this.currentResults=null,this.isLoading=!1,this.init()}init(){this.render(),this.loadQueryHistory()}render(){this.root.innerHTML=`
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
          <!-- Search Controls -->
          <div id="search-controls" class="p-4 border-b border-chat-border bg-white"></div>
          
          <!-- Search Results -->
          <div id="search-results" class="flex-1 overflow-y-auto p-4"></div>
          
          <!-- Query Input -->
          <div class="p-4 border-t border-chat-border bg-white">
            <div id="query-input"></div>
          </div>
        </div>
      </div>
    `,this.initializeComponents()}initializeComponents(){this.queryHistoryComponent=new p(document.getElementById("query-history"),this.onQueryHistorySelect.bind(this),this.onQueryHistoryRename.bind(this),this.onQueryHistoryDelete.bind(this)),this.searchControlsComponent=new v(document.getElementById("search-controls"),this.onSearchParamsChange.bind(this)),this.searchResultsComponent=new m(document.getElementById("search-results"),this.onShowMore.bind(this)),this.queryInputComponent=new y(document.getElementById("query-input"),this.onQuerySubmit.bind(this))}async onQuerySubmit(e,t){if(e.trim()){this.isLoading=!0,this.updateLoadingState();try{const s=await this.searchService.search(e,t);this.currentResults=s;const a={id:Date.now(),query:e,timestamp:new Date().toISOString(),results:s,searchParams:t};this.queryHistory.unshift(a),this.saveQueryHistory(),this.queryHistoryComponent.updateHistory(this.queryHistory),this.searchResultsComponent.displayResults(s)}catch(s){console.error("Search error:",s),this.searchResultsComponent.displayError("Search failed. Please try again.")}finally{this.isLoading=!1,this.updateLoadingState()}}}onQueryHistorySelect(e){this.currentResults=e.results,this.searchResultsComponent.displayResults(e.results),this.queryInputComponent.setQuery(e.query),this.searchControlsComponent.setSearchParams(e.searchParams)}onQueryHistoryRename(e,t){e.query=t,this.saveQueryHistory(),this.queryHistoryComponent.updateHistory(this.queryHistory)}onQueryHistoryDelete(e){this.queryHistory=this.queryHistory.filter(t=>t.id!==e.id),this.saveQueryHistory(),this.queryHistoryComponent.updateHistory(this.queryHistory)}onSearchParamsChange(e){this.currentSearchParams=e}async onShowMore(e,t){try{const s=await this.searchService.expand(e,t);this.searchResultsComponent.expandPassage(e,s)}catch(s){console.error("Expand error:",s),this.searchResultsComponent.displayError("Failed to expand passage.")}}updateLoadingState(){this.queryInputComponent.setLoading(this.isLoading),this.searchResultsComponent.setLoading(this.isLoading)}loadQueryHistory(){try{const e=localStorage.getItem("faiss-rag-query-history");e&&(this.queryHistory=JSON.parse(e),this.queryHistoryComponent.updateHistory(this.queryHistory))}catch(e){console.error("Failed to load query history:",e)}}saveQueryHistory(){try{const e=this.queryHistory.slice(0,50);localStorage.setItem("faiss-rag-query-history",JSON.stringify(e))}catch(e){console.error("Failed to save query history:",e)}}}document.addEventListener("DOMContentLoaded",()=>{const l=document.getElementById("root");new x(l)});
//# sourceMappingURL=index-DBhzJDZb.js.map
