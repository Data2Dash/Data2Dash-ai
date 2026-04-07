from __future__ import annotations
from typing import List, Optional
from pyvis.network import Network
from langchain_experimental.graph_transformers.llm import GraphDocument


# ---------------------------------------------------------------------------
# HTML snippet injected into every generated graph.
# It adds a floating search bar that filters / focuses nodes by name.
# ---------------------------------------------------------------------------
_SEARCH_OVERLAY = """
<style>
  /* ---- search bar overlay ---- */
  #kg-search-wrapper {
    position: fixed;
    top: 14px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(15, 20, 40, 0.88);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(75, 108, 183, 0.55);
    border-radius: 30px;
    padding: 6px 14px 6px 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.45);
    min-width: 320px;
    max-width: 90vw;
  }
  #kg-search-wrapper svg { flex-shrink: 0; opacity: 0.65; }
  #kg-search-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: #e8eaf6;
    font-size: 14px;
    font-family: 'Inter', sans-serif;
    caret-color: #7c9fe4;
    min-width: 0;
  }
  #kg-search-input::placeholder { color: rgba(232,234,246,0.4); }
  #kg-clear-btn {
    background: none;
    border: none;
    color: rgba(232,234,246,0.5);
    cursor: pointer;
    font-size: 18px;
    line-height: 1;
    padding: 0 2px;
    display: none;
  }
  #kg-clear-btn:hover { color: #e8eaf6; }

  /* suggestion dropdown */
  #kg-suggestions {
    position: fixed;
    top: 56px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9998;
    background: rgba(15, 20, 40, 0.96);
    border: 1px solid rgba(75, 108, 183, 0.4);
    border-radius: 14px;
    overflow: hidden;
    min-width: 320px;
    max-width: 90vw;
    max-height: 240px;
    overflow-y: auto;
    display: none;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
  }
  .kg-sug-item {
    padding: 9px 18px;
    color: #c5d0f0;
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: background 0.15s;
  }
  .kg-sug-item:hover, .kg-sug-item.active { background: rgba(75,108,183,0.35); color: #fff; }

  /* match count badge */
  #kg-match-badge {
    position: fixed;
    top: 60px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9997;
    font-size: 11px;
    color: rgba(200,210,255,0.7);
    display: none;
    pointer-events: none;
  }
</style>

<div id="kg-search-wrapper">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7c9fe4" stroke-width="2.2"
       stroke-linecap="round" stroke-linejoin="round">
    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
  <input id="kg-search-input" type="text" autocomplete="off"
         placeholder="Search node…" />
  <button id="kg-clear-btn" title="Clear">&#x2715;</button>
</div>
<div id="kg-suggestions"></div>
<span id="kg-match-badge"></span>

<script>
(function () {
  // ------- helpers -------
  const input   = document.getElementById('kg-search-input');
  const clearBtn = document.getElementById('kg-clear-btn');
  const sugBox  = document.getElementById('kg-suggestions');
  const badge   = document.getElementById('kg-match-badge');

  // original colours stored once
  const ORIG_NODE_COLOR  = '#b9d9ea';
  const HIT_COLOR        = '#f4a261';   // orange – highlighted hit
  const DIM_COLOR        = '#d0d8e8';   // faded
  const HIT_BORDER       = '#e76f51';

  let network = null;          // vis.js Network instance
  let allNodeIds = [];
  let activeIndex = -1;        // for keyboard navigation in dropdown

  // ------- wait for vis Network to be available -------
  function getNetwork () {
    // pyvis stores the Network as a global 'network' in the page
    if (typeof window.network !== 'undefined') return window.network;
    // fallback: some pyvis versions expose it differently
    const keys = Object.keys(window).filter(k => {
      try { return window[k] && window[k].body && window[k].body.nodes; } catch { return false; }
    });
    return keys.length ? window[keys[0]] : null;
  }

  function init () {
    network = getNetwork();
    if (!network) { setTimeout(init, 300); return; }
    allNodeIds = network.body.data.nodes.getIds();
  }
  setTimeout(init, 800);

  // ------- search logic -------
  function getLabel (id) {
    const node = network && network.body.data.nodes.get(id);
    return node ? (node.label || String(id)) : String(id);
  }

  function resetColors () {
    if (!network) return;
    const updates = allNodeIds.map(id => ({
      id,
      color: { background: ORIG_NODE_COLOR, border: '#9dbfd9' },
      font: { color: '#222222', size: 14 }
    }));
    network.body.data.nodes.update(updates);
  }

  function highlightNodes (ids) {
    if (!network) return;
    const hitSet = new Set(ids);
    const updates = allNodeIds.map(id => hitSet.has(id)
      ? { id, color: { background: HIT_COLOR, border: HIT_BORDER }, font: { color: '#1a1a1a', size: 16 } }
      : { id, color: { background: DIM_COLOR, border: '#b0bcd8' }, font: { color: '#888', size: 12 } }
    );
    network.body.data.nodes.update(updates);
  }

  function focusNode (id) {
    if (!network) return;
    network.focus(id, { scale: 1.6, animation: { duration: 700, easingFunction: 'easeInOutQuad' } });
    network.selectNodes([id]);
  }

  function buildSuggestions (query) {
    sugBox.innerHTML = '';
    activeIndex = -1;
    if (!query || !network) { sugBox.style.display = 'none'; badge.style.display = 'none'; return; }

    const q = query.toLowerCase();
    const hits = allNodeIds.filter(id => getLabel(id).toLowerCase().includes(q));

    if (hits.length === 0) {
      sugBox.style.display = 'none';
      badge.textContent = 'No matches';
      badge.style.display = 'block';
      resetColors();
      return;
    }

    badge.style.display = 'none';
    highlightNodes(hits);

    hits.slice(0, 30).forEach((id, i) => {
      const label = getLabel(id);
      const div = document.createElement('div');
      div.className = 'kg-sug-item';
      // bold the matching part
      const idx = label.toLowerCase().indexOf(q);
      div.innerHTML = idx >= 0
        ? label.slice(0, idx) + '<strong>' + label.slice(idx, idx + q.length) + '</strong>' + label.slice(idx + q.length)
        : label;
      div.addEventListener('mousedown', e => { e.preventDefault(); selectItem(id, label); });
      sugBox.appendChild(div);
    });

    if (hits.length > 30) {
      const more = document.createElement('div');
      more.className = 'kg-sug-item';
      more.style.opacity = '0.5';
      more.style.fontStyle = 'italic';
      more.textContent = `… and ${hits.length - 30} more`;
      sugBox.appendChild(more);
    }

    sugBox.style.display = 'block';
  }

  function selectItem (id, label) {
    input.value = label;
    sugBox.style.display = 'none';
    clearBtn.style.display = 'inline';
    resetColors();
    // highlight just this one
    const updates = allNodeIds.map(nid => nid === id
      ? { id: nid, color: { background: HIT_COLOR, border: HIT_BORDER }, font: { color: '#1a1a1a', size: 17 } }
      : { id: nid, color: { background: DIM_COLOR, border: '#b0bcd8' }, font: { color: '#999', size: 12 } }
    );
    if (network) network.body.data.nodes.update(updates);
    focusNode(id);
  }

  // ------- keyboard nav -------
  input.addEventListener('keydown', e => {
    const items = sugBox.querySelectorAll('.kg-sug-item');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, items.length - 1);
      items.forEach((el, i) => el.classList.toggle('active', i === activeIndex));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      items.forEach((el, i) => el.classList.toggle('active', i === activeIndex));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && activeIndex < items.length) {
        items[activeIndex].dispatchEvent(new MouseEvent('mousedown'));
      } else if (allNodeIds.length) {
        // just focus first hit
        const q = input.value.toLowerCase();
        const first = allNodeIds.find(id => getLabel(id).toLowerCase().includes(q));
        if (first) selectItem(first, getLabel(first));
      }
    } else if (e.key === 'Escape') {
      sugBox.style.display = 'none';
      resetColors();
    }
  });

  input.addEventListener('input', () => {
    const q = input.value.trim();
    clearBtn.style.display = q ? 'inline' : 'none';
    buildSuggestions(q);
  });

  input.addEventListener('focus', () => {
    if (input.value.trim()) buildSuggestions(input.value.trim());
  });

  document.addEventListener('click', e => {
    if (!e.target.closest('#kg-search-wrapper') && !e.target.closest('#kg-suggestions')) {
      sugBox.style.display = 'none';
    }
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    clearBtn.style.display = 'none';
    sugBox.style.display = 'none';
    badge.style.display = 'none';
    resetColors();
    input.focus();
  });
})();
</script>
"""


def visualize_graph(
    graph_documents: List[GraphDocument],
    output_file: str = "knowledge_graph.html",
) -> Optional[str]:
    if not graph_documents:
        return None

    g = graph_documents[0]
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
    )

    for n in g.nodes:
        net.add_node(n.id, label=n.id, title=getattr(n, "type", "Concept"), color="#b9d9ea")

    for r in g.relationships:
        net.add_edge(r.source.id, r.target.id, label=r.type, color="#97c2fc")

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 110,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "enabled": true, "iterations": 900 }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "hover": true,
        "zoomView": true
      }
    }
    """)

    net.save_graph(output_file)

    # ---- Inject search overlay into the saved HTML ----
    with open(output_file, "r", encoding="utf-8") as fh:
        html = fh.read()

    # Insert just before </body> so the DOM (and vis.js) is already loaded
    html = html.replace("</body>", _SEARCH_OVERLAY + "\n</body>")

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(html)

    return output_file
