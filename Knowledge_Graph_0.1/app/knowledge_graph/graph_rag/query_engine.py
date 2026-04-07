from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.knowledge_graph.store.vector_store import InMemoryVectorStore

try:
    from langchain_community.graphs import Neo4jGraph
except Exception:
    Neo4jGraph = None

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words to strip when extracting seed terms from a query
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "what", "which", "where", "when", "how", "why", "who", "does", "did",
    "the", "and", "but", "for", "are", "was", "were", "with", "that",
    "this", "from", "have", "has", "been", "its", "they", "their",
    "about", "into", "over", "more", "some", "than", "then", "each",
    "both", "also", "can", "could", "would", "should", "not", "any",
    "all", "other", "used", "use", "using", "give", "tell", "explain",
    "describe", "list", "show", "find", "get", "make", "take",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedContext:
    id: str
    text: str
    source_type: str   # "Vector Chunk" or "Knowledge Graph"
    score: float = 0.0


@dataclass(frozen=True)
class QueryConfig:
    # Vector fetch
    top_k_chunks: int = 10
    max_chunk_chars_each: int = 1500

    # Graph fetch (Neo4j)
    expand_hops: int = 2
    max_graph_facts: int = 60

    # Reranker
    top_k_rerank: int = 6

    # Synthesis model
    synthesis_model: str = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# System + Answer Prompts  — upgraded for richer, structured responses
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Data2Dash Research Assistant — an expert academic analyst powered by a \
Hybrid GraphRAG pipeline (vector chunks + knowledge-graph triplets).

Behavioral rules:
1. Answer ONLY from the provided Context Blocks. Never hallucinate.
2. If the context is insufficient, say so explicitly and state what is missing.
3. Structure your answer clearly using markdown: use **bold** for key terms, \
bullet-lists for enumerations, and numbered steps for processes.
4. Do NOT leak internal identifiers (Chunk IDs, Graph Triplet numbers, scores).
5. Be precise. Prioritize depth over breadth. Cite specific entities, metrics, \
methods, or results found in the context.
6. End with a concise "**Key Takeaway**" sentence summarising the answer in ≤ 25 words.
"""

_ANSWER_PROMPT = """\
## User Question
{question}

---
## Retrieved Context Blocks
{context}

---
## Instructions
Using ONLY the evidence above, write a well-structured, analytical answer. \
Follow the behavioral rules defined in the system prompt exactly.
"""


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def retrieve_chunks_bge_m3(
    vstore: InMemoryVectorStore, query: str, qc: QueryConfig
) -> List[RetrievedContext]:
    """Dense semantic retrieval (BGE-M3 style) from the in-memory vector store."""
    results = vstore.search(query, top_k=qc.top_k_chunks)
    out: List[RetrievedContext] = []
    for cid, text, score in results:
        out.append(
            RetrievedContext(
                id=f"Chunk {cid}",
                text=text,
                source_type="Vector Chunk",
                score=float(score),
            )
        )
    return out


def _extract_seed_terms(question: str) -> List[str]:
    """
    Extract meaningful query terms for graph seed lookup.
    Strips stop-words, numbers, and short tokens; also detects
    quoted phrases (e.g. "Transformer") as high-priority seeds.
    """
    seeds: List[str] = []

    # 1. Quoted phrases take priority
    quoted = re.findall(r'"([^"]+)"', question)
    seeds.extend(quoted)

    # 2. Capitalised runs (likely named entities / model names)
    cap_runs = re.findall(r'\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*\b', question)
    seeds.extend(cap_runs)

    # 3. Remaining content words (len >= 4, not stop-words)
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]{3,}\b', question)
    for tok in tokens:
        t = tok.lower()
        if t not in _STOP_WORDS:
            seeds.append(tok)

    # Deduplicate preserving order
    seen: set = set()
    unique: List[str] = []
    for s in seeds:
        low = s.lower()
        if low not in seen:
            seen.add(low)
            unique.append(s)
    return unique[:12]  # cap to avoid Cypher bloat


def _fetch_graph_facts_2hop(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    seed_terms: List[str],
    qc: QueryConfig,
) -> List[RetrievedContext]:
    """
    Graph Fetch: 2-hop neighbour relationships from Neo4j for seed entities.
    """
    if Neo4jGraph is None or not seed_terms:
        return []

    try:
        g = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
    except Exception as e:
        LOGGER.error(f"Neo4j connection failed: {e}")
        return []

    clean = [t for t in seed_terms if t.strip()]
    if not clean:
        return []

    where = " OR ".join(
        [f"toLower(n.id) CONTAINS toLower($t{i})" for i in range(len(clean))]
    )
    params = {f"t{i}": clean[i] for i in range(len(clean))}

    cypher = f"""
    MATCH path = (n)-[*1..{qc.expand_hops}]-(m)
    WHERE {where}
    UNWIND relationships(path) AS r
    WITH startNode(r) AS src, r, endNode(r) AS tgt
    RETURN DISTINCT src.id AS head, type(r) AS rel, tgt.id AS tail
    LIMIT {qc.max_graph_facts}
    """

    try:
        rows = g.query(cypher, params)
    except Exception as e:
        LOGGER.error(f"Neo4j Query Error: {e}")
        return []

    out: List[RetrievedContext] = []
    for idx, row in enumerate(rows or []):
        h, r, t = row.get("head"), row.get("rel"), row.get("tail")
        if h and r and t:
            # Make the triplet human-readable
            rel_pretty = r.replace("_", " ").lower()
            out.append(
                RetrievedContext(
                    id=f"KG Triplet {idx}",
                    text=f"{h} {rel_pretty} {t}",
                    source_type="Knowledge Graph",
                    score=0.0,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

_RERANKER_MODEL = None


def rerank_qwen3(
    query: str, contexts: List[RetrievedContext], top_k: int
) -> List[RetrievedContext]:
    """
    Improved reranking:
    - Tries CrossEncoder if USE_HEAVY_RERANKER=true
    - Falls back to TF-IDF-style weighted Jaccard with bigram overlaps
      plus a source-type diversity bonus so Graph facts aren't starved.
    """
    if not contexts:
        return []

    global _RERANKER_MODEL
    import os

    use_heavy = os.getenv("USE_HEAVY_RERANKER", "false").lower() == "true"
    if use_heavy:
        try:
            from sentence_transformers import CrossEncoder
            if _RERANKER_MODEL is None:
                model_name = os.getenv(
                    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                LOGGER.info(f"Loading reranker: {model_name}")
                _RERANKER_MODEL = CrossEncoder(model_name)
            pairs = [[query, c.text] for c in contexts]
            scores = _RERANKER_MODEL.predict(pairs)
            ranked = [
                RetrievedContext(id=c.id, text=c.text, source_type=c.source_type, score=float(s))
                for c, s in zip(contexts, scores)
            ]
            ranked.sort(key=lambda x: x.score, reverse=True)
            return ranked[:top_k]
        except Exception as e:
            LOGGER.warning(f"CrossEncoder failed ({e}); using heuristic fallback.")

    # --- Heuristic fallback: unigram + bigram Jaccard + source bonus ---
    q_tokens = query.lower().split()
    q_uni = set(q_tokens)
    q_bi = set(zip(q_tokens, q_tokens[1:]))

    ranked: List[RetrievedContext] = []
    for ctx in contexts:
        c_tokens = ctx.text.lower().split()
        c_uni = set(c_tokens)
        c_bi = set(zip(c_tokens, c_tokens[1:]))

        uni_overlap = len(q_uni & c_uni) / max(len(q_uni | c_uni), 1)
        bi_overlap = len(q_bi & c_bi) / max(len(q_bi | c_bi), 1) if q_bi else 0.0

        # Graph triplets get a small diversity bonus so they aren't drowned out
        source_bonus = 0.05 if ctx.source_type == "Knowledge Graph" else 0.0

        heuristic = ctx.score + uni_overlap * 0.6 + bi_overlap * 0.4 + source_bonus
        ranked.append(
            RetrievedContext(id=ctx.id, text=ctx.text, source_type=ctx.source_type, score=heuristic)
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Context builder — richer formatting sent to the LLM
# ---------------------------------------------------------------------------

def build_synthesis_context(ranked: List[RetrievedContext], max_chars: int) -> str:
    """
    Format ranked evidence blocks for the synthesis LLM.
    Groups Vector Chunks and Knowledge Graph facts separately for clarity.
    """
    vec = [c for c in ranked if c.source_type == "Vector Chunk"]
    kg  = [c for c in ranked if c.source_type == "Knowledge Graph"]

    parts: List[str] = []

    if vec:
        parts.append("### 📄 Vector Evidence (semantic passages)")
        for i, ctx in enumerate(vec, 1):
            text = ctx.text[:max_chars].rstrip() + ("…" if len(ctx.text) > max_chars else "")
            parts.append(f"**[V{i}]** {text}")

    if kg:
        parts.append("\n### 🔗 Knowledge-Graph Facts (structured triplets)")
        for i, ctx in enumerate(kg, 1):
            parts.append(f"**[G{i}]** {ctx.text}")

    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Main query entry-point
# ---------------------------------------------------------------------------

def run_query(
    llm: Any,
    vstore: InMemoryVectorStore,
    question: str,
    qc: Optional[QueryConfig] = None,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    use_neo4j: bool = False,
) -> Tuple[str, List[RetrievedContext], str]:
    qc = qc or QueryConfig()

    # Pin synthesis to best model
    if hasattr(llm, "model_name") and llm.model_name != qc.synthesis_model:
        LOGGER.info(f"Switching synthesis model → {qc.synthesis_model}")
        try:
            llm = ChatGroq(model_name=qc.synthesis_model, temperature=0.2)
        except Exception:
            pass

    # 1. Vector retrieval
    vector_contexts = retrieve_chunks_bge_m3(vstore, question, qc)

    # 2. Graph retrieval (optional)
    graph_contexts: List[RetrievedContext] = []
    if use_neo4j and neo4j_url and neo4j_user is not None and neo4j_password is not None:
        seeds = _extract_seed_terms(question)
        LOGGER.info(f"Graph seed terms: {seeds}")
        graph_contexts = _fetch_graph_facts_2hop(
            neo4j_url, neo4j_user, neo4j_password, seeds, qc
        )

    # 3. Merge + rerank
    all_contexts = vector_contexts + graph_contexts
    top_contexts = rerank_qwen3(question, all_contexts, top_k=qc.top_k_rerank)

    # 4. Build structured context string
    context_str = build_synthesis_context(top_contexts, qc.max_chunk_chars_each)

    # 5. LLM synthesis with system + user messages
    prompt = _ANSWER_PROMPT.format(question=question, context=context_str)
    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        msg = llm.invoke(messages)
        answer = msg.content if hasattr(msg, "content") else str(msg)
    except Exception as e:
        answer = f"❌ Synthesis Failed: {e}"

    return answer, top_contexts, context_str
