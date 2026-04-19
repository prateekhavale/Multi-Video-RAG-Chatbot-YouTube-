import re
import uuid
import os
import requests
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder



# EXTRACT VIDEO ID

def extract_video_id(url: str) -> str:

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]

    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid Youtube URL")

# EXTRACT VIDEO TRANSCRIPT AND NORMALIZE

def extract_transcript(video_id):

    api = YouTubeTranscriptApi()
    
    try:
        transcript = api.fetch(video_id, languages=["en"])

    except:
        transcript_list = api.list(video_id)

        transcript = transcript_list.find_transcript(
            [t.language_code for t in transcript_list]
        ).fetch()

    if not transcript:
        return None

    return [
        {
        "text": t.text,
        "start": t.start,
        "end": t.start + t.duration
        }
            for t in transcript
    ]

# SEMANTIC CHUNKING

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_chunking(segments, threshold=0.7):
    chunks = []

    texts = [s["text"] for s in segments]
    embeddings = embed_model.encode(texts)

    current = [segments[0]]

    for i in range(1, len(segments)):
        sim = util.cos_sim(
            embeddings[i - 1],
            embeddings[i]
        ).item()

        if sim > threshold:
            current.append(segments[i])
        else:
            chunks.append(current)
            current = [segments[i]]

    chunks.append(current)
    return chunks

# CREATE NODES

def create_nodes(chunks, video_id):
    nodes = []

    for chunk in chunks:
        text = " ".join([s["text"] for s in chunk])

        nodes.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "video_id": video_id
        })

    return nodes

# BULD GRAPH

def build_graph(nodes, threshold=0.75):
    graph = {n["id"]: [] for n in nodes}

    embeddings = embed_model.encode([n["text"] for n in nodes])

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()

            if sim > threshold:
                graph[nodes[i]["id"]].append(nodes[j]["id"])
                graph[nodes[j]["id"]].append(nodes[i]["id"])

    return graph

# VECTOR STORE

def build_vector_store(nodes):
    docs = [Document(page_content=n["text"], metadata=n) for n in nodes]
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, emb)

# BM25

def build_bm25(nodes):
    corpus = [n["text"].split() for n in nodes]
    return BM25Okapi(corpus), corpus

# HYBRID RETRIEVAL

def hybrid_retrieval(query, vector_store, bm25, nodes, k=5):

    vector_docs = vector_store.similarity_search_with_score(query, k=k)
    results = {
        doc.metadata["id"]: {
            "node": doc.metadata,
            "sem": float(score),
            "bm": 0.0
        }
        for doc, score in vector_docs
    }

    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    for i in top_idx:
        node = nodes[i]
        nid = node["id"]

        if nid not in results:
            results[nid] = {"node": node, "sem": 0.0, "bm": 0.0}

        results[nid]["bm"] = scores[i]

    ranked = sorted(
        results.values(),
        key=lambda x: 0.7 * x["sem"] + 0.3 * x["bm"],
        reverse=True
    )

    return [r["node"] for r in ranked[:k]]

# RERANKER

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, nodes, top_k=5):
    pairs = [(query, n["text"]) for n in nodes]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)

    return [n for n, _ in ranked[:top_k]]

# GRAPH EXPANSION

def expand_with_graph(retrieved, graph, nodes, max_expand=5):
    node_map = {n["id"]: n for n in nodes}
    expanded = {n["id"]: n for n in retrieved}

    for node in retrieved:
        for nid in graph.get(node["id"], [])[:max_expand]:
            if nid not in expanded:
                expanded[nid] = node_map[nid]

    return list(expanded.values())

# CONTEXT BUILDER

def build_context(nodes, max_chars=1800):
    context = ""
    for i, n in enumerate(nodes):
        block = (
            f"[SOURCE {i}]\n"
            f"Video: {n['video_id']}\n"
            f"Time: {n['start']} - {n['end']}\n"
            f"{n['text']}\n\n"
        )
        if len(context) + len(block) > max_chars:
            break
        context += block
    return context

# CHAT HISTORY

def format_chat_history(chat_history, max_turns=5):

    history = chat_history[-max_turns:]

    formatted = []
    for turn in history:
        role = turn["role"]
        content = turn["content"].strip()

        if content:
            formatted.append(f"{role.upper()}: {content}")

    return "\n".join(formatted)


# PROMPT

def generate_answer(query, context, ask_llm, chat_history):

    history = format_chat_history(chat_history)

    prompt = f"""
You are a highly reliable, fact-grounded AI assistant.

Your task is to answer the user's question using ONLY the provided CONTEXT and CONVERSATION HISTORY.

==============================
CONVERSATION HISTORY:
{history}
==============================

==============================
CONTEXT (SOURCE OF TRUTH):
{context}
==============================

USER QUESTION:
{query}

==============================
CORE INSTRUCTIONS:

1. You MUST use ONLY the information present in the CONTEXT.
2. You MUST NOT use any external knowledge.
3. You MUST NOT assume, infer, or fabricate missing details.

------------------------------
ANSWERING LOGIC:

A. If sufficient information is available:
→ Provide a clear, accurate, and complete answer.

B. If partially available:
→ Provide the best possible answer using available information.
→ Clearly indicate that the answer is partially based on the provided context.

C. If no relevant information is available:
→ Respond EXACTLY:
"The answer is not available in the provided context."

------------------------------
STRICT RULES:

4. Every statement MUST be grounded in the CONTEXT.
5. Do NOT hallucinate, guess, or add unsupported information.
6. Do NOT introduce new facts not present in CONTEXT.
7. Ensure consistency when combining multiple pieces of information.

------------------------------
OUTPUT REQUIREMENTS:

8. Provide a clear, slightly detailed explanation to improve understanding.
9. Keep the response:
   - Structured
   - Concise
   - Easy to read

10. Do NOT include:
   - Citation markers (e.g., [SOURCE X])
   - "Sources Used" section
   - Any metadata (video id, timestamps, etc.)

11. Do NOT mention the CONTEXT explicitly in the answer.

------------------------------
FINAL CHECK (MANDATORY):

Before answering, internally verify:
- Is every part of the answer supported by CONTEXT?
- Are there any assumptions or gaps?

If YES → return answer  
If NO → follow fallback rules

==============================
Now generate the answer strictly following all instructions.
"""

    return ask_llm(prompt)


# ASK

def ask(query, data, ask_llm, chat_history=[]):

    retrieved = hybrid_retrieval(
        query,
        data["vector_store"],
        data["bm25"],
        data["nodes"]
    )

    reranked = rerank(query, retrieved)


    expanded = expand_with_graph(
        reranked,
        data["graph"],
        data["nodes"]
    )

    context = build_context(expanded)

    return generate_answer(query, context, ask_llm, chat_history)

# VIDEO COMPARE

def compare_videos(query, video_store, llm_fn):
    context = ""

    for vid, data in video_store.items():
        retrieved = hybrid_retrieval(
            query,
            data["vector_store"],
            data["bm25"],
            data["nodes"]
        )

        for r in retrieved:
            text = r["text"] if isinstance(r, dict) else r.page_content
            context += f"[VIDEO {vid}]\n{text}\n\n"

    prompt = f"""
You are an AI assistant specialized in comparing information across multiple videos.

Your task is to compare the videos using ONLY the provided CONTEXT.

==============================
CONTEXT:
{context}
==============================

USER QUESTION:
{query}

==============================
INSTRUCTIONS:

1. Use ONLY the information present in the CONTEXT.
2. Do NOT use any external knowledge.
3. Do NOT assume or infer beyond what is explicitly stated.

4. Identify:
   - Key similarities between videos
   - Key differences between videos
   - Unique points specific to each video

5. If information is incomplete:
   - Compare based on available content
   - Clearly indicate limitations

6. Do NOT mention:
   - Video IDs
   - Source labels
   - Any metadata

7. Keep the comparison:
   - Clear
   - Structured
   - Easy to understand

==============================
OUTPUT FORMAT:

Comparison:

Similarities:
- ...

Differences:
- ...

Unique Insights:
- Video A: ...
- Video B: ...

Conclusion:
<short summary based on comparison>

==============================
IMPORTANT:
- Do NOT include any citations
- Do NOT include "Sources Used"
- Do NOT mention the context explicitly
- Stay strictly grounded in the provided information

Now generate the comparison.
"""
    return llm_fn(prompt)

# ASK LLM

API_URL = "https://openrouter.ai/api/v1/chat/completions"

def ask_llm(prompt):
    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": os.getenv("MODEL_NAME", "openai/gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
        )

        if response.status_code != 200:
            return f"LLM API Error: {response.text}"

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"LLM Request Failed: {str(e)}"

# INTENT

def detect_intent(query):
    prompt = f"""
Classify the user input into one of:
- QUESTION
- CASUAL

Input: {query}

Output ONLY one word.
"""
    return ask_llm(prompt)

# CLEAN OUTPUT

def clean_output(text):

    if not text:
        return "No response generated."

    text = re.sub(r"\[\s*SOURCE\s*\d+\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Sources Used:.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*Answer:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()

# CACHE HANDLING IF SAME VIDEO ASKED TWICE

@st.cache_resource
def build_video_pipeline(url):
    vid = extract_video_id(url)

    transcript = extract_transcript(vid)
    chunks = semantic_chunking(transcript)

    nodes = create_nodes(chunks, vid)
    graph = build_graph(nodes)

    vector_store = build_vector_store(nodes)
    bm25, corpus = build_bm25(nodes)

    return {
        "video_id": vid,
        "vector_store": vector_store,
        "bm25": bm25,
        "corpus": corpus,
        "nodes": nodes,
        "graph": graph
    }