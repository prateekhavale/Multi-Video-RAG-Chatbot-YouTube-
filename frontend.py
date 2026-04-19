import streamlit as st
import re
from dotenv import load_dotenv
load_dotenv()

from backend import (
    ask,
    ask_llm,
    clean_output,
    build_video_pipeline,
    compare_videos
)


st.set_page_config(page_title="Video RAG Chatbot", layout="centered")
st.title("💬 Video RAG Chatbot")


if "videos" not in st.session_state:
    st.session_state.videos = {}

if "chat" not in st.session_state:
    st.session_state.chat = []

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None


def extract_youtube_url(text):
    patterns = [
        r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+)",
        r"(https?://youtu\.be/[\w-]+)"
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            return match.group(0)
    return None


def remove_url(text, url):
    return text.replace(url, "").strip() if url else text


def is_general_chat(query):
    query = query.lower().strip()
    casual = [
        "thanks", "thank you", "ok", "okay",
        "hi", "hello", "bye", "cool", "nice"
    ]
    return any(c in query for c in casual) or len(query.split()) <= 2


def is_compare_query(query):
    query = query.lower()
    keywords = ["compare", "difference", "vs", "contrast"]
    return any(k in query for k in keywords)


def detect_video_reference(query, video_ids):
    """
    Optional smart switching:
    If user mentions video id fragment → switch context
    """
    for vid in video_ids:
        if vid.lower()[:5] in query.lower():
            return vid
    return None


for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Paste video link or ask anything...")

if query:

    st.chat_message("user").markdown(query)

    video_url = extract_youtube_url(query)
    cleaned_query = remove_url(query, video_url)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            answer = ""

            
            if video_url:
                try:
                    with st.spinner("Processing video... This may take a few seconds..."):
    
                        data = build_video_pipeline(video_url)

                        st.session_state.videos[data["video_id"]] = data

                        st.session_state.current_video_id = data["video_id"]

                except Exception as e:
                    answer = "Could not process this video. It may not have captions or is restricted."
                    st.markdown(answer)

                # If only link
                if not cleaned_query:
                    answer = "Video added. Ask your question."
                    st.markdown(answer)

                else:
                    query = cleaned_query

            if not answer:

                if is_general_chat(query):
                    answer = ask_llm(query)

                elif not st.session_state.videos:
                    answer = ask_llm(query)

                else:
                    if is_compare_query(query) and len(st.session_state.videos) > 1:

                        raw = compare_videos(
                            query,
                            st.session_state.videos,
                            ask_llm
                        )
                        answer = clean_output(raw)

                    else:
                        detected_vid = detect_video_reference(
                            query,
                            st.session_state.videos.keys()
                        )

                        if detected_vid:
                            st.session_state.current_video_id = detected_vid

                        current_id = st.session_state.current_video_id

                        if not current_id:
                            answer = "Please add a video first."
                        else:
                            data = st.session_state.videos[current_id]

                            raw = ask(
                                        query,
                                        data,
                                        ask_llm,
                                        st.session_state.chat
                                    )
                            answer = clean_output(raw)

                st.markdown(answer)

    st.session_state.chat.append({"role": "user", "content": query})
    st.session_state.chat.append({"role": "assistant", "content": answer})

if st.session_state.videos:
    st.caption(
        f"{len(st.session_state.videos)} video(s) loaded | Active: {st.session_state.current_video_id}"
    )

if st.button("Clear Chat"):
    st.session_state.chat = []
    st.session_state.videos = {}
    st.session_state.current_video_id = None
    st.rerun()