import os
import streamlit as st
from typing import Dict, Any, List
from groq import Groq
from firecrawl import FirecrawlApp

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="AI Deep Research Agents (Groq)", page_icon="📘", layout="wide")
st.title("📘 Deep Research Agent")
st.markdown("This version uses **Groq** models (e.g., `llama-3.1-70b-versatile`) with **Firecrawl** for web research.")

# -----------------------------
# Sidebar API keys & settings
# -----------------------------
with st.sidebar:
    st.header("API Configuration")
    default_groq = os.getenv("GROQ_API_KEY", "")
    default_fire = os.getenv("FIRECRAWL_API_KEY", "")

    groq_api_key = st.text_input("Groq API Key", value=default_groq, type="password")
    firecrawl_api_key = st.text_input("Firecrawl API Key", value=default_fire, type="password")

    st.divider()
    st.subheader("Model Settings")
    model = st.selectbox(
        "Groq model",
        options=[
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ],
        index=0,
        help="Choose a Groq-hosted LLM",
    )

# -----------------------------
# Research topic input
# -----------------------------
research_topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest developments in AI safety")

# -----------------------------
# Helpers
# -----------------------------

def _require_keys() -> bool:
    if not groq_api_key:
        st.warning("Please provide your Groq API key in the sidebar.")
        return False
    if not firecrawl_api_key:
        st.warning("Please provide your Firecrawl API key in the sidebar.")
        return False
    return True


def run_deep_research(query: str, max_depth: int = 3, time_limit: int = 180, max_urls: int = 10) -> Dict[str, Any]:
    """Perform comprehensive web research using Firecrawl's deep_research endpoint."""
    try:
        app = FirecrawlApp(api_key=firecrawl_api_key)

        progress = st.empty()

        def on_activity(activity):
            # Firecrawl streams small events; render last event for brevity
            typ = activity.get("type", "event")
            msg = activity.get("message", "...")
            progress.info(f"[{typ}] {msg}")

        with st.spinner("Performing deep research via Firecrawl..."):
            results = app.deep_research(
                query=query,
                params={"maxDepth": max_depth, "timeLimit": time_limit, "maxUrls": max_urls},
                on_activity=on_activity,
            )

        data = results.get("data", {})
        return {
            "success": True,
            "final_analysis": data.get("finalAnalysis", ""),
            "sources": data.get("sources", []),
        }
    except Exception as e:
        st.error(f"Deep research error: {e}")
        return {"success": False, "error": str(e), "final_analysis": "", "sources": []}


def _format_sources_for_prompt(sources: List[Dict[str, Any]] = None, limit: int = 12) -> str:
    if not sources:
        return "No sources."
    lines = []
    for i, s in enumerate(sources[:limit], start=1):
        url = s.get("url") or s.get("link") or ""
        title = s.get("title") or s.get("name") or "Untitled"
        summary = s.get("summary") or s.get("content") or ""
        summary = summary.strip()
        if len(summary) > 400:
            summary = summary[:400] + "…"
        lines.append(f"{i}. {title}\n   URL: {url}\n   Summary: {summary}")
    return "\n\n".join(lines)


def groq_chat(system: str, user: str, temperature: float = 0.2) -> str:
    client = Groq(api_key=groq_api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


# -----------------------------
# Main flow
# -----------------------------
if st.button("Start Research", disabled=not research_topic):
    if not _require_keys():
        st.stop()

    # Step 1: Firecrawl deep research
    research = run_deep_research(research_topic, max_depth=3, time_limit=180, max_urls=10)
    if not research.get("success"):
        st.stop()

    sources = research.get("sources", [])
    formatted_sources = _format_sources_for_prompt(sources)

    st.subheader("Sources (from Firecrawl)")
    with st.expander("Show sources"):
        st.text(formatted_sources)

    # Step 2: Initial report with Groq
    st.subheader("Initial Research Report")
    with st.spinner("Generating initial report with Groq..."):
        system_prompt = (
            "You are a careful research assistant. Write a concise, well-structured report "
            "with headings, bullet points, and short paragraphs. Include citations as numbered "
            "[n] markers mapped to the provided source list. If something is uncertain, state so."
        )
        user_prompt = (
            f"TOPIC: {research_topic}\n\n"
            f"SOURCES:\n{formatted_sources}\n\n"
            "Write a report (600–1000 words) covering: key points, context/background, current developments, "
            "risks/limitations, and a short FAQ. Use [n] citation markers that refer to the numbered sources above."
        )
        initial_report = groq_chat(system_prompt, user_prompt, temperature=0.2)
    st.markdown(initial_report)

    # Step 3: Elaboration pass with Groq
    st.subheader("Enhanced Research Report")
    with st.spinner("Enhancing the report with Groq..."):
        system_prompt2 = (
            "You are an expert content enhancer. Improve clarity, add examples, brief case studies, "
            "and practical takeaways for different stakeholders. Keep original structure but expand with useful detail. "
            "Do not invent citations; keep the same [n] mapping."
        )
        user_prompt2 = (
            "Enhance the following report. Preserve headings and citation markers.\n\n" + initial_report
        )
        enhanced_report = groq_chat(system_prompt2, user_prompt2, temperature=0.3)
    st.markdown(enhanced_report)

    # Step 4: Download
    st.download_button(
        label="Download Enhanced Report (Markdown)",
        data=enhanced_report,
        file_name=f"{research_topic.replace(' ', '_')}_report.md",
        mime="text/markdown",
    )

st.markdown("---")
st.caption("Powered by Groq + Firecrawl · Streamlit UI")
