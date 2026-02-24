import streamlit as st
import os
import requests
import re
from langchain_groq import ChatGroq
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.tools import tool

# 1. Define the Semantic Scholar Sentence-to-Citation Tool
@tool
def search_semantic_scholar(query: str) -> str:
    """
    Searches Semantic Scholar for papers related to a specific claim, sentence, or concept.
    You can pass the exact natural language sentence directly to this tool.
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        params = {
            "query": query,
            "limit": 5,
            "fields": "title,authors,year,url"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data or not data["data"]:
            return "Error: No matching papers found for this claim."

        formatted_results = []
        for paper in data["data"]:
            title = paper.get("title", "Unknown Title")
            authors_list = paper.get("authors", [])
            authors = [a.get("name", "") for a in authors_list]
            year = paper.get("year", "Unknown Year")
            paper_url = paper.get("url", "")
            
            formatted_results.append(f"Title: {title} | Authors: {authors} | Year: {year} | URL: {paper_url}")
            
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error finding metadata: {str(e)}"

# 2. Streamlit UI Setup
st.set_page_config(page_title="Text-to-Citation Agent", layout="wide")
st.title("📝 Highlight & Cite Agent")

api_key = st.text_input("Enter your Groq API key", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
    
    tools = [search_semantic_scholar]

    # Create the Agent without persistent memory
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=None, 
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": """You are an expert academic citation assistant. 
            When the user gives you a sentence to cite, follow these exact steps:
            1. Use your own vast knowledge to identify the quote. If you know what paper it is from, use the search_semantic_scholar tool to search for the EXACT PAPER TITLE (e.g., "Attention Is All You Need").
            2. If you do not recognize the quote, extract a highly unique 4-to-5 word phrase from the sentence and search for that instead.
            3. Take the top results and format them strictly into the requested citation style.
            4. You MUST make the title of the paper a clickable markdown link. Put the ACTUAL name of the paper inside the brackets (e.g., [Attention Is All You Need](https://url-goes-here.com)). Do not literally write 'Paper Title'.
            
            Do not apologize or explain your search process. Just output the citations."""
        }
    )

    # 3. The Split-Screen Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Your Paper")
        st.text_area("Paste your rough draft here to read from:", height=400, placeholder="Machine learning models have shown great promise in predicting... ")

    with col2:
        st.subheader("2. Citation Search")
        
        citation_format = st.selectbox("Select Citation Format", [
            "APA (7th edition)", 
            "MLA (9th edition)", 
            "Harvard", 
            "Chicago", 
            "IEEE", 
            "Vancouver"
        ])
        
        target_sentence = st.text_area("Paste the specific sentence you want to cite:", placeholder="e.g., Attention mechanisms allow models to focus on specific parts of the input sequence.")
        
        if st.button("Find Top 5 Citations"):
            if target_sentence:
                with st.spinner("Analyzing sentence and searching Semantic Scholar..."):
                    prompt = f"Find 5 citations for this sentence: '{target_sentence}'. Format them in {citation_format} style."
                    
                    try:
                        response = agent_executor.invoke({"input": prompt, "chat_history": []})
                        st.success("Citations Found!")
                        st.markdown(response["output"])
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please paste a sentence first.")
else:
    st.warning("Please enter a valid Groq API key.")