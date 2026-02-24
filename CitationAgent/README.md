# 📝 Highlight & Cite: AI-Powered Academic Citation Agent

An intelligent, multi-tool AI agent built with Python and Streamlit that automatically generates accurate academic citations from highlighted text. Instead of manually searching for references, users can paste a sentence from their draft, and the agent will intelligently extract the core scientific concepts, search the Semantic Scholar academic graph, and return perfectly formatted citations with clickable links.

## ✨ Features
* **Semantic Search:** Utilizes the free Semantic Scholar API to find papers based on natural language claims rather than rigid keyword matching.
* **AI Orchestration:** Powered by a LangChain ReAct (Reasoning + Acting) Agent that autonomously decides how to search for quotes and formats the results.
* **LLM Integration:** Uses the blazing-fast Llama 3 model via the Groq API for rapid inference and keyword extraction.
* **Multi-Format Support:** Instantly formats bibliographies into APA (7th ed.), MLA (9th ed.), Harvard, Chicago, IEEE, and Vancouver styles.
* **Built-in Rate Limit Handling:** Includes custom exponential backoff and retry logic to gracefully handle public API traffic limits.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **AI Framework:** LangChain
* **LLM Provider:** Groq (Llama-3.1-8b)
* **Data Source:** Semantic Scholar Academic Graph API
* **Languages:** Python 3.x

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME 
2. Install the required dependencies
   ```bash
    pip install -r requirements.txt
   
3. Run the application:
     ```bash
      streamlit run app.py
4.Using the App:

Open the local network URL provided in your terminal (usually http://localhost:8501).

Enter a free Groq API key (get one at console.groq.com).

Paste your draft, copy a specific sentence into the target box, select your format, and hit search!

🧠 How It Works Under the Hood
This application does not rely on simple string-matching. When a user submits a sentence, the ReAct agent evaluates it:

If the LLM recognizes the quote from its training data, it searches the API for the exact paper title.

If it is an obscure quote, the LLM extracts a unique 4-to-5 word identifier phrase to bypass API confusion.

The tool fetches the top 5 papers, grabs their unique URLs, and passes the raw JSON metadata back to the LLM.

The LLM strictly formats the final output and injects markdown hyperlinks for easy access to the original source.
