import streamlit as st
import os
import json
import re
from dotenv import load_dotenv

from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("📝 Interactive AI Quiz Generator")
st.write("Upload a PDF, generate a quiz, and test your knowledge directly on the page!")

api_key = st.text_input("Enter your Groq API key", type="password")

# --- NEW: Helper function to clean text before comparing ---
def clean_text(text):
    if not text:
        return ""
    # Removes leading letters like "A.", "B)", "C -", etc., and ignores case/spaces
    cleaned = re.sub(r'^[A-Da-d][\.\)\-]\s*', '', str(text))
    return cleaned.strip().lower()

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    st.subheader("Quiz Settings")
    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.selectbox("Number of Questions", options=[5, 10, 20])
    with col2:
        difficulty = st.selectbox("Difficulty Level", options=["Easy", "Medium", "Hard"])
    
    st.divider()

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp_{uploaded_file.name}"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user instruction, "
            "formulate a standalone instruction which can be understood "
            "without the chat history. Do not answer it, "
            "just reformulate it if needed and otherwise return it as is."
        )    
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
            
        # --- UPDATED JSON SYSTEM PROMPT ---
        system_prompt = (
            f"You are an expert educational assistant. Your task is to generate a multiple-choice quiz based on the provided text.\n"
            f"Generate exactly {num_questions} questions at a '{difficulty}' difficulty level.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You MUST output the quiz strictly as a JSON array of objects.\n"
            "2. Do NOT include any introductory text or markdown formatting outside of the JSON block.\n"
            "3. Do NOT use letter prefixes (like A., B., C., D.) in the options or the answer. Provide JUST the text of the choice.\n\n"
            "Format your response exactly like this example:\n"
            "[\n"
            "  {{\n"
            '    "question": "What is the capital of France?",\n'
            '    "options": ["London", "Berlin", "Paris", "Madrid"],\n'
            '    "answer": "Paris"\n'
            "  }}\n"
            "]\n\n"
            "Context:\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        if st.button("Generate My Quiz!"):
            with st.spinner("Analyzing document and formatting your interactive quiz..."):
                session_history = get_session_history(session_id)
                automated_query = f"Generate the {difficulty} {num_questions}-question JSON quiz."
                
                response = conversational_rag_chain.invoke(
                    {"input": automated_query},
                    config={"configurable": {"session_id": session_id}},
                )
                
                raw_text = response['answer']
                match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        quiz_data = json.loads(json_str)
                        st.session_state.quiz_data = quiz_data
                        st.session_state.quiz_submitted = False 
                    except json.JSONDecodeError:
                        st.error("Failed to parse the AI's response. Please try generating again.")
                else:
                    st.error("The AI did not format the output correctly. Please try again.")

        if "quiz_data" in st.session_state:
            st.markdown("### Your Interactive Quiz")
            
            with st.form("quiz_form"):
                user_answers = {}
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**{i+1}. {q['question']}**")
                    user_answers[i] = st.radio(
                        "Select an answer:", 
                        q['options'], 
                        key=f"radio_{i}", 
                        index=None,
                        label_visibility="collapsed"
                    )
                    st.write("") 
                
                submitted = st.form_submit_button("Submit Answers")
                
                if submitted:
                    st.session_state.user_answers = user_answers
                    st.session_state.quiz_submitted = True

            if st.session_state.get("quiz_submitted"):
                st.markdown("---")
                st.subheader("Quiz Results")
                score = 0
                
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**{i+1}. {q['question']}**")
                    user_ans = st.session_state.user_answers[i]
                    correct_ans = q['answer']
                    
                    # --- NEW: Use the cleaning function before checking ---
                    clean_user = clean_text(user_ans)
                    clean_correct = clean_text(correct_ans)
                    
                    if clean_user == clean_correct and clean_user != "":
                        st.success(f"Your answer: {user_ans} ✅")
                        score += 1
                    elif user_ans is None:
                        st.warning("You left this question blank. ❌")
                        st.info(f"Correct answer: {correct_ans}")
                    else:
                        st.error(f"Your answer: {user_ans} ❌")
                        st.info(f"Correct answer: {correct_ans}")
                
                st.metric(label="Final Score", value=f"{score} / {len(st.session_state.quiz_data)}")

else:
    st.warning("Please enter a valid API key")