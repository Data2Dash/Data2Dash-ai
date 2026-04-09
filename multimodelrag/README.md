# Multimodal RAG Module

## Overview
This module provides a Streamlit-based Multimodal RAG assistant for document understanding and grounded question answering.

It supports:
- PDF upload and processing
- equation, table, and figure extraction
- document-grounded Q&A
- optional image upload for vision-based questions
- pasted text as an alternative input
- response validation to reduce hallucinations

## Main Entry Points
- `app_enhanced.py` → main Streamlit application
- `enhanced_rag_system.py` → core backend orchestration

## Core Components
- `pdf_processor.py` → PDF extraction pipeline
- `specialized_chunker.py` → specialized chunking for text, equations, tables, and figures
- `vector_store.py` → FAISS + lexical fallback retrieval
- `smart_retriever.py` → query routing and retrieval logic
- `self_rag_validator.py` → grounding and response validation
- `advanced_formatter.py` / `response_formatter.py` → output formatting
- `models.py` → shared data models
- `config.py` / `enhanced_rag_config.py` → configuration

## Features
- process academic/research PDFs
- extract equations, tables, and figures
- retrieve relevant chunks for user questions
- generate grounded answers with citations/page references
- support multimodal interaction through uploaded images
- improve reliability through validation and formatting layers

## Installation
Install dependencies first:

```bash
pip install -r requirements.txt