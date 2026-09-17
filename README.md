# AI-Assisted Claim Decision Engine

An AI-assisted insurance claim analysis and decision-support system built for the Aptino AI Engineer take-home assignment.

## Project Overview

This project analyzes insurance claim cases using:

- Policy document retrieval
- Keyword retrieval
- BM25 retrieval
- Dense vector retrieval
- Hybrid retrieval
- Policy analysis
- Claim decision logic
- FastAPI backend
- Streamlit frontend

The system retrieves relevant policy evidence and produces a claim decision with an explanation.

## Architecture

```text
Claim Input
    |
    v
FastAPI / Streamlit
    |
    v
Policy Analysis Agent
    |
    v
Hybrid Retriever
    |
    +--> Keyword Retrieval
    +--> BM25 Retrieval
    +--> Dense Retrieval
    |
    v
Relevant Policy Evidence
    |
    v
Decision Agent
    |
    v
Claim Decision