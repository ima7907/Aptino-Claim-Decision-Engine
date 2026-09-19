# AI-Assisted Claim Decision Engine

An AI-assisted health insurance claim analysis and decision-support system developed for the Aptino AI Engineer take-home assignment.

The system combines policy-document retrieval, hybrid search, deterministic reranking, medical-document analysis, specialized agents, structured workflow state, validation, citations, and abstention when the available evidence is insufficient.

---

## Project Overview

The Claim Decision Engine analyzes structured insurance claim information together with medical-document text and retrieves relevant evidence from the insurance policy.

The system is designed to:

- Retrieve relevant policy evidence
- Combine sparse and dense retrieval
- Apply deterministic reranking before decision reasoning
- Analyze medical-document text
- Detect information requiring human verification
- Produce a structured claim decision
- Cite retrieved policy evidence
- Identify missing evidence
- Abstain with `NEEDS_REVIEW` when evidence is insufficient
- Validate the generated decision
- Record an execution trace for the multi-agent workflow

The system is intended as an AI-assisted decision-support tool and does not replace human claim review.

---

## Architecture

```text
                         Claim Input
                             |
                             v
                    FastAPI / Streamlit
                             |
                             v
                  Claim Decision Workflow
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
 MedicalDocumentAgent  PolicyAnalysisAgent  DecisionAgent
          |                  |                  |
          |                  v                  |
          |            Hybrid Retriever         |
          |                  |                  |
          |        +---------+---------+         |
          |        |         |         |         |
          |     Keyword     BM25    Dense        |
          |     Retrieval  Retrieval Retrieval   |
          |        |         |         |          |
          |        +---------+---------+          |
          |                  |                    |
          |                  v                    |
          |             RRF Fusion                |
          |                  |                    |
          |                  v                    |
          |              Reranking                |
          |                  |                    |
          |                  v                    |
          |            Policy Evidence            |
          |                  |                    |
          +------------------+--------------------+
                             |
                             v
                       Decision Logic
                             |
                             v
                     ValidationAgent
                             |
                             v
                    Structured Response
                             |
          +------------------+------------------+
          |                  |                  |
       Decision           Evidence           Trace
       Findings           Citations        Validation
       Limitations        Missing Evidence