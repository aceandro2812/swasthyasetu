# SwasthyaSetu

**SwasthyaSetu** ("Bridge to Health") is an open-source, AI-powered medical triage and routing assistant designed for accessibility and social impact. It helps users describe their symptoms, receive a preliminary diagnosis, get triaged for urgency, and find local healthcare providers—all with a modern, user-friendly interface.

## Features

- **Multi-agent AI workflow** (LangGraph, Gemini LLM):
  - Differential diagnosis
  - Triage (emergency/urgent/routine)
  - Patient education
  - Bias and equity check (optional, for advanced users)
  - Local doctor/specialist routing (DuckDuckGo search)
- **Modern, accessible UI** (Tailwind CSS, HTML, JS)
- **Location-aware recommendations**
- **Download/print reports**
- **Advanced details** (bias/debug info) hidden by default
- **Granular backend logging for observability**

## Quickstart

1. **Clone the repo:**
   ```sh
   git clone https://github.com/your-org/swasthyasetu.git
   cd swasthyasetu
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment:**
   - Copy `.env.example` to `.env` and add your Google Gemini API key.
4. **Run the app:**
   ```sh
   uvicorn main:app --reload
   ```
5. **Open in browser:**
   - Go to [http://localhost:8000](http://localhost:8000)

## Project Structure

- `main.py` — FastAPI backend, agent workflow, logging
- `templates/index.html` — Main UI (Tailwind CSS)
- `static/app.js` — Frontend logic
- `requirements.txt` — Python dependencies
- `medsarathi.ipynb` — Original notebook (for reference)


---

# Technical Documentation & Developer Guide

## Agent Workflow (LangGraph)

The backend orchestrates a multi-agent workflow using LangGraph:

1. **Diagnostician** — Generates a differential diagnosis from symptoms and RAG context.
2. **Triage Agent** — Classifies urgency (emergency/urgent/routine) and suggests next steps.
3. **Routing Agent** — Uses diagnosis, triage, and user location to find local specialists (DuckDuckGo search).
4. **Validator** — Checks diagnosis against medical guidelines.
5. **Educator** — Generates patient-friendly explanations and next steps.
6. **Bias Checker** — Analyzes for bias and equity (optional output).
7. **Output Formatter** — Assembles the final report.

All agent nodes are fully logged for observability.

## Extending the System

- **Add new agents:**
  - Define a new node function in `main.py`.
  - Add it to the LangGraph workflow and connect edges.
- **Swap LLMs:**
  - Replace Gemini with any LLM (OpenAI, local, etc.) by updating the LLM call logic.
- **Improve RAG:**
  - Add more/better medical documents to the vector store.
- **Integrate with real provider APIs:**
  - Replace DuckDuckGo search with real hospital/doctor APIs for direct routing.
- **Add speech input:**
  - Integrate browser speech-to-text in the frontend.
- **Mobile/PWA support:**
  - Enhance UI for offline/low-bandwidth use.
- **Localization:**
  - Add multi-language support for rural/global deployment.

## Advanced/Agentic LLM Development

- **Agent state is a TypedDict** — Add new fields as needed for new agent nodes.
- **Each agent node is a pure function** — Receives and returns the state dict.
- **LangGraph** — Flexible for branching, streaming, or more complex agentic flows.
- **Observability** — All major steps are logged; add more as needed for debugging or analytics.

## Further Scope

- **Doctor collaboration:**
  - Allow doctors to register and receive direct case routing.
- **Case history:**
  - Store and retrieve past reports for users (with privacy controls).
- **WhatsApp/SMS integration:**
  - For rural/low-tech accessibility.
- **Regulatory compliance:**
  - Add disclaimers, privacy, and consent flows as needed for deployment.

## Contact & Community

- Issues and PRs welcome!
- For questions, open an issue or contact the maintainers.
