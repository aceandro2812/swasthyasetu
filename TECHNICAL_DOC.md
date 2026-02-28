# SwasthyaSetu — Technical Documentation

## Overview
SwasthyaSetu is a modular, agentic AI medical triage and routing assistant. It is designed for extensibility, transparency, and real-world deployment in resource-limited settings. The backend is built with FastAPI and LangGraph, orchestrating a multi-agent workflow powered by LLMs and RAG.

---

## Agent Workflow (LangGraph)

### Agent Nodes
- **Diagnostician**: Generates a differential diagnosis from symptoms and RAG context.
- **Triage Agent**: Classifies urgency (emergency/urgent/routine) and suggests next steps.
- **Routing Agent**: Uses diagnosis, triage, and user location to find local specialists (DuckDuckGo search, can be swapped for real APIs).
- **Validator**: Checks diagnosis against medical guidelines.
- **Educator**: Generates patient-friendly explanations and next steps.
- **Bias Checker**: Analyzes for bias and equity (optional output).
- **Output Formatter**: Assembles the final report.

### State Management
- The agent state is a Python `TypedDict`.
- Each node function is pure: receives and returns the state dict.
- Add new fields to the state as needed for new agents.

### Logging & Observability
- All major agent node entries/exits, errors, and LLM calls are logged.
- Logs are INFO/ERROR level and can be extended for analytics or monitoring.

---

## Production Hardening (Latest)

### 1) Hard Rate Limiting
- `/diagnose` is now protected with strict hard limits:
1. Per-IP rolling limits: per-minute and per-hour
2. Global rolling limits: per-minute and per-hour
- Default guardrail values (env-overridable):
1. `DIAGNOSE_PER_IP_PER_MINUTE=4`
2. `DIAGNOSE_PER_IP_PER_HOUR=30`
3. `DIAGNOSE_GLOBAL_PER_MINUTE=20`
4. `DIAGNOSE_GLOBAL_PER_HOUR=240`
- On breach, API returns `429` plus:
1. `Retry-After`
2. `X-RateLimit-Limit-*` and `X-RateLimit-Remaining-*`
3. JSON payload with `error_code`, `retry_after_seconds`, and `limits`

### 2) Frontend Rate-Limit UX
- UI now handles `429` explicitly and displays a dedicated rate-limit notice.
- Message includes wait duration and the active hard-limit values.

### 3) XSS and Output Safety
- Dynamic report content is sanitized before DOM insertion.
- External links are URL-sanitized to `http/https` and include `rel="noopener noreferrer"`.

### 4) API Reliability + Error Hygiene
- LangGraph execution is offloaded with `run_in_threadpool` inside the async route to prevent event-loop blocking.
- API no longer exposes raw backend exceptions to clients; returns generic failure plus `error_id`.

### 5) CORS and Runtime Dependencies
- CORS remains wildcard-origin but with `allow_credentials=False`.
- Added missing runtime dependencies required by routing:
1. `requests`
2. `beautifulsoup4`

---

## Extending SwasthyaSetu

### Adding New Agents
1. Define a new node function in `main.py`.
2. Add the node to the LangGraph workflow and connect edges.
3. Add any new state fields to the `AgentState` TypedDict.

### Swapping LLMs
- Replace Gemini with OpenAI, local LLMs, or others by updating the LLM call logic.
- Ensure prompt and output parsing are robust to LLM output variations.

### Improving RAG
- Add more or better medical documents to the vector store.
- Tune similarity search parameters for better context.

### Integrating Real Provider APIs
- Replace DuckDuckGo search with real hospital/doctor APIs for direct routing.
- Add authentication and privacy controls as needed.

### UI/UX Customization
- The frontend is decoupled and can be replaced or themed as needed.
- Add speech input, mobile/PWA support, or localization for broader reach.

---

## Advanced Agentic LLM Development

- **Branching/Streaming**: LangGraph supports branching, streaming, and more complex agentic flows.
- **Stateful Agents**: You can add memory, user history, or context as needed.
- **Observability**: Add more granular logging, metrics, or tracing for production.
- **Testing**: Write unit tests for each agent node as pure functions.

---

## Further Scope & Roadmap

- **Doctor Collaboration**: Allow doctors to register and receive direct case routing.
- **Case History**: Store and retrieve past reports for users (with privacy controls).
- **WhatsApp/SMS Integration**: For rural/low-tech accessibility.
- **Regulatory Compliance**: Add disclaimers, privacy, and consent flows as needed for deployment.
- **Multi-language Support**: For rural/global deployment.
- **Analytics Dashboard**: For monitoring usage, errors, and outcomes.

---

## Contributing
- Please open issues or PRs for bugs, features, or improvements.
- For architectural questions, see the code comments and logging in `main.py`.
- For agentic LLM design, see the LangGraph documentation and the node function patterns in this repo.

---

## Contact
- Maintainers: aceandro2812

