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

