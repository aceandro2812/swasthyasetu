<div align="center">

<!-- Animated Hero Banner -->
<img src="https://capsule-render.vercel.app/api?type=venom&height=300&color=gradient&customColorList=0,2,2,5,30&text=SwasthyaSetu&fontSize=70&fontColor=fff&animation=twinkling&stroke=2e7d32&strokeWidth=2" />

<!-- Animated Tagline -->
<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Poppins&weight=600&size=24&duration=3000&pause=800&color=2E7D32&center=true&vCenter=true&width=700&lines=🌉+Bridge+to+Health;🤖+AI-Powered+Medical+Triage;🌍+Accessible+Healthcare+for+All;💙+Social+Impact+Technology" alt="Typing SVG" />
</p>

<!-- Impact Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/🎯-Social%20Impact-FF6B6B?style=for-the-badge&labelColor=2E7D32" />
  <img src="https://img.shields.io/badge/🏥-Healthcare%20AI-4ECDC4?style=for-the-badge&labelColor=2E7D32" />
  <img src="https://img.shields.io/badge/♿-Accessibility%20First-FFD93D?style=for-the-badge&labelColor=2E7D32" />
</p>

<!-- Tech Stack Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Tailwind-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" />
</p>

<!-- Status Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0.0-2E7D32?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-2E7D32?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-2E7D32?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
</p>

<!-- Quick Navigation -->
<p align="center">
  <a href="#-live-demo">🚀 Live Demo</a> •
  <a href="#-features">✨ Features</a> •
  <a href="#-architecture">🏗️ Architecture</a> •
  <a href="#-quickstart">⚡ Quickstart</a> •
  <a href="#-impact">🌍 Impact</a>
</p>

</div>

<!-- Animated Divider -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 📖 Overview

<div align="center">

### "स्वास्थ्य सेतु" — *Bridge to Health*

</div>

**SwasthyaSetu** is an open-source, AI-powered medical triage and routing assistant designed for **accessibility** and **social impact**. It democratizes healthcare access by helping users:

| Feature | Description |
|---------|-------------|
| 🩺 **Symptom Analysis** | Describe symptoms in natural language |
| 🔍 **Smart Diagnosis** | Receive preliminary diagnosis with differential analysis |
| ⚡ **Intelligent Triage** | Get triaged for urgency (emergency/urgent/routine) |
| 🏥 **Provider Discovery** | Find local healthcare providers with location-aware recommendations |
| 📄 **Report Generation** | Download/print reports for medical consultations |

---

## 🚀 Live Demo

<div align="center">

[![Live Demo](https://img.shields.io/badge/🔗-Live%20Demo-2E7D32?style=for-the-badge&logo=vercel&logoColor=white)](https://swasthyasetu.onrender.com)
[![Video Demo](https://img.shields.io/badge/▶️-Video%20Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com/your-video)

</div>

---

## ✨ Features

<div align="center">

<table>
<tr>
<td width="50%" valign="top">

### 🤖 Multi-Agent AI Workflow
*Powered by LangGraph + Gemini LLM*

| Agent | Function |
|-------|----------|
| 🔬 **Diagnostician** | Differential diagnosis generation |
| ⚡ **Triage** | Urgency classification (emergency/urgent/routine) |
| 📚 **Educator** | Patient-friendly explanations |
| ⚖️ **Bias Checker** | Equity & bias analysis (optional) |
| 🗺️ **Router** | Local doctor/specialist finder |
| ✅ **Validator** | Medical guideline compliance |

</td>
<td width="50%" valign="top">

### 🎨 Modern, Accessible UI
*Built with Tailwind CSS + Vanilla JS*

- ♿ **WCAG 2.1 AA** compliant
- 📱 **Responsive design** for all devices
- 🌙 **Dark/Light mode** support
- 🌍 **Localization ready**
- 🔊 **Screen reader** compatible
- ⌨️ **Keyboard navigation**

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🗺️ Location Intelligence

- 📍 **GPS-based** provider discovery
- 🏥 **Specialist matching** by condition
- 🚗 **Distance-based** recommendations
- 🔍 **DuckDuckGo integration** for local search

</td>
<td width="50%" valign="top">

### 📊 Observability & Reports

- 📝 **Granular backend logging**
- 📄 **PDF report generation**
- 🖨️ **Print-friendly output**
- 🔧 **Debug info** (hidden by default)
- 📈 **Usage analytics** ready

</td>
</tr>
</table>

</div>

---

## 🏗️ Architecture

### System Flow

```mermaid
flowchart TB
    subgraph External["🔗 External Services"]
        Gemini["🧠 Gemini LLM"]
        DuckDuck["🔍 DuckDuckGo Search"]
        Location["📍 Location Services"]
    end

    subgraph Backend["⚙️ FastAPI Backend"]
        Input["📥 Input Processor"]
        LangGraph["🧠 LangGraph<br/>Multi-Agent Workflow"]
        Output["📤 Output Formatter"]
        
        subgraph Agents["🤖 AI Agents"]
            Diag["🔬 Diagnostician"]
            Triage["⚡ Triage Agent"]
            Edu["📚 Educator"]
            Bias["⚖️ Bias Checker"]
            Route["🗺️ Routing Agent"]
            Valid["✅ Validator"]
        end
    end

    subgraph Frontend["👤 User Interaction"]
        UI["🌐 Web Interface<br/>Tailwind CSS + JS"]
        UserInput["Symptoms + Location"]
        UserOutput["Report + Recommendations"]
    end

    UI --> UserInput
    UserInput --> Input
    Input --> LangGraph
    LangGraph --> Diag & Triage & Edu & Bias & Route & Valid
    Diag & Triage & Edu & Bias & Route & Valid --> LangGraph
    LangGraph --> Output
    Output --> UserOutput
    UserOutput --> UI
    
    Gemini <---> LangGraph
    DuckDuck <---> Route
    Location <---> Route
```

### Agent Workflow Detail

```mermaid
flowchart TD
    Start([👤 User Input<br/>Symptoms & Location]) --> Orchestrator["🧠 LangGraph Orchestrator"]
    
    Orchestrator --> Diag["🔬 Diagnostician<br/>Differential Diagnosis"]
    Orchestrator --> Triage["⚡ Triage<br/>Emergency Classification"]
    Orchestrator --> Edu["📚 Educator<br/>Patient-Friendly Explanations"]
    Orchestrator --> Bias["⚖️ Bias Checker<br/>Equity Analysis"]
    Orchestrator --> Route["🗺️ Routing Agent<br/>Local Provider Search"]
    Orchestrator --> Valid["✅ Validator<br/>Guideline Compliance"]
    
    Diag --> Merge{📊 Merge Results}
    Triage --> Merge
    Edu --> Merge
    Bias --> Merge
    Route --> Merge
    Valid --> Merge
    
    Merge --> Output["📄 Final Output"]
    
    Output --> Report["📋 Diagnosis Report"]
    Output --> Rec["💡 Triage Recommendation"]
    Output --> Providers["🏥 Healthcare Providers"]
    Output --> Resources["📖 Educational Resources"]
    Output --> Download["⬇️ Download/Print Option"]
    
    style Start fill:#2E7D32,color:#fff
    style Output fill:#2E7D32,color:#fff
    style Merge fill:#FFD93D,color:#000
```

---

## ⚡ Quickstart

### Prerequisites

- Python 3.9+
- Google Gemini API key
- Git

### Installation

<details open>
<summary><b>🚀 One-Command Setup</b></summary>

```bash
# Clone and setup in one go
git clone https://github.com/aceandro2812/swasthyasetu.git && cd swasthyasetu && python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt
```

</details>

<details>
<summary><b>📋 Step-by-Step Installation</b></summary>

```bash
# 1. Clone the repository
git clone https://github.com/aceandro2812/swasthyasetu.git
cd swasthyasetu

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Start the server
# Windows (PowerShell)
.\.venv\Scripts\activate; uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Unix/macOS
source .venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. Open in browser
# Navigate to http://localhost:8000
```

</details>

### Environment Configuration

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
DEBUG=false
LOG_LEVEL=INFO
```

---

## 📁 Project Structure

```
swasthyasetu/
│
├── 📄 main.py                 # FastAPI backend, agent workflow, logging
├── 📄 requirements.txt        # Python dependencies
├── 📄 .env.example           # Environment template
├── 📄 medsarathi.ipynb       # Original notebook (reference)
│
├── 📁 templates/
│   └── 📄 index.html         # Main UI (Tailwind CSS)
│
├── 📁 static/
│   ├── 📄 app.js             # Frontend logic
│   ├── 📄 styles.css         # Custom styles
│   └── 📁 assets/            # Images, icons
│
├── 📁 tests/
│   └── 📄 test_agents.py     # Unit tests
│
└── 📁 docs/
    └── 📄 API.md             # API documentation
```

---

## 🌍 Impact

<div align="center">

### 🎯 Mission
*Democratizing healthcare access through AI-powered triage and routing*

</div>

### 📊 Impact Metrics

| Metric | Target | Status |
|--------|--------|--------|
| 🏥 Rural Accessibility | 100+ villages | 🚧 In Progress |
| 👥 Users Helped | 10,000+ | 🚧 In Progress |
| ⏱️ Avg. Response Time | < 5 seconds | ✅ Achieved |
| ♿ Accessibility Score | WCAG 2.1 AA | ✅ Achieved |
| 🌍 Languages Supported | 5+ | 🚧 In Progress |

### 🌟 Use Cases

<div align="center">

<table>
<tr>
<td align="center" width="33%">

### 🏘️ Rural Healthcare
Bridging the gap between remote communities and quality healthcare

</td>
<td align="center" width="33%">

### ⚡ Emergency Triage
Quick assessment to prioritize critical cases

</td>
<td align="center" width="33%">

### 📚 Health Education
Patient-friendly explanations of conditions

</td>
</tr>
</table>

</div>

---

## 🔧 Development

### Extending the System

<details>
<summary><b>➕ Add New Agents</b></summary>

```python
# In main.py
def new_agent_node(state: AgentState) -> AgentState:
    """Your new agent logic here"""
    # Process state
    # Update state with results
    return state

# Add to workflow
workflow.add_node("new_agent", new_agent_node)
workflow.add_edge("previous_node", "new_agent")
```

</details>

<details>
<summary><b>🔄 Swap LLMs</b></summary>

```python
# Replace Gemini with OpenAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.3)
```

</details>

<details>
<summary><b>📚 Improve RAG</b></summary>

```python
# Add more medical documents to vector store
vectorstore.add_documents(new_medical_docs)
```

</details>

### Roadmap

| Phase | Feature | Timeline |
|-------|---------|----------|
| ✅ v1.0 | Core multi-agent system | Complete |
| 🚧 v1.1 | WhatsApp/SMS integration | Q2 2025 |
| 📋 v1.2 | Multi-language support | Q3 2025 |
| 📋 v1.3 | Doctor collaboration portal | Q4 2025 |
| 📋 v2.0 | Mobile PWA + offline mode | 2026 |

---

## 🤝 Contributing

We welcome contributions from developers, healthcare professionals, and accessibility experts!

<div align="center">

[![Contributing](https://img.shields.io/badge/📖-Contributing%20Guide-2E7D32?style=for-the-badge)](CONTRIBUTING.md)
[![Good First Issue](https://img.shields.io/badge/🎯-Good%20First%20Issue-FFD93D?style=for-the-badge)](https://github.com/your-org/swasthyasetu/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

</div>

```bash
# Fork and contribute
git clone https://github.com/aceandro2812/swasthyasetu.git
git checkout -b feature/amazing-feature
git commit -m "✨ Add amazing feature"
git push origin feature/amazing-feature
# Open a Pull Request 🎉
```

---

## 📜 License

<div align="center">

[![License](https://img.shields.io/badge/📄-MIT%20License-2E7D32?style=for-the-badge)](LICENSE)

### Made with 💚 for healthcare accessibility

</div>

---

<div align="center">

<!-- Animated Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&height=150&color=gradient&customColorList=0,2,2,5,30&section=footer" />

<!-- Social Links -->
<p>
  <a href="https://github.com/your-org">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://twitter.com/your-handle">
    <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" />
  </a>
  <a href="mailto:contact@swasthyasetu.org">
    <img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" />
  </a>
  <a href="https://discord.gg/your-server">
    <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" />
  </a>
</p>

<!-- Made With -->
<p>
  <img src="https://img.shields.io/badge/Made%20with-❤️-FF6B6B?style=flat-square" />
  <img src="https://img.shields.io/badge/Powered%20by-Gemini-4285F4?style=flat-square&logo=google" />
  <img src="https://img.shields.io/badge/Built%20with-LangGraph-1C3C3C?style=flat-square" />
</p>

### ⭐ Star this repository to support healthcare accessibility!

</div>
