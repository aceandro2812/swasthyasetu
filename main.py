# FastAPI backend for SwasthyaSetu
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, TypedDict
import uvicorn
import os
import time
import json
import logging
import warnings
from dataclasses import dataclass
from collections import defaultdict, deque
from threading import Lock
from uuid import uuid4
from dotenv import load_dotenv
# Import AI workflow dependencies
from google import genai
from google.genai import types as genai_types
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
# Import moved to try-catch block to handle missing google.generativeai dependency
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.prompt_helper import PromptHelper
import faiss
import re
from langgraph.graph import StateGraph, END
import requests

# --- Load environment variables and API key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("swasthyasetu")

# --- Initialize Google Gen AI Client ---
client = None
if GOOGLE_API_KEY:
    try:
        # Strip whitespace to guard against .env formatting issues
        _clean_key = GOOGLE_API_KEY.strip()
        client = genai.Client(api_key=_clean_key)
        logger.info("Google Gen AI Client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Google Gen AI Client: {e}")
        client = None
else:
    logger.error("GOOGLE_API_KEY is not set. LLM will not be available.")

# --- Constants ---
LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
PDF_DIR = "./pubmed_data/"
PDF_FILENAME = "pubmed_papers.pdf"
PDF_FILEPATH = os.path.join(PDF_DIR, PDF_FILENAME)

# Hard limits intentionally lower than Gemini service limits to reduce upstream 429 risk.
DIAGNOSE_PER_IP_PER_MINUTE = int(os.getenv("DIAGNOSE_PER_IP_PER_MINUTE", "4"))
DIAGNOSE_PER_IP_PER_HOUR = int(os.getenv("DIAGNOSE_PER_IP_PER_HOUR", "30"))
DIAGNOSE_GLOBAL_PER_MINUTE = int(os.getenv("DIAGNOSE_GLOBAL_PER_MINUTE", "20"))
DIAGNOSE_GLOBAL_PER_HOUR = int(os.getenv("DIAGNOSE_GLOBAL_PER_HOUR", "240"))

# --- Setup RAG (FAISS + LlamaIndex) ---
embed_model = None
try:
    from llama_index.embeddings.gemini import GeminiEmbedding
    logger.info("Initializing Gemini Embedding model...")
    embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL_NAME, api_key=GOOGLE_API_KEY)
    logger.info("Gemini Embedding model initialized.")
except Exception as e:
    logger.warning(f"Gemini Embedding not available ({e}). RAG will be disabled.")

query_engine = None
documents = None
if os.path.exists(PDF_FILEPATH) or os.path.exists(PDF_FILEPATH.replace('.pdf', '.txt')):
    actual_file_path = PDF_FILEPATH if os.path.exists(PDF_FILEPATH) else PDF_FILEPATH.replace('.pdf', '.txt')
    logger.info(f"Loading documents from {actual_file_path}...")
    reader = SimpleDirectoryReader(input_files=[actual_file_path])
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents) if documents else 0} documents.")
    if documents and embed_model:
        d = getattr(embed_model, 'embed_dim', 768)
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=False)
        query_engine = index.as_query_engine(similarity_top_k=3)
        logger.info("RAG query engine initialized.")
    else:
        logger.warning("Documents or embedding model missing; RAG not initialized.")
else:
    logger.warning("No PubMed data file found; RAG will be unavailable.")


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int
    limit_per_minute: int
    remaining_minute: int
    limit_per_hour: int
    remaining_hour: int


class InMemoryHardRateLimiter:
    """Simple in-memory hard limiter.
    Note: per-process only. For multi-worker deploys, use Redis/shared storage.
    """

    def __init__(self, per_minute: int, per_hour: int) -> None:
        self.per_minute = max(1, per_minute)
        self.per_hour = max(1, per_hour)
        self._events: Dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _prune(self, events: deque[float], now: float) -> None:
        one_hour_ago = now - 3600
        while events and events[0] <= one_hour_ago:
            events.popleft()

    def check(self, key: str) -> RateLimitDecision:
        now = time.time()
        with self._lock:
            events = self._events[key]
            self._prune(events, now)

            minute_cutoff = now - 60
            minute_count = sum(1 for t in events if t > minute_cutoff)
            hour_count = len(events)

            minute_allowed = minute_count < self.per_minute
            hour_allowed = hour_count < self.per_hour

            if minute_allowed and hour_allowed:
                events.append(now)
                return RateLimitDecision(
                    allowed=True,
                    retry_after_seconds=0,
                    limit_per_minute=self.per_minute,
                    remaining_minute=max(0, self.per_minute - (minute_count + 1)),
                    limit_per_hour=self.per_hour,
                    remaining_hour=max(0, self.per_hour - (hour_count + 1)),
                )

            retry_after = 60
            if not minute_allowed:
                minute_events = [t for t in events if t > minute_cutoff]
                if minute_events:
                    retry_after = max(1, int(minute_events[0] + 60 - now))
            if not hour_allowed and events:
                retry_after = max(retry_after, int(events[0] + 3600 - now))

            return RateLimitDecision(
                allowed=False,
                retry_after_seconds=retry_after,
                limit_per_minute=self.per_minute,
                remaining_minute=max(0, self.per_minute - minute_count),
                limit_per_hour=self.per_hour,
                remaining_hour=max(0, self.per_hour - hour_count),
            )


diagnose_ip_limiter = InMemoryHardRateLimiter(
    per_minute=DIAGNOSE_PER_IP_PER_MINUTE,
    per_hour=DIAGNOSE_PER_IP_PER_HOUR,
)
diagnose_global_limiter = InMemoryHardRateLimiter(
    per_minute=DIAGNOSE_GLOBAL_PER_MINUTE,
    per_hour=DIAGNOSE_GLOBAL_PER_HOUR,
)


def get_client_identifier(request: Request) -> str:
    # Prefer the first forwarded IP when behind a trusted reverse proxy.
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def parse_json_from_llm_text(
    response_text: str,
    required_keys: List[str],
    list_keys: Optional[List[str]] = None,
    numeric_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not response_text:
        raise ValueError("Empty LLM response.")

    candidates: List[str] = [response_text.strip()]

    fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    candidates.extend(fenced_matches)

    # Try to decode first JSON object found in the text.
    decoder = json.JSONDecoder()
    for start_idx, ch in enumerate(response_text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(response_text[start_idx:])
            if isinstance(obj, dict):
                candidates.append(json.dumps(obj))
                break
        except json.JSONDecodeError:
            continue

    parsed = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                break
        except Exception:
            continue

    if not isinstance(parsed, dict):
        raise ValueError("No valid JSON object found in LLM response.")

    if not all(key in parsed for key in required_keys):
        raise ValueError(f"Parsed JSON missing required keys: {required_keys}")

    for key in list_keys or []:
        if not isinstance(parsed.get(key), list):
            raise ValueError(f"Parsed JSON '{key}' is not a list.")

    for key in numeric_keys or []:
        if not isinstance(parsed.get(key), (float, int)):
            raise ValueError(f"Parsed JSON '{key}' is not a number.")

    return parsed

# --- Helper: Robust LLM Call ---
def generate_gemini_content_with_retry(model_name, prompt, max_retries=3, initial_delay=2):
    logger.info(f"LLM call: model={model_name}, prompt_length={len(prompt)}")
    if not client:
        logger.error("Google Gen AI Client not initialized.")
        return "Error: Google Gen AI Client not initialized."
    
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            duration = time.time() - start
            logger.info(f"LLM response received in {duration:.2f}s (attempt {attempt+1})")
            
            if response.text:
                return response.text
            else:
                return "Error: Empty response from Gemini."
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
            time.sleep(delay)
            delay *= 2
    logger.error("Failed to get response from Gemini API after multiple retries.")
    return "Error: Failed to get response from Gemini API after multiple retries."

# --- LangGraph Agents & Workflow ---
class AgentState(TypedDict):
    original_input: Any
    input_language: Optional[str]
    symptoms_text: str
    location: Optional[str]
    learn_mode: Optional[bool]
    rag_context: Optional[List[str]]
    initial_diagnosis: Optional[Dict[str, Any]]
    triage_result: Optional[Dict[str, Any]]
    routing_result: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    final_diagnosis_report: Optional[Dict[str, Any]]
    patient_education: Optional[Dict[str, Any]]
    bias_analysis: Optional[Dict[str, Any]]
    error_message: Optional[str]




def diagnostician_node(state: AgentState) -> AgentState:
    logger.info("[Node] Diagnostician: Entry")
    symptoms = state.get("symptoms_text")
    learn_mode = state.get("learn_mode", False)
    if not symptoms:
        logger.error("Diagnostician failed: Symptoms missing.")
        return {**state, "error_message": "Diagnostician failed: Symptoms missing."}
    rag_context_str = ""
    if query_engine:
        try:
            logger.info("Diagnostician: Querying RAG context...")
            rag_response = query_engine.query(symptoms)
            retrieved_docs = [node.get_content() for node in rag_response.source_nodes]
            state["rag_context"] = retrieved_docs
            rag_context_str = "\n\nRelevant Medical Context:\n" + "\n---\n".join(retrieved_docs)
            logger.info(f"Diagnostician: Retrieved {len(retrieved_docs)} RAG context snippets.")
        except Exception as e:
            logger.warning(f"Diagnostician: Error retrieving RAG context: {e}")
            rag_context_str = "\n\nRelevant Medical Context: [Error retrieving context]"
            state["rag_context"] = ["[Error retrieving context]"]
    else:
        logger.warning("Diagnostician: RAG context not available.")
        rag_context_str = "\n\nRelevant Medical Context: [Not Available]"
        state["rag_context"] = ["[Not Available]"]
    if not client:
        logger.error("Diagnostician failed: LLM client not initialized.")
        return {**state, "error_message": "Diagnostician failed: LLM client not initialized."}
    prompt = f"""Act as a medical diagnosis assistant. Based ONLY on the provided symptoms and relevant medical context (if available), generate a differential diagnosis.\n\nPatient Symptoms:\n{symptoms}{rag_context_str}\n\nInstructions:\n1. Analyze the symptoms and context.\n2. Generate a list of possible diagnoses (differentials).\n3. For each diagnosis, provide a confidence score (0.0 to 1.0) indicating your certainty based *only* on the provided information. Higher scores mean higher likelihood.\n4. Identify the most likely primary diagnosis.\n5. Structure your output as a JSON object with the following EXACT keys: \"primary_diagnosis\", \"primary_confidence\", \"alternative_diagnoses\" (which should be a list of strings).\n\nProvide ONLY the JSON object in your response."""
    llm_response_text = generate_gemini_content_with_retry(LLM_MODEL_NAME, prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Diagnostician LLM Error: {llm_response_text}")
        return {**state, "error_message": f"Diagnostician LLM Error: {llm_response_text}"}
    diagnosis_json = None
    try:
        diagnosis_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["primary_diagnosis", "primary_confidence", "alternative_diagnoses"],
            list_keys=["alternative_diagnoses"],
            numeric_keys=["primary_confidence"],
        )
        logger.info("Diagnostician: Diagnosis JSON parsed successfully.")
    except Exception as e:
        logger.error(f"Diagnostician failed: {e}")
        return {**state, "error_message": f"Diagnostician failed: {e}"}
    reasoning = []
    guidelines = []
    if learn_mode:
        reasoning.append("Analyzed symptoms and context for likely diagnoses using pattern recognition and RAG context.")
        guidelines.append("Based on general diagnostic reasoning and PubMed abstracts. For real-world use, cross-check with NICE/WHO guidelines.")
    logger.info("[Node] Diagnostician: Exit")
    return {**state, "initial_diagnosis": diagnosis_json, "diagnosis_reasoning": reasoning, "diagnosis_guidelines": guidelines, "error_message": None}

def triage_agent_node(state: AgentState) -> AgentState:
    logger.info("[Node] Triage Agent: Entry")
    diagnosis = state.get("initial_diagnosis", {})
    symptoms = state.get("symptoms_text", "")
    learn_mode = state.get("learn_mode", False)
    if not diagnosis or not symptoms:
        logger.warning("Triage Agent skipped: Missing diagnosis or symptoms.")
        return {**state, "triage_result": {"status": "Skipped", "reason": "Missing diagnosis or symptoms."}}
    if not client:
        logger.error("Triage Agent failed: LLM client not initialized.")
        return {**state, "triage_result": {"status": "Failed", "reason": "LLM client not initialized."}, "error_message": "Triage Agent failed: LLM client not initialized."}
    primary_diag = diagnosis.get("primary_diagnosis", "N/A")
    confidence = diagnosis.get("primary_confidence", 0.0)
    prompt = f"""
Act as a medical triage AI. Given the patient's symptoms and the AI-generated diagnosis, classify the urgency:
- Emergency: Needs immediate medical attention (e.g., heart attack, stroke, severe trauma).
- Urgent: Should see a doctor within 24-48 hours.
- Routine: Can be managed with self-care or a scheduled visit.

Patient Symptoms:
{symptoms}

Diagnosis:
{primary_diag} (Confidence: {confidence})

Instructions:
1. Classify the case as 'emergency', 'urgent', or 'routine'.
2. Suggest the next step (e.g., go to ER, see GP, self-care).
3. If emergency, explain why.
4. Output a JSON object with keys: 'triage_level', 'next_step', 'explanation'.
Provide ONLY the JSON object in your response.
"""
    llm_response_text = generate_gemini_content_with_retry(LLM_MODEL_NAME, prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Triage Agent LLM Error: {llm_response_text}")
        return {**state, "triage_result": {"status": "Failed", "reason": llm_response_text}, "error_message": f"Triage Agent LLM Error: {llm_response_text}"}
    triage_json = None
    try:
        triage_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["triage_level", "next_step", "explanation"],
        )
        logger.info("Triage Agent: Triage JSON parsed successfully.")
        triage_json["status"] = "Success"
        if learn_mode:
            triage_json["reasoning"] = [
                "Classified urgency based on diagnosis and symptom severity.",
                "Emergency if life-threatening, urgent if needs quick attention, routine otherwise."
            ]
            triage_json["guidelines"] = [
                "Triage logic inspired by WHO triage protocols and common clinical practice."
            ]
    except Exception as e:
        logger.error(f"Triage Agent failed: {e}")
        triage_json = {"status": "Failed", "reason": f"Triage Agent failed: {e}"}
    logger.info("[Node] Triage Agent: Exit")
    return {**state, "triage_result": triage_json, "error_message": None}

def routing_agent_node(state: AgentState) -> AgentState:
    logger.info("[Node] Routing Agent: Entry")
    diagnosis = state.get("initial_diagnosis", {})
    triage = state.get("triage_result") or {}
    location = state.get("location", "")
    if not diagnosis or not location:
        logger.warning("Routing Agent skipped: Missing diagnosis or location.")
        return {**state, "routing_result": {"status": "Skipped", "reason": "Missing diagnosis or location."}}
    primary_diag = diagnosis.get("primary_diagnosis", "N/A")
    triage_level = triage.get("triage_level", "routine") if isinstance(triage, dict) else "routine"
    # Compose search query
    if triage_level == "emergency":
        search_query = f"emergency hospital near {location}"
    else:
        search_query = f"{primary_diag} specialist doctor near {location}"
    logger.info(f"Routing Agent: DuckDuckGo search for '{search_query}'")
    try:
        # Use DuckDuckGo's HTML results (no API key needed)
        url = f"https://duckduckgo.com/html/?q={requests.utils.quote(search_query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        results = []
        if resp.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.select("a.result__a")[:3]:
                results.append({
                    "title": a.get_text(strip=True),
                    "url": a.get("href")
                })
        else:
            logger.warning(f"DuckDuckGo search failed: status {resp.status_code}")
        logger.info(f"Routing Agent: Found {len(results)} results.")
        routing_json = {
            "search_query": search_query,
            "results": results,
            "status": "Success" if results else "No results found"
        }
    except Exception as e:
        logger.error(f"Routing Agent failed: {e}")
        routing_json = {"status": "Failed", "reason": str(e)}
    logger.info("[Node] Routing Agent: Exit")
    return {**state, "routing_result": routing_json, "error_message": None}

def validator_node(state: AgentState) -> AgentState:
    logger.info("[Node] Validator: Entry")
    initial_diagnosis = state.get("initial_diagnosis")
    symptoms = state.get("symptoms_text")
    rag_context = state.get("rag_context") or []
    learn_mode = state.get("learn_mode") or False
    if not initial_diagnosis or not symptoms:
        logger.warning("Validator skipped: Missing diagnosis or symptoms.")
        return {**state, "validation_results": {"status": "Skipped", "reason": "Missing diagnosis or symptoms."}}
    if not client:
        logger.error("Validator failed: LLM client not initialized.")
        return {**state, "error_message": "Validator failed: LLM client not initialized."}
    primary_diag = initial_diagnosis.get("primary_diagnosis", "N/A")
    confidence = initial_diagnosis.get("primary_confidence", "N/A")
    alternatives = initial_diagnosis.get("alternative_diagnoses", [])
    rag_context_str = "\n---\n".join(rag_context) if rag_context else "[Not Available]"
    prompt = f"""Act as a clinical reviewer simulating a check against established medical guidelines (like NICE, but using general medical knowledge).\nYou are given an initial diagnosis generated by another AI based on patient symptoms and some retrieved medical context.\n\nPatient Symptoms:\n{symptoms}\n\nRetrieved Medical Context (from PubMed abstracts):\n{rag_context_str}\n\nInitial AI Diagnosis:\nPrimary: {primary_diag} (Confidence: {confidence})\nAlternatives: {', '.join(alternatives) if alternatives else 'None'}\n\nYour Task:\nCritically evaluate the initial diagnosis based *only* on the provided symptoms and context.\n1. Does the primary diagnosis seem reasonable given the symptoms and context?\n2. Are there any obvious contradictions or inconsistencies?\n3. Are there other highly probable diagnoses based on the provided info that were missed in the alternatives?\n4. Based on your critique, would you tentatively 'Confirm', 'Flag for Review', or 'Suggest Revision' for the primary diagnosis?\n\nProvide your output as a JSON object with the following keys:\n- \"validation_status\": (string, one of \"Confirmed\", \"Flagged for Review\", \"Revision Suggested\")\n- \"critique\": (string, your reasoning and evaluation based on the questions above)\n- \"missed_alternatives\": (list of strings, other possible diagnoses you identified, if any)\n\nProvide ONLY the JSON object in your response."""
    llm_response_text = generate_gemini_content_with_retry(LLM_MODEL_NAME, prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Validator LLM Error: {llm_response_text}")
        return {**state, "error_message": f"Validator LLM Error: {llm_response_text}"}
    validation_json = None
    try:
        validation_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["validation_status", "critique", "missed_alternatives"],
            list_keys=["missed_alternatives"],
        )
        logger.info("Validator: Validation JSON parsed successfully.")
        if learn_mode:
            validation_json["reasoning"] = [
                "Compared AI diagnosis to established guidelines and checked for contradictions.",
                "Flagged for review if inconsistencies or missed alternatives found."
            ]
            validation_json["guidelines"] = [
                "Validation logic based on NICE/WHO clinical guidelines."]
    except Exception as e:
        validation_json = {"status": "Failed", "reason": f"Unexpected error - {e}", "critique": "", "missed_alternatives": []}
    logger.info("[Node] Validator: Exit")
    return {**state, "validation_results": validation_json, "error_message": None}

def educator_node(state: AgentState) -> AgentState:
    logger.info("[Node] Educator: Entry")
    diagnosis_info = state.get("initial_diagnosis")
    rag_context = state.get("rag_context") or []
    learn_mode = state.get("learn_mode") or False
    if not diagnosis_info or not diagnosis_info.get("primary_diagnosis"):
        logger.warning("Educator skipped: Missing diagnosis.")
        return {**state, "patient_education": {"status": "Skipped", "reason": "Missing diagnosis."}}
    if not client:
        logger.error("Educator failed: LLM client not initialized.")
        return {**state, "error_message": "Educator failed: LLM client not initialized."}
    primary_diag = diagnosis_info.get("primary_diagnosis")
    rag_context_str = "\n---\n".join(rag_context) if rag_context else "[Not Available]"
    prompt = f"""Act as a patient educator AI. You are given a medical diagnosis and relevant context.\n\nDiagnosis: {primary_diag}\n\nRelevant Medical Context (from PubMed abstracts):\n{rag_context_str}\n\nYour Task: Generate patient education material based *only* on the provided diagnosis and context.\n1.  **Explanation:** Provide a simple, patient-friendly explanation of what '{primary_diag}' is (approx. 2-3 sentences). Avoid jargon.\n2.  **Medication Info:** Scan the 'Relevant Medical Context'. If specific medications for treating '{primary_diag}' are mentioned, list them. If not, state \"Consult your physician for medication options.\" Do NOT invent medications.\n3.  **Next Steps/Lifestyle:** Suggest 2-3 general, safe next steps or lifestyle considerations relevant to this type of condition (e.g., follow-up appointments, rest, hydration, seeking professional advice for specifics). Emphasize consulting a healthcare professional.\n4.  **Visual Placeholder:** Generate a descriptive filename for a hypothetical explanatory visual (e.g., 'Animation_showing_{primary_diag.replace(' ','_')}.mp4').\n\nProvide your output as a JSON object with the following keys:\n- \"explanation\": (string) Patient-friendly explanation.\n- \"medication_info\": (string) Mentioned medications or consultation advice.\n- \"next_steps\": (list of strings) General advice points.\n- \"visual_placeholder_filename\": (string) Generated filename for the visual.\n\nProvide ONLY the JSON object in your response."""
    llm_response_text = generate_gemini_content_with_retry(LLM_MODEL_NAME, prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Educator LLM Error: {llm_response_text}")
        return {**state, "error_message": f"Educator LLM Error: {llm_response_text}"}
    education_json = None
    try:
        education_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["explanation", "medication_info", "next_steps", "visual_placeholder_filename"],
            list_keys=["next_steps"],
        )
        logger.info("Educator: Education JSON parsed successfully.")
        if learn_mode and education_json:
            education_json["reasoning"] = [
                "Generated patient-friendly explanation and next steps based on diagnosis and context.",
                "Avoided medical jargon and emphasized consulting a professional."
            ]
            education_json["guidelines"] = [
                "Patient education based on WHO patient communication best practices."]
    except Exception as e:
        education_json = {"status": "Failed", "reason": f"Unexpected error - {e}"}
    logger.info("[Node] Educator: Exit")
    return {**state, "patient_education": education_json, "error_message": None}

def bias_check_node(state: AgentState) -> AgentState:
    logger.info("[Node] Bias Checker: Entry")
    initial_diagnosis = state.get("initial_diagnosis")
    symptoms = state.get("symptoms_text")
    if not initial_diagnosis or not symptoms:
        logger.warning("Bias Check skipped: Missing diagnosis or symptoms.")
        return {**state, "bias_analysis": {"status": "Skipped", "reason": "Missing diagnosis or symptoms."}}
    if not client:
        logger.error("Bias Check failed: LLM client not initialized.")
        return {**state, "error_message": "Bias Check failed: LLM client not initialized."}
    diagnosis_summary = f"Primary: {initial_diagnosis.get('primary_diagnosis', 'N/A')}, Confidence: {initial_diagnosis.get('primary_confidence', 'N/A')}, Alternatives: {initial_diagnosis.get('alternative_diagnoses', [])}"
    prompt = f"""Analyze the following diagnosis information for potential biases. Focus specifically on:\n1.  **Gender/racial stereotypes:** Does the diagnosis or the way it might have been reached rely on assumptions about specific genders or races?\n2.  **Socioeconomic assumptions:** Does the potential diagnosis path or suggested alternatives implicitly assume a certain socioeconomic status (e.g., access to specific tests, lifestyle factors)?\n3.  **Cultural competency:** Could the symptoms presentation or interpretation be influenced by cultural factors not accounted for? Are there potential cultural adaptations needed for communication or treatment?\n\nPatient Symptoms:\n{symptoms}\n\nAI-Generated Diagnosis Summary:\n{diagnosis_summary}\n\nInstructions:\n- Critically evaluate based on the three points above.\n- Provide a qualitative assessment. Note specific concerns if any.\n- Suggest potential cultural adaptations if relevant (e.g., language considerations, culturally sensitive explanations).\n- Assign a hypothetical bias risk score from 0.0 (very low risk) to 1.0 (high risk detected). This is subjective based on your analysis.\n- Structure your output as a JSON object with keys: \"bias_risk_score\" (float), \"potential_biases_identified\" (list of strings describing concerns), \"suggested_cultural_adaptations\" (list of strings).\n\nProvide ONLY the JSON object in your response."""
    llm_response_text = generate_gemini_content_with_retry(LLM_MODEL_NAME, prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Bias Check LLM Error: {llm_response_text}")
        return {**state, "error_message": f"Bias Check LLM Error: {llm_response_text}"}
    bias_json = None
    try:
        bias_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["bias_risk_score", "potential_biases_identified", "suggested_cultural_adaptations"],
            list_keys=["potential_biases_identified", "suggested_cultural_adaptations"],
            numeric_keys=["bias_risk_score"],
        )
        logger.info("Bias Checker: Bias analysis JSON parsed successfully.")
    except Exception as e:
        bias_json = {"status": "Failed", "reason": f"Unexpected error - {e}"}
    logger.info("[Node] Bias Checker: Exit")
    return {**state, "bias_analysis": bias_json, "error_message": None}

def format_output_node(state: AgentState) -> AgentState:
    logger.info("[Node] Output Formatter: Entry")
    # Use `or {}` / `or []` to safely handle explicitly-stored None values
    # (state.get(key, default) only uses the default when the key is MISSING,
    #  not when it's present but set to None)
    initial_diag = state.get("initial_diagnosis") or {}
    triage = state.get("triage_result") or {}
    routing = state.get("routing_result") or {}
    validation = state.get("validation_results") or {}
    education = state.get("patient_education") or {}
    bias_info = state.get("bias_analysis") or {}
    learn_mode = state.get("learn_mode") or False
    primary_diagnosis = initial_diag.get("primary_diagnosis", "N/A")
    confidence = initial_diag.get("primary_confidence", 0.0)
    alternatives = initial_diag.get("alternative_diagnoses", [])
    diagnosis_part = {
        "primary": primary_diagnosis,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        "alternatives": alternatives if isinstance(alternatives, list) else [],
        "validation_status": validation.get("validation_status", "Pending/Skipped")
    }
    triage_part = {
        "level": triage.get("triage_level", "N/A"),
        "next_step": triage.get("next_step", "N/A"),
        "explanation": triage.get("explanation", "N/A")
    }
    routing_part = {
        "search_query": routing.get("search_query", "N/A"),
        "results": routing.get("results", []),
        "status": routing.get("status", "N/A"),
        "reason": routing.get("reason", "")
    }
    education_part = {
        "visual": education.get("visual_placeholder_filename", "visual_pending.mp4"),
        "explanation": education.get("explanation", "Explanation pending."),
        "medication": education.get("medication_info", "Medication info pending."),
        "next_steps": education.get("next_steps", ["Next steps pending."])
    }
    if education.get("status") == "Failed":
        education_part["status"] = "Failed: " + education.get("reason", "Unknown")
    equity_part = {
        "bias_score": bias_info.get("bias_risk_score", -1.0),
        "potential_biases": bias_info.get("potential_biases_identified", ["Pending analysis"]),
        "cultural_adaptations": bias_info.get("suggested_cultural_adaptations", ["Pending analysis"])
    }
    if bias_info.get("status") == "Failed":
        equity_part["status"] = "Failed: " + bias_info.get("reason", "Unknown")
    final_report = {
        "patient_id": f"ANON-{int(time.time()) % 10000}",
        "diagnosis": diagnosis_part,
        "triage": triage_part,
        "routing": routing_part,
        "education": education_part,
        "equity_check": equity_part,
        "debug_info": {
            "rag_context_snippets_count": len(state.get("rag_context", [])),
            "validator_critique": validation.get("critique", "N/A")
        }
    }
    if learn_mode:
        final_report["reasoning"] = []
        final_report["guidelines"] = []
        if state.get("diagnosis_reasoning"): final_report["reasoning"] += state["diagnosis_reasoning"]
        if triage.get("reasoning"): final_report["reasoning"] += triage["reasoning"]
        if validation.get("reasoning"): final_report["reasoning"] += validation["reasoning"]
        if education.get("reasoning"): final_report["reasoning"] += education["reasoning"]
        if state.get("diagnosis_guidelines"): final_report["guidelines"] += state["diagnosis_guidelines"]
        if triage.get("guidelines"): final_report["guidelines"] += triage["guidelines"]
        if validation.get("guidelines"): final_report["guidelines"] += validation["guidelines"]
        if education.get("guidelines"): final_report["guidelines"] += education["guidelines"]
    error = state.get("error_message")
    if error:
        final_report["workflow_status"] = "Error"
        final_report["error_details"] = error
    elif any(node_output.get("status") == "Failed" for node_output in [validation, education, bias_info]):
        final_report["workflow_status"] = "Completed with Errors in Nodes"
    else:
        final_report["workflow_status"] = "Success"
    logger.info("[Node] Output Formatter: Exit")
    return {**state, "final_diagnosis_report": final_report}

workflow = StateGraph(AgentState)
workflow.add_node("diagnostician", diagnostician_node)
workflow.add_node("triage_agent", triage_agent_node)
workflow.add_node("routing_agent", routing_agent_node)
workflow.add_node("validator", validator_node)
workflow.add_node("bias_checker", bias_check_node)
workflow.add_node("educator", educator_node)
workflow.add_node("output_formatter", format_output_node)
workflow.set_entry_point("diagnostician")
workflow.add_edge("diagnostician", "triage_agent")
workflow.add_edge("triage_agent", "routing_agent")
workflow.add_edge("routing_agent", "validator")
workflow.add_edge("validator", "bias_checker")
workflow.add_edge("bias_checker", "educator")
workflow.add_edge("educator", "output_formatter")
workflow.add_edge("output_formatter", END)
try:
    app_graph = workflow.compile()
except Exception as e:
    logger.error(f"Workflow compilation failed: {e}")
    app_graph = None

app = FastAPI()

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class SymptomInput(BaseModel):
    symptoms: str
    location: str
    learn_mode: Optional[bool] = False

@app.post("/diagnose")
async def diagnose(request: Request, symptom_input: SymptomInput):
    logger.info("/diagnose endpoint called.")
    if not app_graph:
        logger.error("LangGraph workflow not available.")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "Service unavailable. Please try again later."},
        )

    client_id = get_client_identifier(request)
    global_limit = diagnose_global_limiter.check("GLOBAL")
    if not global_limit.allowed:
        logger.warning("Global diagnose rate limit hit.")
        return JSONResponse(
            status_code=429,
            headers={
                "Retry-After": str(global_limit.retry_after_seconds),
                "X-RateLimit-Limit-Minute": str(global_limit.limit_per_minute),
                "X-RateLimit-Limit-Hour": str(global_limit.limit_per_hour),
                "X-RateLimit-Remaining-Minute": str(global_limit.remaining_minute),
                "X-RateLimit-Remaining-Hour": str(global_limit.remaining_hour),
            },
            content={
                "status": "error",
                "error_code": "RATE_LIMITED_GLOBAL",
                "message": "System is currently busy. Please retry after a short wait.",
                "retry_after_seconds": global_limit.retry_after_seconds,
                "limits": {
                    "minute": global_limit.limit_per_minute,
                    "hour": global_limit.limit_per_hour,
                },
            },
        )

    per_ip_limit = diagnose_ip_limiter.check(client_id)
    if not per_ip_limit.allowed:
        logger.warning(f"Rate limit hit for client={client_id}")
        return JSONResponse(
            status_code=429,
            headers={
                "Retry-After": str(per_ip_limit.retry_after_seconds),
                "X-RateLimit-Limit-Minute": str(per_ip_limit.limit_per_minute),
                "X-RateLimit-Limit-Hour": str(per_ip_limit.limit_per_hour),
                "X-RateLimit-Remaining-Minute": str(per_ip_limit.remaining_minute),
                "X-RateLimit-Remaining-Hour": str(per_ip_limit.remaining_hour),
            },
            content={
                "status": "error",
                "error_code": "RATE_LIMITED",
                "message": "Hard request limit reached for diagnosis. Please retry later.",
                "retry_after_seconds": per_ip_limit.retry_after_seconds,
                "limits": {
                    "minute": per_ip_limit.limit_per_minute,
                    "hour": per_ip_limit.limit_per_hour,
                },
            },
        )

    initial_state = AgentState(
        original_input=symptom_input.symptoms,
        input_language="en",
        symptoms_text=symptom_input.symptoms,
        location=symptom_input.location,
        learn_mode=symptom_input.learn_mode,
        rag_context=None,
        initial_diagnosis=None,
        triage_result=None,
        routing_result=None,
        validation_results=None,
        final_diagnosis_report=None,
        patient_education=None,
        bias_analysis=None,
        error_message=None
    )
    try:
        logger.info("Invoking LangGraph workflow...")
        final_state = await run_in_threadpool(app_graph.invoke, initial_state)
        report = final_state.get("final_diagnosis_report", {})
        logger.info("Diagnosis workflow completed successfully.")
        return JSONResponse(
            content={"status": "success", "report": report},
            headers={
                "X-RateLimit-Limit-Minute": str(per_ip_limit.limit_per_minute),
                "X-RateLimit-Limit-Hour": str(per_ip_limit.limit_per_hour),
                "X-RateLimit-Remaining-Minute": str(per_ip_limit.remaining_minute),
                "X-RateLimit-Remaining-Hour": str(per_ip_limit.remaining_hour),
            },
        )
    except Exception as e:
        error_id = str(uuid4())
        logger.error(f"Diagnosis workflow failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "Diagnosis workflow failed. Please retry.",
                "error_id": error_id,
            },
        )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    logger.info("Root endpoint accessed.")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    logger.info("Health check endpoint accessed.")
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
