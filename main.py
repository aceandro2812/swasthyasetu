# FastAPI backend for SwasthyaSetu
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, TypedDict
import uvicorn
import os
import time
import json
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
from threading import Lock
from uuid import uuid4
from dotenv import load_dotenv
# Import AI workflow dependencies
from llm_router import LLMRouter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
import faiss
import re
from langgraph.graph import StateGraph, END
import requests
import urllib.parse

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

# --- Initialize Multi-Provider LLM Router ---
# The router automatically pools Gemini + OpenRouter free models and
# fails over between providers when one hits its rate limit.
llm_router = LLMRouter()
if llm_router.is_available():
    logger.info(f"LLM Router initialized with {len(llm_router._providers)} provider(s).")
else:
    logger.error("LLM Router has NO providers. Set GOOGLE_API_KEY and/or OPENROUTER_API_KEY in .env")

# --- Constants ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
PDF_DIR = "./pubmed_data/"
PDF_FILENAME = "pubmed_papers.pdf"
PDF_FILEPATH = os.path.join(PDF_DIR, PDF_FILENAME)

# Visual Aid Configuration
VISUAL_AID_ENABLED = True
VISUAL_AID_PROVIDER = "pollinations"  # Options: "pollinations"
VISUAL_AID_WIDTH = 400
VISUAL_AID_HEIGHT = 300
VISUAL_AID_SEED = 42  # For consistent but unique images per diagnosis

# Hard limits:
# - Keep conservative defaults for single-provider deployments.
# - Raise defaults only when multiple providers are configured.
# Explicit env vars still override these defaults in all cases.
_provider_count = len(llm_router.status())
_multi_provider_mode = _provider_count > 1

_default_per_ip_per_minute = "6" if _multi_provider_mode else "4"
_default_per_ip_per_hour = "60" if _multi_provider_mode else "40"
_default_global_per_minute = "30" if _multi_provider_mode else "20"
_default_global_per_hour = "360" if _multi_provider_mode else "240"

DIAGNOSE_PER_IP_PER_MINUTE = int(
    os.getenv("DIAGNOSE_PER_IP_PER_MINUTE", _default_per_ip_per_minute)
)
DIAGNOSE_PER_IP_PER_HOUR = int(
    os.getenv("DIAGNOSE_PER_IP_PER_HOUR", _default_per_ip_per_hour)
)
DIAGNOSE_GLOBAL_PER_MINUTE = int(
    os.getenv("DIAGNOSE_GLOBAL_PER_MINUTE", _default_global_per_minute)
)
DIAGNOSE_GLOBAL_PER_HOUR = int(
    os.getenv("DIAGNOSE_GLOBAL_PER_HOUR", _default_global_per_hour)
)

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
        # Detect the actual embedding dimension dynamically via a test vector
        # to avoid silent FAISS corruption if the model changes.
        try:
            _test_vec = embed_model.get_text_embedding("test")
            d = len(_test_vec)
            logger.info(f"Detected embedding dimension: {d}")
        except Exception as _e:
            d = 768  # fallback; only reached if test embedding itself fails
            logger.warning(f"Could not detect embed_dim dynamically ({_e}); falling back to {d}")
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
            in_last_minute = sum(1 for t in events if t > minute_cutoff)

            if in_last_minute >= self.per_minute:
                retry_after = int(60 - (now - events[-self.per_minute]))
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=max(1, retry_after),
                    limit_per_minute=self.per_minute,
                    remaining_minute=0,
                    limit_per_hour=self.per_hour,
                    remaining_hour=max(0, self.per_hour - len(events))
                )

            if len(events) >= self.per_hour:
                retry_after = int(3600 - (now - events[0]))
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=max(1, retry_after),
                    limit_per_minute=self.per_minute,
                    remaining_minute=max(0, self.per_minute - in_last_minute),
                    limit_per_hour=self.per_hour,
                    remaining_hour=0
                )

            self._events[key].append(now)

            return RateLimitDecision(
                allowed=True,
                retry_after_seconds=0,
                limit_per_minute=self.per_minute,
                remaining_minute=self.per_minute - in_last_minute - 1,
                limit_per_hour=self.per_hour,
                remaining_hour=self.per_hour - len(events)
            )


diagnose_rate_limiter = InMemoryHardRateLimiter(
    per_minute=DIAGNOSE_PER_IP_PER_MINUTE,
    per_hour=DIAGNOSE_PER_IP_PER_HOUR
)

global_diagnose_minute_limiter = InMemoryHardRateLimiter(
    per_minute=DIAGNOSE_GLOBAL_PER_MINUTE,
    per_hour=DIAGNOSE_GLOBAL_PER_HOUR
)


# --- Pydantic Models ---
class DiagnosisRequest(BaseModel):
    symptoms: str = Field(..., min_length=5, max_length=2000,
                          description="Patient symptom description (5–2000 chars)")
    location: str = Field("India", max_length=100,
                          description="Patient location for care routing")
    learn_mode: bool = False


# --- LangGraph State ---
class AgentState(TypedDict):
    symptoms_text: str
    location: str
    learn_mode: bool
    rag_context: List[str]
    initial_diagnosis: Optional[Dict[str, Any]]
    triage_result: Optional[Dict[str, Any]]
    routing_result: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    patient_education: Optional[Dict[str, Any]]
    bias_analysis: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    error_message: Optional[str]
    diagnosis_reasoning: Optional[List[str]]
    diagnosis_guidelines: Optional[List[str]]


# --- LangGraph Nodes ---
def generate_llm_content(prompt: str) -> str:
    """Generate content using the LLM router with fallback."""
    if not llm_router.is_available():
        return "Error: No LLM providers available."
    try:
        result = llm_router.generate_content(prompt)
        if result.startswith("Error:"):
            logger.warning(f"LLM Router returned error: {result}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error in generate_llm_content: {e}")
        return f"Error: {e}"


def get_client_ip(request: Request) -> str:
    """Resolve client IP with proxy-aware fallback for rate limiting keys."""
    forwarded = request.headers.get("forwarded")
    if forwarded:
        # RFC 7239 format example: for=203.0.113.43;proto=https;by=203.0.113.60
        for part in forwarded.split(";"):
            key, sep, value = part.strip().partition("=")
            if sep and key.lower() == "for":
                return value.strip().strip('"').strip("[]")

    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        # X-Forwarded-For may contain multiple comma-separated hops.
        forwarded_ip = x_forwarded_for.split(",")[0].strip()
        if forwarded_ip:
            return forwarded_ip

    cf_connecting_ip = request.headers.get("cf-connecting-ip")
    if cf_connecting_ip and cf_connecting_ip.strip():
        return cf_connecting_ip.strip()

    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip and x_real_ip.strip():
        return x_real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def parse_json_from_llm_text(text: str, required_keys: List[str] = None, 
                             list_keys: List[str] = None, numeric_keys: List[str] = None) -> Dict[str, Any]:
    """
    Parse JSON from LLM response text, handling various formats and markdown code blocks.
    
    Args:
        text: The raw LLM response text
        required_keys: List of keys that must be present in the result
        list_keys: List of keys that should be lists (will be fixed if they're not)
        numeric_keys: List of keys that should be numeric
    
    Returns:
        Parsed JSON dictionary
    
    Raises:
        ValueError: If JSON cannot be parsed or required keys are missing
    """
    if not text or text.startswith("Error:"):
        raise ValueError(f"Invalid LLM response: {text}")
    
    list_keys = list_keys or []
    numeric_keys = numeric_keys or []
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text_to_parse = json_match.group(1)
    else:
        # Try to find JSON between curly braces
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            text_to_parse = json_match.group(1)
        else:
            text_to_parse = text
    
    try:
        result = json.loads(text_to_parse)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}. Text: {text[:200]}...")
        raise ValueError(f"Invalid JSON from LLM: {e}")
    
    # Ensure required keys exist
    if required_keys:
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key: {key}")
    
    # Ensure list keys are actually lists
    for key in list_keys:
        if key in result and not isinstance(result[key], list):
            if isinstance(result[key], str):
                # Try to split by newlines or commas
                result[key] = [item.strip() for item in result[key].replace('\n', ',').split(',') if item.strip()]
            else:
                result[key] = [result[key]]
    
    # Ensure numeric keys are numeric
    for key in numeric_keys:
        if key in result:
            try:
                result[key] = float(result[key])
            except (ValueError, TypeError):
                result[key] = 0.0
    
    return result


def generate_visual_aid_url(diagnosis: str, explanation: str = "") -> str:
    """
    Generate a medical illustration URL using Pollinations.ai.
    
    Args:
        diagnosis: The medical diagnosis/condition name
        explanation: Brief explanation to help generate more relevant images
    
    Returns:
        URL string for the AI-generated medical illustration
    """
    if not VISUAL_AID_ENABLED:
        return ""
    
    # Clean and sanitize the diagnosis for the prompt
    clean_diagnosis = diagnosis.strip().lower()
    
    # Create a professional medical illustration prompt
    # Include context from explanation if available
    base_prompt = f"medical illustration of {clean_diagnosis}"
    
    # Add anatomical/educational context based on common medical terms
    anatomical_terms = []
    
    # Common medical illustration enhancers
    if any(term in clean_diagnosis for term in ['fever', 'infection', 'virus', 'bacterial']):
        anatomical_terms.append("medical diagram showing affected body parts")
    elif any(term in clean_diagnosis for term in ['fracture', 'bone', 'joint', 'sprain']):
        anatomical_terms.append("anatomical diagram of skeletal structure")
    elif any(term in clean_diagnosis for term in ['skin', 'rash', 'dermatitis', 'acne']):
        anatomical_terms.append("dermatological illustration of skin layers")
    elif any(term in clean_diagnosis for term in ['heart', 'cardiac', 'blood pressure']):
        anatomical_terms.append("cardiovascular system diagram")
    elif any(term in clean_diagnosis for term in ['lung', 'respiratory', 'asthma', 'pneumonia']):
        anatomical_terms.append("respiratory system anatomical diagram")
    elif any(term in clean_diagnosis for term in ['brain', 'neurological', 'migraine', 'seizure']):
        anatomical_terms.append("neurological brain anatomy diagram")
    elif any(term in clean_diagnosis for term in ['stomach', 'digestive', 'gastric', 'intestinal']):
        anatomical_terms.append("digestive system anatomical diagram")
    elif any(term in clean_diagnosis for term in ['diabetes', 'thyroid', 'hormone']):
        anatomical_terms.append("endocrine system medical diagram")
    else:
        anatomical_terms.append("anatomical medical diagram educational")
    
    # Construct the full prompt
    full_prompt = f"{base_prompt}, {anatomical_terms[0]}, professional medical textbook style, clean white background, labeled anatomical parts, educational healthcare illustration, soft pastel colors, clinical accuracy"
    
    # URL encode the prompt
    encoded_prompt = urllib.parse.quote(full_prompt)
    
    # Generate seed based on diagnosis for consistency
    seed = sum(ord(c) for c in clean_diagnosis) % 1000
    
    # Construct the Pollinations.ai URL
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width={VISUAL_AID_WIDTH}"
        f"&height={VISUAL_AID_HEIGHT}"
        f"&nologo=true"
        f"&seed={seed}"
    )
    
    return url


def retrieval_node(state: AgentState) -> AgentState:
    logger.info("[Node] Retriever: Entry")
    query = state.get("symptoms_text")
    retrieved_context = ["[RAG not available - using LLM knowledge base]"]
    if query_engine and query:
        try:
            response = query_engine.query(query)
            retrieved_context = [node.text for node in response.source_nodes] if response.source_nodes else [str(response)]
            logger.info(f"[Node] Retriever: Retrieved {len(retrieved_context)} context snippets.")
        except Exception as e:
            logger.error(f"[Node] Retriever Error during query: {e}")
            retrieved_context = [f"[Error retrieving context: {e}]"]
    elif not query_engine:
        logger.warning("[Node] Retriever: Query engine not available.")
    else:
        logger.warning("[Node] Retriever: No query provided.")
    logger.info("[Node] Retriever: Exit")
    return {**state, "rag_context": retrieved_context}


def diagnosis_node(state: AgentState) -> AgentState:
    logger.info("[Node] Diagnosis: Entry")
    symptoms = state.get("symptoms_text")
    rag_context = state.get("rag_context") or []
    learn_mode = state.get("learn_mode") or False
    if not symptoms:
        logger.warning("Diagnosis skipped: No symptoms.")
        return {**state, "initial_diagnosis": {"status": "Skipped", "reason": "No symptoms provided."}}
    if not llm_router.is_available():
        logger.error("Diagnosis failed: LLM router has no providers.")
        return {**state, "error_message": "Diagnosis failed: No LLM providers configured."}
    rag_context_str = "\n---\n".join(rag_context) if rag_context else "[Not Available]"
    prompt = f"""You are a clinical decision support AI. Analyze the patient's symptoms and suggest likely diagnoses.

Patient Symptoms:
{symptoms}

Relevant Medical Context (from PubMed abstracts):
{rag_context_str}

Instructions:
- Identify the single most likely primary diagnosis and estimate a confidence percentage (0-100).
- Suggest up to 3 alternative diagnoses that should be considered.
- Base your analysis primarily on the provided symptoms, but consider the medical context as supporting evidence.
- Be specific but cautious. Use standard medical terminology for diagnoses.
- IMPORTANT: Include a disclaimer that this is AI-generated and not a substitute for professional medical advice.

Provide your output as a JSON object with the following keys:
- "primary_diagnosis": (string) The most likely diagnosis.
- "primary_confidence": (number) Confidence score (0.0 to 1.0).
- "alternative_diagnoses": (list of strings) Up to 3 alternative diagnoses.
- "reasoning": (string) Brief clinical reasoning for the primary diagnosis.

Provide ONLY the JSON object in your response."""
    llm_response_text = generate_llm_content(prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Diagnosis LLM Error: {llm_response_text}")
        return {**state, "error_message": "Diagnosis service temporarily unavailable. Please try again shortly."}
    diagnosis_json = None
    try:
        diagnosis_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["primary_diagnosis", "primary_confidence", "alternative_diagnoses"],
            list_keys=["alternative_diagnoses"],
            numeric_keys=["primary_confidence"]
        )
        logger.info("Diagnosis: Diagnosis JSON parsed successfully.")
        # Store reasoning and guidelines for learn mode (immutable copy — do NOT mutate state)
        if learn_mode:
            reasoning_list = []
            guidelines_list = []
            if diagnosis_json.get("reasoning"):
                reasoning_list.append(diagnosis_json["reasoning"])
            reasoning_list.append(f"Diagnosis based on symptom analysis with {diagnosis_json.get('primary_confidence', 0):.0%} confidence.")
            guidelines_list.append("Diagnostic reasoning based on standard clinical assessment practices.")
            return {**state, "initial_diagnosis": diagnosis_json, "error_message": None,
                    "diagnosis_reasoning": reasoning_list, "diagnosis_guidelines": guidelines_list}
    except Exception as e:
        logger.error(f"Diagnosis: Error parsing diagnosis JSON: {e}")
        diagnosis_json = {"status": "Failed", "reason": f"Unexpected error - {e}"}
    logger.info("[Node] Diagnosis: Exit")
    return {**state, "initial_diagnosis": diagnosis_json, "error_message": None}


def triage_node(state: AgentState) -> AgentState:
    logger.info("[Node] Triage: Entry")
    symptoms = state.get("symptoms_text")
    diagnosis_info = state.get("initial_diagnosis")
    learn_mode = state.get("learn_mode") or False
    if not symptoms:
        logger.warning("Triage skipped: No symptoms.")
        return {**state, "triage_result": {"status": "Skipped", "reason": "No symptoms provided."}}
    if not llm_router.is_available():
        logger.error("Triage failed: LLM router has no providers.")
        return {**state, "error_message": "Triage failed: No LLM providers configured."}
    primary_diag = diagnosis_info.get("primary_diagnosis", "N/A") if diagnosis_info else "N/A"
    prompt = f"""You are an emergency medicine triage nurse AI. Assess the urgency of the patient's situation based on symptoms and preliminary diagnosis.

Patient Symptoms:
{symptoms}

Preliminary Diagnosis (AI-generated):
{primary_diag}

Instructions:
- Assign a triage level: "self_care" (mild, home treatment ok), "clinic_visit" (non-urgent medical attention), "urgent_care" (prompt attention needed), or "emergency" (immediate attention).
- Recommend a clear next step (e.g., "Monitor at home", "Schedule clinic appointment", "Go to urgent care", "Call emergency services").
- Explain your reasoning briefly.
- Emphasize seeking human medical advice if uncertain.

Provide your output as a JSON object with the following keys:
- "triage_level": (string) One of: "self_care", "clinic_visit", "urgent_care", "emergency".
- "next_step": (string) Recommended immediate action.
- "explanation": (string) Brief reasoning.

Provide ONLY the JSON object in your response."""
    llm_response_text = generate_llm_content(prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Triage LLM Error: {llm_response_text}")
        return {**state, "error_message": "Triage service temporarily unavailable. Please try again shortly."}
    triage_json = None
    try:
        triage_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["triage_level", "next_step", "explanation"]
        )
        logger.info("Triage: Triage JSON parsed successfully.")
        if learn_mode:
            triage_json["reasoning"] = [
                f"Assessed urgency based on symptom severity and diagnosis: {primary_diag}",
                "Applied standard emergency triage protocols."
            ]
            triage_json["guidelines"] = [
                "Triage based on Emergency Severity Index (ESI) guidelines."]
    except Exception as e:
        triage_json = {"status": "Failed", "reason": f"Unexpected error - {e}"}
    logger.info("[Node] Triage: Exit")
    return {**state, "triage_result": triage_json, "error_message": None}


def care_routing_node(state: AgentState) -> AgentState:
    logger.info("[Node] Care Router: Entry")
    location = state.get("location")
    diagnosis_info = state.get("initial_diagnosis")
    if not location:
        logger.warning("Care Routing skipped: No location.")
        return {**state, "routing_result": {"status": "Skipped", "reason": "No location provided."}}
    primary_diag = diagnosis_info.get("primary_diagnosis", "medical care") if diagnosis_info else "medical care"
    search_query = f"{primary_diag} treatment in {location}"
    # Use urllib.parse.urlencode for correct URL encoding (handles &, #, =, etc.)
    routing_result = {
        "search_query": search_query,
        "results": [
            {"title": f"Search for {primary_diag} specialists near {location}",
             "url": f"https://www.google.com/search?{urllib.parse.urlencode({'q': search_query})}"},
            {"title": f"Find hospitals in {location}",
             "url": f"https://www.google.com/search?{urllib.parse.urlencode({'q': f'hospitals in {location}'})}"},
            {"title": f"{primary_diag} - Patient Information",
             "url": f"https://www.google.com/search?{urllib.parse.urlencode({'q': f'{primary_diag} patient information'})}"}
        ],
        "status": "Success",
        "reason": "Generated search links for local care."
    }
    logger.info(f"Care Router: Generated {len(routing_result['results'])} routing suggestions.")
    logger.info("[Node] Care Router: Exit")
    return {**state, "routing_result": routing_result}


def validator_node(state: AgentState) -> AgentState:
    logger.info("[Node] Validator: Entry")
    initial_diagnosis = state.get("initial_diagnosis")
    rag_context = state.get("rag_context") or []
    learn_mode = state.get("learn_mode") or False
    if not initial_diagnosis:
        logger.warning("Validation skipped: Missing diagnosis.")
        return {**state, "validation_results": {"status": "Skipped", "reason": "Missing diagnosis."}}
    if not llm_router.is_available():
        logger.error("Validator failed: LLM router has no providers.")
        return {**state, "error_message": "Validator failed: No LLM providers configured."}
    primary_diag = initial_diagnosis.get("primary_diagnosis", "N/A")
    rag_context_str = "\n---\n".join(rag_context) if rag_context else "[Not Available]"
    prompt = f"""You are a senior physician validating an AI-generated diagnosis. Review the diagnosis and context critically.

AI Diagnosis:
{primary_diag}

Confidence: {initial_diagnosis.get('primary_confidence', 'N/A')}

Alternative Diagnoses Considered:
{initial_diagnosis.get('alternative_diagnoses', [])}

Relevant Medical Context (PubMed abstracts):
{rag_context_str}

Instructions:
- Critically evaluate: Does the diagnosis align with the provided context? Are there major contradictions?
- Identify any critical conditions in the alternatives that should not be missed.
- Provide a validation status: "Validated", "Needs Review", or "Contradiction Found".
- Provide constructive critique and mention if any important alternative was missed.

Provide your output as a JSON object with the following keys:
- "validation_status": (string) "Validated", "Needs Review", or "Contradiction Found".
- "critique": (string) Brief critical assessment.
- "missed_alternatives": (list of strings) Important alternatives to consider.

Provide ONLY the JSON object in your response."""
    llm_response_text = generate_llm_content(prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Validator LLM Error: {llm_response_text}")
        return {**state, "error_message": "Validation service temporarily unavailable. Please try again shortly."}
    validation_json = None
    try:
        validation_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["validation_status", "critique", "missed_alternatives"],
            list_keys=["missed_alternatives"]
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
    if not llm_router.is_available():
        logger.error("Educator failed: LLM router has no providers.")
        return {**state, "error_message": "Educator failed: No LLM providers configured."}
    primary_diag = diagnosis_info.get("primary_diagnosis")
    rag_context_str = "\n---\n".join(rag_context) if rag_context else "[Not Available]"
    prompt = f"""Act as a patient educator AI. You are given a medical diagnosis and relevant context.

Diagnosis: {primary_diag}

Relevant Medical Context (from PubMed abstracts):
{rag_context_str}

Your Task: Generate patient education material based *only* on the provided diagnosis and context.
1.  **Explanation:** Provide a simple, patient-friendly explanation of what '{primary_diag}' is (approx. 2-3 sentences). Avoid jargon.
2.  **Medication Info:** Scan the 'Relevant Medical Context'. If specific medications for treating '{primary_diag}' are mentioned, list them. If not, state "Consult your physician for medication options." Do NOT invent medications.
3.  **Next Steps/Lifestyle:** Suggest 2-3 general, safe next steps or lifestyle considerations relevant to this type of condition (e.g., follow-up appointments, rest, hydration, seeking professional advice for specifics). Emphasize consulting a healthcare professional.
4.  **Visual Aid Description:** Generate a detailed description for an educational medical illustration about this condition. This should be a specific, descriptive prompt that could be used to generate an anatomical/educational image. Focus on anatomical accuracy and educational value. Example: "Medical diagram showing the effects of Type 2 Diabetes on the pancreas and insulin production".

Provide your output as a JSON object with the following keys:
- "explanation": (string) Patient-friendly explanation.
- "medication_info": (string) Mentioned medications or consultation advice.
- "next_steps": (list of strings) General advice points.
- "visual_aid_description": (string) Detailed description for educational medical illustration.

Provide ONLY the JSON object in your response."""
    llm_response_text = generate_llm_content(prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Educator LLM Error: {llm_response_text}")
        return {**state, "error_message": "Patient education service temporarily unavailable. Please try again shortly."}
    education_json = None
    try:
        education_json = parse_json_from_llm_text(
            llm_response_text,
            required_keys=["explanation", "medication_info", "next_steps", "visual_aid_description"],
            list_keys=["next_steps"],
        )
        # Generate the actual visual aid URL
        if education_json and "visual_aid_description" in education_json:
            visual_url = generate_visual_aid_url(
                diagnosis=primary_diag,
                explanation=education_json.get("explanation", "")
            )
            education_json["visual_aid_url"] = visual_url
            logger.info(f"Educator: Generated visual aid URL for {primary_diag}")
        
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
    if not llm_router.is_available():
        logger.error("Bias Check failed: LLM router has no providers.")
        return {**state, "error_message": "Bias Check failed: No LLM providers configured."}
    diagnosis_summary = f"Primary: {initial_diagnosis.get('primary_diagnosis', 'N/A')}, Confidence: {initial_diagnosis.get('primary_confidence', 'N/A')}, Alternatives: {initial_diagnosis.get('alternative_diagnoses', [])}"
    prompt = f"""Analyze the following diagnosis information for potential biases. Focus specifically on:
1.  **Gender/racial stereotypes:** Does the diagnosis or the way it might have been reached rely on assumptions about specific genders or races?
2.  **Socioeconomic assumptions:** Does the potential diagnosis path or suggested alternatives implicitly assume a certain socioeconomic status (e.g., access to specific tests, lifestyle factors)?
3.  **Cultural competency:** Could the symptoms presentation or interpretation be influenced by cultural factors not accounted for? Are there potential cultural adaptations needed for communication or treatment?

Patient Symptoms:
{symptoms}

AI-Generated Diagnosis Summary:
{diagnosis_summary}

Instructions:
- Critically evaluate based on the three points above.
- Provide a qualitative assessment. Note specific concerns if any.
- Suggest potential cultural adaptations if relevant (e.g., language considerations, culturally sensitive explanations).
- Assign a hypothetical bias risk score from 0.0 (very low risk) to 1.0 (high risk detected). This is subjective based on your analysis.
- Structure your output as a JSON object with keys: "bias_risk_score" (float), "potential_biases_identified" (list of strings describing concerns), "suggested_cultural_adaptations" (list of strings).

Provide ONLY the JSON object in your response."""
    llm_response_text = generate_llm_content(prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Bias Check LLM Error: {llm_response_text}")
        return {**state, "error_message": "Bias check service temporarily unavailable. Please try again shortly."}
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
    # Use `or {}` / or [] to safely handle explicitly-stored None values
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
    
    # The visual_aid_url is generated by educator_node and is seed-deterministic
    # (based on diagnosis name), so regenerating here would produce the same URL.
    # We simply read it from the education dict — no redundant call needed.
    visual_url = education.get("visual_aid_url", "")
    
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
        "visual_aid_url": visual_url,
        "visual_aid_description": education.get("visual_aid_description", "Educational medical illustration"),
        "explanation": education.get("explanation", "Explanation pending."),
        "medication": education.get("medication_info", "Medication info pending."),
        "next_steps": education.get("next_steps", ["Next steps pending."]),
        "ai_generated_image": True
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
        "patient_id": f"ANON-{str(uuid4())[:8].upper()}",
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

    logger.info("[Node] Output Formatter: Exit - Report generated.")
    return {**state, "final_report": final_report}


# --- LangGraph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("diagnosis", diagnosis_node)
workflow.add_node("triage", triage_node)
workflow.add_node("care_routing", care_routing_node)
workflow.add_node("validator", validator_node)
workflow.add_node("educator", educator_node)
workflow.add_node("bias_check", bias_check_node)
workflow.add_node("format_output", format_output_node)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "diagnosis")
workflow.add_edge("diagnosis", "triage")
workflow.add_edge("triage", "care_routing")
workflow.add_edge("care_routing", "validator")
workflow.add_edge("validator", "educator")
workflow.add_edge("educator", "bias_check")
workflow.add_edge("bias_check", "format_output")
workflow.add_edge("format_output", END)

agent_executor = workflow.compile()

# --- FastAPI Application Setup ---
app = FastAPI(
    title="SwasthyaSetu AI",
    description="AI-powered medical diagnosis and patient education system for India",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "llm_router_available": llm_router.is_available(),
        "rag_available": query_engine is not None
    }


@app.post("/diagnose")
async def diagnose(request: Request, diagnosis_request: DiagnosisRequest):
    client_ip = get_client_ip(request)
    
    # Check rate limits
    limiter_decision = diagnose_rate_limiter.check(client_ip)
    if not limiter_decision.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded. Please try again later.",
                "retry_after_seconds": limiter_decision.retry_after_seconds,
                "limit_per_minute": limiter_decision.limit_per_minute,
                "limit_per_hour": limiter_decision.limit_per_hour
            }
        )
    
    global_decision = global_diagnose_minute_limiter.check("global")
    if not global_decision.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Global rate limit exceeded. Please try again later.",
                "retry_after_seconds": global_decision.retry_after_seconds
            }
        )
    
    # Run the workflow
    try:
        initial_state = {
            "symptoms_text": diagnosis_request.symptoms,
            "location": diagnosis_request.location,
            "learn_mode": diagnosis_request.learn_mode,
            "rag_context": [],
            "initial_diagnosis": None,
            "triage_result": None,
            "routing_result": None,
            "validation_results": None,
            "patient_education": None,
            "bias_analysis": None,
            "final_report": None,
            "error_message": None,
            "diagnosis_reasoning": None,
            "diagnosis_guidelines": None
        }
        
        # Run in threadpool to avoid blocking
        result = await run_in_threadpool(agent_executor.invoke, initial_state)
        
        final_report = result.get("final_report", {})
        error_message = result.get("error_message")
        
        if error_message:
            # Log full details server-side only — do NOT expose partial AI chain output to client
            logger.error(f"Workflow error: {error_message} | partial_report keys: {list((final_report or {}).keys())}")
            return JSONResponse(
                status_code=500,
                content={"error": "The AI workflow encountered an error. Please try again shortly."}
            )
        
        # Add rate limit headers
        headers = {
            "X-RateLimit-Remaining-Minute": str(limiter_decision.remaining_minute),
            "X-RateLimit-Remaining-Hour": str(limiter_decision.remaining_hour)
        }
        
        return JSONResponse(
            content={"status": "success", "report": final_report},
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in diagnose endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
