# FastAPI backend for SwasthyaSetu
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, TypedDict
import uvicorn
import os
import time
import json
import logging
import warnings
from dotenv import load_dotenv
# Import AI workflow dependencies
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.prompt_helper import PromptHelper
import faiss
import re
from langgraph.graph import StateGraph, END
import requests

# --- Load environment variables and API key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Configure Google Generative AI ---
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Constants ---
LLM_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
PDF_DIR = "./pubmed_data/"
PDF_FILENAME = "pubmed_papers.pdf"
PDF_FILEPATH = os.path.join(PDF_DIR, PDF_FILENAME)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("swasthyasetu")

# --- Setup RAG (FAISS + LlamaIndex) ---
embed_model = None
try:
    logger.info("Initializing Gemini Embedding model...")
    embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL_NAME, api_key=GOOGLE_API_KEY)
    logger.info("Gemini Embedding model initialized.")
except Exception as e:
    logger.warning(f"Gemini Embedding failed: {e}. Falling back to HuggingFace embedding.")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    logger.info("HuggingFace Embedding model initialized.")

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

# --- Helper: Robust LLM Call ---
def generate_gemini_content_with_retry(model_name, prompt, max_retries=3, initial_delay=2):
    logger.info(f"LLM call: model={model_name}, prompt_length={len(prompt)}")
    if not GOOGLE_API_KEY:
        logger.error("Gemini API key not configured.")
        return "Error: Gemini API key not configured."
    llm = genai.GenerativeModel(model_name)
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            start = time.time()
            response = llm.generate_content(prompt)
            duration = time.time() - start
            logger.info(f"LLM response received in {duration:.2f}s (attempt {attempt+1})")
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                try:
                    return " ".join(part.text for part in response.parts if hasattr(part, 'text'))
                except Exception:
                    return "Error: Failed to extract text from Gemini response."
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
    rag_context: Optional[List[str]]
    initial_diagnosis: Optional[Dict[str, Any]]
    triage_result: Optional[Dict[str, Any]]
    routing_result: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    final_diagnosis_report: Optional[Dict[str, Any]]
    patient_education: Optional[Dict[str, Any]]
    bias_analysis: Optional[Dict[str, Any]]
    error_message: Optional[str]

agent_llm = None
if GOOGLE_API_KEY:
    try:
        agent_llm = genai.GenerativeModel(LLM_MODEL_NAME)
        agent_llm.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1))
    except Exception as e:
        agent_llm = None
else:
    agent_llm = None

def diagnostician_node(state: AgentState) -> AgentState:
    logger.info("[Node] Diagnostician: Entry")
    symptoms = state.get("symptoms_text")
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
    if not agent_llm:
        logger.error("Diagnostician failed: LLM not initialized.")
        return {**state, "error_message": "Diagnostician failed: LLM not initialized."}
    prompt = f"""Act as a medical diagnosis assistant. Based ONLY on the provided symptoms and relevant medical context (if available), generate a differential diagnosis.\n\nPatient Symptoms:\n{symptoms}{rag_context_str}\n\nInstructions:\n1. Analyze the symptoms and context.\n2. Generate a list of possible diagnoses (differentials).\n3. For each diagnosis, provide a confidence score (0.0 to 1.0) indicating your certainty based *only* on the provided information. Higher scores mean higher likelihood.\n4. Identify the most likely primary diagnosis.\n5. Structure your output as a JSON object with the following EXACT keys: \"primary_diagnosis\", \"primary_confidence\", \"alternative_diagnoses\" (which should be a list of strings).\n\nProvide ONLY the JSON object in your response."""
    llm_response_text = generate_gemini_content_with_retry(LLM_MODEL_NAME, prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Diagnostician LLM Error: {llm_response_text}")
        return {**state, "error_message": f"Diagnostician LLM Error: {llm_response_text}"}
    diagnosis_json = None
    try:
        json_match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            diagnosis_json = json.loads(json_str)
            required_keys = ["primary_diagnosis", "primary_confidence", "alternative_diagnoses"]
            if not all(key in diagnosis_json for key in required_keys):
                raise ValueError(f"Parsed JSON missing required keys: {required_keys}")
            if not isinstance(diagnosis_json["alternative_diagnoses"], list):
                raise ValueError("Parsed JSON 'alternative_diagnoses' is not a list.")
            if not isinstance(diagnosis_json["primary_confidence"], (float, int)):
                raise ValueError("Parsed JSON 'primary_confidence' is not a number.")
            logger.info("Diagnostician: Diagnosis JSON parsed successfully.")
        else:
            logger.error("Diagnostician failed: Invalid JSON response from LLM.")
            return {**state, "error_message": "Diagnostician failed: Invalid JSON response from LLM."}
    except Exception as e:
        logger.error(f"Diagnostician failed: {e}")
        return {**state, "error_message": f"Diagnostician failed: {e}"}
    logger.info("[Node] Diagnostician: Exit")
    return {**state, "initial_diagnosis": diagnosis_json, "error_message": None}

def triage_agent_node(state: AgentState) -> AgentState:
    logger.info("[Node] Triage Agent: Entry")
    diagnosis = state.get("initial_diagnosis", {})
    symptoms = state.get("symptoms_text", "")
    if not diagnosis or not symptoms:
        logger.warning("Triage Agent skipped: Missing diagnosis or symptoms.")
        return {**state, "triage_result": {"status": "Skipped", "reason": "Missing diagnosis or symptoms."}}
    if not agent_llm:
        logger.error("Triage Agent failed: LLM not initialized.")
        return {**state, "triage_result": {"status": "Failed", "reason": "LLM not initialized."}, "error_message": "Triage Agent failed: LLM not initialized."}
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
        json_match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            triage_json = json.loads(json_str)
            if not all(k in triage_json for k in ["triage_level", "next_step", "explanation"]):
                raise ValueError("Triage JSON missing required keys.")
            logger.info("Triage Agent: Triage JSON parsed successfully.")
            triage_json["status"] = "Success"
        else:
            logger.error("Triage Agent failed: Invalid JSON response from LLM.")
            triage_json = {"status": "Failed", "reason": "Invalid JSON response from LLM."}
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
    rag_context = state.get("rag_context", [])
    if not initial_diagnosis or not symptoms:
        logger.warning("Validator skipped: Missing diagnosis or symptoms.")
        return {**state, "validation_results": {"status": "Skipped", "reason": "Missing diagnosis or symptoms."}}
    if not agent_llm:
        logger.error("Validator failed: LLM not initialized.")
        return {**state, "error_message": "Validator failed: LLM not initialized."}
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
        json_match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            validation_json = json.loads(json_str)
            if not all(k in validation_json for k in ["validation_status", "critique", "missed_alternatives"]):
                raise ValueError("Validation JSON missing required keys.")
            if not isinstance(validation_json["missed_alternatives"], list):
                raise ValueError("'missed_alternatives' must be a list.")
            logger.info("Validator: Validation JSON parsed successfully.")
        else:
            validation_json = {"status": "Failed", "reason": "Invalid JSON response from LLM critique.", "critique": "", "missed_alternatives": []}
    except Exception as e:
        validation_json = {"status": "Failed", "reason": f"Unexpected error - {e}", "critique": "", "missed_alternatives": []}
    logger.info("[Node] Validator: Exit")
    return {**state, "validation_results": validation_json, "error_message": None}

def educator_node(state: AgentState) -> AgentState:
    logger.info("[Node] Educator: Entry")
    diagnosis_info = state.get("initial_diagnosis")
    rag_context = state.get("rag_context", [])
    if not diagnosis_info or not diagnosis_info.get("primary_diagnosis"):
        logger.warning("Educator skipped: Missing diagnosis.")
        return {**state, "patient_education": {"status": "Skipped", "reason": "Missing diagnosis."}}
    if not agent_llm:
        logger.error("Educator failed: LLM not initialized.")
        return {**state, "error_message": "Educator failed: LLM not initialized."}
    primary_diag = diagnosis_info.get("primary_diagnosis")
    rag_context_str = "\n---\n".join(rag_context) if rag_context else "[Not Available]"
    prompt = f"""Act as a patient educator AI. You are given a medical diagnosis and relevant context.\n\nDiagnosis: {primary_diag}\n\nRelevant Medical Context (from PubMed abstracts):\n{rag_context_str}\n\nYour Task: Generate patient education material based *only* on the provided diagnosis and context.\n1.  **Explanation:** Provide a simple, patient-friendly explanation of what '{primary_diag}' is (approx. 2-3 sentences). Avoid jargon.\n2.  **Medication Info:** Scan the 'Relevant Medical Context'. If specific medications for treating '{primary_diag}' are mentioned, list them. If not, state \"Consult your physician for medication options.\" Do NOT invent medications.\n3.  **Next Steps/Lifestyle:** Suggest 2-3 general, safe next steps or lifestyle considerations relevant to this type of condition (e.g., follow-up appointments, rest, hydration, seeking professional advice for specifics). Emphasize consulting a healthcare professional.\n4.  **Visual Placeholder:** Generate a descriptive filename for a hypothetical explanatory visual (e.g., 'Animation_showing_{primary_diag.replace(' ','_')}.mp4').\n\nProvide your output as a JSON object with the following keys:\n- \"explanation\": (string) Patient-friendly explanation.\n- \"medication_info\": (string) Mentioned medications or consultation advice.\n- \"next_steps\": (list of strings) General advice points.\n- \"visual_placeholder_filename\": (string) Generated filename for the visual.\n\nProvide ONLY the JSON object in your response."""
    llm_response_text = generate_gemini_content_with_retry(LLM_MODEL_NAME, prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Educator LLM Error: {llm_response_text}")
        return {**state, "error_message": f"Educator LLM Error: {llm_response_text}"}
    education_json = None
    try:
        json_match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            education_json = json.loads(json_str)
            if not all(k in education_json for k in ["explanation", "medication_info", "next_steps", "visual_placeholder_filename"]):
                raise ValueError("Education JSON missing required keys.")
            if not isinstance(education_json["next_steps"], list):
                raise ValueError("'next_steps' must be a list.")
            logger.info("Educator: Education JSON parsed successfully.")
        else:
            education_json = {"status": "Failed", "reason": "Invalid JSON response from educator LLM."}
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
    if not agent_llm:
        logger.error("Bias Check failed: LLM not initialized.")
        return {**state, "error_message": "Bias Check failed: LLM not initialized."}
    diagnosis_summary = f"Primary: {initial_diagnosis.get('primary_diagnosis', 'N/A')}, Confidence: {initial_diagnosis.get('primary_confidence', 'N/A')}, Alternatives: {initial_diagnosis.get('alternative_diagnoses', [])}"
    prompt = f"""Analyze the following diagnosis information for potential biases. Focus specifically on:\n1.  **Gender/racial stereotypes:** Does the diagnosis or the way it might have been reached rely on assumptions about specific genders or races?\n2.  **Socioeconomic assumptions:** Does the potential diagnosis path or suggested alternatives implicitly assume a certain socioeconomic status (e.g., access to specific tests, lifestyle factors)?\n3.  **Cultural competency:** Could the symptoms presentation or interpretation be influenced by cultural factors not accounted for? Are there potential cultural adaptations needed for communication or treatment?\n\nPatient Symptoms:\n{symptoms}\n\nAI-Generated Diagnosis Summary:\n{diagnosis_summary}\n\nInstructions:\n- Critically evaluate based on the three points above.\n- Provide a qualitative assessment. Note specific concerns if any.\n- Suggest potential cultural adaptations if relevant (e.g., language considerations, culturally sensitive explanations).\n- Assign a hypothetical bias risk score from 0.0 (very low risk) to 1.0 (high risk detected). This is subjective based on your analysis.\n- Structure your output as a JSON object with keys: \"bias_risk_score\" (float), \"potential_biases_identified\" (list of strings describing concerns), \"suggested_cultural_adaptations\" (list of strings).\n\nProvide ONLY the JSON object in your response."""
    llm_response_text = generate_gemini_content_with_retry(LLM_MODEL_NAME, prompt)
    if llm_response_text and llm_response_text.startswith("Error:"):
        logger.error(f"Bias Check LLM Error: {llm_response_text}")
        return {**state, "error_message": f"Bias Check LLM Error: {llm_response_text}"}
    bias_json = None
    try:
        json_match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            bias_json = json.loads(json_str)
            if not all(k in bias_json for k in ["bias_risk_score", "potential_biases_identified", "suggested_cultural_adaptations"]):
                raise ValueError("Bias analysis JSON missing required keys.")
            if not isinstance(bias_json["bias_risk_score"], (float, int)):
                raise ValueError("'bias_risk_score' must be a number.")
            if not isinstance(bias_json["potential_biases_identified"], list):
                raise ValueError("'potential_biases_identified' must be a list.")
            if not isinstance(bias_json["suggested_cultural_adaptations"], list):
                raise ValueError("'suggested_cultural_adaptations' must be a list.")
            logger.info("Bias Checker: Bias analysis JSON parsed successfully.")
        else:
            bias_json = {"status": "Failed", "reason": "Invalid JSON response from bias check LLM."}
    except Exception as e:
        bias_json = {"status": "Failed", "reason": f"Unexpected error - {e}"}
    logger.info("[Node] Bias Checker: Exit")
    return {**state, "bias_analysis": bias_json, "error_message": None}

def format_output_node(state: AgentState) -> AgentState:
    logger.info("[Node] Output Formatter: Entry")
    initial_diag = state.get("initial_diagnosis", {})
    triage = state.get("triage_result", {})
    routing = state.get("routing_result", {})
    validation = state.get("validation_results", {})
    education = state.get("patient_education", {})
    bias_info = state.get("bias_analysis", {})
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class SymptomInput(BaseModel):
    symptoms: str
    location: str

@app.post("/diagnose")
async def diagnose(symptom_input: SymptomInput):
    logger.info("/diagnose endpoint called.")
    if not app_graph:
        logger.error("LangGraph workflow not available.")
        return JSONResponse(content={"status": "error", "error": "LangGraph workflow not available."})
    initial_state = AgentState(
        original_input=symptom_input.symptoms,
        input_language="en",
        symptoms_text=symptom_input.symptoms,
        location=symptom_input.location,
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
        final_state = app_graph.invoke(initial_state)
        report = final_state.get("final_diagnosis_report", {})
        logger.info("Diagnosis workflow completed successfully.")
        return JSONResponse(content={"status": "success", "report": report})
    except Exception as e:
        logger.error(f"Diagnosis workflow failed: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "error": str(e)})

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
