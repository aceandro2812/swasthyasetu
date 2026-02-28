"""
SwasthyaSetu Backend Test Script
Run this to test the backend agents directly without going through the UI.
Usage: python test_backend.py
The server must be running first: .venv/Scripts/activate && uvicorn main:app --reload
"""
import requests
import json
import sys
import os
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# 1. Verify API key before hitting the server
# ──────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "").strip()
print(f"\n{'='*60}")
print("SwasthyaSetu  —  Backend Test Suite")
print(f"{'='*60}")
print(f"\n[1] Environment Check")
print(f"    GOOGLE_API_KEY present : {'YES' if api_key else 'NO'}")
if api_key:
    print("    Key format             : Present (masked)")
else:
    print("    ERROR: API key missing! Check your .env file.")

# ──────────────────────────────────────────────
# 2. Quick Gemini API check (bypasses FastAPI)
# ──────────────────────────────────────────────
print(f"\n[2] Direct Gemini API Check")
try:
    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with just the word: OK"
    )
    gemini_ok = resp.text.strip() if resp.text else ""
    print(f"    Gemini API response    : {gemini_ok!r}")
    print(f"    Status                 : {'✅ WORKING' if gemini_ok else '❌ EMPTY RESPONSE'}")
except Exception as e:
    print(f"    Status                 : ❌ FAILED — {e}")

# ──────────────────────────────────────────────
# 3. Health-check the FastAPI server
# ──────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
print(f"\n[3] FastAPI Server Health Check ({BASE_URL}/health)")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"    HTTP status            : {r.status_code}")
    print(f"    Response               : {r.json()}")
    print(f"    Status                 : {'✅ RUNNING' if r.status_code == 200 else '❌ UNEXPECTED'}")
except requests.ConnectionError:
    print(f"    Status                 : ❌ SERVER NOT RUNNING — start it first!")
    sys.exit(1)

# ──────────────────────────────────────────────
# 4. Full diagnosis workflow test
# ──────────────────────────────────────────────
print(f"\n[4] Full /diagnose Endpoint Test")
payload = {
    "symptoms": "Patient presents with high fever (103°F), chills, severe headache, body aches, and fatigue for 3 days. Recently returned from a trip to Kerala.",
    "location": "Mumbai, Maharashtra",
    "learn_mode": False
}
print(f"    Symptoms               : {payload['symptoms'][:60]}...")
print(f"    Location               : {payload['location']}")
print(f"    Sending POST to /diagnose ...")

try:
    r = requests.post(
        f"{BASE_URL}/diagnose",
        json=payload,
        timeout=120  # AI workflow can take time
    )
    print(f"    HTTP status            : {r.status_code}")
    data = r.json()

    if data.get("status") == "success":
        report = data.get("report", {})
        diag = report.get("diagnosis", {})
        triage = report.get("triage", {})
        edu = report.get("education", {})

        print(f"\n    ── Diagnosis ──────────────────────────────")
        print(f"    Primary                : {diag.get('primary')}")
        print(f"    Confidence             : {diag.get('confidence', 0)*100:.1f}%")
        print(f"    Alternatives           : {diag.get('alternatives')}")
        print(f"    Validation status      : {diag.get('validation_status')}")

        print(f"\n    ── Triage ─────────────────────────────────")
        print(f"    Urgency level          : {triage.get('level')}")
        print(f"    Next step              : {triage.get('next_step')}")

        print(f"\n    ── Education ──────────────────────────────")
        print(f"    Explanation            : {str(edu.get('explanation', ''))[:80]}...")
        print(f"    Medication             : {edu.get('medication')}")

        print(f"\n    ── Workflow ───────────────────────────────")
        print(f"    Workflow status        : {report.get('workflow_status')}")
        if report.get("error_details"):
            print(f"    Error details          : {report.get('error_details')}")

        print(f"\n    Status                 : ✅ WORKFLOW COMPLETE")
    else:
        print(f"    ❌ API returned error: {data.get('error')}")

except requests.Timeout:
    print(f"    ❌ Request timed out (>120s). The AI workflow is very slow — check LLM connectivity.")
except Exception as e:
    print(f"    ❌ Exception: {e}")

# ──────────────────────────────────────────────
# 5. Learn-mode test
# ──────────────────────────────────────────────
print(f"\n[5] Learn Mode Test")
payload_learn = {
    "symptoms": "Severe chest pain radiating to the left arm, sweating, and shortness of breath.",
    "location": "Delhi, NCR",
    "learn_mode": True
}
print(f"    Symptoms               : {payload_learn['symptoms'][:60]}...")
print(f"    Sending POST to /diagnose (learn_mode=True) ...")
try:
    r = requests.post(f"{BASE_URL}/diagnose", json=payload_learn, timeout=120)
    data = r.json()
    if data.get("status") == "success":
        report = data.get("report", {})
        print(f"    Primary diagnosis      : {report.get('diagnosis', {}).get('primary')}")
        has_reasoning = bool(report.get("reasoning"))
        has_guidelines = bool(report.get("guidelines"))
        print(f"    Reasoning present      : {'✅' if has_reasoning else '❌'}")
        print(f"    Guidelines present     : {'✅' if has_guidelines else '❌'}")
        print(f"    Workflow status        : {report.get('workflow_status')}")
        print(f"    Status                 : ✅ LEARN MODE COMPLETE")
    else:
        print(f"    ❌ API returned error: {data.get('error')}")
except Exception as e:
    print(f"    ❌ Exception: {e}")

print(f"\n{'='*60}")
print("Test complete.")
print(f"{'='*60}\n")
