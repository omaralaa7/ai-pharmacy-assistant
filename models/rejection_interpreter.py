"""
Phase 2.2 — Rejection Code Interpreter (Rule-Based + Fuzzy Matching)
AI-Driven Prior Authorization Assistant (Diploma Project)

This module translates cryptic insurance rejection codes into plain-language
explanations and actionable checklists for pharmacists.

Approach: Rule-based lookup table + fuzzy string matching
  - Primary lookup: exact code match (e.g., "DNC-004")
  - Fuzzy fallback: if pharmacist types a description instead of a code
    (e.g., "step therapy"), we find the closest match using sequence matching

Why rule-based instead of a neural network?
  For a production system with thousands of codes, an NLP model would be better.
  For this diploma MVP with 6 well-defined codes, a lookup table is:
    1. 100% accurate (no model errors)
    2. Instantly explainable (for the defense)
    3. Easy to extend (add a row to the CSV)
  The fuzzy matching adds a "smart" element beyond a simple dictionary lookup.
"""

import os
import json
import csv
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Load rejection codes from CSV
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

_rejection_codes = None


def _load_codes():
    """Load the rejection codes lookup table from CSV."""
    global _rejection_codes
    if _rejection_codes is not None:
        return

    codes_path = os.path.join(DATA_DIR, "rejection_codes.csv")
    _rejection_codes = []

    with open(codes_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse the action_checklist from JSON string to Python list
            try:
                checklist = json.loads(row["action_checklist"])
            except (json.JSONDecodeError, KeyError):
                checklist = []

            _rejection_codes.append({
                "code": row["code"].strip(),
                "raw_message": row["raw_message"].strip(),
                "plain_explanation": row["plain_explanation"].strip(),
                "action_checklist": checklist,
            })


def get_all_codes() -> list:
    """Return all rejection codes for display in the dashboard."""
    _load_codes()
    return _rejection_codes


def interpret_rejection(query: str) -> dict:
    """
    Interpret a rejection code or description and return actionable guidance.

    Parameters
    ----------
    query : str
        Either an exact code (e.g., "DNC-004") or a free-text description
        (e.g., "step therapy", "drug not covered", "lab results missing")

    Returns
    -------
    dict with keys:
        - found (bool): whether a match was found
        - code (str): the matched rejection code
        - raw_message (str): the original insurer message
        - plain_explanation (str): plain-language explanation
        - action_checklist (list[str]): step-by-step actions
        - match_type (str): "exact" or "fuzzy"
        - match_confidence (float): 0.0-1.0 confidence of fuzzy match
    """
    _load_codes()
    query = query.strip()

    if not query:
        return {"found": False, "error": "Please enter a rejection code or description."}

    # --- Attempt 1: Exact code match ---
    query_upper = query.upper()
    for code_entry in _rejection_codes:
        if code_entry["code"].upper() == query_upper:
            return {
                "found": True,
                "match_type": "exact",
                "match_confidence": 1.0,
                **code_entry,
            }

    # --- Attempt 2: Fuzzy matching against raw messages and explanations ---
    # SequenceMatcher compares two strings and returns a similarity ratio (0-1).
    # We check the query against both the raw_message and plain_explanation
    # of each code, and pick the best match.
    best_match = None
    best_score = 0.0

    query_lower = query.lower()
    for code_entry in _rejection_codes:
        # Compare against the raw insurer message
        score_raw = SequenceMatcher(
            None,
            query_lower,
            code_entry["raw_message"].lower()
        ).ratio()

        # Compare against the plain explanation
        score_plain = SequenceMatcher(
            None,
            query_lower,
            code_entry["plain_explanation"].lower()
        ).ratio()

        # Also check if the query is a substring of either field
        # (handles cases like typing "step therapy" which is part of the message)
        substring_bonus = 0.0
        if query_lower in code_entry["raw_message"].lower():
            substring_bonus = 0.3
        elif query_lower in code_entry["plain_explanation"].lower():
            substring_bonus = 0.2

        score = max(score_raw, score_plain) + substring_bonus

        if score > best_score:
            best_score = score
            best_match = code_entry

    # Only return a fuzzy match if confidence is above threshold
    if best_match and best_score >= 0.3:
        return {
            "found": True,
            "match_type": "fuzzy",
            "match_confidence": round(min(best_score, 1.0), 2),
            **best_match,
        }

    return {
        "found": False,
        "error": (
            f"No matching rejection code found for '{query}'. "
            "Try entering the exact code (e.g., DNC-004) or a keyword "
            "like 'step therapy', 'lab results', or 'quantity limit'."
        ),
    }


def generate_pa_letter(
    patient_age: int,
    drug_name: str,
    insurance_name: str,
    diagnosis_code: str,
    rejection_code: str,
    prior_drug_tried: bool = False,
    had_lab_results: bool = False,
) -> str:
    """
    Generate a draft Prior Authorization justification letter.

    This is a template-based approach — it fills in a professional letter
    template with the specific claim details and rejection context.
    For a production system, this could be enhanced with LLM generation.

    Parameters
    ----------
    patient_age : int
    drug_name : str
    insurance_name : str
    diagnosis_code : str
    rejection_code : str
    prior_drug_tried : bool
    had_lab_results : bool

    Returns
    -------
    str: A formatted PA justification letter draft
    """
    # Look up the rejection details
    rejection_info = interpret_rejection(rejection_code)

    # Map diagnosis codes to human-readable conditions
    diagnosis_map = {
        "E11.9": "Type 2 Diabetes Mellitus",
        "M06.9": "Rheumatoid Arthritis",
        "E78.5": "Hyperlipidemia",
        "I48.91": "Atrial Fibrillation",
        "F31.9": "Bipolar Disorder",
        "M79.3": "Panniculitis (Chronic Pain)",
        "J45.909": "Asthma",
    }
    condition = diagnosis_map.get(diagnosis_code, f"condition ({diagnosis_code})")

    # Build the letter
    letter = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         PRIOR AUTHORIZATION REQUEST — DRAFT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To: {insurance_name} — Prior Authorization Department
Re: Prior Authorization Request for {drug_name}
Diagnosis: {condition} (ICD-10: {diagnosis_code})
Patient Age: {patient_age}

Dear Prior Authorization Review Team,

I am writing to request prior authorization for {drug_name} for
a {patient_age}-year-old patient diagnosed with {condition}.
"""

    if rejection_info.get("found"):
        letter += f"""
This request follows the rejection of the initial claim with
code {rejection_info['code']} — "{rejection_info['raw_message']}".
"""

    if prior_drug_tried:
        letter += """
The patient has previously attempted the required step-therapy
alternative(s) as specified by the plan formulary. Documentation
of prior medication trials is attached.
"""
    else:
        letter += """
[NOTE: Please document why the standard step-therapy alternative
is not appropriate for this patient, or confirm prior drug trial.]
"""

    if had_lab_results:
        letter += """
Current laboratory results supporting the medical necessity of
this medication are attached and fall within the required time
window.
"""
    else:
        letter += """
[NOTE: Please attach recent lab results as required by the plan.]
"""

    letter += f"""
Based on the clinical evidence and the patient's treatment history,
{drug_name} is medically necessary for the management of this
patient's {condition}. I respectfully request approval of this
prior authorization.

Sincerely,
[Pharmacist Name]
[Pharmacy Name]
[NPI Number]
[Phone / Fax]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return letter
