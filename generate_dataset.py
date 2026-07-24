"""
Phase 1 - Synthetic Data Generation
AI-Driven Prior Authorization Assistant (Diploma MVP)

Generates three datasets that mimic real insurance prior-authorization behavior:
  1. insurance_rules.csv  - the "ground truth" rules per (insurance, drug)
  2. claims.csv           - simulated patient claims, labeled approved/rejected
  3. rejection_codes.csv  - lookup table: code -> plain explanation -> checklist

Design notes:
  - Rules are randomized per insurer so the dataset isn't trivially separable.
  - claims.csv approval label is generated FROM the rules + noise, so a model
    trained on it has real signal to learn (mirrors how you'd validate against
    known ground truth before ever touching real data).
  - A moderate class imbalance is baked in on purpose (roughly 65/35), since
    real PA approval rates skew "approved" — this gives you a legitimate
    reason to talk about handling imbalance in your report/defense.
"""

import random
import json
import csv
import os
from faker import Faker

SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLAIMS = 2500

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

INSURANCE_COMPANIES = [
    "CareFirst", "BlueShield-X", "MediTrust", "HealthPlus", "UnityCare",
]

# (drug_name, diagnosis_code, category, common step-therapy alternative)
DRUGS = [
    ("Ozempic",      "E11.9",  "diabetes",      "Metformin"),
    ("Metformin",    "E11.9",  "diabetes",      None),
    ("Humira",       "M06.9",  "autoimmune",    "Methotrexate"),
    ("Methotrexate", "M06.9",  "autoimmune",    None),
    ("Lipitor",      "E78.5",  "cardiac",       "Atorvastatin-generic"),
    ("Eliquis",      "I48.91", "cardiac",       "Warfarin"),
    ("Warfarin",     "I48.91", "cardiac",       None),
    ("Abilify",      "F31.9",  "mental_health", "Risperidone"),
    ("Risperidone",  "F31.9",  "mental_health", None),
    ("OxyContin",    "M79.3",  "pain",          "Ibuprofen"),
    ("Ibuprofen",    "M79.3",  "pain",          None),
    ("Advair",       "J45.909","respiratory",   "Albuterol"),
]

# ---------------------------------------------------------------------------
# Rejection code lookup table (fixed, hand-authored — this is the "knowledge
# base" the NLP interpreter module will use in Phase 2)
# ---------------------------------------------------------------------------

REJECTION_CODES = [
    {
        "code": "DNC-004",
        "raw_message": "Drug Not Covered - Step Therapy Required",
        "plain_explanation": "Patient must try a cheaper, equally effective alternative before this drug is approved.",
        "action_checklist": [
            "Confirm which alternative drug is required by the payer",
            "Check patient history for prior attempts on that drug",
            "If not tried, discuss step therapy alternative with prescriber",
        ],
    },
    {
        "code": "LAB-011",
        "raw_message": "Missing Required Clinical Documentation",
        "plain_explanation": "The insurer needs recent lab results before approving this medication.",
        "action_checklist": [
            "Request most recent lab results from patient's provider",
            "Confirm results are within the required time window (usually 3 months)",
            "Attach lab results to the PA request",
        ],
    },
    {
        "code": "QTY-022",
        "raw_message": "Quantity Exceeds Plan Limit Without Authorization",
        "plain_explanation": "The requested days' supply is above what the plan allows without prior authorization.",
        "action_checklist": [
            "Check the plan's max days-supply-without-PA limit",
            "Reduce quantity to the allowed limit, or submit a formal PA request",
        ],
    },
    {
        "code": "DX-007",
        "raw_message": "Diagnosis Code Not Matched to Approved Indication",
        "plain_explanation": "The diagnosis submitted doesn't match an indication this insurer covers for this drug.",
        "action_checklist": [
            "Verify the diagnosis code entered matches the patient's chart",
            "Contact prescriber if an updated/more specific code is needed",
        ],
    },
    {
        "code": "AGE-014",
        "raw_message": "Patient Age Outside Approved Range",
        "plain_explanation": "This drug is restricted to a specific age range under the patient's plan.",
        "action_checklist": [
            "Confirm patient age against plan's approved age range for this drug",
            "Escalate to prescriber for a formal exception request if clinically justified",
        ],
    },
    {
        "code": "GEN-099",
        "raw_message": "General Rejection - Manual Review Required",
        "plain_explanation": "No automated reason was given; this claim needs manual review by the insurer.",
        "action_checklist": [
            "Call payer's PA line for the specific reason",
            "Document the call reference number for follow-up",
        ],
    },
]

REJECTION_CODE_BY_CAUSE = {
    "step_therapy": "DNC-004",
    "lab_results": "LAB-011",
    "days_supply": "QTY-022",
    "none_matched": "GEN-099",
}


def build_insurance_rules():
    """Randomize a rule set per (insurance, drug) pair."""
    rules = []
    for insurer in INSURANCE_COMPANIES:
        for drug_name, diagnosis_code, category, alt_drug in DRUGS:
            requires_step_therapy = bool(alt_drug) and random.random() < 0.55
            requires_lab_results = category in ("cardiac", "diabetes") and random.random() < 0.5
            max_days_supply = random.choice([30, 60, 60, 90])

            rules.append({
                "insurance_name": insurer,
                "drug_name": drug_name,
                "diagnosis_code": diagnosis_code,
                "requires_step_therapy": requires_step_therapy,
                "required_prior_drug": alt_drug if requires_step_therapy else "",
                "requires_lab_results": requires_lab_results,
                "max_days_supply_without_pa": max_days_supply,
            })
    return rules


def rules_lookup(rules):
    return {(r["insurance_name"], r["drug_name"]): r for r in rules}


def generate_claim(claim_id, rules_by_key):
    insurer = random.choice(INSURANCE_COMPANIES)
    drug_name, diagnosis_code, category, alt_drug = random.choice(DRUGS)
    rule = rules_by_key[(insurer, drug_name)]

    patient_age = random.randint(18, 85)
    days_supply_requested = random.choice([14, 30, 30, 60, 90])

    # Simulate whether patient actually meets each requirement
    # (weighted so most patients DO meet requirements - mirrors real PA
    # approval rates, where most claims go through cleanly)
    tried_prior_drug = random.random() < 0.75 if rule["requires_step_therapy"] else True
    had_lab_results = random.random() < 0.8 if rule["requires_lab_results"] else True
    within_days_limit = days_supply_requested <= rule["max_days_supply_without_pa"]

    # Determine approval + failure cause (first failing condition wins)
    cause = "none_matched"
    approved = True

    if rule["requires_step_therapy"] and not tried_prior_drug:
        approved = False
        cause = "step_therapy"
    elif rule["requires_lab_results"] and not had_lab_results:
        approved = False
        cause = "lab_results"
    elif not within_days_limit:
        approved = False
        cause = "days_supply"

    # Add realistic noise: 6% random flip in either direction
    if random.random() < 0.06:
        approved = not approved
        if not approved and cause == "none_matched":
            cause = "none_matched"  # unexplained rejection -> manual review code
        if approved:
            cause = None

    rejection_code = REJECTION_CODE_BY_CAUSE.get(cause) if not approved else ""

    return {
        "claim_id": claim_id,
        "patient_age": patient_age,
        "drug_name": drug_name,
        "insurance_name": insurer,
        "diagnosis_code": diagnosis_code,
        "prior_drug_required": rule["required_prior_drug"],
        "prior_drug_tried": tried_prior_drug if rule["requires_step_therapy"] else "",
        "lab_results_required": rule["requires_lab_results"],
        "had_lab_results": had_lab_results if rule["requires_lab_results"] else "",
        "days_supply_requested": days_supply_requested,
        "max_days_supply_without_pa": rule["max_days_supply_without_pa"],
        "approved": int(approved),
        "rejection_code": rejection_code,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("Generating insurance_rules.csv ...")
    rules = build_insurance_rules()
    rules_by_key = rules_lookup(rules)
    write_csv(
        os.path.join(OUTPUT_DIR, "insurance_rules.csv"),
        rules,
        fieldnames=list(rules[0].keys()),
    )

    print(f"Generating claims.csv ({N_CLAIMS} rows) ...")
    claims = [generate_claim(i + 1, rules_by_key) for i in range(N_CLAIMS)]
    write_csv(
        os.path.join(OUTPUT_DIR, "claims.csv"),
        claims,
        fieldnames=list(claims[0].keys()),
    )

    print("Generating rejection_codes.csv ...")
    rejection_rows = []
    for r in REJECTION_CODES:
        row = dict(r)
        row["action_checklist"] = json.dumps(r["action_checklist"])
        rejection_rows.append(row)
    write_csv(
        os.path.join(OUTPUT_DIR, "rejection_codes.csv"),
        rejection_rows,
        fieldnames=["code", "raw_message", "plain_explanation", "action_checklist"],
    )

    # Quick sanity summary
    approved_count = sum(c["approved"] for c in claims)
    rejected_count = N_CLAIMS - approved_count
    print("\n--- Summary ---")
    print(f"Total claims: {N_CLAIMS}")
    print(f"Approved: {approved_count} ({approved_count / N_CLAIMS:.1%})")
    print(f"Rejected: {rejected_count} ({rejected_count / N_CLAIMS:.1%})")

    cause_counts = {}
    for c in claims:
        if not c["approved"]:
            cause_counts[c["rejection_code"]] = cause_counts.get(c["rejection_code"], 0) + 1
    print("Rejection code breakdown:", cause_counts)
    print(f"\nFiles written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
