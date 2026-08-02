"""
Phase 2 — NLP Chatbot Engine (TF-IDF + Arabic Preprocessing)
AI-Driven Insurance Chatbot (Diploma Project)

This module provides intelligent search over the Egyptian insurance knowledge base.
A pharmacist can ask questions in Arabic or English and get instant, structured answers.

How it works (for diploma defense explanation):
  1. PREPROCESSING: Normalize Arabic text (remove diacritics, normalize alef/ya)
  2. INDEXING: Build a TF-IDF matrix over all policy text chunks
  3. QUERY: When a pharmacist asks a question:
     a. Try to extract a company name and/or category from the query
     b. If found, do a direct knowledge base lookup (exact/fuzzy match)
     c. If not found, use TF-IDF cosine similarity to find best-matching chunks
  4. RESPONSE: Format the retrieved information into a clear, structured answer

Key NLP concepts used:
  - TF-IDF (Term Frequency - Inverse Document Frequency): weights words by how
    important they are to a specific document vs. the entire corpus
  - Cosine Similarity: measures how similar two text vectors are (0 = unrelated, 1 = identical)
  - Arabic NLP: diacritics removal, alef/ya normalization, stop word filtering
  - Fuzzy String Matching: handles typos and partial name matches
"""

import os
import re
import json
import sys
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------------------
# Load knowledge base
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
KB_PATH = os.path.join(PROJECT_DIR, "data", "insurance_knowledge_base.json")

_knowledge_base = None
_tfidf_vectorizer = None
_tfidf_matrix = None
_chunk_index = []  # maps matrix row -> (company_key, category)

# Arabic stop words (common words that don't carry meaning)
ARABIC_STOP_WORDS = {
    "في", "من", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "التى", "الذى", "هو", "هي", "هى", "نحن", "هم", "أن", "ان",
    "كان", "كل", "لم", "لن", "إذا", "اذا", "حتى", "حتي", "بعد", "قبل", "بين",
    "أو", "او", "لا", "ما", "و", "ف", "ب", "ل", "ك", "يتم", "يجب", "لابد",
    "أي", "اي", "فى", "الا", "إلا", "عند", "منذ", "ثم", "أما", "اما",
}

# Category keywords for intent detection
CATEGORY_KEYWORDS = {
    "نماذج الصرف": ["نماذج", "نموذج", "صرف", "روشته", "روشتة", "dispensing", "form", "forms", "prescription"],
    "المحظورات": ["محظور", "محظورات", "ممنوع", "ممنوعات", "حظر", "excluded", "prohibited", "exclusion", "ban", "banned"],
    "التحمل": ["تحمل", "copay", "co-pay", "deductible", "copayment"],
    "التشخيص": ["تشخيص", "diagnosis", "تشخيصات"],
    "صلاحية النموذج": ["صلاحية", "صلاحيه", "validity", "مدة النموذج", "form validity"],
    "صورة البطاقة": ["بطاقة", "بطاقه", "هوية", "هويه", "id", "national id", "identity"],
    "صورة الكارنية": ["كارنية", "كارنيه", "كارنيت", "card", "insurance card", "membership"],
    "الختم / إمضاء العميل": ["ختم", "إمضاء", "امضاء", "توقيع", "stamp", "signature", "sign"],
    "أقصى مدة للصرف": ["مدة", "مده", "أقصى", "اقصي", "اقصى", "duration", "maximum", "max duration", "dispensing period"],
    "الحد الأقصى": ["حد", "أقصى", "اقصى", "اقصي", "limit", "maximum limit", "max", "financial limit"],
    "التواصل للموافقات": ["تواصل", "موافقة", "موافقات", "تليفون", "هاتف", "رقم", "contact", "phone", "approval", "call"],
    "لينك الاونلاين سيستم": ["لينك", "اونلاين", "سيستم", "رابط", "online", "system", "link", "portal", "website"],
    "البدائل": ["بديل", "بدائل", "alternative", "substitut", "generic"],
    "ملاحظات": ["ملاحظات", "ملاحظه", "ملاحظة", "notes", "note", "additional"],
}

# Greetings regex pattern
GREETINGS_REGEX = re.compile(
    r"^(ايه\s*الاخبار|ازيك|ازيك\s*يا\s*دكتور|عامل\s*ايه|اخبارك|مرحبا|اهل[اأ]|أهلين|السلام\s*عليكم|السلام\s*علكم|صباح\s*الخير|مساء\s*الخير|hi|hello|hey|how\s*are\s*you)\b",
    re.IGNORECASE,
)



def _load_kb():
    """Load the knowledge base from JSON."""
    global _knowledge_base
    if _knowledge_base is not None:
        return
    with open(KB_PATH, "r", encoding="utf-8") as f:
        _knowledge_base = json.load(f)


def normalize_arabic(text):
    """
    Normalize Arabic text for better matching.

    Steps:
    1. Remove diacritics (tashkeel) — e.g., فَتْحَة → فتحة
    2. Normalize alef variants — أ إ آ → ا
    3. Normalize ta marbuta — ة → ه (for matching, not display)
    4. Normalize ya — ى → ي
    5. Lowercase Latin characters
    """
    if not text:
        return ""
    # Remove diacritics (Unicode range for Arabic diacritical marks)
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # Normalize alef variants
    text = re.sub(r"[أإآ]", "ا", text)
    # Normalize ta marbuta
    text = text.replace("ة", "ه")
    # Normalize ya
    text = text.replace("ى", "ي")
    # Lowercase
    text = text.lower()
    return text


def _build_tfidf_index():
    """Build TF-IDF index over all knowledge base chunks."""
    global _tfidf_vectorizer, _tfidf_matrix, _chunk_index
    if _tfidf_matrix is not None:
        return

    _load_kb()
    documents = []
    _chunk_index = []

    for company_key, company_data in _knowledge_base.items():
        for category, policy in company_data["policies"].items():
            # Combine all searchable text for this chunk
            chunk_text = " ".join([
                company_data["company_name"],
                company_data.get("company_name_ar", ""),
                company_data.get("company_name_en", ""),
                category,
                policy.get("category_en", ""),
                policy.get("details", ""),
                policy.get("notes", ""),
            ])
            # Normalize for matching
            chunk_text = normalize_arabic(chunk_text)
            documents.append(chunk_text)
            _chunk_index.append((company_key, category))

    # Build TF-IDF matrix
    _tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams + bigrams for better phrase matching
        stop_words=list(ARABIC_STOP_WORDS),
    )
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(documents)


def _fuzzy_match_company(query):
    """Find the best-matching company name using sub-token and fuzzy matching."""
    _load_kb()
    query_norm = normalize_arabic(query)
    words_in_query = set(re.findall(r"\w+", query_norm))

    best_match = None
    best_score = 0.0

    ignore_words = {"شركة", "شركه", "تأمين", "تأمين", "هيئة", "هيئة", "رعاية", "رعايه", "إدارة", "ادارة"}

    for company_key, company_data in _knowledge_base.items():
        name_parts = [
            company_data.get("company_name", ""),
            company_data.get("company_name_ar", ""),
            company_data.get("company_name_en", ""),
        ]

        sub_candidates = []
        for p in name_parts:
            if not p:
                continue
            sub_candidates.append(p)
            sub_candidates.extend(re.split(r"[-/\s]+", p))

        for cand in sub_candidates:
            cand_norm = normalize_arabic(cand.strip())
            if not cand_norm or len(cand_norm) < 2 or cand_norm in ignore_words:
                continue

            if cand_norm in query_norm:
                score = 0.95 if len(cand_norm) >= 3 else 0.85
                if score > best_score:
                    best_score = score
                    best_match = company_key
            else:
                for q_word in words_in_query:
                    if len(q_word) >= 3:
                        ratio = SequenceMatcher(None, q_word, cand_norm).ratio()
                        if ratio > 0.80 and ratio > best_score:
                            best_score = ratio
                            best_match = company_key

    return best_match, best_score


def _detect_category(query):
    """Detect which policy category the user is asking about."""
    query_norm = normalize_arabic(query)

    best_category = None
    best_score = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if normalize_arabic(keyword) in query_norm:
                score += 1
        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score > 0 else None


def get_all_companies():
    """Return a list of all company names for the dropdown."""
    _load_kb()
    return sorted([
        {
            "key": key,
            "name": data["company_name"],
            "name_ar": data.get("company_name_ar", ""),
            "name_en": data.get("company_name_en", ""),
            "id": data.get("company_id", ""),
        }
        for key, data in _knowledge_base.items()
    ], key=lambda x: x["name"])


def get_company_policies(company_key):
    """Return all policies for a specific company."""
    _load_kb()
    if company_key not in _knowledge_base:
        return None
    return _knowledge_base[company_key]


def get_all_categories():
    """Return all unique policy categories."""
    return list(CATEGORY_KEYWORDS.keys())


def check_exclusions(company_key, item_name):
    """
    Check if an item is in a company's exclusion list (المحظورات).

    Returns a dict with:
      - is_excluded (bool)
      - matched_text (str): the part of the exclusion list that matched
      - full_exclusions (str): the complete exclusion text
    """
    _load_kb()
    if company_key not in _knowledge_base:
        return {"is_excluded": False, "error": "Company not found"}

    policies = _knowledge_base[company_key]["policies"]
    if "المحظورات" not in policies:
        return {"is_excluded": False, "message": "No exclusion list found for this company"}

    exclusion_text = policies["المحظورات"].get("details", "")
    item_norm = normalize_arabic(item_name)
    exclusion_norm = normalize_arabic(exclusion_text)

    is_excluded = item_norm in exclusion_norm

    return {
        "is_excluded": is_excluded,
        "item_searched": item_name,
        "full_exclusions": exclusion_text,
        "matched_text": item_name if is_excluded else "",
    }


def chat_query(user_question):
    """
    Main chatbot function — process a natural language question and return
    a structured answer from the knowledge base.
    """
    _load_kb()
    _build_tfidf_index()

    if not user_question or not user_question.strip():
        return {"found": False, "error": "يرجى كتابة سؤالك أولاً / Please enter a question."}

    query = user_question.strip()
    query_norm = normalize_arabic(query)

    # --- Step 0: Check for General Greetings / Off-topic Greetings ---
    if GREETINGS_REGEX.search(query_norm):
        return {
            "found": True,
            "company_name": "مساعد التأمين الطبي",
            "category": "ترحيب",
            "category_en": "Welcome",
            "answer": "أهلاً بك دكتور! 👋\nأنا مساعد التأمين الطبي الذكي. يمكنني إجابتك فوراً عن سياسات الصرف، ضوابط المحظورات، مدة الصرف، وأرقام تواصل الموافقات لشركات التأمين.\n\nتفضل بكتابة اسم شركة التأمين وسؤالك (مثال: 'ما هي محظورات يونايتد؟' أو 'أقصى مدة صرف ويبكو').",
            "confidence": 1.0,
            "method": "direct",
        }

    # --- Step 1: Detect company name ---
    company_match, company_score = _fuzzy_match_company(query)

    # --- Step 2: Detect category ---
    category_match = _detect_category(query)

    # --- Case A: Company detected with high confidence ---
    if company_match and company_score >= 0.65:
        company_data = _knowledge_base[company_match]
        policies = company_data.get("policies", {})

        # Direct category match
        if category_match and category_match in policies:
            policy = policies[category_match]
            return {
                "found": True,
                "company_name": company_data["company_name"],
                "category": category_match,
                "category_en": policy.get("category_en", ""),
                "answer": policy.get("details", "لا توجد تفاصيل متاحة لهذا البند."),
                "notes": policy.get("notes", ""),
                "confidence": min(company_score, 1.0),
                "method": "direct" if company_score >= 0.8 else "fuzzy",
            }

        # Category requested but not directly in policies -> check related fallback categories
        if category_match:
            related_map = {
                "أقصى مدة للصرف": ["صلاحية النموذج", "الحد الأقصى"],
                "الحد الأقصى": ["أقصى مدة للصرف", "التحمل"],
                "صلاحية النموذج": ["أقصى مدة للصرف"],
            }
            fallback_category = None
            if category_match in related_map:
                for rel in related_map[category_match]:
                    if rel in policies:
                        fallback_category = rel
                        break

            if fallback_category:
                policy = policies[fallback_category]
                return {
                    "found": True,
                    "company_name": company_data["company_name"],
                    "category": f"{category_match} ({fallback_category})",
                    "category_en": policy.get("category_en", ""),
                    "answer": policy.get("details", ""),
                    "notes": f"ملاحظة: تم العثور على التفاصيل تحت بند '{fallback_category}' لشركة {company_data['company_name']}.",
                    "confidence": min(company_score, 1.0),
                    "method": "related_fallback",
                }

        # Company matched, but no specific category or category not listed -> return company policy summary
        summary_lines = []
        for cat, pol in policies.items():
            detail_preview = pol.get("details", "")[:120]
            summary_lines.append(f"• {cat}: {detail_preview}")

        cat_title = f"ملخص سياسات {company_data['company_name']}" if not category_match else f"سياسات {company_data['company_name']}"
        note_text = f"ملاحظة: لم يتضمن السجل بنداً صريحاً باسم '{category_match}' لشركة {company_data['company_name']}، وإليك البنود المتاحة:" if category_match else f"تم العثور على {len(policies)} بنود سياسة."

        return {
            "found": True,
            "company_name": company_data["company_name"],
            "category": cat_title,
            "category_en": "Policy Summary",
            "answer": "\n\n".join(summary_lines),
            "notes": note_text,
            "confidence": company_score,
            "method": "company_summary",
        }

    # --- Case B: Category detected but no company match ---
    if category_match and (not company_match or company_score < 0.65):
        return {
            "found": False,
            "category": category_match,
            "error": f"فهمت أنك تسأل عن '{category_match}'، ولكن يرجى تحديد اسم شركة التأمين المحددة في سؤالك (مثال: 'محظورات يونايتد').",
        }

    # --- Case D: Fall back to TF-IDF semantic search ---
    query_vec = _tfidf_vectorizer.transform([query_norm])
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-3:][::-1]
    results = []

    for idx in top_indices:
        if similarities[idx] < 0.20:  # Raised threshold from 0.05 to 0.20 to prevent false matches
            continue
        comp_key, cat = _chunk_index[idx]
        comp_data = _knowledge_base[comp_key]
        policy = comp_data["policies"][cat]
        results.append({
            "company_name": comp_data["company_name"],
            "category": cat,
            "category_en": policy.get("category_en", ""),
            "answer": policy.get("details", ""),
            "notes": policy.get("notes", ""),
            "confidence": float(similarities[idx]),
        })

    if results and results[0]["confidence"] >= 0.25:
        best = results[0]
        best["found"] = True
        best["method"] = "tfidf"
        best["other_results"] = results[1:] if len(results) > 1 else []
        return best

    return {
        "found": False,
        "error": "عذراً دكتور، هذا الاستفسار خارج نطاق سياسات التأمين المتاحة لدينا أو لم يتم تحديد اسم شركة التأمين بدقة.\n\nيرجى تحديد اسم شركة التأمين والسؤال عن بند محدد، مثل:\n• 'ما هي محظورات يونايتد؟'\n• 'أقصى مدة صرف لشركة ويبكو'\n• 'رقم تليفون موافقات دريم مشرق'\n• 'What are the stamp requirements for GlobeMed?'",
    }

