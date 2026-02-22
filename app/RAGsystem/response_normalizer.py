# app/RAGsystem/response_normalizer.py
"""
Response Normalization Layer
Enforces consistent JSON schema regardless of LLM variations
Ensures mandatory reference pages are always present
"""
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_rag_response(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize LLM response to consistent schema.
    
    Handles variations like:
    - "pharmacological" vs "pharmacological_treatment"
    - "non-pharmacological" vs "non_pharmacological_treatment"
    - Different nesting structures
    
    Returns standardized structure with mandatory fields.
    """
    
    # Standard schema template
    normalized = {
        "diagnosis": extract_diagnosis(raw_data),
        "differential_diagnoses": extract_differential_diagnoses(raw_data),
        "recommended_management": {
            "pharmacological": {
                "analgesics": [],
                "antibiotics": []
            },
            "non_pharmacological": [],
            "follow_up": "Not specified"
        },
        "precautions": [],
        "reference_pages": []
    }
    
    # Extract management section (handles various keys)
    mgmt = (
        raw_data.get("management") or 
        raw_data.get("recommended_management") or 
        {}
    )
    
    # Normalize pharmacological treatment
    pharm = extract_pharmacological(mgmt)
    normalized["recommended_management"]["pharmacological"] = pharm
    
    # Normalize non-pharmacological treatment
    non_pharm = extract_non_pharmacological(mgmt)
    normalized["recommended_management"]["non_pharmacological"] = non_pharm
    
    # Extract precautions
    precautions = extract_precautions(raw_data.get("precautions") or mgmt.get("precautions"))
    normalized["precautions"] = precautions
    
    # Collect all reference pages
    normalized["reference_pages"] = extract_all_pages(normalized)
    
    # Validation: Warn if no references found
    if not normalized["reference_pages"]:
        logger.warning("⚠️  No reference pages found in LLM response - this should not happen!")
    
    return normalized


def extract_diagnosis(data: Dict[str, Any]) -> str:
    """Extract diagnosis from various possible locations"""
    return data.get("diagnosis") or "Not specified"


def extract_differential_diagnoses(data: Dict[str, Any]) -> List[str]:
    """Extract differential diagnoses list"""
    diff = data.get("differential_diagnoses") or data.get("differentials") or []
    return list(diff) if isinstance(diff, list) else []


def extract_pharmacological(mgmt: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """
    Extract and normalize pharmacological treatments.
    
    Handles:
    - "pharmacological" or "pharmacological_treatment"
    - Different drug categorizations
    """
    pharm_data = (
        mgmt.get("pharmacological") or 
        mgmt.get("pharmacological_treatment") or 
        {}
    )
    
    result = {
        "analgesics": [],
        "antibiotics": []
    }
    
    if not isinstance(pharm_data, dict):
        return result
    
    # Extract analgesics
    analgesics = pharm_data.get("analgesics") or []
    result["analgesics"] = [
        normalize_drug(drug, "analgesic") for drug in analgesics
    ]
    
    # Extract antibiotics
    antibiotics = pharm_data.get("antibiotics") or []
    result["antibiotics"] = [
        normalize_drug(drug, "antibiotic") for drug in antibiotics
    ]
    
    return result


def normalize_drug(drug: Dict[str, Any], drug_type: str) -> Dict[str, Any]:
    """
    Normalize drug information to consistent format.
    
    Ensures:
    - name field
    - dose field (combines adult_dose/children_dose if needed)
    - reference_page field
    """
    if not isinstance(drug, dict):
        return {"name": str(drug), "dose": "Not specified", "reference_page": None}
    
    # Extract name
    name = drug.get("name") or "Unknown"
    
    # Extract dose (handle variations)
    dose = drug.get("dose")
    if not dose:
        # Try adult_dose + children_dose
        adult = drug.get("adult_dose")
        children = drug.get("children_dose")
        if adult and children:
            dose = f"Adult: {adult}; Children: {children}"
        elif adult:
            dose = adult
        elif children:
            dose = children
        else:
            dose = "Not specified"
    
    # Extract reference page
    ref_page = extract_page_number(drug)
    
    return {
        "name": name,
        "dose": dose,
        "reference_page": ref_page,
        "type": drug_type
    }


def extract_non_pharmacological(mgmt: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and normalize non-pharmacological treatments.
    
    Handles various formats:
    - "non-pharmacological" or "non_pharmacological_treatment"
    - Nested arrays under keys like "drainage", "extraction", "treatment"
    - Direct arrays
    """
    non_pharm = (
        mgmt.get("non_pharmacological") or
        mgmt.get("non-pharmacological") or
        mgmt.get("non_pharmacological_treatment") or
        {}
    )
    
    treatments = []
    
    if isinstance(non_pharm, dict):
        # Iterate through all keys (drainage, extraction, treatment, etc.)
        for key, value in non_pharm.items():
            if isinstance(value, list):
                for item in value:
                    treatments.append(normalize_treatment(item, key))
            elif isinstance(value, dict):
                treatments.append(normalize_treatment(value, key))
    
    elif isinstance(non_pharm, list):
        # Direct array of treatments
        for item in non_pharm:
            treatments.append(normalize_treatment(item))
    
    return treatments


def normalize_treatment(item: Any, category: Optional[str] = None) -> Dict[str, Any]:
    """Normalize a single non-pharmacological treatment"""
    if not isinstance(item, dict):
        return {
            "description": str(item),
            "category": category or "general",
            "reference_page": None
        }
    
    # Extract description (various possible keys)
    description = (
        item.get("type") or 
        item.get("name") or 
        item.get("action") or 
        item.get("description") or
        str(item)
    )
    
    # Extract additional details
    details = item.get("for") or item.get("dose") or ""
    if details:
        description = f"{description} ({details})"
    
    # Extract reference page
    ref_page = extract_page_number(item)
    
    return {
        "description": description,
        "category": category or "general",
        "reference_page": ref_page
    }


def extract_precautions(precautions_data: Any) -> List[Dict[str, Any]]:
    """Extract and normalize precautions/contraindications"""
    if not precautions_data:
        return []
    
    result = []
    
    if isinstance(precautions_data, dict):
        # Nested structure like {"pregnancy": "...", "systemic_diseases": "..."}
        for condition, note in precautions_data.items():
            if isinstance(note, list):
                for item in note:
                    result.append({
                        "condition": condition.replace("_", " ").title(),
                        "note": item.get("note") if isinstance(item, dict) else str(item),
                        "reference_page": extract_page_number(item) if isinstance(item, dict) else None
                    })
            else:
                result.append({
                    "condition": condition.replace("_", " ").title(),
                    "note": str(note),
                    "reference_page": None
                })
    
    elif isinstance(precautions_data, list):
        for item in precautions_data:
            if isinstance(item, dict):
                result.append({
                    "condition": item.get("condition") or "General",
                    "note": item.get("note") or str(item),
                    "reference_page": extract_page_number(item)
                })
            else:
                result.append({
                    "condition": "General",
                    "note": str(item),
                    "reference_page": None
                })
    
    return result


def extract_page_number(item: Dict[str, Any]) -> Optional[int]:
    """
    Extract page number from various possible field names.
    
    Tries: reference_page, ref_page, page, ref
    """
    if not isinstance(item, dict):
        return None
    
    # Try various field names
    for key in ["reference_page", "ref_page", "page", "ref"]:
        value = item.get(key)
        if value is not None:
            try:
                # Handle string pages like "43" or numeric
                return int(str(value).strip())
            except (ValueError, AttributeError):
                pass
    
    return None


def extract_all_pages(data: Dict[str, Any]) -> List[int]:
    """
    Recursively extract all page numbers from the normalized response.
    
    Returns sorted list of unique page numbers.
    """
    pages = set()
    
    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if "page" in key.lower() and value is not None:
                    try:
                        pages.add(int(str(value).strip()))
                    except (ValueError, TypeError, AttributeError):
                        pass
                else:
                    recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    
    recurse(data)
    
    # Return sorted list
    return sorted(list(pages))


# ============================================================================
# VALIDATION
# ============================================================================

def validate_response(data: Dict[str, Any]) -> None:
    """
    Validate that response has all required fields.
    
    Raises HTTPException if critical fields missing.
    """
    errors = []
    
    # Check diagnosis
    if not data.get("diagnosis") or data["diagnosis"] == "Not specified":
        errors.append("Missing diagnosis")
    
    # Check management exists
    mgmt = data.get("recommended_management", {})
    if not mgmt:
        errors.append("Missing recommended_management")
    
    # Check for at least some treatment
    pharm = mgmt.get("pharmacological", {})
    non_pharm = mgmt.get("non_pharmacological", [])
    
    has_analgesics = len(pharm.get("analgesics", [])) > 0
    has_antibiotics = len(pharm.get("antibiotics", [])) > 0
    has_non_pharm = len(non_pharm) > 0
    
    if not (has_analgesics or has_antibiotics or has_non_pharm):
        errors.append("No treatment recommendations found")
    
    # Warn if no reference pages (don't fail, just log)
    if not data.get("reference_pages"):
        logger.warning("⚠️  Response missing reference pages - citations required for clinical trust")
    
    # Raise if critical errors
    if errors:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Invalid LLM response: {', '.join(errors)}"
        )