# app/visionsystem/vision_client.py

import logging
import json
from typing import Dict, Optional
from PIL import Image
from config.visionconfig import vision_settings

logger = logging.getLogger(__name__)


def _build_json_schema(tooth_number: str = None) -> str:
    """Build JSON schema with actual tooth number baked in — no more placeholder copying."""
    focus = tooth_number if tooth_number else "the most symptomatic tooth"
    # Derive FDI quadrant hint from tooth number
    if tooth_number and tooth_number.isdigit():
        t = int(tooth_number)
        if 41 <= t <= 48:
            quadrant_hint = "41-48 range (lower right)"
        elif 31 <= t <= 38:
            quadrant_hint = "31-38 range (lower left)"
        elif 11 <= t <= 18:
            quadrant_hint = "11-18 range (upper right)"
        elif 21 <= t <= 28:
            quadrant_hint = "21-28 range (upper left)"
        else:
            quadrant_hint = "FDI range near " + tooth_number
    else:
        quadrant_hint = "FDI two-digit numbers only (e.g. 46, 47, 48)"

    return f"""RESPOND WITH ONLY THIS JSON:
            {{
            "teeth_visible": ["USE FDI NUMBERING ONLY - teeth in {quadrant_hint} - list only 3-5 actually visible"],
            "image_quality": "good/fair/poor",
            "focused_tooth": "{focus}",
            "caries": {{
                "present": true_or_false,
                "location": "tooth and surface, or null",
                "severity": "mild/moderate/severe or null",
                "notes": "describe dark crown areas, or null"
            }},
            "periapical_pathology": {{
                "present": true_or_false,
                "location": "tooth number, or null",
                "size_mm": null,
                "characteristics": "shape/border of radiolucency, or null",
                "notes": "periapical observations, or null"
            }},
            "bone_loss": {{
                "present": true_or_false,
                "type": "horizontal/vertical/null",
                "location": "between which teeth, or null",
                "severity": "mild/moderate/severe or null",
                "notes": "bone level description, or null"
            }},
            "root_canal_treatment": {{
                "present": true_or_false,
                "location": "tooth number, or null",
                "quality": "adequate/inadequate/null",
                "notes": "RCT appearance, or null"
            }},
            "restorations": {{
                "present": true_or_false,
                "type": "amalgam/composite/crown/null",
                "location": "tooth and surface, or null",
                "condition": "satisfactory/defective/null",
                "notes": "restoration description, or null"
            }},
            "other_abnormalities": [],
            "primary_finding": "Describe main finding on tooth {focus} based on what you see",
            "severity": "normal/mild/moderate/severe",
            "urgency": "routine/prompt/urgent/emergency",
            "image_quality_score": 0.8,
            "diagnostic_confidence": 0.7,
            "interpretation_notes": null,
            "narrative_summary": "2-3 sentence clinical summary of findings on tooth {focus}"
            }}

            RULES — MUST FOLLOW:
            - teeth_visible: FDI two-digit numbers ONLY (e.g. 46, 47, 48) — NEVER use 1-32 Universal numbering
            - Maximum 5 teeth in teeth_visible — periapical X-rays show 3-5 teeth
            - Tooth {focus} MUST be included in teeth_visible
            - If dark areas in crown → caries present=true
            - If dark halo at root tip → periapical_pathology present=true
            - primary_finding: write what YOU SEE, not a template phrase
            """


class VisionClient:
    """Unified interface to ALL vision models."""

    _instance = None
    _clients = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self):
        provider = vision_settings.VISION_MODEL_PROVIDER
        if provider not in self._clients:
            logger.info(f"🔄 Loading {provider} vision client...")
            if provider == "llava":
                from app.visionsystem.llava_client import llava_client
                self._clients["llava"] = llava_client
            elif provider == "llava_med":
                from app.visionsystem.llava_med_client import llava_med_client
                self._clients["llava_med"] = llava_med_client
            elif provider == "biomedclip":
                from app.visionsystem.biomedclip_client import biomedclip_client
                self._clients["biomedclip"] = biomedclip_client
            elif provider == "gpt4v":
                from app.visionsystem.gpt4_client import gpt4v_client
                self._clients["gpt4v"] = gpt4v_client
            elif provider == "claude":
                from app.visionsystem.claude_vision_client import claude_vision_client
                self._clients["claude"] = claude_vision_client
            elif provider == "florence":
                from app.visionsystem.florence_client import florence_client
                self._clients["florence"] = florence_client
            else:
                raise ValueError(f"Unknown vision provider: {provider}")
            logger.info(f"✅ {provider} client loaded")
        return self._clients[provider]

    def analyze_dental_radiograph(
        self,
        image: Image.Image,
        context: str = None,
        clinical_notes: str = None,
        tooth_number: str = None
    ) -> Dict:
        client = self._get_client()
        provider = vision_settings.VISION_MODEL_PROVIDER
        logger.info(f"🔍 Analyzing dental radiograph with {provider}...: Vision Model: {vision_settings.current_vision_model}")
        if context:
            logger.info(f"   📋 Context: {context[:80]}...")
        if tooth_number:
            logger.info(f"   🦷 FOCUS TOOTH: #{tooth_number}")
        if provider == "florence":
            return self._analyze_with_florence(client, image, context, tooth_number)
        elif provider == "biomedclip":
            return self._analyze_with_biomedclip(client, image, tooth_number)
        else:
            return self._analyze_with_structured_output(client, provider, image, context, clinical_notes, tooth_number)

    def _analyze_with_florence(self, client, image, context=None, tooth_number=None):
        logger.info("⚠️  Florence-2: Using simple task token (no context support)")
        try:
            response = client.analyze_image(image, "<MORE_DETAILED_CAPTION>")
            logger.info(f"Analysis complete: {len(response)} chars")
            return {"structured_findings": None, "narrative_analysis": response, "model": "florence",
                    "detailed_description": response, "pathology_summary": "Florence-2 general description",
                    "confidence_score": 0.5, "image_quality_score": 0.5}
        except Exception as e:
            return self._error_response("florence", str(e))

    def _analyze_with_biomedclip(self, client, image, tooth_number=None):
        logger.info("🔬 BiomedCLIP: Running pathology classification...")
        try:
            result = client.classify_pathology(image)
            if "error" in result:
                return self._error_response("biomedclip", result["error"])
            summary = f"BiomedCLIP: {result['prediction']} ({result['confidence']:.1%})"
            return {"structured_findings": None, "narrative_analysis": summary, "model": "biomedclip",
                    "detailed_description": client.analyze_image(image), "pathology_summary": summary,
                    "confidence_score": result["confidence"], "image_quality_score": 0.5}
        except Exception as e:
            return self._error_response("biomedclip", str(e))

    def _analyze_with_structured_output(self, client, provider, image, context=None, clinical_notes=None, tooth_number=None):
        prompt = self._build_structured_prompt(context, clinical_notes, tooth_number)
        logger.info(f"📤 Sending structured prompt to {provider}")
        logger.info(f"   Prompt length: {len(prompt)} chars")
        try:
            response = client.analyze_image(image, prompt)
            logger.info(f"LLaVA analysis complete: {len(response)} characters")
            return self._parse_structured_response(response, provider, tooth_number)
        except Exception as e:
            logger.error(f"❌ {provider} analysis failed: {e}")
            raise

    def _build_structured_prompt(self, context=None, clinical_notes=None, tooth_number=None):
        prompt = """You are an expert dental radiologist analyzing a periapical X-ray.

                MANDATORY:
                1. Respond with ONLY valid JSON - no preamble, no markdown, no explanation
                2. Periapical X-rays show 3-5 adjacent teeth — list all using FDI numbering
                3. PATHOLOGY IS EXPECTED — look carefully and report what you actually see

                READING GUIDE:
                - DARK areas (radiolucent) = PATHOLOGY: caries, abscess, bone loss
                - BRIGHT areas (radiopaque) = enamel, fillings, healthy bone
                - Caries: dark spots/shadows in tooth crown
                - Periapical abscess: dark halo around root tip
                - Bone loss: reduced bone height between teeth

                """
        if tooth_number or context:
            prompt += "CLINICAL CONTEXT:\n"
            if tooth_number:
                prompt += f"⚠️  FOCUS TOOTH: #{tooth_number} — patient has pain here. Pathology is LIKELY PRESENT.\n"
                prompt += f"   Examine tooth #{tooth_number} with extreme care.\n"
            if context:
                prompt += f"Chief Complaint: {context}\n"
            if clinical_notes and vision_settings.INCLUDE_CLINICAL_NOTES_IN_VISION_MODEL_PROMPT:
                prompt += f"Notes: {clinical_notes[:200]}\n"
            prompt += "\n"

        # KEY FIX: inject actual tooth number so model never sees "TOOTH_NUMBER_HERE"
        prompt += _build_json_schema(tooth_number)
        prompt += "\nANALYZE THE X-RAY AND RESPOND WITH JSON ONLY:"
        return prompt

    def _parse_structured_response(self, response: str, provider: str, tooth_number: str = None) -> Dict:
        logger.info(f"📥 Parsing response from {provider} ({len(response)} chars)")
        try:
            cleaned = response.strip()
            for prefix in ["```json", "```"]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            structured_data = json.loads(cleaned)

            # KEY FIX: Override focused_tooth with actual value BEFORE formatting
            # This ensures the placeholder never leaks into narrative/summary strings
            if tooth_number:
                structured_data["focused_tooth"] = tooth_number

            # Format narrative/summary using corrected data
            detailed = self._format_narrative(structured_data)
            summary = self._format_summary(structured_data)

            logger.info("✅ Successfully parsed JSON")
            logger.info(f"   Focused tooth: {structured_data.get('focused_tooth', 'N/A')}")
            logger.info(f"   Finding: {str(structured_data.get('primary_finding', 'N/A'))[:70]}...")

            return {
                "structured_findings": structured_data,
                "narrative_analysis": structured_data.get("narrative_summary", ""),
                "model": provider,
                "detailed_description": detailed,
                "pathology_summary": summary,
                "confidence_score": structured_data.get("diagnostic_confidence", 0.5),
                "image_quality_score": structured_data.get("image_quality_score", 0.5),
            }
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse failed: {e}")
            return {"structured_findings": None, "narrative_analysis": response, "model": provider,
                    "detailed_description": response, "pathology_summary": "Unable to parse structured output",
                    "confidence_score": 0.3, "image_quality_score": 0.5}

    def _format_narrative(self, data: Dict) -> str:
        parts = ["RADIOGRAPHIC ANALYSIS",
                 f"Teeth visible: {', '.join(data.get('teeth_visible', []))}",
                 f"Image quality: {data.get('image_quality', 'unknown')}"]
        if data.get("focused_tooth"):
            parts.append(f"PRIMARY FOCUS: Tooth #{data['focused_tooth']}\n")
        findings = []
        if data.get("caries", {}).get("present"):
            c = data["caries"]
            findings.append(f"CARIES: {c.get('location')}, {c.get('severity')}")
        if data.get("periapical_pathology", {}).get("present"):
            p = data["periapical_pathology"]
            findings.append(f"PERIAPICAL: Tooth {p.get('location')}")
        if data.get("bone_loss", {}).get("present"):
            b = data["bone_loss"]
            findings.append(f"BONE LOSS: {b.get('severity')} {b.get('type')}")
        if data.get("root_canal_treatment", {}).get("present"):
            findings.append(f"RCT: {data['root_canal_treatment'].get('location')}")
        if data.get("restorations", {}).get("present"):
            r = data["restorations"]
            findings.append(f"RESTORATION: {r.get('type')} at {r.get('location')}")
        for a in data.get("other_abnormalities", []):
            findings.append(f"OTHER: {a}")
        if findings:
            parts.append("\nPATHOLOGY DETECTED:")
            parts.extend([f"  ✓ {f}" for f in findings])
        else:
            parts.append("\nNo significant pathology detected")
        parts.append(f"\nPrimary Finding: {data.get('primary_finding', 'None')}")
        parts.append(f"Severity: {data.get('severity', 'unknown')}")
        parts.append(f"Urgency: {data.get('urgency', 'unknown')}")
        if data.get("narrative_summary"):
            parts.append(f"\nSummary: {data['narrative_summary']}")
        return "\n".join(parts)

    def _format_summary(self, data: Dict) -> str:
        parts = []
        if data.get("focused_tooth"):
            parts.append(f"**Focused on tooth #{data['focused_tooth']}**\n")
        caries = data.get("caries", {})
        periapical = data.get("periapical_pathology", {})
        bone = data.get("bone_loss", {})
        rct = data.get("root_canal_treatment", {})
        resto = data.get("restorations", {})
        parts.extend([
            f"CARIES: {'✓ YES - ' + str(caries.get('location', '')) if caries.get('present') else 'No'}",
            f"PERIAPICAL: {'✓ YES - ' + str(periapical.get('location', '')) if periapical.get('present') else 'No'}",
            f"BONE LOSS: {'✓ YES - ' + str(bone.get('severity', '')) if bone.get('present') else 'No'}",
            f"RCT: {'✓ YES' if rct.get('present') else 'No'}",
            f"RESTORATIONS: {'✓ YES - ' + str(resto.get('type', '')) if resto.get('present') else 'No'}",
            f"\nPrimary Finding: {data.get('primary_finding', 'None')}",
            f"Severity: {data.get('severity', 'unknown')}",
            f"Urgency: {data.get('urgency', 'unknown')}",
        ])
        return "\n".join(parts)

    def _error_response(self, model: str, error: str) -> Dict:
        return {"structured_findings": None, "narrative_analysis": f"Analysis failed: {error}",
                "model": model, "detailed_description": f"Error: {error}",
                "pathology_summary": "Analysis failed", "confidence_score": 0.0,
                "image_quality_score": 0.0, "error": error}


# Global instance
vision_client = VisionClient()