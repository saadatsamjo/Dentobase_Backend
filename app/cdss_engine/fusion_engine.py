# app/cdss_engine/fusion_engine.py
"""
CDSS Fusion Engine - Updated with:
1. Structured vision output support
2. Scientific confidence scoring
3. Tooth number enforcement in diagnosis
4. Enhanced prompting for accuracy
"""
import logging
import time
from typing import Dict, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cdss_engine.schemas import (
    CDSSResponse,
    ClinicalRecommendation,
    ImageObservation,
    NoImageProvided,
    PatientHistory,
    RetrievedKnowledge,
)
from app.RAGsystem.llm_client import llm_client
from app.RAGsystem.retriever import RetrieverFactory
from app.system_models.clinical_note_model.clinical_note_model import ClinicalNote
from app.system_models.patient_model.patient_model import Patient
from app.visionsystem.image_processor import ImageProcessor
from app.visionsystem.vision_client import vision_client
from config.ragconfig import rag_settings
from config.visionconfig import vision_settings
from app.cdss_engine.tooth_validator import tooth_validator

logger = logging.getLogger(__name__)


class CDSSFusionEngine:
    """
    Production-ready CDSS with:
    - Structured vision output
    - Scientific confidence scoring
    - Tooth number enforcement
    - Enhanced clinical accuracy
    """

    def __init__(self):
        self.retriever_factory = RetrieverFactory()
        logger.info("🏥 CDSS Fusion Engine initialized")

    async def fetch_patient_history_with_notes(
        self, patient_id: int, db: AsyncSession, num_notes: int = 3
    ) -> tuple[PatientHistory, List[dict]]:
        """Fetch patient demographics + recent clinical notes."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 1: FETCH PATIENT HISTORY + CLINICAL NOTES")
        logger.info(f"{'='*70}")
        logger.info(f"📋 Patient ID: {patient_id}")

        try:
            result = await db.execute(select(Patient).where(Patient.id == patient_id))
            patient = result.scalar_one_or_none()

            if not patient:
                logger.error(f"❌ Patient {patient_id} not found")
                raise ValueError(f"Patient with ID {patient_id} not found")

            # Calculate age
            age = None
            if patient.dob:
                from datetime import datetime
                today = datetime.now().date()
                age = (
                    today.year
                    - patient.dob.year
                    - ((today.month, today.day) < (patient.dob.month, patient.dob.day))
                )

            logger.info(f"✅Patient: {patient.first_name} {patient.last_name}")
            logger.info(f"   Age: {age}, Gender: {patient.gender}")

            # Fetch clinical notes
            logger.info(f"\n📝 Fetching clinical notes (up to {num_notes})...")

            notes_query = (
                select(ClinicalNote)
                .join(ClinicalNote.encounter)
                .where(ClinicalNote.encounter.has(patient_id=patient_id))
                .order_by(desc(ClinicalNote.created_at))
                .limit(num_notes)
            )

            notes_result = await db.execute(notes_query)
            clinical_notes_raw = notes_result.scalars().all()

            formatted_notes = []
            if clinical_notes_raw:
                logger.info(f"✅Retrieved {len(clinical_notes_raw)} clinical note(s):")
                for i, note in enumerate(clinical_notes_raw, 1):
                    formatted_notes.append(
                        {
                            "date": note.created_at.strftime("%Y-%m-%d %H:%M"),
                            "type": note.note_type,
                            "content": note.content,
                            "author_id": note.author_id,
                        }
                    )
                    logger.info(f"   [{i}] {note.created_at.strftime('%Y-%m-%d')} - {note.note_type}")
                    logger.info(f"       {note.content[:100]}...")
            else:
                logger.info(f"   No previous clinical notes found")

            patient_history = PatientHistory(
                patient_id=str(patient.id),
                age=age,
                gender=patient.gender,
                chief_complaint=None,
                medical_history=[],
                current_medications=[],
                allergies=[],
                previous_dental_work=None,
                symptoms_duration=None,
                pain_level=None,
            )

            return patient_history, formatted_notes

        except Exception as e:
            logger.error(f"❌ Database error: {e}", exc_info=True)
            raise

    def analyze_radiograph(
        self,
        image_bytes: bytes,
        clinical_complaint: str,
        patient_info: str,
        tooth_numbers: Optional[List[str]] = None,
        clinical_notes: Optional[List[dict]] = None,
    ) -> ImageObservation:
        """
        Analyze radiograph with structured output and tooth number focus.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 2: ANALYZE RADIOGRAPH")
        logger.info(f"{'='*70}")
        logger.info(f"🔍 Vision Model Provider: {vision_settings.VISION_MODEL_PROVIDER}")
        logger.info(f"🔍 Vision Model: {vision_settings.current_vision_model}")
        logger.info(f"📝 Chief Complaint: {clinical_complaint}")
        if tooth_numbers:
            logger.info(f"🦷 Focus Teeth: {', '.join(tooth_numbers)}")
            logger.info(f"CRITICAL: Analysis must focus on tooth/teeth {tooth_numbers[0]}")

        image = ImageProcessor.preprocess_image(image_bytes)
        logger.info(f"✅ Image preprocessed: {image.size}")

        # Build context
        enhanced_context = clinical_complaint
        if tooth_numbers:
            tooth_list = ", ".join(tooth_numbers)
            enhanced_context = f"{clinical_complaint}. Focus on tooth/teeth: {tooth_list}"

        # Build clinical notes text
        clinical_notes_text = None
        if clinical_notes:
            notes_parts = []
            for note in clinical_notes:
                notes_parts.append(f"[{note['type']}]: {note['content']}")
            clinical_notes_text = " | ".join(notes_parts)

        # Log context
        logger.info(f"\n{'─'*70}")
        logger.info(f"📤 CONTEXT SENT TO VISION MODEL:")
        logger.info(f"{'─'*70}")
        logger.info(f"Enhanced Context: {enhanced_context}")
        if clinical_notes_text:
            logger.info(f"Clinical Notes: {clinical_notes_text[:200]}...")
        if tooth_numbers:
            logger.info(f"PRIMARY FOCUS TOOTH: #{tooth_numbers[0]}")
        logger.info(f"{'─'*70}\n")

        # Call vision client with tooth number
        result = vision_client.analyze_dental_radiograph(
            image,
            context=enhanced_context,
            clinical_notes=clinical_notes_text,
            tooth_number=tooth_numbers[0] if tooth_numbers else None
        )

        # Extract scores
        confidence_score = result.get("confidence_score", 0.5)
        quality_score = result.get("image_quality_score", 0.5)

        # Derive confidence level from scores
        overall_score = (confidence_score + quality_score) / 2
        if overall_score >= 0.8:
            confidence_level = "high"
        elif overall_score >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        logger.info(f"✅ Analysis complete")
        logger.info(f"   Vision model Provider: {vision_settings.VISION_MODEL_PROVIDER}")
        logger.info(f"   Vision model: {vision_settings.current_vision_model}")
        logger.info(f"   Image Quality Score: {quality_score:.2f}")
        logger.info(f"   Diagnostic Confidence: {confidence_score:.2f}")
        logger.info(f"   Overall Confidence: {confidence_level}")



        # Log structured findings if available
        if result.get("structured_findings"):
            logger.info(f"\n{'─'*70}")
            logger.info(f"📊 STRUCTURED IMAGE ANALYSIS:")
            logger.info(f"{'─'*70}")
            findings = result["structured_findings"]
            logger.info(f"Focused Tooth: #{findings.get('focused_tooth', 'Not specified')}")
            logger.info(f"Teeth Visible: {', '.join(findings.get('teeth_visible', []))}")
            logger.info(f"Image Quality: {findings.get('image_quality', 'unknown')}")
            logger.info(f"\nPathology Findings:")
            logger.info(f"  Caries: {'Yes' if findings.get('caries', {}).get('present') else 'No'}")
            if findings.get('caries', {}).get('present'):
                logger.info(f"    Location: {findings['caries'].get('location')}")
            logger.info(f"  Periapical: {'Yes' if findings.get('periapical_pathology', {}).get('present') else 'No'}")
            if findings.get('periapical_pathology', {}).get('present'):
                logger.info(f"    Location: Tooth {findings['periapical_pathology'].get('location')}")
            logger.info(f"  Bone Loss: {'Yes' if findings.get('bone_loss', {}).get('present') else 'No'}")
            logger.info(f"\nPrimary Finding: {findings.get('primary_finding', 'None')}")
            logger.info(f"Severity: {findings.get('severity', 'unknown')}")
            logger.info(f"Urgency: {findings.get('urgency', 'unknown')}")
            logger.info(f"{'─'*70}\n")

        return ImageObservation(
            structured_findings=result.get("structured_findings"),
            raw_description=result["detailed_description"],
            pathology_summary=result["pathology_summary"],
            model_used=vision_settings.current_vision_model,
            focused_tooth=tooth_numbers[0] if tooth_numbers else None,
            image_quality_score=quality_score,
            diagnostic_confidence=confidence_score,
            overall_confidence=confidence_level,
        )

    def retrieve_knowledge(
        self,
        clinical_complaint: str,
        image_obs=None,  # ImageObservation | None
    ):
        """
        Retrieve knowledge with a clean, pathology-specific query.
        
        FIX: Do NOT use pathology_summary (may contain TOOTH_NUMBER_HERE junk).
        Instead, build a clean query from structured_findings directly.
        """

        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 3: RETRIEVE CLINICAL KNOWLEDGE")
        logger.info(f"{'='*70}")

        # ── Build a clean, specific retrieval query ──────────────────────────
        query_parts = []

        # 1. Extract actual pathologies from structured findings (clean, no placeholders)
        if image_obs and image_obs.structured_findings:
            sf = image_obs.structured_findings
            tooth = image_obs.focused_tooth or sf.get("focused_tooth", "")
            
            pathologies = []
            if sf.get("caries", {}).get("present"):
                severity = sf["caries"].get("severity", "")
                pathologies.append(f"{severity} dental caries".strip())

            if sf.get("periapical_pathology", {}).get("present"):
                pathologies.append("periapical abscess periapical pathology treatment")

            if sf.get("bone_loss", {}).get("present"):
                bl = sf["bone_loss"]
                pathologies.append(f"{bl.get('severity', '')} {bl.get('type', '')} periodontal bone loss".strip())

            if sf.get("root_canal_treatment", {}).get("present"):
                pathologies.append("root canal treatment endodontic")

            urgency = sf.get("urgency", "")
            severity = sf.get("severity", "")

            if pathologies:
                query_parts.append(f"Dental diagnosis and treatment for: {', '.join(pathologies)}")
                if tooth and tooth not in ("TOOTH_NUMBER_HERE", "the most symptomatic tooth"):
                    query_parts.append(f"Tooth {tooth}")
                if urgency:
                    query_parts.append(f"Urgency: {urgency}")
            else:
                # Fallback if no specific pathology
                query_parts.append(f"Dental pain management lower jaw tooth")

        else:
            # No image - use complaint only
            query_parts.append(clinical_complaint)

        enhanced_query = ". ".join(query_parts)

        logger.info(f"🔍 Building retrieval query:")
        logger.info(f"   Query: {enhanced_query}")
        logger.info(f"\n{'─'*70}")
        logger.info(f"📤 QUERY SENT TO RETRIEVER:")
        logger.info(f"{'─'*70}")
        logger.info(f"{enhanced_query}")
        logger.info(f"{'─'*70}\n")
        logger.info(f"⚙️  Configuration:")
        logger.info(f"   Retriever: {rag_settings.RETRIEVER_TYPE}")
        logger.info(f"   K: {rag_settings.RETRIEVAL_K}")

        try:
            retriever = self.retriever_factory.get_retriever()
            docs = retriever.invoke(enhanced_query)
            logger.info(f"✅Retrieved {len(docs)} chunks")

            if len(docs) == 0:
                logger.warning("⚠️  ZERO CHUNKS RETRIEVED")
                return [], []

            knowledge = []
            available_pages = set()

            logger.info(f"📖 Retrieved Knowledge:")
            for i, doc in enumerate(docs, 1):
                pages = doc.metadata.get("pages", [])
                if not pages and "page" in doc.metadata:
                    pages = [doc.metadata["page"] + 1]
                source = doc.metadata.get("source", "Guidelines")
                preview = doc.page_content[:80].replace("\n", " ")
                logger.info(f"   [{i}] Pages {pages}: {preview}...")
                knowledge.append(
                    RetrievedKnowledge(
                        content=doc.page_content,
                        pages=pages,
                        relevance_score=doc.metadata.get("score"),
                        source=source,
                    )
                )
                available_pages.update(pages)

            sorted_pages = sorted(list(available_pages))
            logger.info(f"\n📄 Total unique pages retrieved: {sorted_pages}")
            return knowledge, sorted_pages

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}", exc_info=True)
            return [], []






    def _calculate_confidence(
        self,
        image_obs: Optional[ImageObservation],
        knowledge: List[RetrievedKnowledge],
        clinical_notes: List[dict],
        tooth_number_specified: bool
    ) -> tuple[float, dict]:
        """
        Calculate scientific confidence score based on multiple factors.
        
        Returns:
            (overall_score, factor_breakdown)
        """
        factors = {}
        
        # 1. Image quality factor (0-1)
        if image_obs:
            factors["image_quality"] = image_obs.image_quality_score
            factors["radiographic_confidence"] = image_obs.diagnostic_confidence
        else:
            factors["image_quality"] = 0.0
            factors["radiographic_confidence"] = 0.0
        
        # 2. Knowledge availability factor (0-1)
        if len(knowledge) >= 5:
            factors["knowledge_availability"] = 1.0
        elif len(knowledge) >= 3:
            factors["knowledge_availability"] = 0.8
        elif len(knowledge) >= 1:
            factors["knowledge_availability"] = 0.5
        else:
            factors["knowledge_availability"] = 0.2
        
        # 3. Clinical context factor (0-1)
        clinical_context_score = 0.5  # baseline
        if clinical_notes:
            clinical_context_score += 0.3
        if tooth_number_specified:
            clinical_context_score += 0.2
        factors["clinical_context"] = min(clinical_context_score, 1.0)
        
        # 4. Consistency factor (0-1) - Do findings align?
        # If we have both image and knowledge, they should agree
        if image_obs and knowledge:
            # This is simplified - could be more sophisticated
            factors["consistency"] = 0.8
        else:
            factors["consistency"] = 0.6
        
        # Calculate weighted overall score
        weights = {
            "image_quality": 0.25,
            "radiographic_confidence": 0.30,
            "knowledge_availability": 0.25,
            "clinical_context": 0.10,
            "consistency": 0.10
        }
        
        overall_score = sum(factors[k] * weights[k] for k in factors.keys())
        
        return overall_score, factors

    def fuse_and_reason(
        self,
        patient_history: PatientHistory,
        clinical_notes: List[dict],
        clinical_complaint: str,
        image_obs: Optional[ImageObservation],
        knowledge: List[RetrievedKnowledge],
        available_pages: List[int],
        tooth_numbers: Optional[List[str]] = None
    ) -> ClinicalRecommendation:
        """Generate recommendation with tooth number enforcement."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 4: GENERATE STRUCTURED RECOMMENDATION")
        logger.info(f"{'='*70}")
        logger.info(f"🤖 LLM Provider: {rag_settings.LLM_PROVIDER}")
        logger.info(f"🤖 LLM Model: {rag_settings.current_llm_model}")

        knowledge_available = len(knowledge) > 0

        # Calculate scientific confidence
        confidence_score, confidence_factors = self._calculate_confidence(
            image_obs,
            knowledge,
            clinical_notes,
            bool(tooth_numbers)
        )

        # Derive confidence level
        if confidence_score >= 0.75:
            confidence_level = "high"
        elif confidence_score >= 0.45:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        logger.info(f"\n📊 CONFIDENCE CALCULATION:")
        logger.info(f"   Overall Score: {confidence_score:.3f}")
        logger.info(f"   Level: {confidence_level}")
        logger.info(f"   Factors:")
        for factor, score in confidence_factors.items():
            logger.info(f"     - {factor}: {score:.3f}")

        # Build contexts
        patient_ctx = f"""Patient Demographics:
                - ID: {patient_history.patient_id}
                - Age: {patient_history.age or 'Unknown'}, Gender: {patient_history.gender or 'Unknown'}

                Current Complaint: {clinical_complaint}"""

        if clinical_notes:
            patient_ctx += f"\n\nRecent Clinical Notes ({len(clinical_notes)}):\n"
            for i, note in enumerate(clinical_notes, 1):
                patient_ctx += f"[{i}] {note['date']} - {note['type']}: {note['content']}\n"

        # CRITICAL: Emphasize tooth number in image context
        image_ctx = ""
        if image_obs:
            image_ctx = f"""Radiographic Analysis ({image_obs.model_used}):"""
            
            if image_obs.focused_tooth:
                image_ctx += f"\n\n CRITICAL - FOCUSED ANALYSIS ON TOOTH #{image_obs.focused_tooth}"
                image_ctx += f"\nAll radiographic findings relate to tooth #{image_obs.focused_tooth} unless explicitly stated otherwise.\n"
            
            if image_obs.structured_findings:
                import json
                image_ctx += f"\n\nSTRUCTURED FINDINGS:\n{json.dumps(image_obs.structured_findings, indent=2)}\n"
            
            image_ctx += f"\n\nNARRATIVE ANALYSIS:\n{image_obs.raw_description}\n"
            image_ctx += f"\nKEY PATHOLOGY:\n{image_obs.pathology_summary}"
            image_ctx += f"\n\nImage Quality: {image_obs.image_quality_score:.2f}/1.0"
            image_ctx += f"\nDiagnostic Confidence: {image_obs.diagnostic_confidence:.2f}/1.0"
        else:
            image_ctx = "No radiograph provided."

        # Guidelines context
        if knowledge_available:
            guidelines_ctx = "Clinical Guidelines:\n\n"
            for i, k in enumerate(knowledge, 1):
                pages = f"Pages {k.pages}" if k.pages else ""
                guidelines_ctx += f"[{i}] {pages}\n{k.content}\n\n"
        else:
            guidelines_ctx = "⚠️ No clinical guidelines retrieved."

        # Log contexts
        logger.info(f"\n{'─'*70}")
        logger.info(f"📤 CONTEXT SENT TO LLM:")
        logger.info(f"{'─'*70}")
        logger.info(f"\n=== PATIENT CONTEXT ===")
        logger.info(f"{patient_ctx[:300]}...")
        logger.info(f"\n=== IMAGE CONTEXT ===")
        logger.info(f"{image_ctx[:500]}...")
        logger.info(f"\n=== KNOWLEDGE CONTEXT ===")
        logger.info(f"{guidelines_ctx[:300]}...")
        logger.info(f"{'─'*70}\n")

        # ENHANCED QUERY WITH TOOTH NUMBER ENFORCEMENT
        enhanced_query = clinical_complaint
        if tooth_numbers:
            enhanced_query += f"\n\nCRITICAL INSTRUCTION: The diagnosis MUST reference tooth #{tooth_numbers[0]}. Do NOT diagnose a different tooth number."

        try:
            logger.info(f"⚙️  Calling LLM with structured output...")

            result = llm_client.generate_clinical_recommendation(
                patient_context=patient_ctx,
                image_findings=image_ctx,
                retrieved_knowledge=guidelines_ctx,
                query=enhanced_query,
                knowledge_available=knowledge_available,
                available_pages=available_pages,
            )

            logger.info(f"✅Structured recommendation generated")
            logger.info(f"   Diagnosis: {result['diagnosis'][:100]}...")
            logger.info(f"   Reference pages: {result['reference_pages']}")

            return ClinicalRecommendation(
                diagnosis=result["diagnosis"],
                differential_diagnoses=result["differential_diagnoses"],
                recommended_management=result["recommended_management"],
                reference_pages=result["reference_pages"],
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                confidence_factors=confidence_factors,
                llm_provider=rag_settings.LLM_PROVIDER,
            )

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)

            return ClinicalRecommendation(
                diagnosis=f"Error: {str(e)}",
                differential_diagnoses=[],
                recommended_management="Consult supervising dentist.",
                reference_pages=[],
                confidence_score=0.0,
                confidence_level="low",
                confidence_factors={"error": 0.0},
                llm_provider=rag_settings.LLM_PROVIDER,
            )


    async def provide_final_recommendation(
        self,
        user_id: int,
        patient_id: int,
        chief_complaint: str,
        db: AsyncSession,
        image_bytes: Optional[bytes] = None,
        tooth_numbers: Optional[List[str]] = None,
        
    ) -> CDSSResponse:
        """Main CDSS pipeline with NoImageProvided support."""
        start_time = time.time()

        logger.info(f"\n{'#'*70}")
        logger.info(f"# CDSS PIPELINE - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'#'*70}")
        logger.info(f"User: (Dentist ID) {user_id}")
        logger.info(f"Patient: {patient_id}")
        logger.info(f"Complaint: {chief_complaint}")
        if tooth_numbers:
            logger.info(f"Tooth Numbers: {', '.join(tooth_numbers)}")
        logger.info(f"Config: {rag_settings.RETRIEVER_TYPE}")
        if rag_settings.RETRIEVER_TYPE == "similarity_score_threshold":
            logger.info(f"Similarity threshold: {rag_settings.SIMILARITY_THRESHOLD}")
        elif rag_settings.RETRIEVER_TYPE == "mmr":
            logger.info(f"Diversity (Lambda multiplier): {rag_settings.LAMBDA_MULT}")
            logger.info(f"Fetch K (number of results to return): {rag_settings.FETCH_K}")

        # Pipeline - fetch patient history and notes
        patient_history, clinical_notes = await self.fetch_patient_history_with_notes(
            patient_id, db, 3
        )
        patient_history.chief_complaint = chief_complaint

        # CRITICAL: Initialize both variables
        image_obs = None
        no_image_message = None
        
        # Try image analysis if image provided
        if image_bytes:
            try:
                image_obs = self.analyze_radiograph(
                    image_bytes,
                    chief_complaint,
                    f"{patient_history.age}y {patient_history.gender}",
                    tooth_numbers,
                    clinical_notes,
                )
            except Exception as e:
                logger.error(f"❌ Image analysis failed: {e}")
                # image_obs stays None, will create NoImageProvided below

        # CRITICAL FIX: Create NoImageProvided if no successful image analysis
        if image_obs is None:
            no_image_message = NoImageProvided(
                message="No radiograph image was provided for this consultation" if not image_bytes else "Image analysis failed",
                image_required=False,
            )
            logger.info(f"  \n⚠️  No image analysis available")

        knowledge, available_pages = self.retrieve_knowledge(chief_complaint, image_obs)

        recommendation = self.fuse_and_reason(
            patient_history, clinical_notes, chief_complaint, image_obs, knowledge, available_pages, tooth_numbers
        )

        # ============================================================
        # TOOTH NUMBER VALIDATION (ADD THIS ENTIRE BLOCK)
        # ============================================================
        if tooth_numbers and image_obs:
            logger.info(f"{'='*70}")
            logger.info(f"🔍 VALIDATING TOOTH NUMBER CONSISTENCY")
            logger.info(f"{'='*70}")
            
            validation = tooth_validator.validate_and_fix(
                focus_tooth=tooth_numbers[0],
                vision_teeth=image_obs.structured_findings.get('teeth_visible', []) if image_obs.structured_findings else [],
                diagnosis=recommendation.diagnosis,
                image_obs=image_obs.__dict__ if hasattr(image_obs, '__dict__') else image_obs
            )
            
            if not validation["valid"]:
                logger.warning(f"⚠️  Validation issues detected:")
                for issue in validation["issues"]:
                    logger.warning(f"     - {issue}")
                
                # Apply fixes
                if "corrected_diagnosis" in validation["fixes"]:
                    recommendation.diagnosis = validation["fixes"]["corrected_diagnosis"]
                    logger.info(f"✅ FIXED: Updated diagnosis")
                
                if "corrected_teeth_visible" in validation["fixes"]:
                    if image_obs.structured_findings:
                        image_obs.structured_findings["teeth_visible"] = validation["fixes"]["corrected_teeth_visible"]
                        logger.info(f"✅ FIXED: Corrected visible teeth")
            else:
                logger.info(f"✅ All validations passed")
            
            logger.info(f"{'='*70}")
        # ============================================================
        # END VALIDATION
        # ============================================================

        elapsed = time.time() - start_time

        logger.info(f"\n{'#'*70}")
        logger.info(f"✅✅✅✅ COMPLETED IN {elapsed:.2f}s✅✅✅✅")
        logger.info(f"{'#'*70}")
        logger.info(f"Diagnosis: {recommendation.diagnosis[:80]}...")
        logger.info(f"Knowledge: {len(knowledge)} chunks")
        logger.info(f"Notes: {len(clinical_notes)} items")
        logger.info(f"Reference Pages: {recommendation.reference_pages}")
        logger.info(f"Confidence: {recommendation.confidence_level}")
        logger.info(f"{'#'*70}\n")

        return CDSSResponse(
            recommendation=recommendation,
            image_observations=image_obs if image_obs else no_image_message,  # CRITICAL: Use proper object
            knowledge_sources=knowledge,
            reasoning_chain=f"""Analysis:
                    1. Patient: {patient_id}, {patient_history.age}y {patient_history.gender}
                    2. Complaint: {chief_complaint}
                    3. Tooth Numbers: {', '.join(tooth_numbers) if tooth_numbers else 'Not specified'}
                    4. Notes: {len(clinical_notes)} clinical notes
                    5. Image: {'Yes (' + image_obs.model_used + ')' if image_obs else 'No image'}
                    6. Knowledge: {len(knowledge)} chunks
                    7. Diagnosis: {recommendation.diagnosis}
                    8. Confidence: {recommendation.confidence_level}""",
            processing_metadata={
                "total_time_seconds": round(elapsed, 2),
                "knowledge_chunks": len(knowledge),
                "clinical_notes_count": len(clinical_notes),
                "retriever_type": rag_settings.RETRIEVER_TYPE,
                "diversity": rag_settings.LAMBDA_MULT if rag_settings.RETRIEVER_TYPE == "mmr" else None,
                "fetch_k": rag_settings.FETCH_K if rag_settings.RETRIEVER_TYPE == "mmr" else None,
                "similarity_threshold": rag_settings.SIMILARITY_THRESHOLD if rag_settings.RETRIEVER_TYPE == "similarity_score_threshold" else None,
                "llm_provider": rag_settings.LLM_PROVIDER,
                "llm_model": rag_settings.current_llm_model,
                "vision_provider": vision_settings.VISION_MODEL_PROVIDER if image_obs else "N/A",
                "vision_model": vision_settings.current_vision_model if image_obs else "N/A",
                "embedding_provider": rag_settings.EMBEDDING_PROVIDER,
                "embedding_model": rag_settings.current_embedding_model,
                "user_id": user_id,
                "patient_id": patient_id,
                "tooth_number": tooth_numbers or [],
            },
        )