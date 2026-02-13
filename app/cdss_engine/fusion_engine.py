# app/cdss_engine/fusion_engine.py
"""
Updated CDSSFusionEngine with:
1. Proper pathology summary extraction and usage
2. Complete radiographic findings in LLM context
3. Enhanced logging showing context sent to each model
4. NoImageProvided support
5. Tooth number support
6. Clinical notes integration
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

logger = logging.getLogger(__name__)


class CDSSFusionEngine:
    """
    Production-ready CDSS with:
    - Structured Pydantic output (always valid)
    - Graceful handling of zero knowledge chunks
    - Clinical notes integration
    - Tooth number support for focused analysis
    - Complete radiographic findings in LLM context
    - Full configurability for experiments
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
            # Fetch patient
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
                .join(ClinicalNote.encounter)  # Join through encounter
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
                    logger.info(
                        f"   [{i}] {note.created_at.strftime('%Y-%m-%d')} - {note.note_type}"
                    )
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
        Analyze radiograph with clinical context from notes.

        Args:
            image_bytes: Image data
            clinical_complaint: Clinical complaint
            patient_info: Patient demographics
            tooth_numbers: List of tooth numbers to focus on
            clinical_notes: Recent clinical notes for context
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 2: ANALYZE RADIOGRAPH")
        logger.info(f"{'='*70}")
        logger.info(f"🔍 Vision Model: {vision_settings.VISION_MODEL_PROVIDER}")
        logger.info(f"📝 Chief Complaint: {clinical_complaint}")
        if tooth_numbers:
            logger.info(f"🦷 Focus Teeth: {', '.join(tooth_numbers)}")
        if clinical_notes:
            logger.info(f"📋 Clinical Notes: {len(clinical_notes)} note(s) available")

        image = ImageProcessor.preprocess_image(image_bytes)
        logger.info(f"✅ Image preprocessed: {image.size}")

        # Build enhanced context with tooth numbers
        enhanced_context = clinical_complaint
        if tooth_numbers:
            tooth_list = ", ".join(tooth_numbers)
            enhanced_context = f"{clinical_complaint}. Focus on tooth/teeth: {tooth_list}"

        # Build clinical notes text if available
        clinical_notes_text = None
        if clinical_notes:
            notes_parts = []
            for note in clinical_notes:
                notes_parts.append(f"[{note['type']}]: {note['content']}")
            clinical_notes_text = " | ".join(notes_parts)
            logger.info(f"   Notes for context: {clinical_notes_text[:150]}...")

        # LOG CONTEXT BEING SENT TO VISION MODEL
        logger.info(f"\n{'─'*70}")
        logger.info(f"📤 CONTEXT SENT TO VISION MODEL:")
        logger.info(f"{'─'*70}")
        logger.info(f"Enhanced Context: {enhanced_context}")
        if clinical_notes_text:
            logger.info(f"Clinical Notes: {clinical_notes_text[:200]}...")
        logger.info(f"{'─'*70}\n")

        # Call vision client with context and clinical notes
        result = vision_client.analyze_dental_radiograph(
            image, context=enhanced_context, clinical_notes=clinical_notes_text
        )

        logger.info(f"✅ Analysis complete")
        logger.info(f"   Model: {result['model']}")
        logger.info(f"   Detailed description: {len(result['detailed_description'])} chars")
        logger.info(f"   Pathology summary: {len(result['pathology_summary'])} chars")

        # ENHANCED LOGGING - Display both descriptions
        logger.info(f"\n{'─'*70}")
        logger.info(f"📊 IMAGE ANALYSIS RESULTS:")
        logger.info(f"{'─'*70}")
        logger.info(f"Model Used: {result['model']}")
        logger.info(
            f"Confidence: {'high' if 'no pathology' not in result['detailed_description'].lower() else 'low'}"
        )

        logger.info(f"\n📋 Pathology Summary ({len(result['pathology_summary'])} chars):")
        logger.info(f"{result['pathology_summary'][:500]}...")

        logger.info(
            f"\n📄 Complete Radiographic Findings ({len(result['detailed_description'])} chars):"
        )
        logger.info(f"{result['detailed_description'][:800]}...")
        logger.info(f"{'─'*70}\n")

        return ImageObservation(
            raw_description=result["detailed_description"],  # Complete findings
            pathology_summary=result["pathology_summary"],  # Extracted summary
            confidence=(
                "high" if "no pathology" not in result["detailed_description"].lower() else "low"
            ),
            model_used=result["model"],
        )

    def retrieve_knowledge(
        self, clinical_complaint: str, image_obs: Optional[ImageObservation] = None
    ) -> tuple[List[RetrievedKnowledge], List[int]]:
        """
        Retrieve knowledge with enhanced query from image findings.

        Returns:
            Tuple of (knowledge chunks, available page numbers)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 3: RETRIEVE CLINICAL KNOWLEDGE")
        logger.info(f"{'='*70}")

        # Build query - USE PATHOLOGY SUMMARY (not detailed description)
        query_parts = [clinical_complaint]
        if image_obs and image_obs.pathology_summary:
            # Use pathology summary for focused retrieval
            query_parts.append(f"Radiographic findings: {image_obs.pathology_summary[:300]}")

        enhanced_query = ". ".join(query_parts)

        logger.info(f"🔍 Building retrieval query:")
        logger.info(f"   Base: {clinical_complaint}")
        if image_obs:
            logger.info(f"   + Pathology: {image_obs.pathology_summary[:150]}...")

        logger.info(f"\n{'─'*70}")
        logger.info(f"📤 QUERY SENT TO RETRIEVER:")
        logger.info(f"{'─'*70}")
        logger.info(f"{enhanced_query[:400]}...")
        logger.info(f"{'─'*70}\n")

        logger.info(f"⚙️  Configuration:")
        logger.info(f"   Retriever: {rag_settings.RETRIEVER_TYPE}")
        logger.info(f"   K: {rag_settings.RETRIEVAL_K}")
        if rag_settings.RETRIEVER_TYPE == "similarity_score_threshold":
            logger.info(f"   Threshold: {rag_settings.SIMILARITY_THRESHOLD}")

        try:
            retriever = self.retriever_factory.get_retriever()
            docs = retriever.invoke(enhanced_query)

            logger.info(f"✅Retrieved {len(docs)} chunks")

            if len(docs) == 0:
                logger.warning(f"⚠️  ZERO CHUNKS RETRIEVED")
                logger.warning(f"   This is expected if:")
                logger.warning(f"   - similarity_score_threshold is high")
                logger.warning(f"   - Query doesn't match document content")
                logger.warning(f"   LLM will use general knowledge instead")
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

                # Collect all available pages
                available_pages.update(pages)

            sorted_pages = sorted(list(available_pages))
            logger.info(f"\n📄 Total unique pages retrieved: {sorted_pages}")

            return knowledge, sorted_pages

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}", exc_info=True)
            return [], []

    def fuse_and_reason(
        self,
        patient_history: PatientHistory,
        clinical_notes: List[dict],
        clinical_complaint: str,
        image_obs: Optional[ImageObservation],
        knowledge: List[RetrievedKnowledge],
        available_pages: List[int],
    ) -> ClinicalRecommendation:
        """
        Generate structured recommendation with complete X-ray findings in context.

        Args:
            patient_history: Patient demographics
            clinical_notes: Clinical notes
            clinical_complaint: Chief complaint
            image_obs: Image analysis results
            knowledge: Retrieved knowledge chunks
            available_pages: List of actual page numbers retrieved
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 4: GENERATE STRUCTURED RECOMMENDATION")
        logger.info(f"{'='*70}")
        logger.info(f"🤖 LLM: {rag_settings.LLM_PROVIDER}")

        knowledge_available = len(knowledge) > 0

        # Patient context
        patient_ctx = f"""Patient Demographics:
            - ID: {patient_history.patient_id}
            - Age: {patient_history.age or 'Unknown'}, Gender: {patient_history.gender or 'Unknown'}
            - Medical History: {', '.join(patient_history.medical_history) or 'None'}
            - Medications: {', '.join(patient_history.current_medications) or 'None'}
            - Allergies: {', '.join(patient_history.allergies) or 'None'}

            Current Complaint: {clinical_complaint}"""

        if clinical_notes:
            patient_ctx += f"\n\nRecent Clinical Notes ({len(clinical_notes)}):\n"
            for i, note in enumerate(clinical_notes, 1):
                patient_ctx += f"[{i}] {note['date']} - {note['type']}: {note['content']}\n"

        # Image findings - USE COMPLETE DESCRIPTION (RAW FINDINGS)
        image_ctx = ""
        if image_obs:
            # Use raw_description for complete radiographic findings
            image_ctx = f"""Radiographic Analysis ({image_obs.model_used}):

COMPLETE RADIOGRAPHIC FINDINGS:
{image_obs.raw_description}

Key Pathology Summary:
{image_obs.pathology_summary}

Confidence: {image_obs.confidence}"""
        else:
            image_ctx = "No radiograph provided."

        # Guidelines context
        if knowledge_available:
            guidelines_ctx = "Clinical Guidelines:\n\n"
            for i, k in enumerate(knowledge, 1):
                pages = f"Pages {k.pages}" if k.pages else ""
                guidelines_ctx += f"[{i}] {pages}\n{k.content}\n\n"
        else:
            guidelines_ctx = "⚠️ No clinical guidelines retrieved from knowledge base.\nRecommendation will be based on general dental knowledge."

        # LOG THE EXACT CONTEXT BEING SENT TO LLM
        logger.info(f"📊 Building context for LLM:")
        logger.info(f"   Patient context: {len(patient_ctx)} chars")
        logger.info(f"   Clinical notes: {len(clinical_notes)} items")
        logger.info(f"   Image context: {len(image_ctx)} chars")
        logger.info(f"   Knowledge: {len(knowledge)} chunks")
        logger.info(f"   Available pages: {available_pages}")

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

        try:
            logger.info(f"⚙️  Calling LLM with structured output...")

            # Call LLM with available pages list
            result = llm_client.generate_clinical_recommendation(
                patient_context=patient_ctx,
                image_findings=image_ctx,  # contains COMPLETE radiographic findings
                retrieved_knowledge=guidelines_ctx,
                query=clinical_complaint,
                knowledge_available=knowledge_available,
                available_pages=available_pages,
            )

            logger.info(f"✅Structured recommendation generated")
            logger.info(f"   Diagnosis: {result['diagnosis'][:100]}...")
            logger.info(f"   Management: {len(result['recommended_management'])} chars")
            logger.info(f"   Reference pages: {result['reference_pages']}")

            return ClinicalRecommendation(
                diagnosis=result["diagnosis"],
                differential_diagnoses=result["differential_diagnoses"],
                recommended_management=result["recommended_management"],
                reference_pages=result["reference_pages"],
                confidence_level=(
                    "high"
                    if (knowledge_available and len(knowledge) > 3)
                    else "medium" if knowledge_available else "low"
                ),
                llm_provider=rag_settings.LLM_PROVIDER,
            )

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)

            return ClinicalRecommendation(
                diagnosis=f"Error: {str(e)}",
                differential_diagnoses=[],
                recommended_management="Consult supervising dentist.",
                reference_pages=[],
                confidence_level="low",
                llm_provider=rag_settings.LLM_PROVIDER,
            )

    async def provide_final_recommendation(
        self,
        patient_id: int,
        chief_complaint: str,
        db: AsyncSession,
        image_bytes: Optional[bytes] = None,
        tooth_numbers: Optional[List[str]] = None,
        user_id: int = 1,
    ) -> CDSSResponse:
        """
        Main CDSS pipeline with NoImageProvided support and clinical context.
        """
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

        # Image analysis with clinical context or NoImageProvided
        image_obs = None
        no_image_message = None
        
        if image_bytes:
            try:
                image_obs = self.analyze_radiograph(
                    image_bytes,
                    chief_complaint,
                    f"{patient_history.age}y {patient_history.gender}",
                    tooth_numbers,
                    clinical_notes,  # Pass clinical notes to vision analysis
                )
            except Exception as e:
                logger.error(f"❌ Image analysis failed: {e}")
               
        
        # Create NoImageProvided message if no image
        if image_obs is None and image_bytes is None:
            no_image_message = NoImageProvided(
                message="No radiograph image was provided for this consultation",
                image_required=False,
            )
            logger.info(f"\n⚠️  No image provided - using text-based analysis only")

        knowledge, available_pages = self.retrieve_knowledge(chief_complaint, image_obs)

        recommendation = self.fuse_and_reason(
            patient_history, clinical_notes, chief_complaint, image_obs, knowledge, available_pages
        )

        elapsed = time.time() - start_time

        logger.info(f"\n{'#'*70}")
        logger.info(f"# COMPLETED IN {elapsed:.2f}s")
        logger.info(f"{'#'*70}")
        logger.info(f"Diagnosis: {recommendation.diagnosis[:80]}...")
        logger.info(f"Knowledge: {len(knowledge)} chunks")
        logger.info(f"Notes: {len(clinical_notes)} items")
        logger.info(f"Reference Pages: {recommendation.reference_pages}")
        logger.info(f"Confidence: {recommendation.confidence_level}")
        logger.info(f"{'#'*70}\n")

        return CDSSResponse(
            recommendation=recommendation,
            image_observations=(
                image_obs if image_obs else no_image_message
            ),  # NoImageProvided instead of None
            knowledge_sources=knowledge,
            reasoning_chain=f"""Analysis:
                    1. Patient: {patient_id}, {patient_history.age}y {patient_history.gender}
                    2. Complaint: {chief_complaint}
                    3. Tooth Numbers: {', '.join(tooth_numbers) if tooth_numbers else 'Not specified'}
                    4. Notes: {len(clinical_notes)} clinical notes
                    5. Image: {'Yes (' + image_obs.model_used + ')' if image_obs else 'No image provided'}
                    6. Knowledge: {len(knowledge)} chunks
                    7. Diagnosis: {recommendation.diagnosis}
                    8. Confidence: {recommendation.confidence_level}""",
                                processing_metadata={
                "total_time_seconds": round(elapsed, 2),
                "knowledge_chunks": len(knowledge),
                "clinical_notes_count": len(clinical_notes),
                "retriever_type": rag_settings.RETRIEVER_TYPE,
                "diversity": (
                    rag_settings.LAMBDA_MULT if rag_settings.RETRIEVER_TYPE == "mmr" else None
                ),
                "fetch_k": rag_settings.FETCH_K if rag_settings.RETRIEVER_TYPE == "mmr" else None,
                "llm_provider": rag_settings.LLM_PROVIDER,
                "llm_model": rag_settings.current_llm_model,
                "vision_provider": vision_settings.VISION_MODEL_PROVIDER if image_obs else "N/A",
                "embedding_provider": rag_settings.EMBEDDING_PROVIDER,
                "user_id": user_id,
                "patient_id": patient_id,
                "tooth_number": tooth_numbers or [],
            },
        )