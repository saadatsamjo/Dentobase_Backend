# app/cdss_engine/fusion_engine.py
"""
CDSS Fusion Engine - PRODUCTION READY
- Structured Pydantic output
- Handles zero knowledge chunks gracefully
- Clinical notes integration
- Works with ALL retriever configurations
"""
import logging
import time
from typing import List, Optional
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.cdss_engine.schemas import (
    CDSSResponse,
    ClinicalRecommendation,
    ImageObservation,
    PatientHistory,
    RetrievedKnowledge,
)
from app.RAGsystem.retriever import RetrieverFactory
from app.RAGsystem.llm_client import llm_client
from app.visionsystem.vision_client import vision_client
from app.visionsystem.image_processor import ImageProcessor
from app.system_models.patient_model.patient_model import Patient
from app.system_models.clinical_note_model.clinical_note_model import ClinicalNote
from config.ragconfig import rag_settings
from config.visionconfig import vision_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CDSSFusionEngine:
    """
    Production-ready CDSS with:
    - Structured Pydantic output (always valid)
    - Graceful handling of zero knowledge chunks
    - Clinical notes integration
    - Full configurability for experiments
    """

    def __init__(self):
        self.retriever_factory = RetrieverFactory()
        logger.info("🏥 CDSS Fusion Engine initialized")

    async def fetch_patient_history_with_notes(
        self, 
        patient_id: int, 
        db: AsyncSession,
        num_notes: int = 3
    ) -> tuple[PatientHistory, List[dict]]:
        """Fetch patient demographics + recent clinical notes."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 1: FETCH PATIENT HISTORY + CLINICAL NOTES")
        logger.info(f"{'='*70}")
        logger.info(f"📋 Patient ID: {patient_id}")
        
        try:
            # Fetch patient
            result = await db.execute(
                select(Patient).where(Patient.id == patient_id)
            )
            patient = result.scalar_one_or_none()
            
            if not patient:
                logger.error(f"❌ Patient {patient_id} not found")
                raise ValueError(f"Patient with ID {patient_id} not found")
            
            # Calculate age
            age = None
            if patient.dob:
                from datetime import datetime
                today = datetime.now().date()
                age = today.year - patient.dob.year - (
                    (today.month, today.day) < (patient.dob.month, patient.dob.day)
                )
            
            logger.info(f"✓ Patient: {patient.first_name} {patient.last_name}")
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
                logger.info(f"✓ Retrieved {len(clinical_notes_raw)} clinical note(s):")
                for i, note in enumerate(clinical_notes_raw, 1):
                    formatted_notes.append({
                        "date": note.created_at.strftime("%Y-%m-%d %H:%M"),
                        "type": note.note_type,
                        "content": note.content,
                        "author_id": note.author_id
                    })
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
                pain_level=None
            )
            
            return patient_history, formatted_notes
            
        except Exception as e:
            logger.error(f"❌ Database error: {e}", exc_info=True)
            raise

    def analyze_radiograph(
        self, 
        image_bytes: bytes,
        clinical_complaint: str,
        patient_info: str
    ) -> ImageObservation:
        """Analyze radiograph."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 2: ANALYZE RADIOGRAPH")
        logger.info(f"{'='*70}")
        logger.info(f"🔍 Vision Model: {vision_settings.VISION_MODEL_PROVIDER}")
        logger.info(f"📝 Context: {clinical_complaint}")
        
        image = ImageProcessor.preprocess_image(image_bytes)
        logger.info(f"✓ Image preprocessed: {image.size}")
        
        result = vision_client.analyze_dental_radiograph(image)
        
        logger.info(f"✓ Analysis complete")
        logger.info(f"   Model: {result['model']}")
        logger.info(f"   Description: {len(result['detailed_description'])} chars")
        
        return ImageObservation(
            raw_description=result["detailed_description"],
            pathology_summary=result["pathology_summary"],
            confidence="high" if "no pathology" not in result["detailed_description"].lower() else "low",
            model_used=result["model"]
        )

    def retrieve_knowledge(
        self, 
        clinical_complaint: str,
        image_obs: Optional[ImageObservation] = None
    ) -> List[RetrievedKnowledge]:
        """
        Retrieve knowledge - works with ANY retriever configuration.
        
        May return 0 chunks if:
        - similarity_score_threshold is too high
        - ChromaDB is empty
        - Query doesn't match content
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 3: RETRIEVE CLINICAL KNOWLEDGE")
        logger.info(f"{'='*70}")
        
        # Build query
        query_parts = [clinical_complaint]
        if image_obs and image_obs.pathology_summary:
            query_parts.append(f"Findings: {image_obs.pathology_summary[:200]}")
        
        enhanced_query = ". ".join(query_parts)
        
        logger.info(f"🔍 Query: {enhanced_query[:150]}...")
        logger.info(f"⚙️  Configuration:")
        logger.info(f"   Retriever: {rag_settings.RETRIEVER_TYPE}")
        logger.info(f"   K: {rag_settings.RETRIEVAL_K}")
        if rag_settings.RETRIEVER_TYPE == "similarity_score_threshold":
            logger.info(f"   Threshold: {rag_settings.SIMILARITY_THRESHOLD}")
        
        try:
            retriever = self.retriever_factory.get_retriever()
            docs = retriever.invoke(enhanced_query)
            
            logger.info(f"✓ Retrieved {len(docs)} chunks")
            
            if len(docs) == 0:
                logger.warning(f"⚠️  ZERO CHUNKS RETRIEVED")
                logger.warning(f"   This is expected if:")
                logger.warning(f"   - similarity_score_threshold is high")
                logger.warning(f"   - Query doesn't match document content")
                logger.warning(f"   LLM will use general knowledge instead")
                return []
            
            knowledge = []
            logger.info(f"📖 Retrieved Knowledge:")
            for i, doc in enumerate(docs, 1):
                pages = doc.metadata.get("pages", [])
                if not pages and "page" in doc.metadata:
                    pages = [doc.metadata["page"] + 1]
                
                source = doc.metadata.get("source", "Guidelines")
                preview = doc.page_content[:80].replace('\n', ' ')
                
                logger.info(f"   [{i}] Pages {pages}: {preview}...")
                
                knowledge.append(
                    RetrievedKnowledge(
                        content=doc.page_content,
                        pages=pages,
                        relevance_score=doc.metadata.get("score"),
                        source=source
                    )
                )
            
            return knowledge
            
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}", exc_info=True)
            return []

    def fuse_and_reason(
        self,
        patient_history: PatientHistory,
        clinical_notes: List[dict],
        clinical_complaint: str,
        image_obs: Optional[ImageObservation],
        knowledge: List[RetrievedKnowledge],
    ) -> ClinicalRecommendation:
        """
        Generate recommendation with structured Pydantic output.
        
        CRITICAL: Handles zero knowledge chunks gracefully.
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

        # Image findings
        image_ctx = ""
        if image_obs:
            image_ctx = f"""Radiograph Analysis ({image_obs.model_used}):
{image_obs.raw_description}

Pathology Summary:
{image_obs.pathology_summary}"""
        else:
            image_ctx = "No radiograph provided."

        # Guidelines context
        if knowledge_available:
            guidelines_ctx = "Clinical Guidelines:\n\n"
            for i, k in enumerate(knowledge, 1):
                pages = f"Pages {', '.join(map(str, k.pages))}" if k.pages else ""
                guidelines_ctx += f"[{i}] {pages}\n{k.content}\n\n"
        else:
            guidelines_ctx = "⚠️ No clinical guidelines retrieved from knowledge base.\nRecommendation will be based on general dental knowledge."
        
        logger.info(f"📊 Context:")
        logger.info(f"   Patient: {len(patient_ctx)} chars")
        logger.info(f"   Notes: {len(clinical_notes)} items")
        logger.info(f"   Image: {len(image_ctx)} chars")
        logger.info(f"   Knowledge: {len(knowledge)} chunks")
        logger.info(f"   Knowledge available: {knowledge_available}")
        
        try:
            logger.info(f"⚙️  Calling LLM with structured output...")
            
            # Call LLM with knowledge_available flag
            result = llm_client.generate_clinical_recommendation(
                patient_context=patient_ctx,
                image_findings=image_ctx,
                retrieved_knowledge=guidelines_ctx,
                query=clinical_complaint,
                knowledge_available=knowledge_available  # CRITICAL FLAG
            )
            
            logger.info(f"✓ Structured recommendation generated")
            logger.info(f"   Diagnosis: {result['diagnosis'][:100]}...")
            logger.info(f"   Management: {len(result['recommended_management'])} chars")
            logger.info(f"   References: {result['page_references']}")
            
            return ClinicalRecommendation(
                diagnosis=result["diagnosis"],
                differential_diagnoses=result["differential_diagnoses"],
                recommended_management=result["recommended_management"],
                page_references=result["page_references"],
                confidence_level="high" if (knowledge_available and len(knowledge) > 3) else "medium" if knowledge_available else "low",
                llm_provider=rag_settings.LLM_PROVIDER
            )
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)
            
            return ClinicalRecommendation(
                diagnosis=f"Error: {str(e)}",
                differential_diagnoses=[],
                recommended_management="Consult supervising dentist.",
                page_references=["Error - unable to generate"],
                confidence_level="low",
                llm_provider=rag_settings.LLM_PROVIDER
            )

    async def provide_final_recommendation(
        self,
        patient_id: int,
        chief_complaint: str,
        db: AsyncSession,
        image_bytes: Optional[bytes] = None,
        user_id: int = 1
    ) -> CDSSResponse:
        """Main CDSS pipeline - production ready."""
        start_time = time.time()
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"# CDSS PIPELINE - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'#'*70}")
        logger.info(f"User: (Dentist ID) {user_id}")
        logger.info(f"Patient: {patient_id}")
        logger.info(f"Complaint: {chief_complaint}")
        logger.info(f"Config: {rag_settings.RETRIEVER_TYPE}")
        if rag_settings.RETRIEVER_TYPE == "similarity_score_threshold":
            logger.info(f"Similarity threshold: {rag_settings.SIMILARITY_THRESHOLD}")
        elif rag_settings.RETRIEVER_TYPE == "mmr":
            logger.info(f"Diversity: {rag_settings.LAMBDA_MULT}")
            logger.info(f"Fetch K: {rag_settings.FETCH_K}")

        # Pipeline
        patient_history, clinical_notes = await self.fetch_patient_history_with_notes(patient_id, db, 3)
        patient_history.chief_complaint = chief_complaint
        
        image_obs = None
        if image_bytes:
            try:
                image_obs = self.analyze_radiograph(
                    image_bytes, 
                    chief_complaint, 
                    f"{patient_history.age}y {patient_history.gender}"
                )
            except Exception as e:
                logger.error(f"❌ Image analysis failed: {e}")

        knowledge = self.retrieve_knowledge(chief_complaint, image_obs)

        recommendation = self.fuse_and_reason(
            patient_history, 
            clinical_notes,
            chief_complaint,
            image_obs, 
            knowledge
        )

        elapsed = time.time() - start_time

        logger.info(f"\n{'#'*70}")
        logger.info(f"# COMPLETED IN {elapsed:.2f}s")
        logger.info(f"{'#'*70}")
        logger.info(f"Diagnosis: {recommendation.diagnosis[:80]}...")
        logger.info(f"Knowledge: {len(knowledge)} chunks")
        logger.info(f"Notes: {len(clinical_notes)} items")
        logger.info(f"Confidence: {recommendation.confidence_level}")
        logger.info(f"{'#'*70}\n")

        return CDSSResponse(
            recommendation=recommendation,
            image_observations=image_obs,
            knowledge_sources=knowledge,
            reasoning_chain=f"""Analysis:
1. Patient: {patient_id}, {patient_history.age}y {patient_history.gender}
2. Complaint: {chief_complaint}
3. Notes: {len(clinical_notes)} clinical notes
4. Image: {'Yes (' + image_obs.model_used + ')' if image_obs else 'No'}
5. Knowledge: {len(knowledge)} chunks
6. Diagnosis: {recommendation.diagnosis}
7. Confidence: {recommendation.confidence_level}""",
            processing_metadata={
                "total_time_seconds": round(elapsed, 2),
                "knowledge_chunks": len(knowledge),
                "clinical_notes_count": len(clinical_notes),
                "retriever_type": rag_settings.RETRIEVER_TYPE,
                "similarity_threshold": rag_settings.SIMILARITY_THRESHOLD,
                "llm_provider": rag_settings.LLM_PROVIDER,
                "llm_model": rag_settings.current_llm_model,
                "vision_provider": vision_settings.VISION_MODEL_PROVIDER if image_obs else "N/A",
                "embedding_provider": rag_settings.EMBEDDING_PROVIDER,
                "user_id": user_id,
                "patient_id": patient_id
            },
        )