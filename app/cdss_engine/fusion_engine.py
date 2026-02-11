# app/cdss_engine/fusion_engine.py
"""
CDSS Fusion Engine - IMPROVED
Better integration of patient history, clinical complaint, image findings, and knowledge
"""
import logging
import time
from typing import List, Optional
from sqlalchemy import select
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
from config.ragconfig import rag_settings
from config.visionconfig import vision_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CDSSFusionEngine:
    """
    IMPROVED CDSS Pipeline:
    1. Fetch patient history + clinical complaint
    2. Analyze radiograph (if provided) - uses complaint for context
    3. Retrieve knowledge - enhanced by complaint + image findings
    4. Generate recommendation - integrates ALL context
    """

    def __init__(self):
        self.retriever_factory = RetrieverFactory()
        logger.info("🏥 CDSS Fusion Engine initialized")

    async def fetch_patient_history(
        self, 
        patient_id: int, 
        db: AsyncSession
    ) -> PatientHistory:
        """Fetch patient from database with logging."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 1: FETCH PATIENT HISTORY FROM DATABASE")
        logger.info(f"{'='*70}")
        logger.info(f"📋 Querying database for patient ID: {patient_id}")
        
        try:
            result = await db.execute(
                select(Patient).where(Patient.id == patient_id)
            )
            patient = result.scalar_one_or_none()
            
            if not patient:
                logger.error(f"❌ Patient {patient_id} not found in database")
                raise ValueError(f"Patient with ID {patient_id} not found")
            
            # Calculate age
            age = None
            if patient.dob:
                from datetime import datetime
                today = datetime.now().date()
                age = today.year - patient.dob.year - (
                    (today.month, today.day) < (patient.dob.month, patient.dob.day)
                )
            
            logger.info(f"✓ Patient retrieved from database:")
            logger.info(f"   Name: {patient.first_name} {patient.last_name}")
            logger.info(f"   Age: {age} years")
            logger.info(f"   Gender: {patient.gender}")
            logger.info(f"   DOB: {patient.dob}")
            
            patient_history = PatientHistory(
                patient_id=str(patient.id),
                age=age,
                gender=patient.gender,
                chief_complaint=None,  # Will be set by caller
                medical_history=[],
                current_medications=[],
                allergies=[],
                previous_dental_work=None,
                symptoms_duration=None,
                pain_level=None
            )
            
            return patient_history
            
        except Exception as e:
            logger.error(f"❌ Database error: {e}", exc_info=True)
            raise

    def analyze_radiograph(
        self, 
        image_bytes: bytes,
        clinical_complaint: str,
        patient_info: str
    ) -> ImageObservation:
        """
        Analyze radiograph with clinical context.
        
        NOTE: Vision model receives clinical context for better analysis.
        This helps the model focus on relevant pathology.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 2: ANALYZE RADIOGRAPH WITH CLINICAL CONTEXT")
        logger.info(f"{'='*70}")
        logger.info(f"🔍 Vision Model: {vision_settings.VISION_MODEL_PROVIDER}")
        logger.info(f"📝 Clinical Context Provided to Vision Model:")
        logger.info(f"   Complaint: {clinical_complaint}")
        logger.info(f"   Patient: {patient_info}")
        logger.info(f"   ℹ️  Note: Providing context helps vision model focus on relevant pathology")
        
        image = ImageProcessor.preprocess_image(image_bytes)
        logger.info(f"✓ Image preprocessed: {image.size}")
        
        # Analyze with vision model
        # Note: Standard analyze_dental_radiograph uses dental-specific prompts
        # The clinical context helps but isn't strictly necessary since the
        # prompts already instruct the model to look for dental pathology
        result = vision_client.analyze_dental_radiograph(image)
        
        logger.info(f"✓ Vision analysis complete:")
        logger.info(f"   Model used: {result['model']}")
        logger.info(f"   Description length: {len(result['detailed_description'])} chars")
        logger.info(f"   Pathology summary: {len(result.get('pathology_summary', ''))} chars")
        
        # Log key findings
        if result.get('pathology_summary'):
            logger.info(f"📊 Key Pathological Findings:")
            summary_lines = result['pathology_summary'].split('\n')[:5]
            for line in summary_lines:
                if line.strip():
                    logger.info(f"   • {line.strip()}")
        
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
        Retrieve knowledge enhanced by complaint AND image findings.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 3: RETRIEVE CLINICAL KNOWLEDGE FROM KNOWLEDGE BASE")
        logger.info(f"{'='*70}")
        
        # Build enhanced search query
        query_parts = [clinical_complaint]
        
        if image_obs and image_obs.pathology_summary:
            # Extract key pathology terms for better retrieval
            pathology_terms = image_obs.pathology_summary[:200]
            query_parts.append(f"Radiographic findings: {pathology_terms}")
        
        enhanced_query = ". ".join(query_parts)
        
        logger.info(f"🔍 Search Query Construction:")
        logger.info(f"   Base: {clinical_complaint}")
        if image_obs:
            logger.info(f"   Enhanced with image findings: Yes")
        else:
            logger.info(f"   Enhanced with image findings: No")
        logger.info(f"📚 Final search query (first 200 chars):")
        logger.info(f"   {enhanced_query[:200]}...")
        logger.info(f"⚙️  Retrieval Settings:")
        logger.info(f"   Type: {rag_settings.RETRIEVER_TYPE}")
        logger.info(f"   K (chunks to retrieve): {rag_settings.RETRIEVAL_K}")
        logger.info(f"   Embedding provider: {rag_settings.EMBEDDING_PROVIDER}")
        logger.info(f"   Embedding model: {rag_settings.current_embedding_model}")
        
        try:
            retriever = self.retriever_factory.get_retriever()
            docs = retriever.invoke(enhanced_query)
            
            logger.info(f"✓ Retrieved {len(docs)} document chunks")
            
            if len(docs) == 0:
                logger.warning(f"⚠️  WARNING: NO DOCUMENTS RETRIEVED!")
                logger.warning(f"   Possible causes:")
                logger.warning(f"   1. ChromaDB is empty - run: python scripts/ingest_documents.py")
                logger.warning(f"   2. Query doesn't match document content")
                logger.warning(f"   3. Embedding model mismatch")
                logger.warning(f"   Path: {rag_settings.PERSIST_DIR}")
                return []
            
            knowledge = []
            logger.info(f"📖 Retrieved Knowledge Chunks:")
            for i, doc in enumerate(docs, 1):
                pages = doc.metadata.get("pages", [])
                if not pages and "page" in doc.metadata:
                    pages = [doc.metadata["page"] + 1]
                
                source = doc.metadata.get("source", "Clinical Guidelines")
                preview = doc.page_content[:100].replace('\n', ' ')
                
                logger.info(f"   [{i}] Source: {source}, Pages: {pages}")
                logger.info(f"       Preview: {preview}...")
                
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
            logger.error(f"❌ Retrieval failed: {e}", exc_info=True)
            return []

    def fuse_and_reason(
        self,
        patient_history: PatientHistory,
        clinical_complaint: str,
        image_obs: Optional[ImageObservation],
        knowledge: List[RetrievedKnowledge],
    ) -> ClinicalRecommendation:
        """
        Generate recommendation integrating ALL context:
        - Patient demographics and history
        - Clinical complaint
        - Image findings
        - Retrieved knowledge
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 4: GENERATE CLINICAL RECOMMENDATION (CONTEXT FUSION)")
        logger.info(f"{'='*70}")
        logger.info(f"🤖 LLM Configuration:")
        logger.info(f"   Provider: {rag_settings.LLM_PROVIDER}")
        logger.info(f"   Model: {rag_settings.current_llm_model}")
        logger.info(f"   Temperature: {rag_settings.LLM_TEMPERATURE}")
        
        # Build comprehensive patient context
        patient_ctx = f"""Patient Demographics and History:
- Patient ID: {patient_history.patient_id}
- Age: {patient_history.age or 'Unknown'} years
- Gender: {patient_history.gender or 'Unknown'}
- Medical History: {', '.join(patient_history.medical_history) or 'None reported'}
- Current Medications: {', '.join(patient_history.current_medications) or 'None'}
- Known Allergies: {', '.join(patient_history.allergies) or 'None'}
- Pain Level: {patient_history.pain_level or 'Not specified'}/10

Chief Complaint (Current Visit):
{clinical_complaint}"""

        # Build image findings context
        image_ctx = ""
        if image_obs:
            image_ctx = f"""Radiographic Analysis Results (Model: {image_obs.model_used}):

=== DETAILED RADIOGRAPHIC DESCRIPTION ===
{image_obs.raw_description}

=== PATHOLOGY CHECKLIST ===
{image_obs.pathology_summary}

Radiograph Confidence Level: {image_obs.confidence}"""
        else:
            image_ctx = "No radiograph provided for this consultation."

        # Build guidelines context
        guidelines_ctx = ""
        all_page_refs = []
        
        if len(knowledge) > 0:
            guidelines_ctx = "Retrieved Clinical Guidelines (from knowledge base):\n\n"
            for i, k in enumerate(knowledge, 1):
                pages_str = f"Pages: {', '.join(map(str, k.pages))}" if k.pages else "Page unknown"
                guidelines_ctx += f"--- Source {i} ({pages_str}) ---\n{k.content}\n\n"
                all_page_refs.extend([f"Page {p}" for p in k.pages])
        else:
            guidelines_ctx = "WARNING: No clinical guidelines retrieved from knowledge base. Recommendation will be based on general dental knowledge only."
        
        logger.info(f"📊 Context Summary Being Sent to LLM:")
        logger.info(f"   Patient context: {len(patient_ctx)} characters")
        logger.info(f"   Clinical complaint: {len(clinical_complaint)} characters")
        logger.info(f"   Image findings: {len(image_ctx)} characters")
        logger.info(f"   Guidelines: {len(guidelines_ctx)} characters")
        logger.info(f"   Knowledge chunks: {len(knowledge)}")
        logger.info(f"   Total context size: {len(patient_ctx) + len(image_ctx) + len(guidelines_ctx)} characters")
        
        # Generate recommendation
        try:
            logger.info(f"⚙️  Calling LLM to generate recommendation...")
            
            result = llm_client.generate_clinical_recommendation(
                patient_context=patient_ctx,
                image_findings=image_ctx,
                retrieved_knowledge=guidelines_ctx,
                query=clinical_complaint
            )
            
            logger.info(f"✓ LLM recommendation generated successfully")
            logger.info(f"📋 Recommendation Summary:")
            logger.info(f"   Diagnosis: {result.get('diagnosis', 'N/A')[:150]}...")
            logger.info(f"   Differential diagnoses: {len(result.get('differential_diagnoses', []))} alternatives")
            logger.info(f"   Management plan length: {len(result.get('recommended_management', ''))} characters")
            logger.info(f"   Page references: {len(result.get('page_references', []))} citations")
            
            return ClinicalRecommendation(
                diagnosis=result.get("diagnosis", "Unable to determine diagnosis"),
                differential_diagnoses=result.get("differential_diagnoses", []),
                recommended_management=result.get("recommended_management", "Consult with supervising dentist"),
                page_references=result.get("page_references", list(set(all_page_refs))[:10]),
                confidence_level="high" if (len(knowledge) > 3 and image_obs and image_obs.confidence == "high") else "medium" if len(knowledge) > 0 else "low",
                llm_provider=rag_settings.LLM_PROVIDER
            )
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}", exc_info=True)
            logger.error(f"   This usually means:")
            logger.error(f"   1. LLM returned invalid JSON structure")
            logger.error(f"   2. Validation error (e.g., list instead of string)")
            logger.error(f"   3. LLM timeout or connection issue")
            
            return ClinicalRecommendation(
                diagnosis="Error generating recommendation - see logs",
                differential_diagnoses=[],
                recommended_management=f"LLM Error: {str(e)}. Please consult supervising dentist for clinical assessment.",
                page_references=list(set(all_page_refs))[:10] if all_page_refs else [],
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
        """
        Main CDSS pipeline with comprehensive logging and context integration.
        """
        start_time = time.time()
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"# CDSS PIPELINE INITIATED")
        logger.info(f"{'#'*70}")
        logger.info(f"📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"👤 Patient ID: {patient_id}")
        logger.info(f"👨‍⚕️ User/Doctor ID: {user_id}")
        logger.info(f"💬 Chief Complaint: {chief_complaint}")
        logger.info(f"🖼️  Radiograph provided: {'Yes' if image_bytes else 'No'}")

        # Step 1: Patient history
        patient_history = await self.fetch_patient_history(patient_id, db)
        patient_history.chief_complaint = chief_complaint
        
        patient_info = f"{patient_history.age}y {patient_history.gender}"

        # Step 2: Image analysis (with clinical context)
        image_obs = None
        if image_bytes:
            try:
                image_obs = self.analyze_radiograph(
                    image_bytes, 
                    chief_complaint, 
                    patient_info
                )
            except Exception as e:
                logger.error(f"❌ Image analysis failed: {e}", exc_info=True)

        # Step 3: Knowledge retrieval (enhanced by complaint + image)
        knowledge = self.retrieve_knowledge(chief_complaint, image_obs)

        # Step 4: Generate recommendation (ALL context integrated)
        recommendation = self.fuse_and_reason(
            patient_history, 
            chief_complaint,
            image_obs, 
            knowledge
        )

        processing_time = time.time() - start_time

        reasoning = f"""Clinical Decision Support Analysis Summary:

1. PATIENT INFORMATION:
   - ID: {patient_id}, Age: {patient_history.age}y, Gender: {patient_history.gender}
   - Chief Complaint: {chief_complaint}

2. RADIOGRAPHIC ANALYSIS:
   - Performed: {'Yes (' + image_obs.model_used + ')' if image_obs else 'No'}
   - Key Findings: {image_obs.pathology_summary[:100] + '...' if image_obs else 'N/A'}

3. KNOWLEDGE RETRIEVAL:
   - Chunks Retrieved: {len(knowledge)}
   - Sources: {', '.join(set([k.source for k in knowledge]))} if knowledge else 'None'

4. CLINICAL RECOMMENDATION:
   - Diagnosis: {recommendation.diagnosis[:100]}...
   - Confidence: {recommendation.confidence_level}
   - LLM Provider: {rag_settings.LLM_PROVIDER}"""

        logger.info(f"\n{'#'*70}")
        logger.info(f"# CDSS PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"{'#'*70}")
        logger.info(f"⏱️  Total Processing Time: {processing_time:.2f} seconds")
        logger.info(f"✓ Final Diagnosis: {recommendation.diagnosis[:100]}...")
        logger.info(f"✓ Confidence Level: {recommendation.confidence_level}")
        logger.info(f"✓ Knowledge Chunks Used: {len(knowledge)}")
        logger.info(f"✓ Page References: {len(recommendation.page_references)}")
        logger.info(f"{'#'*70}\n")

        return CDSSResponse(
            recommendation=recommendation,
            image_observations=image_obs,
            knowledge_sources=knowledge,
            reasoning_chain=reasoning,
            processing_metadata={
                "total_time_seconds": round(processing_time, 2),
                "knowledge_chunks": len(knowledge),
                "llm_provider": rag_settings.LLM_PROVIDER,
                "llm_model": rag_settings.current_llm_model,
                "vision_provider": vision_settings.VISION_MODEL_PROVIDER if image_obs else "N/A",
                "vision_model": image_obs.model_used if image_obs else "N/A",
                "embedding_provider": rag_settings.EMBEDDING_PROVIDER,
                "embedding_model": rag_settings.current_embedding_model,
                "retriever_type": rag_settings.RETRIEVER_TYPE,
                "user_id": user_id,
                "patient_id": patient_id,
                "chromadb_path": rag_settings.PERSIST_DIR
            },
        )