# app/RAGsystem/chains.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from config.ragconfig import rag_settings
import logging

logger = logging.getLogger(__name__)

class ClinicalRAGChain:
    """LCEL-based Clinical RAG Chain."""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOllama(
            model=rag_settings.current_llm_model,
            temperature=rag_settings.LLM_TEMPERATURE,
            format=rag_settings.FORMAT
        )
        
        # Clinical system prompt
        self.system_prompt = """You are a Clinical Decision Support System analyzing dental/oral health guidelines.
Your task is to provide an evidence-based, structured JSON response based strictly on the retrieved context.

RESPONSE JSON SCHEMA:
{{
  "diagnosis": "string",
  "differential_diagnoses": ["string"],
  "recommended_management": {{
    "pharmacological": {{
      "analgesics": [{{"name": "string", "dose": "string", "reference_page": int}}],
      "antibiotics": [{{"name": "string", "dose": "string", "reference_page": int}}]
    }},
    "non_pharmacological": [{{"description": "string", "category": "string", "reference_page": int}}],
    "follow_up": "string"
  }},
  "precautions": [{{"condition": "string", "note": "string", "reference_page": int}}]
}}

RULES:
1. STRICTLY use the provided clinical guidelines context. Do not add outside knowledge.
2. Cite the specific page number for EVERY recommendation by setting the "reference_page" field.
3. If information is insufficient, state: "Insufficient information in retrieved guidelines."
4. Distinguish between "pharmacological" and "non_pharmacological" treatments.
5. Note any "precautions" or contraindications mentioned in the context.
6. Your FINAL output must be ONLY the resulting JSON object. Do not include markdown formatting (e.g., ```json ... ```) or any other descriptive text.
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Clinical Guidelines Context:
            {context}

            Question: {question}

            Provide a structured clinical answer with page citations.""")
        ])
    
    def format_docs(self, docs):
        """Format documents with clear provenance."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            pages = doc.metadata.get('pages', [])
            if not pages and 'page' in doc.metadata:
                pages = [doc.metadata['page'] + 1]  # Convert 0-indexed to 1-indexed
            
            page_str = f"Pages {', '.join(map(str, pages))}" if pages else "Unknown"
            formatted.append(f"[{i}] [{page_str}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def create_chain(self):
        """Build LCEL chain, including raw retrieved docs in output."""
        
        # This will get the question and context, and raw documents
        setup_and_retrieve = RunnableParallel(
            question=RunnablePassthrough(), # Original question comes in here
            retrieved_docs=self.retriever, # Retrieve docs based on question
        ).assign(
            context=lambda x: self.format_docs(x["retrieved_docs"]) # Format docs into context string
        )
        
        # Now, combine this with the LLM call
        full_chain = setup_and_retrieve | {
            "answer": self.prompt | self.llm | StrOutputParser(),
            "retrieved_docs": lambda x: x["retrieved_docs"] # Pass raw docs through to the end
        }
        
        return full_chain
    
    def invoke(self, question: str):
        """Execute chain and log details."""
        logger.info(f"Processing RAG query: {question[:100]}...")
        llm_model_name = self.llm.model if hasattr(self.llm, 'model') else "Unknown LLM Model"
        logger.info(f"----------------------- LLM Model Used: {llm_model_name} ------------------\n") # Log model name

        chain = self.create_chain()
        result = chain.invoke(question)

        answer = result["answer"]
        retrieved_docs = result["retrieved_docs"]

        logger.info(f"Retrieved {len(retrieved_docs)} chunks.")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"------------------------------ Chunk {i+1} -----------------------------------")
            logger.info(f"Content: {doc.page_content[:200]}...") # Log highlights (first 200 chars)
            logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
            logger.info("===============================================================================\n\n")
            
        return answer

# Legacy support
def answer_question(question: str, retriever):
    """Backward compatible wrapper."""
    chain = ClinicalRAGChain(retriever)
    return chain.invoke(question)