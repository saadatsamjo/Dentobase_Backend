# app/RAGsystem/chains.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config.ragconfig import rag_settings
import logging

logger = logging.getLogger(__name__)

class ClinicalRAGChain:
    """LCEL-based Clinical RAG Chain."""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOllama(
            model=rag_settings.OLLAMA_LLM_MODEL,
            temperature=rag_settings.LLM_TEMPERATURE
        )
        
        # Clinical system prompt
        self.system_prompt = """You are a Clinical Decision Support System analyzing dental/oral health guidelines.
            Provide evidence-based recommendations strictly from the retrieved context.

            RULES:
            1. Use ONLY the provided clinical guidelines
            2. Cite specific page numbers [Page X] for every recommendation
            3. If insufficient information, state: "Insufficient information in retrieved guidelines"
            4. Distinguish pharmacological vs non-pharmacological treatments
            5. Note any precautions or contraindications mentioned"""
        
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
        """Build LCEL chain."""
        return (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(self, question: str):
        """Execute chain."""
        logger.info(f"Processing RAG query: {question[:100]}...")
        chain = self.create_chain()
        return chain.invoke(question)

# Legacy support
def answer_question(question: str, retriever):
    """Backward compatible wrapper."""
    chain = ClinicalRAGChain(retriever)
    return chain.invoke(question)