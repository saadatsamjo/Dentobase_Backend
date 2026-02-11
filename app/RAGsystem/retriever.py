# app/RAGsystem/retriever.py
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_ollama import ChatOllama, OllamaLLM
from app.RAGsystem.embeddings import embedding_provider
from config.ragconfig import rag_settings
import logging

logger = logging.getLogger(__name__)

class RetrieverFactory:
    """Factory for creating different retrieval strategies."""
    
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or rag_settings.PERSIST_DIR
        self.embeddings = embedding_provider.get_embeddings()
        self.vectorstore = None
    
    def _get_vectorstore(self):
        """Lazy load vectorstore."""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        return self.vectorstore
    
    def create_similarity_retriever(self, k: int = None, threshold: float = None):
        """Standard similarity-based retrieval."""
        k = k or rag_settings.RETRIEVAL_K
        search_kwargs = {"k": k}
        
        if threshold or rag_settings.SIMILARITY_THRESHOLD:
            search_kwargs["score_threshold"] = threshold or rag_settings.SIMILARITY_THRESHOLD
        
        logger.info(f"Creating similarity retriever (k={k})")
        return self._get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    def create_mmr_retriever(self, k: int = None, fetch_k: int = None, lambda_mult: float = None):
        """
        Maximal Marginal Relevance - balances relevance with diversity.
        Best for clinical queries to get comprehensive coverage.
        """
        k = k or rag_settings.RETRIEVAL_K
        fetch_k = fetch_k or rag_settings.FETCH_K
        lambda_mult = lambda_mult or rag_settings.LAMBDA_MULT
        
        logger.info(f"Creating MMR retriever (k={k}, fetch_k={fetch_k}, lambda={lambda_mult})")
        
        return self._get_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )
    
    def create_multi_query_retriever(self, llm_model: str = None):
        """
        Generates multiple query variations to improve recall.
        Uses LLM to reformulate the query in different ways.
        """
        base_retriever = self.create_mmr_retriever(k=4, fetch_k=15)
        
        llm = ChatOllama(
            model=llm_model or rag_settings.OLLAMA_LLM_MODEL,
            temperature=0.1
        )
        
        logger.info("Creating multi-query retriever")
        mutiquery = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        print(f"Created MultiQueryRetriever with {mutiquery.get_relevant_documents(query='management of pulpitis')}")
        return mutiquery
    
    def create_compression_retriever(self, base_retriever=None):
        """
        Uses LLM to filter irrelevant content from retrieved documents.
        Higher precision but slower. Good for noisy documents.
        """
        llm = OllamaLLM(model=rag_settings.OLLAMA_LLM_MODEL)
        compressor = LLMChainExtractor.from_llm(llm)
        
        base = base_retriever or self.create_mmr_retriever()
        
        logger.info("Creating compression retriever")
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base
        )
        
    def create_similarity_score_threshold_retriever(self, k: int = None, threshold: float = None):
        """
        Similarity-based retriever with score threshold.
        """
        k = k or rag_settings.RETRIEVAL_K
        search_kwargs = {"k": k}
        
        if threshold or rag_settings.SIMILARITY_THRESHOLD:
            search_kwargs={"k": rag_settings.RETRIEVAL_K, "score_threshold": threshold or rag_settings.SIMILARITY_THRESHOLD}
        
        logger.info(f"Creating similarity score threshold retriever (k={k}), with threshold={threshold}")
        return self._get_vectorstore().as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )
    
    def get_retriever(self, retriever_type: str = None):
        """
        Get retriever based on configuration or specified type.
        """
        rtype = retriever_type or rag_settings.RETRIEVER_TYPE
        
        if rtype == "similarity":
            return self.create_similarity_retriever()
        elif rtype == "mmr":
            return self.create_mmr_retriever()
        elif rtype == "multi_query":
            return self.create_multi_query_retriever()
        elif rtype == "similarity_score_threshold":
            return self.create_similarity_score_threshold_retriever()
        else:
            raise ValueError(f"Unknown retriever type: {rtype}")

def get_retriever(persist_dir: str = None, strategy: str = None):
    """
    Convenience function for backward compatibility.
    """
    factory = RetrieverFactory(persist_dir)
    return factory.get_retriever(strategy)






# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings


# def get_retriever(persist_dir: str = "chroma_db"):

#     # Retrieving chunks from Chroma DB
#     # 1. Using Ollama embedding models
#     embeddings = OllamaEmbeddings(model="mxbai-embed-large")
#     vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

#     # 2. Using HuggingFace embedding models
#     # embeddings = HuggingFaceEmbeddings(model_name=rag_settings.EMBEDDING_MODEL, model_kwargs={"device": rag_settings.EMBEDDING_DEVICE}, encode_kwargs={'normalize_embeddings': True})
#     # vectorstore = Chroma(
#     #     persist_directory=persist_dir,
#     #     embedding_function=embeddings
#     # )

#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#     # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})
#     # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 10,})
#     # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 10, "lambda_mult": 0.5})
#     # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 8, "score_threshold": 0.8})

#     return retriever
