"""
enhanced_rag_config.py - Advanced Configuration V9.0
====================================================
✅ Chain of Verification (CoVe) settings
✅ Query expansion for failed retrievals
✅ Multi-step reasoning configuration
✅ Caching and performance optimization
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class EnhancedRAGConfig:
    """Enhanced RAG System Configuration with CoVe and Query Decomposition"""
    
    groq_api_key: str = ""
    model_name: str = "llama-3.3-70b-versatile"
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    
    enable_self_rag: bool = True
    self_rag_threshold: float = 0.8
    
    enable_cove: bool = True
    cove_max_retries: int = 2
    cove_verification_questions: int = 3
    
    enable_query_expansion: bool = True
    query_expansion_templates: List[str] = field(default_factory=lambda: [
        "What is {query}?",
        "Explain {query} in detail",
        "Show examples of {query}",
        "{query} definition and applications"
    ])
    
    enable_query_decomposition: bool = True
    max_sub_queries: int = 3
    
    cache_embeddings: bool = True
    cache_ttl_seconds: int = 3600
    
    enable_multimodal: bool = True
    extract_tables: bool = True
    extract_equations: bool = True
    extract_figures: bool = True
    
    table_extraction_confidence: float = 0.75
    equation_extraction_min_length: int = 3
    
    max_context_length: int = 8000
    temperature: float = 0.1
    
    enable_verification_cache: bool = True
    verification_cache_size: int = 100
    
    parallel_retrieval: bool = True
    max_parallel_chunks: int = 10
    
    enable_answer_fusion: bool = True
    fusion_method: str = "weighted_average"
    
    logging_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'groq_api_key': '***HIDDEN***',
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k_retrieval': self.top_k_retrieval,
            'enable_self_rag': self.enable_self_rag,
            'enable_cove': self.enable_cove,
            'enable_query_expansion': self.enable_query_expansion,
            'enable_query_decomposition': self.enable_query_decomposition,
            'enable_multimodal': self.enable_multimodal,
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.groq_api_key:
            raise ValueError("groq_api_key is required")
        
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be >= 100")
        
        if self.top_k_retrieval < 1:
            raise ValueError("top_k_retrieval must be >= 1")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        return True