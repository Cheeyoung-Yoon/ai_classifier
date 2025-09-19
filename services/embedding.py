"""
Shared Embedding Service
Provides cached SentenceTransformer for all nodes in the pipeline.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """
    Singleton pattern embedding provider that caches SentenceTransformer instances.
    This reduces memory usage and improves performance by avoiding multiple model loads.
    """
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingProvider, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            logger.info("Initializing EmbeddingProvider")
    
    def get_encoder(self, model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
        """
        Get or create a cached SentenceTransformer encoder.
        
        Args:
            model_name: Name of the sentence transformer model
            
        Returns:
            Cached SentenceTransformer instance
        """
        if model_name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                self._models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded model: {model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed. Using dummy encoder.")
                self._models[model_name] = DummyEncoder()
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                self._models[model_name] = DummyEncoder()
        
        return self._models[model_name]
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts using the cached model.
        
        Args:
            texts: Text or list of texts to encode
            model_name: Name of the model to use
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        encoder = self.get_encoder(model_name)
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = encoder.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=show_progress
            )
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}")
            # Return dummy embeddings as fallback
            return np.random.rand(len(texts), 384)  # Default dimension
    
    def clear_cache(self):
        """Clear all cached models to free memory."""
        logger.info("Clearing embedding model cache")
        self._models.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        return {
            "cached_models": list(self._models.keys()),
            "cache_size": len(self._models)
        }


class DummyEncoder:
    """Dummy encoder for fallback when SentenceTransformer is not available."""
    
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        logger.warning(f"Using dummy encoder for {len(texts)} texts")
        return np.random.rand(len(texts), 384)


# Singleton instance for global use
embedding_provider = EmbeddingProvider()


def get_embedding_provider() -> EmbeddingProvider:
    """
    Get the global embedding provider instance.
    
    Returns:
        Global EmbeddingProvider instance
    """
    return embedding_provider


class VectorEmbedding:
    """
    Legacy wrapper for backward compatibility.
    New code should use EmbeddingProvider directly.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.provider = get_embedding_provider()
        logger.warning("VectorEmbedding is deprecated. Use EmbeddingProvider directly.")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Legacy encode method."""
        return self.provider.encode(
            texts, 
            model_name=self.model_name,
            batch_size=batch_size,
            show_progress=show_progress
        )