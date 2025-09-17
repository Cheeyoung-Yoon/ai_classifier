from sentence_transformers import SentenceTransformer
import tqdm
from typing import List, Union

class VectorEmbedding:
    def __init__(self, model_name: str = 'jhgan/ko-sroberta-multitask'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list:
        """
        텍스트를 임베딩하여 리스트 형태로 반환
        """
        embedding_tensor = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding_tensor.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[list]:
        """
        여러 텍스트를 배치로 임베딩 처리
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

class EmbeddingProcessor:
    """
    Stage 3에서 사용할 임베딩 처리기
    """
    def __init__(self, model_name: str = 'jhgan/ko-sroberta-multitask'):
        self.embedder = VectorEmbedding(model_name)
    
    def embed_words(self, words: List[str]) -> List[list]:
        """단어 수준 임베딩"""
        return self.embedder.embed_batch(words)
    
    def embed_sentences(self, sentences: List[str]) -> List[list]:
        """문장 수준 임베딩"""
        return self.embedder.embed_batch(sentences)