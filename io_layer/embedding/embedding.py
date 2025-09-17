# %%
from sentence_transformers import SentenceTransformer
import tqdm

class VectorEmbedding:
    def __init__(self, model_name: str = 'jhgan/ko-sroberta-multitask'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list:
        """
        텍스트를 임베딩하여 리스트 형태로 반환
        """
        embedding_tensor = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding_tensor.tolist()
    
    