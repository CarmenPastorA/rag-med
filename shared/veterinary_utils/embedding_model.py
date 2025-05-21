


"""
Embedding model for veterinary medicine RAG system.
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

try:
    # it is imported as part of the package
    from shared import dunder_info
    dunder_info.inject_dunder(__name__) # injects the variables
except ImportError:
    # is executed directly or the absolute path is not in sys.path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from shared import dunder_info
    dunder_info.inject_dunder(__name__) # injects the variables

class EmbeddingModel:
    """
    Sentence Transformers embedding model wrapper.
    SentenceTransformer(
      (0): Transformer
      (1): Pooling
      (2): Normalize
    )
    """
    
    def __init__(self, embed_model_name: str, embed_device: str, embed_max_seq_length: int):
        """
        Initialize embedding model.
        
        Args:
            embed_model_name: Path to the embedding model
            embed_device: Device to run the model on ('cpu' or 'cuda')
            embed_max_seq_length: Model context size
        """
        print(f"Loading embedding model on device: {embed_device}")
        self.embedding_model = SentenceTransformer(embed_model_name, device=embed_device)
        self.embedding_model.max_seq_length = embed_max_seq_length

    def get_word_embedding_dimension(self) -> int:
        """
        Return the value of the word_embedding_dimension in Pooling layer.
        
        Returns:
            Dimension of the word embeddings
        """
        return self.embedding_model[1].word_embedding_dimension
    
    def get_embeddings(self, 
                       texts: List[str], 
                       batch_size: int = 64, 
                       show_progress_bar: bool = False, 
                       convert_to_numpy: bool = True, 
                       normalize_embeddings: bool = True) -> Union[np.ndarray, List]:
        """
        Get embeddings from texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress_bar: Whether to show a progress bar
            convert_to_numpy: Whether to convert output to numpy array
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Embeddings as numpy array or list
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )
        return embeddings



