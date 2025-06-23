# shared/veterinary_utils/jina_embedding_model.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Union

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

class JinaEmbeddingModel:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3", device: str = "cuda", max_seq_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.device = device
        self.max_seq_length = max_seq_length

    def get_word_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def get_embeddings(self, 
                       texts: List[str], 
                       batch_size: int = 64, 
                       show_progress_bar: bool = False, 
                       convert_to_numpy: bool = True, 
                       normalize_embeddings: bool = True) -> Union[np.ndarray, List]:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]  # CLS token

                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.detach().cpu().to(torch.float32))

        final = torch.cat(all_embeddings, dim=0)
        if convert_to_numpy:
            return final.numpy().astype(np.float32)
        else:
            return final
