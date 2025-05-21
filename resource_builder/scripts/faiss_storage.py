

"""
FAISS storage module for veterinary medicine RAG system.
Handles document processing, embedding generation, and FAISS index management.

Currently, to use FAISS with Numpy, 
the Numpy version must be lower than 2: pip install 'numpy<2.0.0'
https://github.com/facebookresearch/faiss/issues/3526
"""

import json
import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from tqdm import tqdm
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared import dunder_info
from shared.veterinary_utils.embedding_model import EmbeddingModel
dunder_info.inject_dunder(__name__) # injects the variables

class FaissStorage:
    """
    FAISS storage system for veterinary medicine SmPCs.
    Processes JSON documents, generates embeddings, and builds the FAISS index.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, embedding_dim: int = 384, 
                 separator: str = "@", store_full_documents: bool = True, verbose: bool = True):
        """
        Initializes the FAISS storage system.
        
        Args:
            embedding_model: Model to generate embeddings with get_embeddings method
            embedding_dim: Dimension of the generated embeddings
            separator: Separator used in chunk identifiers
            store_full_documents: Whether to store full documents in cache
            verbose: Increase output verbosity
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.separator = separator
        self.store_full_documents = store_full_documents
        self.verbose = verbose
        
        # FAISS index
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product index (cosine similarity)
        
        # Mappings for retrieval
        self.id_to_embedding_info = {}  # Mapping from FAISS ID to embedding information
        self.document_cache = {}  # Cache of complete documents (optional)
        self.chunks_cache = {}  # Cache of chunks with their text and metadata
    
    def process_document(self, json_path: str) -> Dict[str, Dict]:
        """
        Processes a JSON document and extracts fragments for embeddings.
        
        Args:
            json_path: Path to the JSON file of the SmPC
        
        Returns:
            Dictionary with fragments and their metadata
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        
        # Extract document ID
        doc_id = doc.get('document_id', os.path.basename(json_path).split('.')[0])
        
        # Store complete document in cache if enabled
        if self.store_full_documents:
            self.document_cache[doc_id] = doc
        
        chunks = {}
        
        # Extract relevant metadata
        doc_link = doc.get('url', '')
        med_name = doc.get('nombre_medicamento', '')
        lab_titular = doc.get('laboratorio_titular', '')
        fecha_autorizacion = doc.get('fecha_primera_autorizacion', '')
        atc = "Códigos ATC:\n" + "\n".join(
            f"- {item.get('codigo', '')}: {item.get('nombre', '')} (Nivel {item.get('nivel', '')})" \
                for item in doc.get("codigos_atc", [])
        )
        target_species = doc.get('especies_destino', '')
        active_ingredients = doc.get('principios_activos', '')
        excipients = doc.get('escipientes', '')
        pharm_form = doc.get('forma_farmaceutica', '')
        dis_conditions = doc.get('condiciones_dispensacion', '')
        admin_conditions = doc.get('condiciones_administracion', '')
        antibiotic = doc.get('antibiotico', False)
        
        # Complete document as a chunk
        doc_text = f"Ficha técnica: {med_name}\n"
        doc_text += f"Enlace: {doc_link}\n"
        doc_text += f"Laboratorio titular: {lab_titular}\n"
        doc_text += f"Fecha de primera autorización: {fecha_autorizacion}\n"
        doc_text += f"Especies de destino: {target_species}\n"
        doc_text += f"{atc}\n"
        doc_text += f"Principios activos: {active_ingredients}\n"
        doc_text += f"Excipientes: {excipients}\n"
        doc_text += f"Forma farmacéutica: {pharm_form}\n"
        doc_text += f"Condiciones de dispensación: {dis_conditions}\n"
        doc_text += f"Condiciones de administración: {admin_conditions}\n"
        doc_text += f"Este medicamento{' no' if not antibiotic else ''} es un antibiótico."
        
        chunk_id = f"{doc_id}{self.separator}full"
        chunks[chunk_id] = {
            "text": doc_text,
            "metadata": {
                "document_id": doc_id,
                "link": doc_link,
                "chunk_type": "complete_document",
                "chunk_id": "full"
            }
        }
        
        # Process sections
        for section in doc.get('secciones', []):
            section_id = section.get('seccion_id', '')
            section_title = section.get('titulo', '')
            section_content = section.get('contenido', '')
            
            section_text = f"{section_id}. {section_title}\n{section_content}"
            chunk_id = f"{doc_id}{self.separator}{section_id}"
            
            chunks[chunk_id] = {
                "text": med_name + "\n\n" + section_text,
                "metadata": {
                    "document_id": doc_id,
                    "chunk_type": "section",
                    "chunk_id": section_id,
                    "title": section_title,
                    "path": [section_id]
                }
            }
            
            # Process subsections
            for subsection in section.get('subsecciones', []):
                subsection_id = subsection.get('seccion_id', '')
                subsection_title = subsection.get('titulo', '')
                subsection_content = subsection.get('contenido', '')
                
                subsection_text = f"{section_id}. {section_title} - {subsection_id} {subsection_title}\n{subsection_content}"
                chunk_id = f"{doc_id}{self.separator}{subsection_id}"
                
                chunks[chunk_id] = {
                    "text": med_name + "\n\n" + subsection_text,
                    "metadata": {
                        "document_id": doc_id,
                        "chunk_type": "subsection",
                        "chunk_id": subsection_id,
                        "title": subsection_title,
                        "path": [section_id, subsection_id],
                        "parent_id": section_id
                    }
                }
        
        # Store chunks in the cache
        self.chunks_cache.update(chunks)
        
        return chunks
    
    def add_document(self, json_path: str) -> None:
        """
        Processes a document, generates embeddings, and adds them to the FAISS index.
        
        Args:
            json_path: Path to the JSON file of the technical data sheet
        """
        # Process document and get chunks
        chunks = self.process_document(json_path)
        
        # Prepare texts and their IDs
        texts = []
        chunk_ids = []
        
        for chunk_id, chunk_data in chunks.items():
            texts.append(chunk_data["text"])
            chunk_ids.append(chunk_id)
        
        # Generate embeddings in batches
        # convert to numpy format for FAISS
        # normalize for cosine similarity
        embeddings = self.embedding_model.get_embeddings(
            texts, 
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Get current number of vectors in the index
        start_idx = self.index.ntotal
        
        # Add embeddings to the index
        self.index.add(embeddings)
        
        # Save mapping of IDs to information
        for i, chunk_id in enumerate(chunk_ids):
            faiss_id = start_idx + i
            self.id_to_embedding_info[faiss_id] = {
                "chunk_id": chunk_id,
                "metadata": chunks[chunk_id]["metadata"]
            }
    
    def add_documents_from_directory(self, directory_path: str) -> None:
        """
        Processes all JSON files in a directory and adds them to the index.
        
        Args:
            directory_path: Path to the directory with JSON files
        """
        files_in_dir = os.listdir(directory_path)
        for filename in tqdm(files_in_dir, desc="Storing in FAISS", disable= not self.verbose):
            if filename.endswith('.json'):
                json_path = os.path.join(directory_path, filename)
                self.add_document(json_path)
    
    def clear_document_cache(self) -> None:
        """
        Clears the document cache to free memory.
        Only call this after all documents have been processed if you don't need rich context.
        """
        self.document_cache = {}
        self.store_full_documents = False
    
    def save_index(self, index_path: str, mapping_path: str, chunks_path: str = None) -> None:
        """
        Saves the FAISS index and mapping information.
        
        Args:
            index_path: Path to save the index
            mapping_path: Path to save the mapping
            chunks_path: Optional path to save the chunks cache
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save mapping
        with open(mapping_path, 'w', encoding='utf-8') as f:
            # Convert int keys to strings for JSON
            json.dump(
                {str(k): v for k, v in self.id_to_embedding_info.items()}, 
                f,
                indent=4, 
                ensure_ascii=False
            )
        
        # Optionally save chunks cache
        if chunks_path:
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks_cache, f, indent=4, ensure_ascii=False)

def example_usage():
    """Example usage of the FAISS storage system"""
    
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    sim_path = os.path.join(script_dir, "../../models/similarity_model")
    faiss_index_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/index.faiss")
    mapping_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/mapping.json")
    chunks_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/chunks.json")
    jsons_path = os.path.join(script_dir, "../../data/posteriori_resources/processed_json")
    
    # Normalize paths
    sim_path = os.path.abspath(sim_path)
    faiss_index_path = os.path.abspath(faiss_index_path)
    mapping_path = os.path.abspath(mapping_path)
    chunks_path = os.path.abspath(chunks_path)
    jsons_path = os.path.abspath(jsons_path)
    
    # Create embedding model
    embedding_model = EmbeddingModel(
        sim_path,
        "cuda", #"cpu",
        512
    )
    
    # Create FAISS storage
    faiss_storage = FaissStorage(
        embedding_model,
        embedding_dim=embedding_model.get_word_embedding_dimension()
    )
    
    # Add documents
    faiss_storage.add_documents_from_directory(jsons_path)
    
    # Save index and chunks cache
    faiss_storage.save_index(
        faiss_index_path, 
        mapping_path,
        chunks_path
    )
    
    # Clear document cache to save memory (optional)
    # Only if you don't need hierarchical context
    # faiss_storage.clear_document_cache()
    
if __name__ == "__main__":
    example_usage()





