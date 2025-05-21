


"""
FAISS search module for veterinary medicine RAG system.
Handles FAISS index loading and querying.

Currently, to use FAISS with Numpy, 
the Numpy version must be lower than 2: pip install 'numpy<2.0.0'
https://github.com/facebookresearch/faiss/issues/3526
"""


import json
import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared import dunder_info
from shared.veterinary_utils.embedding_model import EmbeddingModel
dunder_info.inject_dunder(__name__) # injects the variables

class FaissSearch:
    """
    FAISS search system for veterinary medicine SmPCs.
    Loads FAISS index and provides search functionality.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, separator: str = "@"):
        """
        Initializes the FAISS search system.
        
        Args:
            embedding_model: Model to generate embeddings with get_embeddings method
            separator: Separator used in chunk identifiers
        """
        self.embedding_model = embedding_model
        self.separator = separator
        
        # FAISS index (will be loaded)
        self.index = None
        
        # Mappings for retrieval
        self.id_to_embedding_info = {}  # Mapping from FAISS ID to embedding information
        self.document_cache = {}  # Cache of complete documents (optional)
        self.chunks_cache = {}  # Cache of chunks with their text and metadata
    
    def load_index(self, index_path: str, mapping_path: str, chunks_path: str = None, 
                   documents_directory: str = None) -> None:
        """
        Loads a FAISS index and mapping information.
        
        Args:
            index_path: Path to the index
            mapping_path: Path to the mapping
            chunks_path: Optional path to the chunks cache
            documents_directory: Optional path to load full documents
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            # Convert keys from string to int during loading
            mapping_data = json.load(f)
            self.id_to_embedding_info = {int(k): v for k, v in mapping_data.items()}
        
        # Optionally load chunks cache
        if chunks_path and os.path.exists(chunks_path):
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks_cache = json.load(f)
        
        # Optionally load full documents
        if documents_directory and os.path.exists(documents_directory):
            for filename in os.listdir(documents_directory):
                if filename.endswith('.json'):
                    json_path = os.path.join(documents_directory, filename)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                        doc_id = doc.get('document_id', filename.split('.')[0])
                        self.document_cache[doc_id] = doc
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the k most relevant fragments for the query.
        
        Args:
            query: User query
            k: Number of results to retrieve
            
        Returns:
            List of dictionaries with retrieved fragments and their information
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
            
        # Generate embedding for the query
        query_embedding = self.embedding_model.get_embeddings(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in the index
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # If the index is not valid, continue
            if idx == -1 or idx not in self.id_to_embedding_info:
                continue
            
            # Get chunk information
            chunk_info = self.id_to_embedding_info[idx]
            metadata = chunk_info["metadata"]
            chunk_id = chunk_info["chunk_id"]
            
            # Retrieve the original text directly from chunks cache
            chunk_data = self.chunks_cache.get(chunk_id, {})
            original_text = chunk_data.get("text", "")
            
            results.append({
                "score": float(score),
                "chunk_id": chunk_id,
                "metadata": metadata,
                "text": original_text
            })
            
        return results
    
    def get_context(self, chunk_id: str) -> Dict[str, Any]:
        """
        Retrieves contextual information for a fragment.
        
        Args:
            chunk_id: ID of the fragment for which context is desired
            
        Returns:
            Dictionary with contextual information
        """
        # Extract information from chunk ID
        parts = chunk_id.split(self.separator)
        doc_id = parts[0]
        
        # Get chunk metadata
        chunk_data = self.chunks_cache.get(chunk_id)
        if not chunk_data:
            return {"error": "Chunk not found in cache"}
            
        chunk_metadata = chunk_data.get("metadata", {})
        
        # Initialize context with basic information
        context = {
            "document_id": doc_id,
            "chunk_metadata": chunk_metadata,
            "related_content": []
        }
        
        # If we have the full document, add document title and enrich with hierarchical context
        if doc_id in self.document_cache:
            document = self.document_cache[doc_id]
            context["document_title"] = document.get("nombre_medicamento", "")
            
            # Add contextual information based on the type of chunk
            chunk_type = chunk_metadata.get("chunk_type", "")
            
            if chunk_type == "section":
                # For a section, add all its subsections
                section_id = chunk_metadata.get("chunk_id", "")
                for section in document.get("secciones", []):
                    if section.get("seccion_id") == section_id:
                        context["section_data"] = section
                        break
            
            elif chunk_type == "subsection":
                # For a subsection, add the parent section and sibling subsections
                path = chunk_metadata.get("path", [])
                if len(path) >= 1:
                    parent_id = path[0]
                    for section in document.get("secciones", []):
                        if section.get("seccion_id") == parent_id:
                            context["parent_section"] = section
                            subsec_id = chunk_metadata.get("chunk_id")
                            # Add related subsections excluding the current one
                            context["related_subsections"] = [
                                subsec for subsec in section.get("subsecciones", [])
                                if subsec.get("subseccion_id") != subsec_id
                            ]
                            break
        
        else:
            # If we don't have the full document, try to reconstruct context from chunks cache
            # Get document title from any chunk of the same document
            for other_chunk_id, other_chunk_data in self.chunks_cache.items():
                if other_chunk_id.startswith(doc_id) and "text" in other_chunk_data:
                    # Extract document title from the first line of any chunk
                    first_line = other_chunk_data["text"].split("\n")[0]
                    context["document_title"] = first_line
                    break
                    
            # Find related chunks based on path information
            path = chunk_metadata.get("path", [])
            if path:
                for other_chunk_id, other_chunk_data in self.chunks_cache.items():
                    if other_chunk_id != chunk_id and other_chunk_id.startswith(doc_id):
                        other_metadata = other_chunk_data.get("metadata", {})
                        other_path = other_metadata.get("path", [])
                        
                        # Check if this chunk is related (parent, child, or sibling)
                        if (len(other_path) > 0 and len(path) > 0 and 
                            (other_path[0] == path[0] or  # Same section
                             (len(path) > 1 and len(other_path) > 1 and other_path[1] == path[1]))):  # Same subsection
                            
                            context["related_content"].append({
                                "chunk_id": other_chunk_id,
                                "metadata": other_metadata,
                                "relationship": "related"  # Could be refined to parent/child/sibling
                            })
        
        return context
        
    def search_with_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the k most relevant fragments and enriches with context.
        
        Args:
            query: User query
            k: Number of results to retrieve
            
        Returns:
            List of results enriched with context
        """
        # Get basic results
        results = self.search(query, k)
        
        # Enrich each result with context
        for result in results:
            chunk_id = result.get("chunk_id")
            context = self.get_context(chunk_id)
            result["context"] = context
            
        return results

def show_results(query: str, results: List[Dict[str, Any]]) -> None:
    """
    Format and display search results.
    
    Args:
        query: The user query
        results: List of search results
    """
    print(f"\nQuery: {query}\n")
    print("=" * 50)
    
    # Process results
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result['score']:.4f}):")
        text_info = result.get('text', 'None')
        #print(f"Fragment: {text_info[:100]}...")
        print(f"Fragment: {text_info}")
        print(f"Document: {result['metadata']['document_id']}")
        print(f"Type: {result['metadata']['chunk_type']}")
        
        if 'path' in result['metadata']:
            print(f"Path: {result['metadata']['path']}")
            
        print("-" * 50)

def example_usage():
    """Example usage of the FAISS search system"""
    
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths based on the script directory
    #sim_path = os.path.join(script_dir, "../../models/similarity_model")
    sim_path = "intfloat/multilingual-e5-large"
    faiss_index_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/index.faiss")
    mapping_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/mapping.json")
    chunks_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/chunks.json")
    jsons_path = os.path.join(script_dir, "../../data/posteriori_resources/processed_json")
    
    # Normalize paths
    #sim_path = os.path.abspath(sim_path)
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
    
    # Create FAISS search
    faiss_search = FaissSearch(embedding_model)
    
    # Load index and chunks cache
    faiss_search.load_index(
        faiss_index_path, 
        mapping_path,
        chunks_path,
        documents_directory=jsons_path  # Optional, for rich context
    )
    
    # Perform search
    query = "¿puedo usar lincomicina en perros?"
    results = faiss_search.search_with_context(query, k=3)
    show_results(query, results)

if __name__ == "__main__":
    example_usage()





