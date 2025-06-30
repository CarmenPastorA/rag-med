# faiss_search.py

"""
Hierarchical FAISS search module for veterinary medicine RAG system.
Handles FAISS index loading and querying.

Currently, to use FAISS with Numpy, 
the Numpy version must be lower than 2: pip install 'numpy<2.0.0'
https://github.com/facebookresearch/faiss/issues/3526
"""

import json
import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared import dunder_info
from shared.veterinary_utils.utils import vprint
from shared.veterinary_utils.embedding_model import EmbeddingModel
dunder_info.inject_dunder(__name__)  # injects the variables

class FaissSearch:
    """
    Simple FAISS searcher over precomputed chunk embeddings.
    Loads:
    - chunks_index.faiss: FAISS index
    - chunks_mapping.json: FAISS ID -> chunk metadata
    - chunks_cache.json: chunk_id -> full text
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.id_to_info = {}
        self.chunk_cache = {}

    def load_index(self, index_path: str, mapping_path: str, cache_path: str) -> None:
        """
        Loads the FAISS index and mapping data.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        self.index = faiss.read_index(index_path)

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            self.id_to_info = {int(k): v for k, v in mapping.items()}

        with open(cache_path, "r", encoding="utf-8") as f:
            self.chunk_cache = json.load(f)

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform a search over the chunk index.
        Returns top-k results sorted by FAISS similarity score.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded.")

        query_emb = self.embedding_model.get_embeddings(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, actual_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx not in self.id_to_info:
                continue
            chunk_info = self.id_to_info[idx]
            chunk_id = chunk_info["chunk_id"]
            text = self.chunk_cache.get(chunk_id, {}).get("text", "")

            results.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": chunk_info["metadata"],
                "score": float(scores[0][i])
            })

        return results


class HierarchicalFaissSearch:
    """
    Hierarchical FAISS search system for veterinary medicine SmPCs.
    
    This class handles searching through previously stored FAISS indices using
    a two-stage retrieval approach:
    1. First stage: Retrieves relevant documents based on essential information
    2. Second stage: Retrieves specific chunks from the relevant documents
    
    The class loads pre-built FAISS indices and their associated mappings to
    perform efficient similarity searches.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, separator: str = "@", verbose: bool = False):
        """
        Initialize the hierarchical FAISS search system.
        
        Args:
            embedding_model: Model to generate embeddings with get_embeddings method
            separator: Separator used in chunk identifiers (must match the one used during storage)
            verbose: Enable verbose output for debugging and monitoring
        """
        self.embedding_model = embedding_model
        self.separator = separator
        self.verbose = verbose
        
        # First stage: Essential info index for document-level retrieval
        self.essential_index = None
        self.essential_id_to_info = {}  # Maps FAISS ID to document info
        self.essential_cache = {}  # Cache of essential info texts
        
        # Second stage: Detailed chunks index for chunk-level retrieval
        self.chunks_index = None
        self.chunks_id_to_info = {}  # Maps FAISS ID to chunk info
        self.chunks_cache = {}  # Cache of chunks with their text and metadata
        
        # Status flags
        self.essential_index_loaded = False
        self.chunks_index_loaded = False
    
    def load_essential_index(self, essential_index_path: str, essential_mapping_path: str, 
                           essential_cache_path: str) -> None:
        """
        Load the essential information index and its associated data.
        
        This loads the first-stage index used for document-level retrieval based on
        essential information like medication name, laboratory, target species, etc.
        
        Args:
            essential_index_path: Path to the essential info FAISS index file
            essential_mapping_path: Path to the JSON file containing FAISS ID to document info mapping
            essential_cache_path: Path to the JSON file containing cached essential info texts
            
        Raises:
            FileNotFoundError: If any of the required files cannot be found
            ValueError: If the loaded data is invalid or corrupted
        """
        try:
            # Load FAISS index
            if not os.path.exists(essential_index_path):
                raise FileNotFoundError(f"Essential index file not found: {essential_index_path}")
            
            self.essential_index = faiss.read_index(essential_index_path)
            
            # Load ID to info mapping
            if not os.path.exists(essential_mapping_path):
                raise FileNotFoundError(f"Essential mapping file not found: {essential_mapping_path}")
            
            with open(essential_mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                # Convert string keys back to integers
                self.essential_id_to_info = {int(k): v for k, v in mapping_data.items()}
            
            # Load essential info cache
            if not os.path.exists(essential_cache_path):
                raise FileNotFoundError(f"Essential cache file not found: {essential_cache_path}")
            
            with open(essential_cache_path, 'r', encoding='utf-8') as f:
                self.essential_cache = json.load(f)
            
            self.essential_index_loaded = True
            
            vprint(f"Essential index loaded successfully:", self.verbose)
            vprint(f"  - Index size: {self.essential_index.ntotal} documents", self.verbose)
            vprint(f"  - Mapping entries: {len(self.essential_id_to_info)}", self.verbose)
            vprint(f"  - Cache entries: {len(self.essential_cache)}", self.verbose)
                
        except Exception as e:
            self.essential_index_loaded = False
            raise ValueError(f"Failed to load essential index: {str(e)}")
    
    def load_chunks_index(self, chunks_index_path: str, chunks_mapping_path: str, 
                         chunks_cache_path: str) -> None:
        """
        Load the chunks index and its associated data.
        
        This loads the second-stage index used for chunk-level retrieval from
        specific sections and subsections of the documents.
        
        Args:
            chunks_index_path: Path to the chunks FAISS index file
            chunks_mapping_path: Path to the JSON file containing FAISS ID to chunk info mapping
            chunks_cache_path: Path to the JSON file containing cached chunk texts and metadata
            
        Raises:
            FileNotFoundError: If any of the required files cannot be found
            ValueError: If the loaded data is invalid or corrupted
        """
        try:
            # Load FAISS index
            if not os.path.exists(chunks_index_path):
                raise FileNotFoundError(f"Chunks index file not found: {chunks_index_path}")
            
            self.chunks_index = faiss.read_index(chunks_index_path)
            
            # Load ID to info mapping
            if not os.path.exists(chunks_mapping_path):
                raise FileNotFoundError(f"Chunks mapping file not found: {chunks_mapping_path}")
            
            with open(chunks_mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                # Convert string keys back to integers
                self.chunks_id_to_info = {int(k): v for k, v in mapping_data.items()}
            
            # Load chunks cache
            if not os.path.exists(chunks_cache_path):
                raise FileNotFoundError(f"Chunks cache file not found: {chunks_cache_path}")
            
            with open(chunks_cache_path, 'r', encoding='utf-8') as f:
                self.chunks_cache = json.load(f)
            
            self.chunks_index_loaded = True
            
            vprint(f"Chunks index loaded successfully:", self.verbose)
            vprint(f"  - Index size: {self.chunks_index.ntotal} chunks", self.verbose)
            vprint(f"  - Mapping entries: {len(self.chunks_id_to_info)}", self.verbose)
            vprint(f"  - Cache entries: {len(self.chunks_cache)}", self.verbose)
        
        except Exception as e:
            self.chunks_index_loaded = False
            raise ValueError(f"Failed to load chunks index: {str(e)}")
    
    def load_indices(self, essential_index_path: str, essential_mapping_path: str, essential_cache_path: str,
                    chunks_index_path: str, chunks_mapping_path: str, chunks_cache_path: str) -> None:
        """
        Load both essential and chunks indices in one call.
        
        This is a convenience method that loads both indices required for hierarchical search.
        
        Args:
            essential_index_path: Path to the essential info FAISS index file
            essential_mapping_path: Path to the essential info mapping JSON file
            essential_cache_path: Path to the essential info cache JSON file
            chunks_index_path: Path to the chunks FAISS index file
            chunks_mapping_path: Path to the chunks mapping JSON file
            chunks_cache_path: Path to the chunks cache JSON file
        """
        self.load_essential_index(essential_index_path, essential_mapping_path, essential_cache_path)
        self.load_chunks_index(chunks_index_path, chunks_mapping_path, chunks_cache_path)
    
    def get_relevant_document_ids(self, query: str, top_k: int = 10) -> List[str]:
        """
        First stage: Retrieve relevant document IDs based on essential information.
        
        This method searches through the essential information index to find documents
        that are most relevant to the query based on high-level document characteristics
        like medication name, target species, active ingredients, etc.
        
        Args:
            query: Search query string
            top_k: Number of top documents to retrieve
            
        Returns:
            List of relevant document IDs ordered by relevance score
            
        Raises:
            RuntimeError: If the essential index is not loaded
        """
        if not self.essential_index_loaded:
            raise RuntimeError("Essential index not loaded. Call load_essential_index() first.")
        
        if self.essential_index.ntotal == 0:
            vprint("Warning: Essential index is empty", self.verbose)
            return []
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.get_embeddings(
            [query], 
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in essential info index
        actual_k = min(top_k, self.essential_index.ntotal)
        scores, indices = self.essential_index.search(query_embedding, actual_k)
        
        # Extract document IDs
        relevant_doc_ids = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.essential_id_to_info:
                doc_id = self.essential_id_to_info[idx]["document_id"]
                relevant_doc_ids.append(doc_id)
                
                vprint(f"Document {doc_id}: score {scores[0][i]:.4f}", self.verbose)
        
        vprint(f"Retrieved {len(relevant_doc_ids)} relevant documents for query: '{query}'", self.verbose)
        
        return relevant_doc_ids
    
    def get_relevant_chunks_from_documents(self, query: str, document_ids: List[str], 
                                         top_k: int = 5) -> List[Dict]:
        """
        Second stage: Retrieve relevant chunks from specified documents.
        
        This method searches through the chunks index to find the most relevant
        sections/subsections from the documents identified in the first stage.
        
        Args:
            query: Search query string
            document_ids: List of document IDs to search within (from first stage)
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of dictionaries containing chunk information:
            - chunk_id: Unique identifier for the chunk
            - text: Full text content of the chunk
            - metadata: Metadata including document_id, chunk_type, title, etc.
            - score: Similarity score from the search
            
        Raises:
            RuntimeError: If the chunks index is not loaded
        """
        if not self.chunks_index_loaded:
            raise RuntimeError("Chunks index not loaded. Call load_chunks_index() first.")
        
        if self.chunks_index.ntotal == 0:
            vprint("Warning: Chunks index is empty", self.verbose)
            return []
        
        if not document_ids:
            vprint("Warning: No document IDs provided for chunk search", self.verbose)
            return []
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.get_embeddings(
            [query], 
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in chunks index
        # We search more broadly and then filter by document IDs to ensure we get enough results
        search_k = min(top_k * len(document_ids) * 3, self.chunks_index.ntotal)
        scores, indices = self.chunks_index.search(query_embedding, search_k)
        
        # Filter results by document IDs and collect relevant chunks
        relevant_chunks = []
        document_ids_set = set(document_ids)
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.chunks_id_to_info:
                chunk_info = self.chunks_id_to_info[idx]
                chunk_doc_id = chunk_info["metadata"]["document_id"]
                
                if chunk_doc_id in document_ids_set:
                    chunk_id = chunk_info["chunk_id"]
                    if chunk_id in self.chunks_cache:
                        chunk_data = {
                            "chunk_id": chunk_id,
                            "text": self.chunks_cache[chunk_id]["text"],
                            "metadata": chunk_info["metadata"],
                            "score": float(scores[0][i])
                        }
                        relevant_chunks.append(chunk_data)
                        
                        vprint(f"Chunk {chunk_id}: score {chunk_data['score']:.4f}", self.verbose)
                        
                        if len(relevant_chunks) >= top_k:
                            break
        
        vprint(f"Retrieved {len(relevant_chunks)} relevant chunks from {len(document_ids)} documents", self.verbose)
        
        return relevant_chunks
    
    def hierarchical_search(self, query: str, top_documents: int = 10, top_chunks: int = 5) -> List[Dict]:
        """
        Perform complete hierarchical search: first retrieve relevant documents, then relevant chunks.
        
        This is the main search method that combines both stages of the hierarchical search:
        1. Find relevant documents based on essential information
        2. Find relevant chunks within those documents
        
        Args:
            query: Search query string
            top_documents: Number of top documents to retrieve in first stage
            top_chunks: Number of top chunks to retrieve in second stage
            
        Returns:
            List of dictionaries containing the most relevant chunks with their information
            
        Raises:
            RuntimeError: If either index is not loaded
        """
        if not self.essential_index_loaded or not self.chunks_index_loaded:
            raise RuntimeError("Both indices must be loaded. Call load_indices() first.")
        
        vprint(f"Starting hierarchical search for query: '{query}'", self.verbose)
        vprint(f"Stage 1: Retrieving top {top_documents} documents", self.verbose)
        
        # First stage: Get relevant document IDs
        relevant_doc_ids = self.get_relevant_document_ids(query, top_documents)
        
        if not relevant_doc_ids:
            vprint("No relevant documents found in first stage", self.verbose)
            return []
        
        vprint(f"Stage 2: Retrieving top {top_chunks} chunks from {len(relevant_doc_ids)} documents", self.verbose)
        
        # Second stage: Get relevant chunks from those documents
        relevant_chunks = self.get_relevant_chunks_from_documents(query, relevant_doc_ids, top_chunks)
        
        vprint(f"Hierarchical search completed: {len(relevant_chunks)} final results", self.verbose)
        
        return relevant_chunks
    
    def search_chunks_only(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search directly in the chunks index without document-level filtering.
        
        This method bypasses the hierarchical approach and searches directly in all chunks.
        Useful for cases where you want to search across all content regardless of document boundaries.
        
        Args:
            query: Search query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of dictionaries containing chunk information ordered by relevance
            
        Raises:
            RuntimeError: If the chunks index is not loaded
        """
        if not self.chunks_index_loaded:
            raise RuntimeError("Chunks index not loaded. Call load_chunks_index() first.")
        
        if self.chunks_index.ntotal == 0:
            vprint("Warning: Chunks index is empty", self.verbose)
            return []
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.get_embeddings(
            [query], 
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in chunks index
        actual_k = min(top_k, self.chunks_index.ntotal)
        scores, indices = self.chunks_index.search(query_embedding, actual_k)
        
        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.chunks_id_to_info:
                chunk_info = self.chunks_id_to_info[idx]
                chunk_id = chunk_info["chunk_id"]
                
                if chunk_id in self.chunks_cache:
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "text": self.chunks_cache[chunk_id]["text"],
                        "metadata": chunk_info["metadata"],
                        "score": float(scores[0][i])
                    }
                    results.append(chunk_data)
                    
                    vprint(f"Chunk {chunk_id}: score {chunk_data['score']:.4f}", self.verbose)
        
        vprint(f"Direct chunk search completed: {len(results)} results", self.verbose)
        
        return results
    
    def get_document_essential_info(self, document_id: str) -> Optional[str]:
        """
        Retrieve the essential information for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Essential information text if found, None otherwise
        """
        if not self.essential_index_loaded:
            vprint("Warning: Essential index not loaded", self.verbose)
            return None
        
        return self.essential_cache.get(document_id)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Dictionary containing chunk text and metadata if found, None otherwise
        """
        if not self.chunks_index_loaded:
            vprint("Warning: Chunks index not loaded", self.verbose)
            return None
        
        if chunk_id in self.chunks_cache:
            # Find the metadata for this chunk
            for faiss_id, info in self.chunks_id_to_info.items():
                if info["chunk_id"] == chunk_id:
                    return {
                        "chunk_id": chunk_id,
                        "text": self.chunks_cache[chunk_id]["text"],
                        "metadata": info["metadata"]
                    }
        
        return None
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded indices.
        
        Returns:
            Dictionary containing statistics about both indices
        """
        stats = {
            "essential_index_loaded": self.essential_index_loaded,
            "chunks_index_loaded": self.chunks_index_loaded,
            "essential_documents": 0,
            "total_chunks": 0,
            "essential_cache_size": 0,
            "chunks_cache_size": 0
        }
        
        if self.essential_index_loaded:
            stats["essential_documents"] = self.essential_index.ntotal
            stats["essential_cache_size"] = len(self.essential_cache)
        
        if self.chunks_index_loaded:
            stats["total_chunks"] = self.chunks_index.ntotal
            stats["chunks_cache_size"] = len(self.chunks_cache)
        
        return stats

    def get_chunk_index(self):
        """
        Return the raw FAISS index used for chunk-level retrieval.
        
        Useful for operations like reconstruct(faiss_id), which require direct access
        to the internal FAISS index structure.
        """
        if not self.chunks_index_loaded:
            raise RuntimeError("Chunks index not loaded.")
        return self.chunks_index

    def get_doc_embeddings(self, doc_ids: list[str]) -> dict[str, np.ndarray]:
        """
        Compute document-level embeddings by averaging the FAISS chunk embeddings
        for each document ID in the list.
    
        Args:
            doc_ids (list[str]): List of normalized document IDs.
    
        Returns:
            dict: Mapping from doc_id to average embedding (np.ndarray).
        """
        from collections import defaultdict
        from shared.veterinary_utils.utils import normalize_doc_id
    
        if self.chunks_cache is None or self.chunks_index is None or self.chunks_id_to_info is None:
            raise ValueError("Chunk cache/index/mapping not loaded.")
    
        doc_to_embeddings = defaultdict(list)
    
        for faiss_id, meta in self.chunks_id_to_info.items():
            doc_id = normalize_doc_id(meta["metadata"]["document_id"])
            chunk_id = meta["metadata"]["chunk_id"]
            full_chunk_id = f"{doc_id}@{chunk_id}"
    
            if doc_id in doc_ids and full_chunk_id in self.chunks_cache:
                emb = self.chunks_index.reconstruct(faiss_id)
                doc_to_embeddings[doc_id].append(emb)
    
        doc_embeddings = {}
        for doc_id, vectors in doc_to_embeddings.items():
            if vectors:
                avg_emb = np.mean(vectors, axis=0)
                norm_emb = avg_emb / np.linalg.norm(avg_emb)
                doc_embeddings[doc_id] = norm_emb.astype(np.float32)
    
        return doc_embeddings


def example_usage():
    """Example usage of the HierarchicalFaissSearch class"""
    
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths based on the script directory
    sim_path = os.path.join(script_dir, "../../models/multilingual-e5-large-local")
    
    # Paths for hierarchical indices
    essential_index_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/essential_index.faiss")
    essential_mapping_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/essential_mapping.json")
    essential_cache_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/essential_cache.json")
    
    chunks_index_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/chunks_index.faiss")
    chunks_mapping_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/chunks_mapping.json")
    chunks_cache_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/chunks_cache.json")
    
    # Normalize paths
    sim_path = os.path.abspath(sim_path)
    
    try:
        # Create embedding model
        embedding_model = EmbeddingModel(
            sim_path,
            "cpu", #"cuda",
            512
        )
        
        # Create hierarchical search instance
        search_engine = HierarchicalFaissSearch(
            embedding_model,
            separator="@",
            verbose=True
        )
        
        # Load both indices
        search_engine.load_indices(
            essential_index_path, essential_mapping_path, essential_cache_path,
            chunks_index_path, chunks_mapping_path, chunks_cache_path
        )
        
        # Get index statistics
        stats = search_engine.get_index_statistics()
        print(f"\nIndex Statistics:")
        print(f"  Essential documents: {stats['essential_documents']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        
        # Example hierarchical search
        query = "mi perro tiene insuficiencia cardíaca, qué le puedo dar?"
        #query = "a mi perro le han mandado enacard 5 mg, cuánto le doy?"
        print(f"\n{'='*60}")
        print(f"Hierarchical search for: '{query}'")
        print(f"{'='*60}")
        
        len_text_preview = 500
        results = search_engine.hierarchical_search(query, top_documents=5, top_chunks=3)
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
            print(f"Document ID: {result['metadata']['document_id']}")
            print(f"Chunk Type: {result['metadata']['chunk_type']}")
            if 'title' in result['metadata']:
                print(f"Section Title: {result['metadata']['title']}")
            print(f"Text preview ({len_text_preview} chars): {result['text'][:len_text_preview]} ...")
        
        # Example direct chunk search
        print(f"\n{'='*60}")
        print(f"Direct chunk search for: '{query}'")
        print(f"{'='*60}")
        
        direct_results = search_engine.search_chunks_only(query, top_k=3)
        
        for i, result in enumerate(direct_results):
            print(f"\n--- Direct Result {i+1} (Score: {result['score']:.4f}) ---")
            print(f"Document ID: {result['metadata']['document_id']}")
            print(f"Chunk Type: {result['metadata']['chunk_type']}")
            print(f"Text preview ({len_text_preview} chars): {result['text'][:len_text_preview]} ...")
        
        # Example of getting essential info for a document
        if results:
            doc_id = results[0]['metadata']['document_id']
            essential_info = search_engine.get_document_essential_info(doc_id)
            if essential_info:
                print(f"\n{'='*60}")
                print(f"Essential info for document {doc_id}:")
                print(f"{'='*60}")
                print(essential_info)
        
    except Exception as e:
        print(f"Error during example execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Running HierarchicalFaissSearch example...")
    example_usage()
