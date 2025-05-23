

"""
BM25 search module for veterinary medicine RAG system.

Uses: 
- rank-bm25: only depends on numpy
"""


import os
from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.veterinary_utils.utils import (get_text, 
                                           get_file_filename_ext,
                                           get_pickle,
                                           vprint)
from resource_builder.scripts.bm25_storage import BM25Storage
from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

class BM25Search:
    """
    Class for searching documents using a pre-built BM25 index.
    Compatible with indices created by BM25Storage, including those with automatic stopwords generation.
    """
    
    def __init__(self, directory: str, filename_prefix: str = "bm25_index", 
                 fasttext_model_path: Optional[str] = None, 
                 stopwords_path: Optional[str] = None,
                 preserve_words_path: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize the BM25Search with a pre-built index.
        
        Args:
            directory: Directory containing the saved BM25 index
            filename_prefix: Prefix for the saved files
            fasttext_model_path: Path to the fasttext language detection model
            stopwords_path: Path to the file containing stopwords
            preserve_words_path: Path to the file containing words to preserve
            verbose: Output verbosity
        """
        self.verbose = verbose
        # Load preprocessing parameters first to determine auto_generate_settings
        preproc_path = os.path.join(directory, f"{filename_prefix}_preprocessing.pkl")
        self.preproc_params = get_pickle(preproc_path)
        
        # Extract auto_generate_settings if available, otherwise use defaults
        auto_generate_settings = self.preproc_params.get("auto_generate_settings", {
            "auto_generate_stopwords": False,
            "markdown_dir": None,
            "stopwords_output_path": stopwords_path,
            "base_stopwords_path": None,
            "time_threshold": 15,
            "change_threshold": 50,
            "metadata_file": None,
            "translation_service": "llm"
        })
        
        # Load the BM25 model
        model_path = os.path.join(directory, f"{filename_prefix}_model.pkl")
        self.bm25 = get_pickle(model_path)
        
        # Load document IDs
        ids_path = os.path.join(directory, f"{filename_prefix}_document_ids.pkl")
        self.document_ids = get_pickle(ids_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{filename_prefix}_metadata.pkl")
        self.corpus_metadata = get_pickle(metadata_path)
        
        # Create a BM25Storage instance for text preprocessing with all parameters
        # including automatic stopwords generation settings
        self.preprocessor = BM25Storage(
            model_type=self.preproc_params["model_type"],
            fasttext_model_path=fasttext_model_path,
            stopwords_path=stopwords_path or auto_generate_settings.get("stopwords_output_path"),
            default_language=self.preproc_params.get("default_language", "es"),
            remove_accents=self.preproc_params.get("remove_accents", True),
            remove_punctuation=self.preproc_params.get("remove_punctuation", True),
            remove_stopwords=self.preproc_params.get("remove_stopwords", True),
            apply_stemming=self.preproc_params.get("apply_stemming", True),
            preserve_words_path=preserve_words_path,
            # Auto-generate stopwords parameters
            auto_generate_stopwords=auto_generate_settings.get("auto_generate_stopwords", False),
            markdown_dir=auto_generate_settings.get("markdown_dir"),
            stopwords_output_path=auto_generate_settings.get("stopwords_output_path"),
            base_stopwords_path=auto_generate_settings.get("base_stopwords_path"),
            time_threshold=auto_generate_settings.get("time_threshold", 15),
            change_threshold=auto_generate_settings.get("change_threshold", 50),
            metadata_file=auto_generate_settings.get("metadata_file"),
            translation_service=auto_generate_settings.get("translation_service", "llm")
        )
        
        vprint(f"Initialized BM25 search with {len(self.document_ids)} documents", self.verbose)
        if auto_generate_settings.get("auto_generate_stopwords", False):
            vprint("Automatic stopwords generation is enabled", self.verbose)
        
    def search(
        self, 
        query: str, 
        k: int = 10, 
        include_scores: bool = False,
        translate: bool = True
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Search for documents matching the query.
        
        Args:
            query: The search query
            k: Maximum number of results to return
            include_scores: Whether to include relevance scores in the results
            translate: Whether to translate non-Spanish text to Spanish
            
        Returns:
            If include_scores is False, returns a list of document IDs.
            If include_scores is True, returns a list of (document_id, score) tuples.
        """
        # Preprocess the query using the same preprocessing as during indexing
        processed_query = self.preprocessor.preprocess_text(query, translate=translate)
        
        if not processed_query:
            vprint("Warning: Query is empty after preprocessing", self.verbose)
            raise ValueError("Query is empty after preprocessing")
        
        # Get document scores
        doc_scores = self.bm25.get_scores(processed_query)
        
        # Get the top-k document indices
        top_indices = np.argsort(doc_scores)[::-1][:k]
        
        if include_scores:
            results = [(self.document_ids[idx], doc_scores[idx]) for idx in top_indices]
        else:
            results = [self.document_ids[idx] for idx in top_indices]
            
        return results
    
    def search_batch(
        self, 
        queries: List[str], 
        k: int = 10, 
        include_scores: bool = False,
        translate: bool = True
    ) -> List[Union[List[str], List[Tuple[str, float]]]]:
        """
        Search for documents matching multiple queries.
        
        Args:
            queries: List of search queries
            k: Maximum number of results per query
            include_scores: Whether to include relevance scores in the results
            translate: Whether to translate non-Spanish text to Spanish
            
        Returns:
            List of search results, one for each query
        """
        results = []
        for query in queries:
            query_results = self.search(query, k, include_scores, translate=translate)
            results.append(query_results)
            
        return results
    
    def get_preprocessed_query(self, query: str, translate: bool = True) -> List[str]:
        """
        Get the preprocessed version of a query without performing a search.
        Useful for understanding how queries are processed.
        
        Args:
            query: The search query
            translate: Whether to translate non-Spanish text to Spanish
            
        Returns:
            List of preprocessed tokens
        """
        return self.preprocessor.preprocess_text(query, translate=translate)

def example_usage():
    """
    Example usage of BM25Search with improved results display.
    Shows document content and query preprocessing for better evaluation.
    """
    import time
    from tabulate import tabulate
    
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    bm25_folder = os.path.join(script_dir, "../../data/posteriori_resources/bm25_stuffs")
    md_folder = os.path.join(script_dir, "../../data/posteriori_resources/markdown_files")
    fasttext_model_path = os.path.join(script_dir, "../../models/lang_model")
    stopwords_path = os.path.join(script_dir, "../../data/priori_resources/stopwords.txt")
    preserve_words_path = os.path.join(script_dir, "../../data/priori_resources/preserve_words.txt")
    
    print(f"Loading BM25 index from {bm25_folder}...")
    bm25 = BM25Search(
        bm25_folder, 
        filename_prefix="bm25_index", 
        fasttext_model_path=fasttext_model_path, 
        stopwords_path=stopwords_path, 
        preserve_words_path=preserve_words_path,
        verbose=True
    )
    
    # Example queries in different languages
    queries = [
        "qué medicamentos son buenos para el tratamiento gastrointestinal producido por nematodos", # Spanish
        "what medications are good for treating gastrointestinal nematode infection?",              # English
        "quins medicaments són bons per al tractament gastrointestinal produït per nematodes",      # Catalan
        "que medicamentos son bos para tratar a infección por nematodos gastrointestinais?"         # Galician
    ]
    query_languages = ["Spanish", "English", "Catalan", "Galician"]
    
    # Store preprocessed queries for context highlighting
    processed_queries = []
    
    # Show preprocessed queries (including translations)
    print("\n=== Query Preprocessing ===")
    for i, query in enumerate(queries):
        processed_query = bm25.get_preprocessed_query(query)
        processed_queries.append(processed_query)  # Store for later use
        original_text = f"Original ({query_languages[i]}): {query}"
        processed_text = f"Processed: {' '.join(processed_query)}"
        print(f"\n{original_text}\n{processed_text}")
    
    # Perform the searches
    print("\n\n=== Performing searches... ===")
    start_time = time.time()
    results = bm25.search_batch(queries, k=5, include_scores=True, translate=True)
    end_time = time.time()
    
    # Function to get document content with highlighted query terms
    def get_document_content(doc_id, processed_query_tokens):
        """
        Get content of a document by its ID with highlighted context around query tokens.
        
        Args:
            doc_id: The document identifier
            processed_query_tokens: List of preprocessed query tokens to highlight
        
        Returns:
            A formatted string with document content highlighting relevant sections
        """
        md_file = os.path.join(md_folder, f"{doc_id}.md")
        
        if not os.path.exists(md_file):
            return f"Document file not found: {md_file}"
        
        try:
            content = get_text(md_file)
            
            # If no query tokens provided, just return truncated content
            if not processed_query_tokens:
                if len(content) > 300:
                    return content[:300] + "..."
                return content
            
            # Only process tokens with length >= 3
            valid_tokens = [token for token in processed_query_tokens if len(token) >= 3]
            if not valid_tokens:
                if len(content) > 300:
                    return content[:300] + "..."
                return content
            
            # Find best context for each query token
            token_contexts = {}
            content_lower = content.lower()
            
            for token in valid_tokens:
                # Find best occurrence of this token (we'll track all, but show the best)
                occurrences = []
                start_pos = 0
                
                while True:
                    pos = content_lower.find(token, start_pos)
                    if pos == -1:
                        break
                        
                    # Define window boundaries (context)
                    context_size = 40  # Characters on each side
                    
                    # Find left boundary (sentence start or paragraph)
                    left_boundary = max(0, pos - context_size)
                    for boundary_char in ['.', '!', '?', '\n']:
                        potential_boundary = content.rfind(boundary_char, left_boundary, pos)
                        if potential_boundary != -1:
                            left_boundary = potential_boundary + 1
                            break
                    
                    # Find right boundary (sentence end or paragraph)
                    right_boundary = min(len(content), pos + len(token) + context_size)
                    for boundary_char in ['.', '!', '?', '\n']:
                        potential_boundary = content.find(boundary_char, pos + len(token), right_boundary)
                        if potential_boundary != -1:
                            right_boundary = potential_boundary + 1
                            break
                    
                    # Extract the context snippet
                    context_snippet = content[left_boundary:right_boundary].strip()
                    
                    # Store this occurrence
                    occurrences.append({
                        'context': context_snippet,
                        'position': pos - left_boundary,  # Position of token within snippet
                        'length': len(token),
                        'full_context': context_snippet  # Save original for display
                    })
                    
                    # Move to next potential occurrence
                    start_pos = pos + len(token)
                
                # Store the best context for this token (prioritize those with more tokens nearby)
                if occurrences:
                    # For simplicity, just use the first occurrence for now
                    # In a more advanced version, you could score contexts by relevance
                    token_contexts[token] = occurrences[0]
            
            # If we found relevant snippets, format and return them
            if token_contexts:
                result = []
                
                # Title showing how many tokens were found
                found_tokens = list(token_contexts.keys())
                result.append(f"Found {len(found_tokens)} of {len(valid_tokens)} query tokens:")
                result.append("")  # Empty line
                
                # For each token found, show its best context
                for token in found_tokens:
                    ctx = token_contexts[token]
                    
                    # Format the context with a marker for the token
                    # Use [TOKEN] instead of **TOKEN** for better readability
                    context_text = ctx['full_context']
                    position = ctx['position']
                    token_length = ctx['length']
                    
                    # Extract parts of the context
                    before_text = context_text[:position]
                    token_text = context_text[position:position+token_length]
                    after_text = context_text[position+token_length:]
                    
                    # Create formatted version with token highlight
                    formatted_context = f"{before_text}[{token_text.upper()}]{after_text}"
                    
                    # Add to results
                    result.append(f"TOKEN '{token}':")
                    result.append(f"...{formatted_context}...")
                    result.append("")  # Empty line
                
                # Join all parts
                return "\n".join(result)
            
            # If no matches found, return truncated content
            if len(content) > 300:
                return f"No matches for query tokens. Preview:\n{content[:300]}..."
            return f"No matches for query tokens. Content:\n{content}"
            
        except Exception as e:
            return f"Error reading document: {str(e)}"
    
    # Display the results for each query
    for i, (query, query_results) in enumerate(zip(queries, results)):
        print(f"\n\n=== Results for Query ({query_languages[i]}) ===")
        print(f"Query: {query}")
        print(f"Processed tokens: {' '.join(processed_queries[i])}")
        
        if not query_results:
            print("No results found.")
            continue
        
        # Prepare data for tabulate
        table_data = []
        for j, (doc_id, score) in enumerate(query_results, 1):
            # Use the improved function with processed query tokens
            content = get_document_content(doc_id, processed_queries[i])
            table_data.append([j, doc_id, f"{score:.4f}", content])
        
        # Display table with results
        headers = ["Rank", "Document ID", "Score", "Relevant Content"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print(f"\nTotal search time: {end_time - start_time:.4f} seconds")
    print(f"Average time per query: {(end_time - start_time)/len(queries):.4f} seconds")

if __name__ == "__main__":
    example_usage()
