

"""
BM25 storage module for veterinary medicine RAG system.
Handles document pre-processing, building and storing BM25 indices for efficient sparse retrieval.

Uses: 
- rank-bm25: only depends on numpy
"""


import os
import numpy as np
import argparse
import glob
from typing import List, Union, Dict, Any, Optional, Set, Tuple
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from tqdm import tqdm
import logging
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.veterinary_utils.utils import (get_text, 
                                           get_file_filename_ext,
                                           get_pickle,
                                           save_pickle)
from shared.veterinary_utils.text_preprocessor import TextPreprocessor
from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Storage:
    """
    Class for building and storing BM25 indices for efficient sparse retrieval.
    Supports BM25Okapi, BM25L, and BM25Plus algorithms with advanced text preprocessing.
    """

    BM25_MODELS = {
        "okapi": BM25Okapi,
        "l": BM25L,
        "plus": BM25Plus
    }

    def __init__(
        self, 
        model_type: str = "okapi",
        fasttext_model_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        default_language: str = 'es',
        lowercase: bool = True,
        remove_accents: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        apply_stemming: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        preserve_words_path: Optional[str] = None,
        # parameters for automatic stopwords generation
        auto_generate_stopwords: bool = False,
        markdown_dir: Optional[str] = None,
        stopwords_output_path: Optional[str] = None,
        base_stopwords_path: Optional[str] = None,
        time_threshold: int = 15,
        change_threshold: int = 50,
        metadata_file: Optional[str] = None,
        translation_service: str = "llm"
    ):
        """
        Initialize the BM25Storage with preprocessing options.
        
        Args:
            model_type: Type of BM25 model to use ('okapi', 'l', or 'plus')
            fasttext_model_path: Path to the fasttext language detection model
            stopwords_path: Path to the file containing stopwords
            default_language: Default language code to use if detection fails
            lowercase: Whether to convert text to lowercase
            remove_accents: Whether to remove accents and diacritics
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming
            custom_stopwords: Additional stopwords (as a list of strings)
            preserve_words_path: Spanish words file; words to preserve even if they would be removed by other processes
            auto_generate_stopwords: Whether to automatically generate stopwords
            markdown_dir: Directory containing Markdown documents for stopwords extraction
            stopwords_output_path: Path to save generated stopwords (defaults to stopwords_path if None)
            base_stopwords_path: Path to base stopwords file (used as seed for generation)
            time_threshold: Days to wait before regenerating stopwords
            change_threshold: Percentage of document changes to trigger regeneration
            metadata_file: File to store execution metadata
            translation_service: Translation service to use ('dummy' or 'llm')
        """
        if model_type not in self.BM25_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: {', '.join(self.BM25_MODELS.keys())}")
        
        self.model_type = model_type
        
        # Set default stopwords_output_path if not provided
        if auto_generate_stopwords and not stopwords_output_path and stopwords_path:
            stopwords_output_path = stopwords_path
        
        # Initialize the text preprocessor with correct parameters including auto_generate_stopwords
        self.preprocessor = TextPreprocessor(
            fasttext_model_path=fasttext_model_path,
            translation_service=translation_service,
            stopwords_path=stopwords_path,
            default_language=default_language,
            remove_accents=remove_accents,
            remove_punctuation=remove_punctuation,
            remove_stopwords=remove_stopwords,
            apply_stemming=apply_stemming,
            custom_stopwords=custom_stopwords,
            preserve_words_path=preserve_words_path,
            verbose=True, # set verbose to true for better monitoring
            # parameters for automatic stopwords generation
            auto_generate_stopwords=auto_generate_stopwords,
            markdown_dir=markdown_dir,
            stopwords_output_path=stopwords_output_path,
            base_stopwords_path=base_stopwords_path,
            time_threshold=time_threshold,
            change_threshold=change_threshold,
            metadata_file=metadata_file
        )
        
        # Placeholder for the BM25 model
        self.bm25 = None
        self.document_ids = []
        self.corpus_metadata = {}
        
        # Store settings for automatic stopwords generation for serialization
        self.auto_generate_settings = {
            "auto_generate_stopwords": auto_generate_stopwords,
            "markdown_dir": markdown_dir,
            "stopwords_output_path": stopwords_output_path,
            "base_stopwords_path": base_stopwords_path,
            "time_threshold": time_threshold,
            "change_threshold": change_threshold,
            "metadata_file": metadata_file,
            "translation_service": translation_service
        }
    
    def preprocess_text(self, text: str, translate: bool=False) -> List[str]:
        """
        Preprocess text according to the configured options.
        
        Args:
            text: The input text to preprocess
            translate (bool): Whether to translate non-Spanish text to Spanish
            
        Returns:
            List of preprocessed tokens
        """
        # Use the TextPreprocessor to normalize the text
        return self.preprocessor.normalize_text(text, translate=translate)

    def build_index(self, documents: List[str], document_ids: List[str], batch_size: int = 1000):
        """
        Build a BM25 index for the given documents.
        
        Args:
            documents: List of document texts
            document_ids: List of unique identifiers for the documents
            batch_size: Number of documents to process in each batch
        """
        if len(documents) != len(document_ids):
            raise ValueError("The number of documents and document IDs must be the same")
            
        self.document_ids = document_ids
        
        # Preprocess documents in batches to avoid memory issues
        preprocessed_corpus = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Preprocessing documents"):
            batch_docs = documents[i:i+batch_size]
            batch_processed = [self.preprocess_text(doc) for doc in batch_docs]
            preprocessed_corpus.extend(batch_processed)
        
        # Create the BM25 model
        self.bm25 = self.BM25_MODELS[self.model_type](preprocessed_corpus)
        
        # Store metadata about the corpus
        self.corpus_metadata = {
            "num_documents": len(documents),
            "model_type": self.model_type,
            "preprocessing": {
                "lowercase": True,  # The preprocessor always uses lowercase
                "remove_accents": self.preprocessor.do_remove_accents,
                "remove_punctuation": self.preprocessor.do_remove_punctuation,
                "remove_stopwords": self.preprocessor.do_remove_stopwords,
                "apply_stemming": self.preprocessor.do_apply_stemming,
                "auto_generate_stopwords": self.auto_generate_settings["auto_generate_stopwords"]
            }
        }
        
        logger.info(f"Built BM25 index with {len(documents)} documents using {self.model_type} model")

    def build_index_from_markdown_folder(self, path_md_folder: str, batch_size: int = 1000) -> int:
        """
        Build a BM25 index from Markdown files in a folder.
        
        Args:
            path_md_folder: Path to the folder containing Markdown files
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of documents processed
        """
        # Find all markdown files in the folder
        md_files = glob.glob(os.path.join(path_md_folder, "*.md"))
        logger.info(f"Found {len(md_files)} Markdown files in {path_md_folder}")
        
        if not md_files:
            logger.warning(f"No Markdown files found in {path_md_folder}")
            return 0
        
        # Read and process the markdown files
        documents = []
        document_ids = []
        
        for md_file in tqdm(md_files, desc="Reading Markdown files"):
            try:
                # Use the get_text utility to read the content
                content = get_text(md_file)
                
                # Use the filename (without extension) as the document ID
                doc_id = get_file_filename_ext(md_file)[1]
                
                documents.append(content)
                document_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")
        
        # Update markdown_dir in preprocessor if not already set and auto_generate_stopwords is active
        if (self.auto_generate_settings["auto_generate_stopwords"] and 
            not self.auto_generate_settings["markdown_dir"]):
            self.auto_generate_settings["markdown_dir"] = path_md_folder
            # Reinitialize the preprocessor with updated markdown_dir
            self._update_preprocessor()
        
        # Build the index with the documents
        self.build_index(documents, document_ids, batch_size)
        
        return len(documents)

    def _update_preprocessor(self):
        """
        Update the preprocessor with current settings.
        Used when settings change and preprocessor needs to be reinitialized.
        """
        self.preprocessor = TextPreprocessor(
            fasttext_model_path=self.preprocessor.lan_detect_model._path if self.preprocessor.lan_detect_model else None,
            translation_service=self.auto_generate_settings["translation_service"],
            stopwords_path=self.auto_generate_settings["stopwords_output_path"],
            default_language=self.preprocessor.default_language,
            remove_accents=self.preprocessor.do_remove_accents,
            remove_punctuation=self.preprocessor.do_remove_punctuation,
            remove_stopwords=self.preprocessor.do_remove_stopwords,
            apply_stemming=self.preprocessor.do_apply_stemming,
            verbose=True,
            # parameters for automatic stopwords generation
            auto_generate_stopwords=self.auto_generate_settings["auto_generate_stopwords"],
            markdown_dir=self.auto_generate_settings["markdown_dir"],
            stopwords_output_path=self.auto_generate_settings["stopwords_output_path"],
            base_stopwords_path=self.auto_generate_settings["base_stopwords_path"],
            time_threshold=self.auto_generate_settings["time_threshold"],
            change_threshold=self.auto_generate_settings["change_threshold"],
            metadata_file=self.auto_generate_settings["metadata_file"]
        )

    def save(self, directory: str, filename_prefix: str = "bm25_index") -> None:
        """
        Save the BM25 index and related data to disk.
        
        Args:
            directory: Directory to save the files
            filename_prefix: Prefix for the saved files
        """
        if self.bm25 is None:
            raise ValueError("No BM25 index has been built yet")
            
        os.makedirs(directory, exist_ok=True)
        
        # Save the BM25 model
        model_path = os.path.join(directory, f"{filename_prefix}_model.pkl")
        save_pickle(model_path, self.bm25)
        
        # Save document IDs
        ids_path = os.path.join(directory, f"{filename_prefix}_document_ids.pkl")
        save_pickle(ids_path, self.document_ids)
        
        # Save metadata
        metadata_path = os.path.join(directory, f"{filename_prefix}_metadata.pkl")
        save_pickle(metadata_path, self.corpus_metadata)
        
        # Save preprocessing parameters including auto_generate_stopwords settings
        preproc_path = os.path.join(directory, f"{filename_prefix}_preprocessing.pkl")
        preproc_params = {
            "model_type": self.model_type,
            "default_language": self.preprocessor.default_language,
            "remove_accents": self.preprocessor.do_remove_accents,
            "remove_punctuation": self.preprocessor.do_remove_punctuation,
            "remove_stopwords": self.preprocessor.do_remove_stopwords,
            "apply_stemming": self.preprocessor.do_apply_stemming,
            # Include auto_generate_stopwords settings
            "auto_generate_settings": self.auto_generate_settings
        }
        save_pickle(preproc_path, preproc_params)
        
        logger.info(f"Saved BM25 index to {directory}")
        
    @classmethod
    def load(cls, directory: str, filename_prefix: str = "bm25_index", 
             fasttext_model_path: Optional[str] = None, 
             stopwords_path: Optional[str] = None,
             preserve_words_path: Optional[str] = None):
        """
        Load a previously saved BM25 index.
        
        Args:
            directory: Directory containing the saved files
            filename_prefix: Prefix for the saved files
            fasttext_model_path: Path to the fasttext language detection model
            stopwords_path: Path to the file containing stopwords
            preserve_words_path: Path to the file containing spanish words
            
        Returns:
            A BM25Storage instance with the loaded index
        """
        # Load preprocessing parameters
        preproc_path = os.path.join(directory, f"{filename_prefix}_preprocessing.pkl")
        preproc_params = get_pickle(preproc_path)
        
        # Extract auto_generate_settings if available, otherwise use defaults
        auto_generate_settings = preproc_params.get("auto_generate_settings", {
            "auto_generate_stopwords": False,
            "markdown_dir": None,
            "stopwords_output_path": stopwords_path,
            "base_stopwords_path": None,
            "time_threshold": 15,
            "change_threshold": 50,
            "metadata_file": None,
            "translation_service": "llm"
        })
        
        # Create a new instance with the same preprocessing parameters
        instance = cls(
            model_type=preproc_params["model_type"],
            fasttext_model_path=fasttext_model_path,
            stopwords_path=stopwords_path or auto_generate_settings.get("stopwords_output_path"),
            default_language=preproc_params.get("default_language", "es"),
            remove_accents=preproc_params.get("remove_accents", True),
            remove_punctuation=preproc_params.get("remove_punctuation", True),
            remove_stopwords=preproc_params.get("remove_stopwords", True),
            apply_stemming=preproc_params.get("apply_stemming", True),
            preserve_words_path=preserve_words_path,
            # auto_generate_stopwords parameters
            auto_generate_stopwords=auto_generate_settings.get("auto_generate_stopwords", False),
            markdown_dir=auto_generate_settings.get("markdown_dir"),
            stopwords_output_path=auto_generate_settings.get("stopwords_output_path"),
            base_stopwords_path=auto_generate_settings.get("base_stopwords_path"),
            time_threshold=auto_generate_settings.get("time_threshold", 15),
            change_threshold=auto_generate_settings.get("change_threshold", 50),
            metadata_file=auto_generate_settings.get("metadata_file"),
            translation_service=auto_generate_settings.get("translation_service", "llm")
        )
        
        # Load the BM25 model
        model_path = os.path.join(directory, f"{filename_prefix}_model.pkl")
        instance.bm25 = get_pickle(model_path)
        
        # Load document IDs
        ids_path = os.path.join(directory, f"{filename_prefix}_document_ids.pkl")
        instance.document_ids = get_pickle(ids_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{filename_prefix}_metadata.pkl")
        instance.corpus_metadata = get_pickle(metadata_path)
        
        logger.info(f"Loaded BM25 index with {instance.corpus_metadata['num_documents']} documents")
        return instance


def main():
    """
    Command-line interface for building and saving BM25 indices.
    """
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    md_folder = os.path.join(script_dir, "../../data/posteriori_resources/markdown_files")
    output_dir = os.path.join(script_dir, "../../data/posteriori_resources/bm25_stuffs")
    fasttext_model_path = os.path.join(script_dir, "../../models/lang_model")
    stopwords_path = os.path.join(script_dir, "../../data/priori_resources/stopwords.txt")
    preserve_words_path = os.path.join(script_dir, "../../data/priori_resources/preserve_words.txt")

    print(preserve_words_path)
    #
    stopwords_output_path = os.path.join(script_dir, "../../data/posteriori_resources/stopwords.txt")
    base_stopwords_path = stopwords_path
    metadata_file_path = os.path.join(script_dir, "../../data/posteriori_resources/.stopwords_metadata.json")
    
    parser = argparse.ArgumentParser(description="Build and save BM25 indices from Markdown files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Principal arguments
    parser.add_argument("--md-folder", type=str, default=md_folder, 
                        help="Path to the folder containing Markdown files")
    parser.add_argument("--output-dir", type=str, default=output_dir,
                        help="Directory to save the BM25 index")
    
    # Optional arguments
    parser.add_argument("--model-type", type=str, default="okapi", choices=["okapi", "l", "plus"],
                        help="Type of BM25 model to use")
    parser.add_argument("--fasttext-model", type=str, default=fasttext_model_path,
                        help="Path to the fasttext language model")
    parser.add_argument("--stopwords-path", type=str, default=stopwords_path,
                        help="Path to the file containing stopwords")
    parser.add_argument("--default-language", type=str, default="es",
                        help="Default language if detection fails")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of documents to process in each batch")
    parser.add_argument("--filename-prefix", type=str, default="bm25_index",
                        help="Prefix for the saved files")
    
    # Preprocessing options
    parser.add_argument("--no-remove-accents", action="store_true",
                        help="Do not remove accents and diacritics")
    parser.add_argument("--no-remove-punctuation", action="store_true",
                        help="Do not remove punctuation")
    parser.add_argument("--no-remove-stopwords", action="store_true",
                        help="Do not remove stopwords")
    parser.add_argument("--no-apply-stemming", action="store_true",
                        help="Do not apply stemming")
    parser.add_argument("--preserve-words-path", type=str, default=preserve_words_path,
                        help="Path to the file containing words to preserve")
    parser.add_argument("--custom-stopwords", type=str, default=None,
                        help="Comma-separated list of additional stopwords")
    
    # arguments for automatic stopwords generation
    parser.add_argument("--auto-generate-stopwords", action="store_true",
                        help="Automatically generate stopwords from documents")
    parser.add_argument("--stopwords-output-path", type=str, default=stopwords_output_path,
                        help="Path to save generated stopwords (defaults to stopwords-path)")
    parser.add_argument("--base-stopwords-path", type=str, default=base_stopwords_path,
                        help="Path to base stopwords file (for auto-generation)")
    parser.add_argument("--time-threshold", type=int, default=15,
                        help="Days to wait before regenerating stopwords")
    parser.add_argument("--change-threshold", type=int, default=50,
                        help="Percentage of document changes to trigger regeneration")
    parser.add_argument("--metadata-file", type=str, default=metadata_file_path,
                        help="File to store stopwords generation metadata")
    parser.add_argument("--translation-service", type=str, default="llm", choices=["dummy", "llm"],
                        help="Translation service to use")
    
    # Load index option
    parser.add_argument("--load", action="store_true",
                        help="Load an existing index instead of building a new one")
    
    # Parse arguments
    args = parser.parse_args()
    
    custom_stopwords = None
    if args.custom_stopwords:
        custom_stopwords = [word.strip() for word in args.custom_stopwords.split(",")]
    
    # Loading or building the index
    if args.load:
        # Load an existing index
        logger.info(f"Loading BM25 index from {args.output_dir}")
        bm25_storage = BM25Storage.load(
            directory=args.output_dir,
            filename_prefix=args.filename_prefix,
            fasttext_model_path=args.fasttext_model,
            stopwords_path=args.stopwords_path,
            preserve_words_path=args.preserve_words_path
        )
        logger.info(f"Successfully loaded BM25 index with {len(bm25_storage.document_ids)} documents")
        
    else:
        # Create a new BM25Storage instance
        bm25_storage = BM25Storage(
            model_type=args.model_type,
            fasttext_model_path=args.fasttext_model,
            stopwords_path=args.stopwords_path,
            default_language=args.default_language,
            remove_accents=not args.no_remove_accents,
            remove_punctuation=not args.no_remove_punctuation,
            remove_stopwords=not args.no_remove_stopwords,
            apply_stemming=not args.no_apply_stemming,
            custom_stopwords=custom_stopwords,
            preserve_words_path=args.preserve_words_path,
            # parameters for automatic stopwords generation
            auto_generate_stopwords=args.auto_generate_stopwords,
            markdown_dir=args.md_folder if args.auto_generate_stopwords else None,
            stopwords_output_path=args.stopwords_output_path,
            base_stopwords_path=args.base_stopwords_path,
            time_threshold=args.time_threshold,
            change_threshold=args.change_threshold,
            metadata_file=args.metadata_file,
            translation_service=args.translation_service
        )
        
        # Build the index from markdown files
        logger.info(f"Building BM25 index from Markdown files in {args.md_folder}")
        num_docs = bm25_storage.build_index_from_markdown_folder(
            path_md_folder=args.md_folder,
            batch_size=args.batch_size
        )
        
        if num_docs > 0:
            # Save the index
            logger.info(f"Saving BM25 index to {args.output_dir}")
            bm25_storage.save(
                directory=args.output_dir,
                filename_prefix=args.filename_prefix
            )
            logger.info(f"Successfully built and saved BM25 index with {num_docs} documents")
        else:
            logger.error("No documents were processed. Check the markdown folder path.")
            sys.exit(1)


if __name__ == "__main__":
    main()

