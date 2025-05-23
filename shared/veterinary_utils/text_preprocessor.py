

"""
Text preprocessor for BM25.
Translate the query into Spanish if necessary.
"""

import re
import fasttext
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from typing import List, Set, Dict, Optional, Union, Any
from tqdm import tqdm
import os
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.veterinary_utils.utils import (vprint, 
                                           get_lines, 
                                           get_text,
                                           StemProcessor, 
                                           remove_accents as rem_acc)
from resource_builder.scripts.custom_stopwords_generator import CustomStopwordsGenerator
from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

class TextPreprocessor:
    """
    A text preprocessor for BM25 with translation capabilities to Spanish.
    """
    def __init__(
        self,
        fasttext_model_path: Optional[str] = None,
        translation_service: str = "dummy",
        stopwords_path: str = None,
        default_language: str = 'es',
        remove_accents: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        apply_stemming: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        preserve_words_path: str = None,
        verbose: bool = False,
        # parameters for automatic stopwords generation
        auto_generate_stopwords: bool = False,
        markdown_dir: Optional[str] = None,
        stopwords_output_path: Optional[str] = None, 
        base_stopwords_path: Optional[str] = None,
        time_threshold: int = 15,
        change_threshold: int = 50,
        metadata_file: Optional[str] = None
    ):
        """
        Initialize the TextPreprocessor with the desired configuration.
        
        Args:
            fasttext_model_path: Path to the fasttext language detection model
            translation_service: Translation service to use ('dummy' or 'llm')
            stopwords_path: Spanish stopwords file
            default_language: Default language code to use if detection fails
            remove_accents: Whether to remove accents and diacritics
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming
            custom_stopwords: List with additional stopwords ['word1', 'word2']
            preserve_words_path: Spanish words file; words to preserve even if they would be removed by other processes
            verbose: Output verbosity
            auto_generate_stopwords: Activate/deactivate automatic stopwords generation
            markdown_dir: Directory containing Markdown documents for stopwords extraction
            stopwords_output_path: Path to save generated stopwords (defaults to stopwords_path if None)
            base_stopwords_path: Path to base stopwords file (used as seed for generation)
            time_threshold: Days to wait before regenerating stopwords
            change_threshold: Percentage of document changes to trigger regeneration
            metadata_file: File to store execution metadata
        """
        self.default_language = default_language
        self.do_remove_accents = remove_accents
        self.do_remove_punctuation = remove_punctuation
        self.do_remove_stopwords = remove_stopwords
        self.do_apply_stemming = apply_stemming
        self.preserve_words = set() if preserve_words_path is None else set(get_lines(preserve_words_path))
        self.verbose = verbose
        
        # Load language detection model
        self.lan_detect_model = None
        if fasttext_model_path:
            self.load_language_model(fasttext_model_path)
        
        # Initialize tokenizer for punctuation removal
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        # Spanish stemmer
        stemmer_base= SnowballStemmer('spanish')
        self.stemmer = StemProcessor(stemmer_base)
        
        # Initialize stopwords handling
        self._initialize_stopwords(
            stopwords_path=stopwords_path,
            custom_stopwords=custom_stopwords,
            auto_generate=auto_generate_stopwords,
            markdown_dir=markdown_dir,
            stopwords_output_path=stopwords_output_path or stopwords_path,
            base_stopwords_path=base_stopwords_path,
            time_threshold=time_threshold,
            change_threshold=change_threshold,
            metadata_file=metadata_file
        )
        
        # Initialize translation service
        self.translation_service = translation_service
        self.translator = self._init_translator(translation_service)
    
    def _initialize_stopwords(
        self,
        stopwords_path: str = None,
        custom_stopwords: Optional[List[str]] = None,
        auto_generate: bool = False,
        markdown_dir: Optional[str] = None,
        stopwords_output_path: Optional[str] = None,
        base_stopwords_path: Optional[str] = None,
        time_threshold: int = 15,
        change_threshold: int = 50,
        metadata_file: Optional[str] = None
    ) -> None:
        """
        Initialize stopwords with optional automatic generation.
        
        Args:
            stopwords_path: Path to load predefined stopwords from
            custom_stopwords: Additional stopwords provided directly
            auto_generate: Whether to use automatic stopwords generation
            markdown_dir: Directory containing Markdown documents for stopwords extraction
            stopwords_output_path: Path to save generated stopwords
            base_stopwords_path: Path to base stopwords file (used as seed for generation)
            time_threshold: Days to wait before regenerating stopwords
            change_threshold: Percentage of document changes to trigger regeneration
            metadata_file: File to store execution metadata
        """
        # Default to an empty set if we can't load or generate stopwords
        self.stopwords = set()
        
        # Determine if we should use automatic generation
        if auto_generate:
            if not markdown_dir:
                vprint("Warning: markdown_dir not specified. Automatic stopwords generation disabled.", self.verbose)
                auto_generate = False
            elif not stopwords_output_path:
                vprint("Warning: stopwords_output_path not specified. Automatic stopwords generation disabled.", self.verbose)
                auto_generate = False
        
        # Use automatic stopwords generation if enabled
        if auto_generate:
            # Get script directory for default paths if needed
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Set defaults for optional parameters
            if metadata_file is None:
                # Default metadata file in the same directory as output file
                output_dir = os.path.dirname(stopwords_output_path)
                metadata_file = os.path.join(output_dir, '.stopwords_metadata.json')
            
            vprint(f"Initializing automatic stopwords generation from {markdown_dir}", self.verbose)
            
            # Create and run the generator
            generator = CustomStopwordsGenerator(
                input_dir=markdown_dir,
                output_file=stopwords_output_path,
                base_stopwords=base_stopwords_path,
                time_threshold=time_threshold,
                change_threshold=change_threshold,
                metadata_file=metadata_file,
                verbose=self.verbose
            )
            
            generator.run()
            
            # Now load the generated stopwords
            if os.path.exists(stopwords_output_path):
                self.stopwords = set(get_lines(stopwords_output_path))
                vprint(f"Loaded {len(self.stopwords)} generated stopwords from {stopwords_output_path}", self.verbose)
            else:
                vprint(f"Warning: Generated stopwords file not found at {stopwords_output_path}", self.verbose)
        
        # If not using auto-generation or if it failed, try loading from file
        if not self.stopwords and stopwords_path:
            if os.path.exists(stopwords_path):
                self.stopwords = set(get_lines(stopwords_path))
                vprint(f"Loaded {len(self.stopwords)} stopwords from {stopwords_path}", self.verbose)
            else:
                vprint(f"Warning: Stopwords file not found at {stopwords_path}", self.verbose)
        
        # Add custom stopwords provided in constructor
        if custom_stopwords:
            self.stopwords.update(set(custom_stopwords))
            vprint(f"Added {len(custom_stopwords)} custom stopwords", self.verbose)
    
    def load_language_model(self, model_path: str) -> None:
        """
        Load the fastText language detection model.
        
        Args:
            model_path: Path to the fastText model file
        """
        try:
            if os.path.isfile(model_path):
                self.lan_detect_model = fasttext.load_model(model_path)
                vprint(f"Successfully loaded language detection model from {model_path}", self.verbose)
            else:
                print(f"Language detection model file not found at {model_path}")
                vprint("You can download the model from: https://fasttext.cc/docs/en/language-identification.html", self.verbose)
        except Exception as e:
            print(f"Error loading language detection model: {e}")
            vprint("Language detection will default to specified default language", self.verbose)
    
    def _init_translator(self, service: str):
        """
        Initialize the appropriate translation service.
        
        Args:
            service (str): The translation service to use
                
        Returns:
            object: Translator or None
        """
        if service == 'dummy':
            return None
        elif service == 'llm':
            try:
                from tools.arqa.Reader import OllamaReader
                # prompts as strings
                system_prompt = "Eres un asistente experto en traducir al español."
                user_prompt = ("Por favor, traduce al español la siguiente pregunta, que está (probablemente) en el idioma ISO '{question}':\n"
                              "---------------------\n"
                              "{context}\n"
                              "---------------------\n"
                              "Responde únicamente con la traducción de la pregunta, sin comentarios adicionales.")
                # Possible Ollama LLM -> hf.co/BSC-LT/salamandraTA-7B-instruct-GGUF
                # initialize OllamaReader
                info_model = {
                    "model_name": "llama3.2", #"qwen2",
                    "tokenizer": "cl100k_base",
                    "context": 1024,#8192, # small context due to small prompt
                    "use_dynamic_context_size": True,
                    "endpoint": "http://localhost:11434/api/chat",
                    "temperature": 0,
                    "keep_alive": -1,
                    "seed": 42
                }
                translator = OllamaReader(
                    prompt_system=system_prompt,
                    prompt_user=user_prompt,
                    info_model=info_model,
                    verbose=False
                )
                return translator
            except ImportError:
                print("The LLM could not be imported for translation.")
                return None
        else:
            print(f"Unknown translation service: {service}")
            return None
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO language code (e.g., 'es', 'en', 'ca', 'gl')
        """
        if not text or not self.lan_detect_model:
            return self.default_language
        
        try:
            # Clean text for language detection
            clean_text = ' '.join(text.replace('\n', ' ').split())
            if not clean_text:
                return self.default_language
                
            # Get prediction from fastText
            result = self.lan_detect_model.predict(clean_text)
            lang_code = result[0][0].replace('__label__', '')
            
            # Map fastText language codes to our standard codes
            lang_mapping = {
                'es': 'es', 'spa': 'es',
                'en': 'en', 'eng': 'en',
                'ca': 'ca', 'cat': 'ca',
                'gl': 'gl', 'glg': 'gl',
                'fr': 'fr', 'fra': 'fr',
                'it': 'it', 'ita': 'it',
                'pt': 'pt', 'por': 'pt',
                'de': 'de', 'deu': 'de',
                'nl': 'nl', 'nld': 'nl',
                'ru': 'ru', 'rus': 'ru',
            }
            
            mapped_code = lang_mapping.get(lang_code, None)
            if mapped_code:
                return mapped_code
            else:
                vprint(f"Unmapped language code: {lang_code}, using default: {self.default_language!r}", self.verbose)
                return self.default_language
                
        except Exception as e:
            vprint(f"Error detecting language: {e}")
            return self.default_language
    
    def translate_to_spanish(self, text: str, source_lang: str) -> str:
        """
        Translate text to Spanish if not already in Spanish.
        
        Args:
            text (str): The text to translate
            source_lang (str): The source language code
            
        Returns:
            str: The translated text or original if already Spanish
        """
        if source_lang == 'es' or not text:
            return text
        
        if not self.translator:
            # Simple dictionary-based translation (very limited, just for example)
            vprint("Warning: Using dummy translation. Consider implementing a real translation API.", self.verbose)
            simple_dict = {
                'dog': 'perro',
                'cat': 'gato',
                'infection': 'infección',
                'bacterial': 'bacteriana',
                'antibiotic': 'antibiótico',
                'medicine': 'medicamento',
                'treatment': 'tratamiento',
                'dose': 'dosis',
                'tablet': 'comprimido',
                'injection': 'inyección',
                'oral': 'oral',
                'veterinary': 'veterinario'
            }
            
            words = text.lower().split()
            translated_words = []
            for word in words:
                if word in simple_dict:
                    translated_words.append(simple_dict[word])
                else:
                    translated_words.append(word)  # Keep original if not found
                    
            return ' '.join(translated_words)
            
        elif self.translation_service == 'llm':
            try:
                result = self.translator.get_answer(source_lang, text, num_ctx=1024)
                return result
            except Exception as e:
                print(f"Translation error: {e}")
                return text
        
        return text
    
    def remove_accents(self, text: str) -> str:
        """
        Remove accents and diacritical marks from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with accents and diacritics removed
        """
        return rem_acc(text)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text, removing punctuation if configured.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if self.do_remove_punctuation:
            tokens = self.tokenizer.tokenize(text)
            return tokens
        else:
            # Simple whitespace tokenization if keeping punctuation
            return text.split()
    
    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords for the specified language.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Filtered list of tokens
        """
        if not self.do_remove_stopwords:
            return tokens
        
        # Filter out stopwords but preserve words in the preserve_words set
        return [token for token in tokens if token in self.preserve_words or token not in self.stopwords]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Spanish stemming to tokens.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        if not self.do_apply_stemming:
            return tokens
        
        return self.stemmer.stem_list(tokens)
    
    def normalize_text(self, text: str, translate: bool=False) -> List[str]:
        """
        Apply complete normalization pipeline to text for BM25 indexing.
        
        Args:
            text (str): Text to process
            translate (bool): Whether to translate non-Spanish text to Spanish
            
        Returns:
            list: Processed tokens ready for BM25
        """
        if not text:
            return []
        
        # Detect language if translation is needed
        if translate:
            lang = self.detect_language(text)
            if lang != 'es':
                text = self.translate_to_spanish(text, lang)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove accents if configured
        if self.do_remove_accents:
            text = self.remove_accents(text)
        
        # Tokenize (and remove punctuation if configured)
        tokens = self.tokenize(text)
        
        # Remove stopwords if configured
        tokens = self.filter_stopwords(tokens)
        
        # Apply stemming if configured
        tokens = self.apply_stemming(tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def preprocess_corpus(self, documents: List[str]) -> List[List[str]]:
        """
        Apply normalization to a collection of documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of lists of normalized tokens
        """
        processed_docs = []
        
        for doc in tqdm(documents, disable=not self.verbose):
            # No translation needed for corpus documents (already in Spanish)
            processed_doc = self.normalize_text(doc, translate=False)
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess a search query.
        
        Args:
            query: Query text
            
        Returns:
            List of normalized tokens
        """
        # Use the same pipeline as for documents
        # Translate query if needed
        return self.normalize_text(query, translate=True)



# Example usage
if __name__ == "__main__":
    import time
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    fasttext_model_path = os.path.join(script_dir, "../../models/lang_model")
    stopwords_path = os.path.join(script_dir, "../../data/priori_resources/stopwords.txt")
    preserve_words_path = os.path.join(script_dir, "../../data/priori_resources/preserve_words.txt")
    markdown_path = os.path.join(script_dir, "../../data/posteriori_resources/markdown_files")
    stopwords_output_path = os.path.join(script_dir, "../../data/posteriori_resources/stopwords.txt")
    base_stopwords_path = stopwords_path
    metadata_file_path = os.path.join(script_dir, "../../data/posteriori_resources/.stopwords_metadata.json")
    
    preprocessor = TextPreprocessor(
        fasttext_model_path=fasttext_model_path, 
        translation_service="llm",
        stopwords_path=stopwords_path,
        preserve_words_path=preserve_words_path,
        verbose=True,
        # params for auto generate sw
        auto_generate_stopwords=True, # set to False to not automatically generate sw
        markdown_dir=markdown_path,
        stopwords_output_path=stopwords_output_path,
        base_stopwords_path=base_stopwords_path,
        time_threshold=15, 
        change_threshold=50, 
        metadata_file=metadata_file_path
    )
    
    # Example with dummy documents
    docs = [
        "Amoxicilina 500mg - Indicado para infecciones bacterianas en perros y gatos. Administrar 10mg/kg cada 12 horas.",
        "Enrofloxacino 50mg - Antibiótico de amplio espectro para uso veterinario en pequeños animales. No administrar a animales grandes.",
        "Meloxicam 1.5mg - Antiinflamatorio no esteroideo para el tratamiento del dolor y la inflamación en perros."
    ]
    
    processed_docs = preprocessor.preprocess_corpus(docs)
    print("Processed dummy documents:")
    for i, doc in enumerate(processed_docs):
        print(f"{i+1}: {doc}")
    
    # Example with real documents
    s = time.time()
    md_content = [get_text(os.path.join(markdown_path, x)) for x in os.listdir(markdown_path) if x.endswith(".md")]
    processed_docs = preprocessor.preprocess_corpus(md_content)
    e = time.time()
    print(f"Processed real documents.\nTime: {e -s} seconds")
    
    # Example queries in different languages
    queries = [
        "infección bacteriana perro 500mg",  # Spanish
        "dog infection antibiotic",          # English
        "gat antibiòtic dosi",               # Catalan
        "infección can"                      # Galician
    ]
    
    print("\nProcessed queries:")
    for i, query in enumerate(queries):
        lang = preprocessor.detect_language(query)
        processed_query = preprocessor.preprocess_query(query)
        print(f"{i+1} [{lang}]: {query} -> {processed_query}")
    
