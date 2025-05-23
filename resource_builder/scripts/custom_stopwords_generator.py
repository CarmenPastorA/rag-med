

"""
Custom Stopwords Generator for BM25

This script automatically generates custom stopwords for a specific corpus of documents
based on TF-IDF analysis. It extracts words that appear in many documents with high frequency
but provide little discriminative value for BM25 ranking.

The script includes conditional execution based on:
1. Time elapsed since last execution
2. Percentage of documents changed in the corpus

Usage:
    python custom_stopwords_generator.py --input_dir ./documents --output_file ./stopwords.txt
    python custom_stopwords_generator.py --input_dir ./documents --output_file ./stopwords.txt --base_stopwords ./base_stopwords.txt
    python custom_stopwords_generator.py --input_dir ./documents --output_file ./stopwords.txt --time_threshold 15 --change_threshold 50
"""

import os
import re
import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timedelta
import math
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.veterinary_utils.utils import (vprint, 
                                           remove_accents,
                                           get_lines,
                                           get_dict_from_json,
                                           save_dict_to_json)
from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

class CustomStopwordsGenerator:
    """
    Generates custom stopwords based on TF-IDF analysis of a corpus.
    Implements conditional execution based on time and document changes.
    """

    def __init__(self, input_dir, output_file, base_stopwords=None, 
                 time_threshold=15, change_threshold=50, 
                 metadata_file='.stopwords_metadata.json',
                 verbose=False):
        """
        Initialize the stopwords generator.

        Args:
            input_dir (str): Directory containing markdown documents
            output_file (str): Path to save the generated stopwords
            base_stopwords (str, optional): Path to base stopwords file
            time_threshold (int): Days to wait before regenerating stopwords
            change_threshold (int): Percentage of document changes to trigger regeneration
            metadata_file (str): File to store execution metadata
            verbose (bool): Output verbosity
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.base_stopwords_file = base_stopwords
        self.time_threshold = time_threshold
        self.change_threshold = change_threshold
        self.metadata_file = metadata_file
        self.verbose = verbose
        
        # Load base stopwords if provided
        self.base_stopwords = set()
        if base_stopwords and os.path.exists(base_stopwords):
            self.base_stopwords = set(get_lines(base_stopwords))
        
        # Initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """
        Load metadata from previous executions.
        
        Returns:
            dict: Metadata including timestamps and document hashes
        """
        default_metadata = {
            'last_execution': None,
            'document_hashes': {},
            'generated_stopwords': []
        }
        
        if os.path.exists(self.metadata_file):
            try:
                return get_dict_from_json(self.metadata_file)
            except (json.JSONDecodeError, IOError):
                vprint("Error loading metadata, using default values", self.verbose)
                return default_metadata
        return default_metadata
    
    def _save_metadata(self):
        """Save current metadata to file."""
        save_dict_to_json(self.metadata_file, self.metadata)
    
    def _hash_document(self, file_path):
        """
        Generate SHA-256 hash for a document.
        
        Args:
            file_path (str): Path to the document
        
        Returns:
            str: Hexadecimal digest of the hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b''):
                sha256.update(block)
        return sha256.hexdigest()
    
    def should_execute(self):
        """
        Determine if stopwords should be regenerated based on time elapsed
        or document changes.
        
        Returns:
            bool: True if regeneration is needed, False otherwise
        """
        # Check if stopwords have never been generated
        if not self.metadata['last_execution'] or not os.path.exists(self.output_file):
            vprint("First-time execution or output file missing.", self.verbose)
            return True
        
        # Check time threshold
        last_execution = datetime.fromisoformat(self.metadata['last_execution'])
        time_elapsed = datetime.now() - last_execution
        if time_elapsed > timedelta(days=self.time_threshold):
            vprint(f"Time threshold exceeded. Last execution: {last_execution.isoformat()}", self.verbose)
            return True
        
        # Get current document hashes
        current_hashes = {}
        markdown_files = self._get_markdown_files()
        for file_path in markdown_files:
            current_hashes[file_path] = self._hash_document(file_path)
        
        # Check for new files
        old_files = set(self.metadata['document_hashes'].keys())
        new_files = set(current_hashes.keys())
        
        if old_files != new_files:
            added = len(new_files - old_files)
            removed = len(old_files - new_files)
            vprint(f"Document set changed: {added} added, {removed} removed", self.verbose)
            return True
        
        # Check for changed files
        common_files = old_files.intersection(new_files)
        changed_files = [f for f in common_files 
                         if current_hashes[f] != self.metadata['document_hashes'][f]]
        
        change_percentage = (len(changed_files) / len(common_files)) * 100 if common_files else 0
        
        if change_percentage >= self.change_threshold:
            vprint(f"Change threshold exceeded: {change_percentage:.2f}% of documents changed", self.verbose)
            return True
        
        vprint("No regeneration needed based on time and change thresholds", self.verbose)
        return False
    
    def _get_markdown_files(self):
        """
        Get all markdown files in the input directory.
        
        Returns:
            list: List of paths to markdown files
        """
        markdown_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.md'):
                    markdown_files.append(os.path.join(root, file))
        return markdown_files
    
    def _preprocess_text(self, text):
        """
        Preprocess text by removing punctuation, numbers, etc.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        # Remove code blocks - they can contain special terms that shouldn't be stopwords
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r'`.*?`', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Split into tokens and filter empty strings
        return [token for token in text.split() if token.strip()]
    
    def generate_stopwords(self):
        """
        Generate custom stopwords based on TF-IDF analysis.
        
        Returns:
            set: Set of custom stopwords
        """
        markdown_files = self._get_markdown_files()
        if not markdown_files:
            vprint("No markdown files found in the input directory", self.verbose)
            return self.base_stopwords
        
        vprint(f"Processing {len(markdown_files)} markdown files...", self.verbose)
        
        # Document frequency of each term
        doc_freq = Counter()
        # Total term frequency
        term_freq = Counter()
        # Document content and hashes for metadata
        doc_hashes = {}
        
        # Process each document
        for file_path in markdown_files:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                try:
                    content = f.read()
                except UnicodeDecodeError:
                    vprint(f"Warning: Unicode decode error in {file_path}, skipping", self.verbose)
                    continue
            
            # Hash the document for metadata
            doc_hashes[file_path] = self._hash_document(file_path)
            
            # Preprocess and tokenize the document
            tokens = self._preprocess_text(content)
            
            # Update document frequency (count unique terms in this document)
            doc_freq.update(set(tokens))
            
            # Update term frequency
            term_freq.update(tokens)
        
        num_docs = len(markdown_files)
        
        # Calculate TF-IDF scores
        tfidf_scores = {}
        for term, df in doc_freq.items():
            # Skip very rare terms (appear in only one document)
            if df <= 1:
                continue
                
            # TF score - normalize by total term count
            tf = term_freq[term] / sum(term_freq.values())
            
            # IDF score
            idf = math.log(num_docs / df)
            
            # TF-IDF score
            tfidf_scores[term] = tf * idf
        
        # Words with low TF-IDF are likely stopwords
        # We'll also consider words that appear in many documents (high DF)
        potential_stopwords = set()
        
        # Add words with high document frequency
        df_threshold = 0.5  # Words that appear in 50% or more of documents
        df_min_count = max(2, int(num_docs * df_threshold))
        
        for term, df in doc_freq.items():
            if df >= df_min_count:  # High document frequency
                if len(term) > 1:  # Skip single-character terms
                    potential_stopwords.add(term)
        
        # Add words with low TF-IDF (bottom 20%)
        if tfidf_scores:
            sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1])
            num_stopwords = max(10, int(len(sorted_terms) * 0.2))
            
            for term, score in sorted_terms[:num_stopwords]:
                if len(term) > 1:  # Skip single-character terms
                    potential_stopwords.add(term)
        
        # Combine with base stopwords
        combined_stopwords = self.base_stopwords.union(potential_stopwords)
        
        # Update metadata
        self.metadata['last_execution'] = datetime.now().isoformat()
        self.metadata['document_hashes'] = doc_hashes
        self.metadata['generated_stopwords'] = list(potential_stopwords)
        self._save_metadata()
        
        return combined_stopwords
    
    def run(self):
        """Run the stopwords generation process if needed."""
        if self.should_execute():
            vprint("Generating custom stopwords...", self.verbose)
            stopwords = self.generate_stopwords()
            
            # Remove accents and diacritical marks from stopwords
            stopwords = set([remove_accents(t) for t in stopwords])
            
            # Save stopwords to file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for word in sorted(stopwords):
                    f.write(f"{word}\n")
            
            vprint(f"Generated {len(stopwords)} stopwords ({len(self.metadata['generated_stopwords'])} custom)", self.verbose)
            vprint(f"Stopwords saved to {self.output_file}", self.verbose)
        else:
            vprint(f"Using existing stopwords from {self.output_file}", self.verbose)
            vprint(f"Last generated on {self.metadata['last_execution']}", self.verbose)


def main():
    """Main entry point for the script."""
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    md_folder = os.path.join(script_dir, "../../data/posteriori_resources/markdown_files")
    stopwords_path = os.path.join(script_dir, "../../data/posteriori_resources/stopwords.txt")
    base_stopwords_path = os.path.join(script_dir, "../../data/priori_resources/stopwords.txt")
    metadata_file_path = os.path.join(script_dir, "../../data/posteriori_resources/.stopwords_metadata.json")
    
    parser = argparse.ArgumentParser(
        description="Generate custom stopwords for BM25 based on corpus analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input-dir", default=md_folder, 
                        help="Directory containing markdown documents")
    parser.add_argument("-o", "--output-file", default=stopwords_path, 
                        help="File to save generated stopwords")
    parser.add_argument("-b", "--base-stopwords", default=base_stopwords_path,
                        help="Base stopwords file (e.g., from NLTK)")
    parser.add_argument("-t", "--time-threshold", type=int, default=15,
                        help="Days to wait before regenerating stopwords")
    parser.add_argument("-c", "--change-threshold", type=int, default=50,
                        help="Percentage of document changes to trigger regeneration")
    parser.add_argument("-m", "--metadata-file", default=metadata_file_path,
                        help="File to store execution metadata")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")
    
    args = parser.parse_args()
    
    generator = CustomStopwordsGenerator(
        input_dir=args.input_dir,
        output_file=args.output_file,
        base_stopwords=args.base_stopwords,
        time_threshold=args.time_threshold,
        change_threshold=args.change_threshold,
        metadata_file=args.metadata_file,
        verbose=args.verbose
    )
    
    generator.run()


if __name__ == "__main__":
    main()
