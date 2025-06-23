
# unfied_cimavet_processor.py

"""
It allows the following:

1. Download JSONs with regulatory data on veterinary SmPCs
    1.1 Download extra metadata and presentation information per registration number
    1.2 Load static mapping for Principio Activo (COD_PACTIV) to Antibiotic classes
    1.3 Enrich the metadata with antibiotic information from the mapping
2. Merge all previous JSONs into a single one (for efficiency in subsequent steps)
3. Extract registration numbers (all or only new ones)
4. Extract medicine names from the merged JSON
5. Download PDFs (all or only new ones) of the SmPCs in parallel
6. Convert the PDFs (all or only new ones) to Markdown in parallel
7. Parse the PDFs (all or only new ones) with Markdown to generate JSON with a predefined structure
8. Creates text files with essential information for each medicine
9. Store the JSON information in FAISS

The script is very flexible and allows you to skip any steps you want.
"""

import os
import json
import argparse
import subprocess
import re
import requests
from tqdm import tqdm
import logging
from concurrent.futures import (ThreadPoolExecutor, 
                                ProcessPoolExecutor,
                                as_completed)
import pandas as pd
import pymupdf4llm
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import quote
from datetime import datetime
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.veterinary_utils.utils import format_registration_for_url
from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cimavet_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CIMAVet-Processor")

class CIMAVetProcessor:
    def __init__(self, output_dir, num_workers=20, max_eu_workers=2, max_retries=3, max_empty_responses=30, use_hierarchical_storage=True):
        """
        Initializes the CIMAVet processor
        
        Args:
            output_dir (str): Base directory to save all files
            num_workers (int): Number of workers for parallel downloads
            max_eu_workers (int): Maximum workers for EU registrations to prevent rate limiting
            max_retries (int): Maximum retries for failed downloads
            max_empty_responses (int): Maximum number of consecutive empty responses, 
                                       in the 'registroCambios' service, to be considered as an error
            use_hierarchical_storage (bool): Flag to enable/disable hierarchical storage
        """
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.max_eu_workers = max_eu_workers
        self.max_retries = max_retries
        self.max_empty_responses = max_empty_responses
        self.use_hierarchical_storage = use_hierarchical_storage
        self.pact_mapping = self._load_static_pact_mapping()
        self.embed_model_path = "intfloat/multilingual-e5-large"
        
        # Create directory structure
        self.json_dir = os.path.join(output_dir, "json_data")
        self.pdf_dir = os.path.join(output_dir, "pdf_files")
        self.markdown_dir = os.path.join(output_dir, "markdown_files")
        self.processed_json_dir = os.path.join(output_dir, "processed_json")
        self.essential_info_dir = os.path.join(output_dir, "essential_info")
        
        for directory in [self.output_dir, self.json_dir, self.pdf_dir, 
                         self.markdown_dir, self.processed_json_dir, self.essential_info_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Create file paths
        self.merged_json_path = os.path.join(self.json_dir, "master_merge.json")
        self.reg_numbers_path = os.path.join(self.output_dir, "registration_numbers.txt")
        self.names_path = os.path.join(output_dir, "medication_names.txt")
        self.state_file = os.path.join(output_dir, ".cimavet_state.json")
        
        self.registration_numbers = []
        self.base_cimavet_url = "https://cimavet.aemps.es/cimavet"
        self.base_ema_url = "https://medicines.health.europa.eu"
        self.last_execution = None # [1]
        self.consecutive_empty_responses = 0 # [2]
        self._load_state() # overrides [1] and [2]
    
    # 1. Download JSONs with regulatory data on veterinary SmPCs
    def download_json_data(self):
        """Download JSON files with regulatory data"""
        logger.info("Starting JSON master data download...")
        
        # Get data from the first page to calculate parameters
        first_page_url = f"{self.base_cimavet_url}/rest/medicamentos?nombre=*&pagesize=1000&pagina=1"
        response = requests.get(first_page_url)
        if response.status_code != 200:
            logger.error(f"Error downloading the first page: {response.status_code}")
            return False
            
        master_data = response.json()
        
        # Calculate pagination parameters
        total_regs = master_data["totalFilas"]
        page_size = master_data["tamanioPagina"]
        total_pages = total_regs // page_size
        last_page_size = total_regs % page_size
        
        if last_page_size > 0:
            total_pages += 1
        else:
            last_page_size = page_size
            
        logger.info(f"Total records: {total_regs}, Pages: {total_pages}, "
                   f"Page size: {page_size}, Last page: {last_page_size}")
        
        # Save the first page
        first_page_file = os.path.join(self.json_dir, "master_page1.json")
        with open(first_page_file, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, ensure_ascii=False, indent=2)
        
        # Download the rest of the pages
        for page in tqdm(range(2, total_pages + 1), desc="Downloading JSON pages"):
            current_page_size = page_size if page < total_pages else last_page_size
            url = f"{self.base_cimavet_url}/rest/medicamentos?nombre=*&pagesize={current_page_size}&pagina={page}"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    page_data = response.json()
                    output_file = os.path.join(self.json_dir, f"master_page{page}.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(page_data, f, ensure_ascii=False, indent=2)
                else:
                    logger.error(f"Error downloading the page {page}: {response.status_code}")
            except Exception as e:
                logger.error(f"Exception downloading page {page}: {str(e)}")
                
        logger.info("JSON data download completed")
        return True
    
    # 1.1 Download extra metadata and presentation information per registration number
    def get_presentation_extra_metadata(self, nregistro):
        """
        Retrieve detailed metadata for a given 'nregistro'

        Args:
            nregistro (str): Registration number

        Returns:
            dict or None: JSON metadata if available, None otherwise
        """
        url = f"{self.base_cimavet_url}/rest/medicamento?nregistro={nregistro}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch metadata for {nregistro}: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Exception while fetching metadata for {nregistro}: {e}")
            return None
    
    # 1.2 Load static mapping for Principio Activo (COD_PACTIV) to Antibiotic classes
    def _load_static_pact_mapping(self):
        """
        Loads the static mapping of COD_PACTIV to Antibiotic classes from a CSV file.
        """
        try:
            # Use same logic as in other parts of the project to define project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if project_root not in sys.path:
                sys.path.append(project_root)

            pact_mapping_path = os.path.join(project_root, "data/priori_resources/pact_antibioticos.csv")

            df = pd.read_csv(pact_mapping_path, sep=';',dtype=str)
            df = df.drop_duplicates(subset=["COD_PACTIV"])
            mapping = df.set_index("COD_PACTIV")[["Categoría", "Familia", "Principio activo"]].to_dict(orient="index")

            logger.info(f"Loaded {len(mapping)} antibiotic mappings from CSV")
            return mapping
        except Exception as e:
            logger.error(f"Error loading static antibiotic mapping: {e}")
            return {}
        
    # 1.3 Enrich the metadata with antibiotic information from the mapping
    def _build_antibiotic_details(self, pactivos):
        """
        Build antibiotic mapping information for given principles.

        Args:
            pactivos (list): List of dicts from metadata["principiosActivos"]

        Returns:
            list: List of enriched antibiotic entries (only those found in mapping)
        """
        enriched = []
        for p in pactivos:
            codigo = p.get("codigo", "").strip()
            if codigo in self.pact_mapping:
                enriched.append(self.pact_mapping[codigo])
        return enriched

        
    # Worker function to enrich a single record
    def _enrich_single_record(self, item):
        """
        Enrich a single medication record with metadata
        
        Args:
            item (tuple): Tuple containing (reg_num, medication_data)
            
        Returns:
            tuple: Tuple containing (reg_num, enriched_data)
        """
        reg_num, medication_data = item
        
        metadata = self.get_presentation_extra_metadata(reg_num)
        if metadata:
            medication_data["metadata"] = metadata
            
            # If the medication is an antibiotic, enrich with antibiotic details
            if medication_data.get("antibiotico") is True:
                pactivos = metadata.get("principiosActivos", [])
                enriched_antibioticos = self._build_antibiotic_details(pactivos)
                medication_data["antibiotico"] = enriched_antibioticos
            
            return reg_num, medication_data, True
        
        return reg_num, medication_data, False
        
    # 2. Merge all previous JSONs into a single one (for efficiency in subsequent steps)
    def merge_json_files(self):
        """
        Merge all JSON files into a single dictionary format and enrich with extra metadata
        using parallel processing to speed up HTTP requests
        """
        logger.info("Merging JSON files...")
        merged_data = {}
        
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            file_path = os.path.join(self.json_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if "resultados" in data:
                    for item in data["resultados"]:
                        reg_num = item.get("nregistro")
                        if reg_num:
                            merged_data[reg_num] = {k: v for k, v in item.items() if k != "nregistro"}

            except Exception as e:
                logger.error(f"Error processing {json_file}: {str(e)}")

        # Log how many registration numbers we will enrich
        total_records = len(merged_data)
        logger.info(f"Enriching {total_records} medications with full metadata and presentations info...")

        # Create a list of items to be processed
        items_to_process = list(merged_data.items())
        enriched_count = 0
        
        # Initialize the progress bar
        progress_bar = tqdm(total=total_records, desc="Fetching full metadata")
        
        # Process in blocks to avoid exhausting resources
        batch_size = 500
        for i in range(0, len(items_to_process), batch_size):
            batch = items_to_process[i:i+batch_size]
            
            # Using ThreadPoolExecutor to parallelize HTTP requests
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self._enrich_single_record, item): item[0] for item in batch}
                
                # Process results as they are completed
                for future in as_completed(futures):
                    reg_num, updated_data, was_enriched = future.result()
                    merged_data[reg_num] = updated_data
                    if was_enriched:
                        enriched_count += 1
                    progress_bar.update(1)
        
        progress_bar.close()
        
        # Save the merged JSON
        with open(self.merged_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        count = len(merged_data)
        logger.info(f"Merged {count} medication records into master_merge.json")
        logger.info(f"Successfully enriched {enriched_count} records with metadata")
        return count > 0
    
    # 3. Extract registration numbers
    def extract_registration_numbers(self):
        """Extract registration numbers (all or only new ones)"""
        try:
            if self.consecutive_empty_responses >= self.max_empty_responses:
                logger.warning(f"{self.max_empty_responses} consecutive empty responses. Forcing extract all reg numbers.")
                success = self._extract_all_registration_numbers()
                if success:
                    self.consecutive_empty_responses = 0  # Reset counter
            elif not self.last_execution:
                success = self._extract_all_registration_numbers()
            else:
                success = self._extract_new_registration_numbers()
            
            if success: 
                self._save_state()
                # Save registration numbers to a file
                with open(self.reg_numbers_path, 'w', encoding='utf-8') as f:
                    for reg in self.registration_numbers:
                        f.write(f"{reg}\n")
            return success
        except Exception as e:
            logger.error(f"Error in extract_registration_numbers: {e}")
            return False
    
    def _get_smpc_reg_numbers_with_changes(self, data):
        """Obtain registration numbers of SmPCs that have changed"""
        regs = []
        for d in data:
            if "ft" in d.get("cambio", []): # only SmPCs
                regs.append(d.get("nregistro", ""))
        regs = [x for x in regs if x.strip()]
        return regs
    
    def _extract_new_registration_numbers(self):
        """Query external service to get modified reg numbers from last_execution"""
        logger.info("Extracting new registration numbers...")
        
        # Get data from service
        url = f"{self.base_cimavet_url}/rest/registroCambios?fecha={self.last_execution}&pagesize=1500"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # get registration numbers from data
                new_regs = self._get_smpc_reg_numbers_with_changes(data.get("resultados", []))
                if not new_regs: # empty
                    self.consecutive_empty_responses += 1
                    logger.warning(f"Empty response from the service "
                                   f"({self.consecutive_empty_responses}/{self.max_empty_responses})")
                else:
                    self.consecutive_empty_responses = 0 # reset counter
                
                self.registration_numbers = new_regs
                logger.info(f"{len(self.registration_numbers)} new registration numbers have been extracted")
                return True
            else:
                logger.error(f"Error downloading the page registroCambios: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Exception downloading page registroCambios: {str(e)}")
            return False
    
    def _extract_all_registration_numbers(self):
        """Extract all registration numbers from merged JSON file."""
        logger.info("Extracting all registration numbers from merged JSON...")
        registration_numbers = []
        
        try:
            with open(self.merged_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            registration_numbers = list(data.keys())
        except Exception as e:
            logger.error(f"Error processing merged JSON: {str(e)}")
        
        #self.registration_numbers = [r for r in registration_numbers if not r.startswith("EU")] # skip EU reg numbers
        self.registration_numbers = registration_numbers
        logger.info(f"{len(registration_numbers)} registration numbers have been extracted")
        
        return len(registration_numbers) > 0
    
    # 4. Extract medicine names from the merged JSON
    def extract_medication_names(self):
        """Extract medication names from merged JSON file."""
        logger.info("Extracting medication names...")
        medication_names = []
        
        try:
            with open(self.merged_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            medication_names = [med["nombre"] for med in data.values() if "nombre" in med and med["nombre"]]
        except Exception as e:
            logger.error(f"Error processing merged JSON: {str(e)}")
        
        self.medication_names = medication_names
        logger.info(f"{len(medication_names)} medication names have been extracted")
        
        # Save medication names to a file
        with open(self.names_path, 'w', encoding='utf-8') as f:
            for name in medication_names:
                f.write(f"{name}\n")
        
        return len(medication_names) > 0
    
    # 5. Download PDFs of the SmPCs in parallel --- V1
    def _download_pdf_v1(self, registration_number):
        """Download a specific PDF"""
        # Replace spaces with + for the URL
        reg_for_url = registration_number.replace(" ", "+")
        pdf_url = f"{self.base_cimavet_url}/pdfs/es/ft/{reg_for_url}/FT_{reg_for_url}.pdf"
        
        output_path = os.path.join(self.pdf_dir, f"FT_{reg_for_url.replace('+', '_')}.pdf")
        
        try:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                logger.warning(f"Could not download PDF for {registration_number} (HTTP {response.status_code})")
                return False
        except Exception as e:
            logger.error(f"Error downloading PDF for {registration_number}: {str(e)}")
            return False
    
    def download_pdfs_v1(self):
        """Download all PDFs using multiple threads"""
        if not self.registration_numbers:
            logger.error("No registration numbers are available. Run extract_registration_numbers first.")
            return False
            
        logger.info(f"Starting download of {len(self.registration_numbers)} PDFs...")
        
        success_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(self._download_pdf_v1, self.registration_numbers),
                total=len(self.registration_numbers),
                desc="Downloading PDFs"
            ))
            
        success_count = sum(1 for result in results if result)
        failed_count = len(results) - success_count
        
        logger.info(f"PDF download completed. Successes: {success_count}, Failures: {failed_count}")
        return success_count > 0
    
    # 5. Download PDFs of the SmPCs in parallel
    # The EMA server has a rate limiting system in place that detects patterns of simultaneous requests 
    # and responds with a 429 code to protect against what could be interpreted as a DoS attack or 
    # aggressive scraping
    # To avoid this: 
    # - process non-EU registrations with full parallelism
    # - process the EU registrations with reduced parallelism
    def _get_pdf_url_from_ema_page(self, html_content):
        """
        Extract the PDF download URL from EMA HTML page
        
        Args:
            html_content (str): HTML content of the EMA page
            
        Returns:
            str or None: Full URL to download the PDF or None if not found
        """
        try:
            # Use BeautifulSoup to parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find links that contain the PDF download path
            download_path = None
            links = soup.find_all('a', href=True)
            for link in links:
                if '/veterinary/es/documents/download/' in link['href']:
                    download_path = link['href']
                    break
            
            if download_path:
                # Build the full URL
                return f"{self.base_ema_url}{download_path}"
            else:
                logger.warning("No PDF download link found on the EMA page")
                return None
        except Exception as e:
            logger.error(f"Error extracting PDF URL from EMA page: {str(e)}")
            return None

    def _download_with_retry(self, url, stream=True, retry_count=0):
        """
        Download with retry logic and exponential backoff
        
        Args:
            url (str): URL to download
            stream (bool): Whether to stream the response
            retry_count (int): Current retry attempt
            
        Returns:
            requests.Response or None: Response object or None if all retries failed
        """
        if retry_count > self.max_retries:
            logger.warning(f"Maximum number of retries reached for URL: {url}")
            return None
            
        try:
            # Add a random delay to avoid detectable request patterns
            # The delay increases exponentially with each retry
            if retry_count > 0:
                delay = (2 ** retry_count) + random.uniform(0, 1)
                logger.info(f"Retrying download after {delay:.2f} seconds (attempt {retry_count})")
                time.sleep(delay)
            else:
                # Small random delay even for the first attempt
                time.sleep(random.uniform(0.5, 1.5))
                
            # Make the request with a reasonable timeout
            response = requests.get(url, stream=stream, timeout=30)
            
            # Specifically handle the case of rate limiting
            if response.status_code == 429:
                # Get the recommended timeout from the Retry-After header if available
                retry_after = response.headers.get('Retry-After')
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** retry_count) + 5
                
                logger.warning(f"Rate limit reached (429). Waiting {wait_time} seconds before retrying.")
                time.sleep(wait_time)
                
                # Retry recursively
                return self._download_with_retry(url, stream, retry_count + 1)
                
            return response
            
        except (requests.RequestException, Exception) as e:
            logger.error(f"Error during download (attempt {retry_count+1}): {str(e)}")
            return self._download_with_retry(url, stream, retry_count + 1)

    def _download_pdf(self, registration_number):
        """
        Download a specific PDF, handling both ESP and EU registration numbers with retry logic
        
        Args:
            registration_number (str): The registration number
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        is_eu_registration = registration_number.startswith('EU')
        
        # Format the registration number for the URL
        reg_for_url = format_registration_for_url(registration_number)
        
        # Build the output path of the PDF file
        safe_filename = reg_for_url.replace('+', '_').replace('@', '-')
        output_path = os.path.join(self.pdf_dir, f"FT_{safe_filename}.pdf")
        
        # If the file already exists, avoid re-downloading it.
        #if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            #logger.info(f"PDF already exists for {registration_number}, skipping download")
            #return True
        
        # Build the initial URL for the download
        pdf_url = f"{self.base_cimavet_url}/pdfs/es/ft/{reg_for_url}/FT_{reg_for_url}.pdf"
        
        try:
            # First download attempt with retries
            response = self._download_with_retry(pdf_url)
            
            if response is None:
                return False
            
            # For EU registrations, the response could be an HTML redirect page.
            if is_eu_registration and response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                
                # If the response is HTML instead of PDF
                if 'text/html' in content_type:
                    # Extract real PDF URL from HTML page
                    real_pdf_url = self._get_pdf_url_from_ema_page(response.text)
                    
                    if real_pdf_url:
                        # Make a second request for the real PDF
                        logger.info(f"Redirecting to EMA to download PDF of {registration_number}")
                        response = self._download_with_retry(real_pdf_url)
                        
                        if response is None:
                            return False
                    else:
                        logger.warning(f"Could not extract PDF URL for {registration_number}")
                        return False
            
            # Save the downloaded PDF
            if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                #logger.info(f"PDF downloaded successfully for {registration_number}")
                return True
            else:
                logger.warning(f"Could not download PDF for {registration_number} (HTTP {response.status_code})")
                return False
        except Exception as e:
            logger.error(f"Error downloading PDF for {registration_number}: {str(e)}")
            return False
    
    def download_pdfs(self):
        """
        Download all PDFs using multiple threads, with special handling for EU registrations
        """
        if not self.registration_numbers:
            # Check if the status indicates an error or if there are simply no changes.
            if self.last_execution and self.consecutive_empty_responses < self.max_empty_responses:
                logger.info("No changes detected since last execution. Skipping PDF download.")
                return True  # return True because it is not an error, but a valid state
            else:
                logger.error("No registration numbers are available. Run extract_registration_numbers first.")
                return False
        
        logger.info(f"Starting download of {len(self.registration_numbers)} PDFs...")
        
        # Separate registration numbers into EU and non-EU
        eu_registrations = [reg for reg in self.registration_numbers if reg.startswith('EU')]
        non_eu_registrations = [reg for reg in self.registration_numbers if not reg.startswith('EU')]
        
        success_count = 0
        failed_count = 0
        
        # First process non-EU registrations with full parallelism
        if non_eu_registrations:
            logger.info(f"Processing {len(non_eu_registrations)} non-EU registrations with {self.num_workers} workers")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(tqdm(
                    executor.map(self._download_pdf, non_eu_registrations),
                    total=len(non_eu_registrations),
                    desc="Downloading non-EU PDFs"
                ))
                
            non_eu_success = sum(1 for result in results if result)
            non_eu_failed = len(results) - non_eu_success
            success_count += non_eu_success
            failed_count += non_eu_failed
            
        # Then process the EU registrations with reduced parallelism
        if eu_registrations:
            logger.info(f"Processing {len(eu_registrations)} EU registrations with {self.max_eu_workers} workers to avoid rate limiting")
            with ThreadPoolExecutor(max_workers=self.max_eu_workers) as executor:
                results = list(tqdm(
                    executor.map(self._download_pdf, eu_registrations),
                    total=len(eu_registrations),
                    desc="Downloading EU PDFs (reduced parallelism)"
                ))
                
            eu_success = sum(1 for result in results if result)
            eu_failed = len(results) - eu_success
            success_count += eu_success
            failed_count += eu_failed
        
        logger.info(f"PDF download completed. Successes: {success_count}, Failures: {failed_count}")
        return success_count > 0
    
    # 6. Convert the PDFs to Markdown in parallel
    def _extract_with_pymupdf4llm(self, pdf_path, show_progress=False):
        """
        Extracts text from a PDF file using pymupdf4llm.
        Returns the extracted text in markdown format.
        """
        return pymupdf4llm.to_markdown(pdf_path, show_progress=show_progress)
    
    def convert_to_markdown_serial_version(self):
        """
        Convert PDFs to Markdown-formatted text. Use pymupdf4llm.
        Sequential processing, 5x slower than the parallel version.
        """
        logger.info("Starting PDF conversion to Markdown...")
        
        try:
            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
            
            for pdf_file in tqdm(pdf_files, desc="Converting to markdown"):
                pdf_path = os.path.join(self.pdf_dir, pdf_file)
                output_base = os.path.splitext(pdf_file)[0]
                #if output_base.startswith("FT_EU"): # skip
                #    logger.info(f"skip {output_base}.pdf")
                #    continue
                markdown_path = os.path.join(self.markdown_dir, f"{output_base}.md")
                #logger.info(f"Convert {pdf_path} to {markdown_path}")
                md = self._extract_with_pymupdf4llm(pdf_path)
                # save
                with open(markdown_path, "w", encoding="utf-8") as fp:
                    fp.write(md)
                
            logger.info(f"Markdown conversion completed for {len(pdf_files)} files")
            return True
        except Exception as e:
            logger.error(f"Error converting to markdown: {str(e)}")
            return False
    
    @staticmethod
    def _process_single_pdf(pdf_dir, markdown_dir, pdf_file):
        """
        Process a single PDF file and convert it to markdown.
        This function is static to avoid shared state issues when running in parallel.
        """
        try:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            output_base = os.path.splitext(pdf_file)[0]
            
            #if output_base.startswith("FT_EU"):  # skip
                #print(f"skip {output_base}.pdf")  # Using print instead of logger for multiprocessing
            #    return None
                
            markdown_path = os.path.join(markdown_dir, f"{output_base}.md")
            
            # Process the PDF in a separate process
            md = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
            
            # save
            with open(markdown_path, "w", encoding="utf-8") as fp:
                fp.write(md)
            
            return pdf_file
        except Exception as e:
            print(f"Error converting {pdf_file} to markdown: {str(e)}")  # Using print for multiprocessing
            return None
    
    def convert_to_markdown(self):
        """
        Convert PDFs to Markdown-formatted text 
        using parallel processing with ProcessPoolExecutor, 
        5x faster than the serial version
        """
        logger.info("Starting parallel PDF conversion to Markdown using ProcessPoolExecutor...")
        
        try:
            # If there are no registration numbers to process and it is a valid status, skip the conversion
            if not self.registration_numbers:
                if self.last_execution and self.consecutive_empty_responses < self.max_empty_responses:
                    logger.info("No changes detected since last execution. Skipping Markdown conversion.")
                    return True
            
            # Determine which PDFs to process
            if self.registration_numbers:
                # Only convert PDFs corresponding to updated registration numbers
                pdf_files = []
                for reg_number in self.registration_numbers:
                    # build the filename of the PDF file
                    reg_for_url = format_registration_for_url(reg_number)
                    pdf_file = f"FT_{reg_for_url.replace('+', '_').replace('@', '-')}.pdf"
                    if os.path.exists(os.path.join(self.pdf_dir, pdf_file)):
                        pdf_files.append(pdf_file)
                
                if not pdf_files:
                    logger.warning("No PDF files found for the current registration numbers.")
                    return True  # Return True because it is not an error, just that there are no PDFs to convert
            else:
                # If there are no specific registrations, process all PDFs (first-execution case)
                pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
            
            successful_conversions = 0
            
            # If there are no PDFs to process, finish
            if not pdf_files:
                logger.info("No PDF files to convert.")
                return True
            
            # Create a list of parameters for each PDF file
            pdf_params = [(self.pdf_dir, self.markdown_dir, pdf_file) for pdf_file in pdf_files]
            
            # Use ProcessPoolExecutor instead of ThreadPoolExecutor
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(tqdm(
                    executor.map(self._process_single_pdf, *zip(*pdf_params)),
                    total=len(pdf_files),
                    desc="Converting to markdown"
                ))
                
                # Count successful conversions
                successful_conversions = sum(1 for result in results if result is not None)
            
            logger.info(f"Markdown conversion completed: {successful_conversions} successful out of {len(pdf_files)} files")
            return True
        except Exception as e:
            logger.error(f"Error in parallel markdown conversion: {str(e)}")
            return False
    
    # 7. Parse the PDFs with Markdown to generate JSON with a predefined structure
    def parse_markdown_to_json(self):
        """
        Parse markdown files to structured JSON format.
        Takes each markdown file generated from PDFs and extracts structured data
        about veterinary medications, saving the results as JSON files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting parsing markdown to JSON...")
        
        try:
            # If there are no registration numbers to process and it is a valid state, skip parsing
            if not self.registration_numbers:
                if self.last_execution and self.consecutive_empty_responses < self.max_empty_responses:
                    logger.info("No changes detected since last execution. Skipping markdown to JSON parsing.")
                    return True
            
            # Add the project directory to the path for imports
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            # Import markdown parser
            from resource_builder.scripts.markdown_parser import MarkdownParser
            
            # load merged JSON
            with open(self.merged_json_path, 'r', encoding='utf-8') as f:
                merged_json = json.load(f)
            
            # Determine which markdown files to process
            if self.registration_numbers:
                # Only process markdowns corresponding to updated registration numbers
                md_files = []
                for reg_number in self.registration_numbers:
                    # build the filename of the markdown file
                    reg_for_url = format_registration_for_url(reg_number)
                    md_file = f"FT_{reg_for_url.replace('+', '_').replace('@', '-')}.md"
                    if os.path.exists(os.path.join(self.markdown_dir, md_file)):
                        md_files.append(md_file)
                
                if not md_files:
                    logger.warning("No markdown files found for the current registration numbers.")
                    return True  # It's not an error, just that there are no files to process.
            else:
                # If there are no specific registrations, process all markdown files
                md_files = [f for f in os.listdir(self.markdown_dir) if f.endswith('.md')]
            
            # If there are no markdown files to process, terminate
            if not md_files:
                logger.info("No markdown files to parse.")
                return True
        
            for md_file in tqdm(md_files, desc="Parsing to JSON"):
                md_path = os.path.join(self.markdown_dir, md_file)
                output_base = os.path.splitext(md_file)[0]
                json_path = os.path.join(self.processed_json_dir, f"{output_base}.json")
                
                # Create and use the MarkdownParser
                parser = MarkdownParser(md_path, merged_json, json_path)
                success = parser.process()
                
                if not success:
                    logger.warning(f"Failed to process {md_file}")
            
            logger.info(f"JSON conversion completed for {len(md_files)} files")
            return True
        except Exception as e:
            logger.error(f"Error converting to JSON: {str(e)}")
            return False
    
    # 8. Creates text files with essential information for each medicine
    def create_essential_info_files(self):
        """
        Creates text files with essential information for each medicine.
        These files contain key metadata that can be used by a primary FAISS index
        to filter relevant medications before detailed document search.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting creation of essential information files...")
        
        try:
            # Create directory for essential info files
            self.essential_info_dir = os.path.join(self.output_dir, "essential_info")
            os.makedirs(self.essential_info_dir, exist_ok=True)
            
            # If there are no registration numbers to process and it is a valid state, skip creation
            if not self.registration_numbers:
                if self.last_execution and self.consecutive_empty_responses < self.max_empty_responses:
                    logger.info("No changes detected since last execution. Skipping essential info files creation.")
                    return True
            
            # Determine which JSON files to process
            if self.registration_numbers:
                # Only process JSONs corresponding to updated registration numbers
                json_files = []
                for reg_number in self.registration_numbers:
                    # Build the filename of the JSON file
                    reg_for_url = format_registration_for_url(reg_number)
                    json_file = f"FT_{reg_for_url.replace('+', '_').replace('@', '-')}.json"
                    json_path = os.path.join(self.processed_json_dir, json_file)
                    if os.path.exists(json_path):
                        json_files.append(json_file)
                
                if not json_files:
                    logger.warning("No JSON files found for the current registration numbers.")
                    return True  # It's not an error, just that there are no files to process
            else:
                # If there are no specific registrations, process all JSON files
                json_files = [f for f in os.listdir(self.processed_json_dir) if f.endswith('.json')]
            
            # If there are no JSON files to process, terminate
            if not json_files:
                logger.info("No JSON files to process for essential info creation.")
                return True
            
            processed_count = 0
            for json_file in tqdm(json_files, desc="Creating essential info files"):
                json_path = os.path.join(self.processed_json_dir, json_file)
                
                try:
                    # Load the JSON document
                    with open(json_path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                    
                    # Create essential info text
                    essential_text = self._create_essential_info_text(doc)
                    
                    # Save essential info to text file
                    output_filename = os.path.splitext(json_file)[0] + ".txt"
                    output_path = os.path.join(self.essential_info_dir, output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(essential_text)
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process {json_file}: {str(e)}")
                    continue
            
            logger.info(f"Essential info files creation completed for {processed_count} files")
            return True
            
        except Exception as e:
            logger.error(f"Error creating essential info files: {str(e)}")
            return False

    def _create_essential_info_text(self, doc):
        """
        Creates essential information text from a document JSON.
        This text contains key searchable information for primary filtering.
        
        Args:
            doc (dict): Document JSON data
            
        Returns:
            str: Essential information text
        """
        # Extract document ID
        doc_id = doc.get('document_id', '')
        
        # Extract key metadata
        med_name = doc.get('nombre_medicamento', '')
        lab_titular = doc.get('laboratorio_titular', '')
        fecha_autorizacion = doc.get('fecha_primera_autorizacion', '')
        
        # Process ATC codes
        atc_info = []
        for item in doc.get("codigos_atc", []):
            codigo = item.get('codigo', '')
            nombre = item.get('nombre', '')
            nivel = item.get('nivel', '')
            if codigo and nombre:
                atc_info.append(f"{codigo}: {nombre} (Nivel {nivel})")
        
        atc_text = "Códigos ATC: " + "; ".join(atc_info) if atc_info else ""
        
        # Process species - use normalized names from especies_cimavet
        species_cimavet = doc.get('especies_cimavet', [])
        if species_cimavet:
            target_species = []
            for sp in species_cimavet:
                nombre = sp.get('nombre', '')
                nombre_normalizado = sp.get('nombre_normalizado', '')
                
                # If both names exist and are different (case-insensitive), show both
                if nombre and nombre_normalizado and nombre.lower() != nombre_normalizado.lower():
                    target_species.append(f"{nombre} ({nombre_normalizado})")
                # Otherwise, use the normalized name, or the regular name if normalized doesn't exist
                else:
                    target_species.append(nombre_normalizado or nombre)
            
            species_text = "Especies: " + ", ".join(target_species)
        else:
            species_raw = doc.get('especies_destino', '')
            species_text = f"Especies: {species_raw}" if species_raw else ""
        
        # Process active ingredients - both text and structured data
        active_ingredients = doc.get('principios_activos', '')
        active_ingredients_cimavet = doc.get('principios_activos_cimavet', [])
        
        if active_ingredients_cimavet:
            active_list = []
            for item in active_ingredients_cimavet:
                nombre = item.get('nombre', '')
                cantidad = item.get('cantidad', '')
                unidad = item.get('unidad', '')
                if nombre:
                    if cantidad and unidad:
                        active_list.append(f"{nombre} ({cantidad} {unidad})")
                    else:
                        active_list.append(nombre)
            active_text = "Principios activos: " + "; ".join(active_list)
        else:
            active_text = f"Principios activos: {active_ingredients}" if active_ingredients else ""
        
        # Pharmaceutical form and administration
        pharm_form = doc.get('forma_farmaceutica', '')
        pharm_form_text = f"Forma farmacéutica: {pharm_form}" if pharm_form else ""
        
        # Administration routes
        admin_routes = doc.get('vias_administracion', [])
        if admin_routes:
            routes_list = [route.get('nombre', '') for route in admin_routes if route.get('nombre')]
            admin_routes_text = "Vías de administración: " + ", ".join(routes_list)
        else:
            admin_routes_text = ""
        
        # Antibiotic information
        antibiotic = doc.get('antibiotico', False)
        antibiotic_text = f"Antibiótico: {'Sí' if antibiotic else 'No'}"
        
        # Process indications by species (simplified)
        indications = doc.get('indicaciones', [])
        indication_list = []
        if indications:
            indications_by_species = {}
            for indication in indications:
                especie = indication.get('especie', {})
                especie_name = especie.get('nombre_normalizado', especie.get('nombre', 'General'))
                indication_name = indication.get('nombre', '')
                if indication_name:
                    if especie_name not in indications_by_species:
                        indications_by_species[especie_name] = []
                    indications_by_species[especie_name].append(indication_name)
            
            for especie, inds in indications_by_species.items():
                indication_list.extend([f"{ind} ({especie})" for ind in inds])
        
        indications_text = "Indicaciones: " + "; ".join(indication_list) if indication_list else ""
        
        # Process contraindications (simplified)
        contraindications = doc.get('contraindicaciones', [])
        contra_list = []
        contraindicated_species = []
        
        if contraindications:
            for contra in contraindications:
                if contra.get('es_especie', False):
                    # This contraindication is itself a species
                    species_name = contra.get('nombre_normalizado', contra.get('nombre', ''))
                    if species_name:
                        contraindicated_species.append(species_name)
                else:
                    contra_name = contra.get('nombre', '')
                    if contra_name:
                        if 'especie' in contra:
                            especie = contra.get('especie', {})
                            especie_name = especie.get('nombre_normalizado', especie.get('nombre', ''))
                            contra_list.append(f"{contra_name} ({especie_name})")
                        else:
                            contra_list.append(contra_name)
        
        contraindications_text = ""
        if contra_list:
            contraindications_text = "Contraindicaciones: " + "; ".join(contra_list)
        if contraindicated_species:
            contraindicated_text = "No usar en: " + ", ".join(contraindicated_species)
            if contraindications_text:
                contraindications_text += ". " + contraindicated_text
            else:
                contraindications_text = contraindicated_text
        
        # Dispensing and administration conditions
        dis_conditions = doc.get('condiciones_dispensacion', '')
        dis_conditions_text = f"Dispensación: {dis_conditions}" if dis_conditions else ""
        
        admin_conditions = doc.get('condiciones_administracion', '')
        admin_conditions_text = f"Administración: {admin_conditions}" if admin_conditions else ""
        
        # Build the essential info text
        essential_parts = [
            f"ID: {doc_id}",
            f"Medicamento: {med_name}",
            f"Laboratorio: {lab_titular}",
            f"Autorización: {fecha_autorizacion}",
            species_text,
            atc_text,
            active_text,
            pharm_form_text,
            admin_routes_text,
            antibiotic_text,
            indications_text,
            contraindications_text,
            dis_conditions_text,
            admin_conditions_text
        ]
        
        # Filter out empty parts and join
        essential_text = "\n".join([part for part in essential_parts if part.strip()])
        
        return essential_text
    
    # 9. Store the JSON information in FAISS
    def store_in_faiss(self):
        """Stores the content of the JSONs in FAISS"""
        logger.info("Starting storage at FAISS...")
    
        try:
            # Add the project directory to the path for imports
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if project_root not in sys.path:
                sys.path.append(project_root)
    
            # Import embedding model and FAISS storage classes
            from shared.veterinary_utils.embedding_model import EmbeddingModel
            if self.embed_model_path == "jinaai/jina-embeddings-v3":
                from shared.veterinary_utils.jina_embedding_model import JinaEmbeddingModel as EmbeddingModel
            else:
                from shared.veterinary_utils.embedding_model import EmbeddingModel


            
            if self.use_hierarchical_storage:
                from resource_builder.scripts.faiss_storage import HierarchicalFaissStorage
            else:
                from resource_builder.scripts.faiss_storage import FaissStorage
    
            # Configure model and output folder
            model_path = self.embed_model_path if hasattr(self, 'embed_model_path') else "intfloat/multilingual-e5-large"
            model_tag = model_path.split("/")[-1].replace("-", "_").replace(".", "_")
            faiss_output_dir = os.path.join(project_root, f"data/posteriori_resources/faiss_stuff_{model_tag}")
            os.makedirs(faiss_output_dir, exist_ok=True)
    
            json_dir = os.path.join(project_root, "data/posteriori_resources/processed_json")
            essential_info_dir = os.path.join(project_root, "data/posteriori_resources/essential_info")
    
            # Define paths based on retrieval mode
            if self.use_hierarchical_storage:
                essential_index_path = os.path.join(faiss_output_dir, "essential_index.faiss")
                essential_mapping_path = os.path.join(faiss_output_dir, "essential_mapping.json")
                essential_cache_path = os.path.join(faiss_output_dir, "essential_cache.json")
    
                chunks_index_path = os.path.join(faiss_output_dir, "chunks_index.faiss")
                chunks_mapping_path = os.path.join(faiss_output_dir, "chunks_mapping.json")
                chunks_cache_path = os.path.join(faiss_output_dir, "chunks_cache.json")
    
                # Skip if indices already exist and no updates
                hierarchical_indices_exist = (
                    os.path.exists(essential_index_path) and os.path.getsize(essential_index_path) > 0 and
                    os.path.exists(chunks_index_path) and os.path.getsize(chunks_index_path) > 0
                )
                if hierarchical_indices_exist and not self.registration_numbers:
                    if self.last_execution and self.consecutive_empty_responses < self.max_empty_responses:
                        logger.info("No changes detected since last execution. Skipping hierarchical FAISS storage.")
                        return True
            else:
                faiss_index_path = os.path.join(faiss_output_dir, "index.faiss")
                mapping_path = os.path.join(faiss_output_dir, "mapping.json")
                chunks_path = os.path.join(faiss_output_dir, "chunks.json")
    
                if os.path.exists(faiss_index_path) and os.path.getsize(faiss_index_path) > 0 and not self.registration_numbers:
                    if self.last_execution and self.consecutive_empty_responses < self.max_empty_responses:
                        logger.info("No changes detected since last execution. Skipping traditional FAISS storage.")
                        return True
    
            # Load and use embedding model
            logger.info(f"Using embedding model: {model_path}")
            embedding_model = EmbeddingModel(model_path, "cuda", 512)
    
            if self.use_hierarchical_storage:
                logger.info("Using hierarchical FAISS storage (two-stage retrieval)...")
    
                if not os.path.exists(essential_info_dir):
                    logger.warning(f"Essential info directory not found: {essential_info_dir}")
                    os.makedirs(essential_info_dir, exist_ok=True)
    
                storage_emb = HierarchicalFaissStorage(
                    embedding_model,
                    embedding_dim=embedding_model.get_word_embedding_dimension()
                )
                storage_emb.add_documents_from_directory(json_dir, essential_info_dir)
                storage_emb.save_indices(
                    essential_index_path, essential_mapping_path, essential_cache_path,
                    chunks_index_path, chunks_mapping_path, chunks_cache_path
                )
                logger.info("Hierarchical FAISS storage completed successfully")
                logger.info(f"Essential info index: {storage_emb.essential_index.ntotal} documents")
                logger.info(f"Chunks index: {storage_emb.chunks_index.ntotal} chunks")
    
            else:
                logger.info("Using traditional FAISS storage (single-stage retrieval)...")
    
                storage_emb = FaissStorage(
                    embedding_model,
                    embedding_dim=embedding_model.get_word_embedding_dimension()
                )
                storage_emb.add_documents_from_directory(json_dir)
                storage_emb.save_index(faiss_index_path, mapping_path, chunks_path)
    
                logger.info("Traditional FAISS storage completed successfully")
                logger.info(f"Index contains: {storage_emb.index.ntotal} chunks")
    
            return True
    
        except Exception as e:
            logger.error(f"Error in storage in FAISS: {str(e)}")
            return False

    
    def _load_state(self):
        """Loads the previous state if it exists"""
        if os.path.isfile(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.last_execution = state.get("last_execution")
                    self.consecutive_empty_responses = state.get("consecutive_empty_responses", 0)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading state: {e}. Starting fresh.")
                self.last_execution = None
                self.consecutive_empty_responses = 0
    
    def _save_state(self):
        """Save the current state"""
        state = {
            "last_execution": datetime.today().strftime("%d/%m/%Y"), 
            "consecutive_empty_responses": self.consecutive_empty_responses
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)
    
    def run_pipeline(self):
        """Execute the complete pipeline"""
        steps = [
            ("Download JSON data", self.download_json_data),
            ("Merge JSON files", self.merge_json_files),
            ("Extraction of registration numbers", self.extract_registration_numbers),
            ("Extraction of medication names", self.extract_medication_names),
            ("Download PDFs", self.download_pdfs),
            ("Converting to markdown", self.convert_to_markdown),
            ("Parsing to JSON", self.parse_markdown_to_json),
            ("Create essential info files", self.create_essential_info_files),
            ("Storage in FAISS", self.store_in_faiss)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Starting step: {step_name}")
            result = step_func()
            if not result:
                logger.error(f"Error in step: {step_name}. Pipeline stopped.")
                return False
            logger.info(f"Step completed: {step_name}")
        
        self._save_state()
        logger.info("Pipeline successfully completed")
        return True

def main():
    parser = argparse.ArgumentParser(description='Unified processor for CIMAVet medicines',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output-dir', default='../data/posteriori_resources/', 
                        help='Output directory for all files')
    parser.add_argument('-w', '--workers', type=int, default=5,
                        help='Number of workers for parallel downloads')
    parser.add_argument('--skip-json', action='store_true',
                        help='Skip downloading JSON data')
    parser.add_argument('--skip-merge', action='store_true',
                        help='Skip merging JSON files')
    parser.add_argument('--skip-registration', action='store_true',
                        help='Skip extraction of registration numbers')
    parser.add_argument('--skip-names', action='store_true',
                        help='Skip extraction of medication names')
    parser.add_argument('--skip-pdf', action='store_true',
                        help='Skip downloading PDFs')
    parser.add_argument('--skip-markdown', action='store_true',
                        help='Skip conversion to markdown')
    parser.add_argument('--skip-json-parsing', action='store_true',
                        help='Skip parsing to JSON')
    parser.add_argument('--skip-essential-info', action='store_true',
                        help='Skip creation of essential info files')
    parser.add_argument('--skip-faiss', action='store_true',
                        help='Skip storage in FAISS')
    parser.add_argument('--embed-model',
                        default='intfloat/multilingual-e5-large',
                        help='HuggingFace model name or path for embeddings')

    
    args = parser.parse_args()
    
    processor = CIMAVetProcessor(args.output_dir, args.workers)
    processor.embed_model_path = args.embed_model
    
    # Run the pipeline with skip-step options
    if not args.skip_json:
        processor.download_json_data()
    
    # merged JSONs
    if not args.skip_merge:
        processor.merge_json_files()
    elif os.path.exists(processor.merged_json_path):
        logger.info(f"Using existing merged JSON file at {processor.merged_json_path}")
    
    # reg numbers
    if not args.skip_registration:
        processor.extract_registration_numbers()
    elif os.path.exists(processor.reg_numbers_path):
        # Upload registration numbers if they already exist
        with open(processor.reg_numbers_path, 'r') as f:
            processor.registration_numbers = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(processor.registration_numbers)} registration numbers from existing file")
    
    # med names
    if not args.skip_names:
        processor.extract_medication_names()
    elif os.path.exists(processor.names_path):
        # Upload medication names if they already exist
        with open(processor.names_path, 'r') as f:
            processor.medication_names = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(processor.medication_names)} medication names from existing file")
    
    if not args.skip_pdf:
        processor.download_pdfs()
    
    if not args.skip_markdown:
        processor.convert_to_markdown()
    
    if not args.skip_json_parsing:
        processor.parse_markdown_to_json()
    
    if not args.skip_essential_info:
        processor.create_essential_info_files()
    
    if not args.skip_faiss:
        processor.store_in_faiss()
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
