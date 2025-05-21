

import json
import os
import re
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.veterinary_utils.utils import (convert_timestamp_to_date,
                                           format_registration_for_url,
                                           format_registration_number)
from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

class MarkdownParser:
    """
    Class to parse markdown files from veterinary medicine technical data sheets
    and convert them into structured JSON format.
    """
    
    def __init__(self, markdown_path, merged_json, json_output_path):
        """
        Initialize the markdown parser.
        
        Args:
            markdown_path (str): Path to the markdown file
            merged_json (dict): Dictionary with regulatory data
            json_output_path (str): Path where the JSON output will be saved
        """
        self.markdown_path = markdown_path
        self.merged_json = merged_json
        self.json_output_path = json_output_path
        self.text = None
        # Define valid section IDs for veterinary technical sheets (1-10)
        self.valid_section_ids = [str(i) for i in range(1, 11)]
        # Maximum number of subsections per section based on documentation
        self.max_subsections = 15
        
    def load_markdown(self):
        """
        Load the content of the markdown file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.markdown_path, "r", encoding="utf-8") as f:
                self.text = f.read()
            return True
        except Exception as e:
            print(f"Error loading markdown file: {str(e)}")
            return False
            
    def get_cimavet_data(self, document_id):
        """
        Get cimavet data from the merged JSON file.
        
        Args:
            document_id (str): The document identifier
        Returns:
            dict: Cimavet data
        """
        return self.merged_json[document_id]

    def parse_ficha_tecnica(self):
        """
        Parses the loaded markdown text from a SmPC (ficha técnica),
        identifying sections and subsections based on bold formatting.
        
        Returns:
            sections (list): List of sections with their subsections
            especies_destino (str): Species for which the SmPC is indicated
        """
        if not self.text:
            return [], ""
            
        # Split the text into lines for line-by-line processing
        lines = self.text.split('\n')
        
        # Lists to store sections and subsections
        all_elements = []
        
        # Current position in the text for indexing
        current_pos = 0
        
        # First pass: detect all main sections (1-10)
        for i, line in enumerate(lines):
            original_line = line
            
            # Calculate the starting position of this line in the original text
            line_start = current_pos
            current_pos += len(original_line) + 1  # +1 for the newline
            
            # More comprehensive pattern for main sections, including ".-" format
            # Match patterns like:
            # **1. TITLE**
            # **1.** TITLE
            # **1** TITLE
            # **1.- TITLE**
            # **1.** - TITLE
            section_match = re.match(r'^\*\*(\d+)(?:\.|-\.|\.-)?\*\*[\s\.]*(-?\s*)(.*?)$', line)
            
            if not section_match:
                # Try alternative pattern where the entire line might be bold
                section_match = re.match(r'^\*\*(\d+)(?:\.|-\.|\.-)?\s+(-?\s*)(.*?)\*\*$', line)
                
            if not section_match:
                # Try another pattern where numbering might be outside bold markers
                section_match = re.match(r'^(\d+)(?:\.|-\.|\.-)?\s+\*\*(.*?)\*\*$', line)
            
            if section_match:
                section_id = section_match.group(1)
                
                # Only consider valid sections (1-10)
                if section_id not in self.valid_section_ids:
                    continue
                    
                # Extract the section title, removing any remaining bold markers and dash if present
                title = section_match.group(3) if len(section_match.groups()) >= 3 else section_match.group(2)
                title = title.replace('**', '').strip()
                
                # Remove trailing colon if present
                if title.endswith(':'):
                    title = title[:-1].strip()
                
                all_elements.append({
                    "type": "section",
                    "id": section_id,
                    "title": title,
                    "start": line_start,
                    "end": current_pos - 1,  # -1 to exclude the newline
                })
        
        # Reset position counter for the second pass
        current_pos = 0
                
        # Second pass: identify subsections
        for i, line in enumerate(lines):
            original_line = line
            
            # Calculate the starting position of this line in the original text
            line_start = current_pos
            current_pos += len(original_line) + 1  # +1 for the newline
            
            # More comprehensive patterns for subsections
            # Try to match with the original bold markers first, including the ".-" pattern
            subsection_match = re.match(r'^\*\*(\d+\.\d+)(?:\.|-\.|\.-)?\*\*[\s\.]*(-?\s*)(.*?)$', line)
            
            if not subsection_match:
                # Try with the entire line being bold
                subsection_match = re.match(r'^\*\*(\d+\.\d+)(?:\.|-\.|\.-)?\s+(-?\s*)(.*?)\*\*$', line)
                
            if not subsection_match:
                # Then try without bold markers (clean line)
                clean_line = re.sub(r'\*\*', '', line).strip()
                subsection_match = re.match(r'^(\d+\.\d+)(?:\.|-\.|\.-)?\s*(?:-\s*)?(.*?)$', clean_line)
            
            if subsection_match:
                section_id = subsection_match.group(1)
                title = subsection_match.group(3) if len(subsection_match.groups()) >= 3 else subsection_match.group(2)
                title = title.strip()
                
                # Validate subsection format: must be section.subsection where
                # section is 1-10 and subsection is 1-15
                parts = section_id.split('.')
                if len(parts) != 2:
                    continue
                    
                parent_id, subsection_num = parts
                
                # Check parent is valid and subsection is within acceptable range
                if (parent_id not in self.valid_section_ids or 
                    not subsection_num.isdigit() or 
                    int(subsection_num) > self.max_subsections):
                    continue
                
                # Remove trailing colon if present
                if title.endswith(':'):
                    title = title[:-1].strip()
                
                # Remove bold markers if any left
                title = title.replace('**', '').strip()
                
                all_elements.append({
                    "type": "subsection",
                    "id": section_id,
                    "title": title,
                    "start": line_start,
                    "end": current_pos - 1,  # -1 to exclude the newline
                    "parent_id": parent_id
                })
        
        # Sort all elements by their position in the text
        all_elements.sort(key=lambda x: x["start"])
        
        # For debugging: print all detected elements
        # for elem in all_elements:
        #     print(f"{elem['type']} {elem['id']} - {elem['title']}")
        
        # Process elements to extract content
        section_map = {}
        especies_destino = ""  # Will be populated later
        
        for i, elem in enumerate(all_elements):
            # Find the end of this element (start of next element that should terminate it)
            content_end = len(self.text)  # Default to end of text
            
            if elem["type"] == "section":
                # A section ends at the next section
                for j in range(i + 1, len(all_elements)):
                    if all_elements[j]["type"] == "section":
                        content_end = all_elements[j]["start"]
                        break
                        
            elif elem["type"] == "subsection":
                # A subsection ends at the next subsection of same parent or next section
                for j in range(i + 1, len(all_elements)):
                    next_elem = all_elements[j]
                    if (next_elem["type"] == "subsection" and next_elem["parent_id"] == elem["parent_id"]) or \
                       (next_elem["type"] == "section"):
                        content_end = next_elem["start"]
                        break
            
            # Extract content
            content = self.text[elem["end"]:content_end].strip()
            
            # Process based on element type
            if elem["type"] == "section":
                section_map[elem["id"]] = {
                    "seccion_id": elem["id"],
                    "titulo": elem["title"],
                    "contenido": content,
                    "subsecciones": []
                }
            elif elem["type"] == "subsection":
                # Add to parent section if it exists
                if elem["parent_id"] in section_map:
                    section_map[elem["parent_id"]]["subsecciones"].append({
                        "seccion_id": elem["id"],
                        "titulo": elem["title"],
                        "contenido": content
                    })
                else:
                    # Create parent section if it doesn't exist yet
                    section_map[elem["parent_id"]] = {
                        "seccion_id": elem["parent_id"],
                        "titulo": f"SECCIÓN {elem['parent_id']}",  # Generic title
                        "contenido": "",
                        "subsecciones": [{
                            "seccion_id": elem["id"],
                            "titulo": elem["title"],
                            "contenido": content
                        }]
                    }
        
        # Post-process sections to remove subsection content from section content
        for section_id, section in section_map.items():
            if section["subsecciones"]:
                # If there are subsections, clean up the section content
                first_subsection = min(section["subsecciones"], key=lambda x: 
                    next((e["start"] for e in all_elements if e["type"] == "subsection" and e["id"] == x["seccion_id"]), float('inf')))
                
                # Find the section and subsection elements by ID
                first_subsection_element = next((e for e in all_elements if e["type"] == "subsection" and e["id"] == first_subsection["seccion_id"]), None)
                section_element = next((e for e in all_elements if e["type"] == "section" and e["id"] == section_id), None)
                
                # Extract only the text between the section and its first subsection
                if section_element and first_subsection_element:
                    section["contenido"] = self.text[section_element["end"]:first_subsection_element["start"]].strip()
        
        # Sort subsections within each section
        for section in section_map.values():
            # Custom sorting to handle subsection IDs properly
            def subsection_sort_key(subsection):
                parts = subsection["seccion_id"].split(".")
                # Convert each part to int when possible for proper numeric sorting
                return [int(part) if part.isdigit() else part for part in parts]
                
            section["subsecciones"] = sorted(section["subsecciones"], key=subsection_sort_key)
        
        # Convert section_map to list and sort by section number
        sections = list(section_map.values())
        sections.sort(key=lambda x: int(x["seccion_id"]))
        
        # We'll find especies_destino later, in create_json_structure
        return sections, especies_destino
    
    def _clean_species_content(self, content):
        """
        Clean the species content to extract only the relevant species information.
        
        Args:
            content (str): The raw content from the species section
            
        Returns:
            str: Cleaned species information
        """
        # Remove any "Indicaciones" or paragraphs after the first line/paragraph
        lines = content.split('\n')
        if not lines:
            return ""
            
        # Get first paragraph (usually just the species name)
        species_text = lines[0].strip()
        
        # If the first paragraph is too long, it probably contains more than just the species
        # In that case, try to extract just the first sentence which often contains the species
        if len(species_text) > 50:
            # Try to split by period
            sentences = re.split(r'\.', species_text)
            if sentences:
                species_text = sentences[0].strip()
        
        # Clean up any trailing punctuation
        species_text = re.sub(r'[.,;:\(\)\[\]]$', '', species_text).strip()
        
        return species_text
    
    def create_json_structure(self, sections, especies_destino, document_id):
        """
        Create the final JSON structure with all the parsed information.
        
        Args:
            sections (list): List of sections with their subsections
            especies_destino (str): Species for which the SmPC is indicated
            document_id (str): Document identifier (registration number)
            
        Returns:
            dict: Structured JSON data
        """
        # Look for the especies_destino in section 3.1 first (most common location)
        for section in sections:
            if section["seccion_id"] == "3":
                for subsection in section["subsecciones"]:
                    if subsection["seccion_id"] == "3.1":
                        subsection_title = subsection["titulo"].lower()
                        if ("especie" in subsection_title and "destino" in subsection_title) or \
                           subsection_title in ["especie de destino", "especies de destino", "especies destino"]:
                            especies_destino = self._clean_species_content(subsection["contenido"])
                            break
            if especies_destino:
                break
        
        # If not found in 3.1, check all subsections
        if not especies_destino:
            for section in sections:
                for subsection in section["subsecciones"]:
                    subsection_title = subsection["titulo"].lower()
                    if ("especie" in subsection_title and "destino" in subsection_title) or \
                       subsection_title in ["especie de destino", "especies de destino", "especies destino"]:
                        especies_destino = self._clean_species_content(subsection["contenido"])
                        break
                if especies_destino:
                    break

        # If we still don't have especies_destino, try a direct search in the text
        if not especies_destino and self.text:
            # Try to find a match for "Especies de destino" and extract the following text
            species_patterns = [
                r'(?:Especies|Especie)(?:\s+de)?\s+destino[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\n\d|\Z)',
                r'\d+\.\d+\.?\s+(?:Especies|Especie)(?:\s+de)?\s+destino\s*([^\n]*(?:\n[^\n]*)*?)(?=\n\d|\Z)'
            ]
            
            for pattern in species_patterns:
                match = re.search(pattern, self.text, re.IGNORECASE)
                if match:
                    especies_destino = self._clean_species_content(match.group(1).strip())
                    break
        
        cimavet_data = self.get_cimavet_data(document_id)
        metadata = cimavet_data.get("metadata", {})
        
        # Format the registration number for url (assuming format "FT_<num-reg>_ESP")
        num_reg = format_registration_for_url(document_id)
        url = f"https://cimavet.aemps.es/cimavet/pdfs/es/ft/{num_reg}/FT_{num_reg}.pdf"
        
        # Construct the JSON output
        json_output = {
            "document_id": document_id,
            "nombre_medicamento": cimavet_data.get("nombre", ""),
            "url": url,
            "laboratorio_titular": cimavet_data.get("labtitular", ""),
            "fecha_primera_autorizacion": convert_timestamp_to_date(cimavet_data.get("primeraAutorizacion", -1)),
            "codigos_atc": cimavet_data.get("atcs", ""),
            "especies_destino": especies_destino,
            "especies_cimavet": metadata.get("especies", ""), # Added the species from metadata to compare
            "principios_activos": cimavet_data.get("pactivos", ""),
            "principios_activos_cimavet": metadata.get("principiosActivos", ""), # Added from metadata to compare
            "excipientes": cimavet_data.get("excip", ""),
            "forma_farmaceutica": cimavet_data.get("forma", {}).get("nombre", ""),
            "condiciones_dispensacion": cimavet_data.get("dispensacion", {}).get("nombre", ""),
            "condiciones_administracion": cimavet_data.get("administracion", {}).get("nombre", ""),
            "antibiotico": cimavet_data.get("antibiotico", False),
            "vias_administracion": metadata.get("viasAdministracion", ""),
            "indicaciones": metadata.get("indicaciones", ""),
            "contraindicaciones": metadata.get("contraindicaciones", ""),
            "reacciones_adversas": metadata.get("reaccionesAdversas", ""),
            "interacciones": metadata.get("interacciones", ""),
            "tiempo_espera": metadata.get("tiemposEspera", ""),
            "presentaciones": metadata.get("presentaciones", ""),
            "secciones": sections
        }
        
        return json_output
    
    def save_json(self, json_data):
        """
        Save the JSON data to the specified output path.
        
        Args:
            json_data (dict): The JSON data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.json_output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")
            return False
    
    def trim_eu_smpc(self):
        """
        Trims the EU-type SmPC document text, keeping only the first part until section 10.
        It identifies section 10 and removes everything that follows the next '-----' separator.

        Returns:
            str: Trimmed text containing only the main SmPC up to section 10.
        """
        if not self.text:
            return ""
        
        # Look for section 10 with a more comprehensive pattern
        section10_pattern = r'(?:^|\n)\*\*10(?:\.|-\.|\.-)?(?:\*\*|\s+.*?\*\*)'
        match = re.search(section10_pattern, self.text)
        
        if not match:
            return self.text  # If section 10 not found, return original text
        
        section10_pos = match.start()
        
        # Find the first '-----' after section 10
        match_separator = re.search(r'-----', self.text[section10_pos:])
        
        if match_separator:
            end_position = section10_pos + match_separator.start()
            return self.text[:end_position].strip()
        
        return self.text  # If no separator is found, return the full text
    
    def process(self):
        """
        Execute the full process: load markdown, parse content, create and save JSON.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.load_markdown():
                return False
            
            # Get document_id from filename (remove path and extension)
            document_id = os.path.basename(self.markdown_path).replace(".md", "").replace("FT_", "")
            document_id = format_registration_number(document_id)
            
            # Trim text if it is EU type
            if document_id.startswith("EU"): 
                self.text = self.trim_eu_smpc()
            
            # Parse the markdown text
            sections, especies_destino = self.parse_ficha_tecnica()
            
            # Create JSON structure
            json_data = self.create_json_structure(sections, especies_destino, document_id)
            
            # Save the JSON file
            return self.save_json(json_data)
        except Exception as e:
            print(f"Error processing markdown file: {str(e)}")
            return False


# Example
if __name__ == "__main__":
    #id_ = "FT_EU-2-24-320-001-002"#"FT_480_ESP"
    id_ = "FT_2045_ESP"  # Example ID for testing
    
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    md_path = os.path.join(script_dir, "../../data/posteriori_resources/markdown_files", f"{id_}.md")
    json_path = os.path.join(script_dir, "../../data/posteriori_resources/processed_json", f"{id_}.json")
    merged_json_path = os.path.join(script_dir, "../../data/posteriori_resources/json_data/master_merge.json")
    
    # Normalize paths
    md_path = os.path.abspath(md_path)
    json_path = os.path.abspath(json_path)
    merged_json_path = os.path.abspath(merged_json_path)
    
    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_json = json.load(f)
    
    # Initialize MarkdownParser instance
    parser = MarkdownParser(md_path, merged_json, json_path) 
    
    # Process the Markdown file
    if parser.process():
        print(f"Markdown file {md_path!r} successfully processed.")
        print(f"JSON output saved to {json_path!r}.")
    else:
        print(f"Failed to process Markdown file {md_path!r}.")


