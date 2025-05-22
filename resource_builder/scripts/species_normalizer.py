

"""
Normalizing species names in veterinary medicine data.
"""

import json
import re
import csv
import os
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import Levenshtein
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.veterinary_utils.utils import get_dict_from_json, vprint
from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

class SpeciesNormalizer:
    """
    Class for normalizing species names in veterinary medicine data.
    Handles loading of normalization maps from JSON, finding closest matches,
    and normalizing species names in structured data.
    """
    
    def __init__(self, mapping_file=None, verbose=False):
        """
        Initialize the SpeciesNormalizer with a mapping file.
        
        Args:
            mapping_file (str, optional): Path to the JSON file containing the species mapping.
                                          If None, will try to find a default file.
            verbose (bool, optional): Increase output verbosity
        """
        self.verbose = verbose
        self.mapping = {}
        
        # Try to load mapping file
        if mapping_file is None:
            # Look for the mapping file in the 'a priori' directory
            current_dir = Path(__file__).parent
            default_path = current_dir / "../../data/priori_resources/species_mapping.json"
            
            if default_path.exists():
                mapping_file = str(default_path)
            else:
                raise FileNotFoundError(
                    "Species mapping file not found. Please provide a valid path or "
                    "place 'species_mapping.json' in the 'a priori' directory."
                )
        
        self.load_mapping(mapping_file)
    
    def load_mapping(self, mapping_file):
        """
        Load species mapping from a JSON file.
        
        Args:
            mapping_file (str): Path to the JSON file containing the species mapping
            
        Raises:
            FileNotFoundError: If the mapping file doesn't exist
            json.JSONDecodeError: If the mapping file isn't valid JSON
        """
        try:
            self.mapping = get_dict_from_json(mapping_file)
            vprint(f"Loaded {len(self.mapping)} species mappings from {mapping_file}", self.verbose)
        except FileNotFoundError:
            raise FileNotFoundError(f"Species mapping file not found: {mapping_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in mapping file: {mapping_file}")
    
    def save_mapping(self, output_file):
        """
        Save the current mapping to a JSON file.
        
        Args:
            output_file (str): Path to save the mapping to
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, indent=2, ensure_ascii=False)
        
        vprint(f"Saved {len(self.mapping)} species mappings to {output_file}", self.verbose)
    
    def add_mapping(self, original_name, normalized_category):
        """
        Add a new mapping to the normalization dictionary.
        
        Args:
            original_name (str): Original species name
            normalized_category (str): Category to normalize to
        """
        self.mapping[original_name.strip().lower()] = normalized_category
    
    def normalize_species_name(self, name):
        """
        Normalize a species name using the mapping dictionary.
        Converts to lowercase and returns the mapped standard name.
        
        Args:
            name (str): Species name to normalize
            
        Returns:
            str: Normalized species category name or None if no match found
        """
        if not name:
            return None
            
        name_lower = name.strip().lower()
        
        # Direct match
        if name_lower in self.mapping:
            return self.mapping[name_lower]
        
        # Try to find closest match using Levenshtein distance
        return self.find_closest_species_match(name_lower)
    
    def find_closest_species_match(self, name, threshold=0.85):
        """
        Find the closest matching species name using Levenshtein distance.
        
        Args:
            name (str): The species name to match
            threshold (float): Similarity threshold (0-1) to consider a match valid
            
        Returns:
            str: Normalized species category or None if no good match found
        """
        if not name:
            return None
            
        best_match = None
        best_score = 0
        
        # Calculate relative distance for all known species names
        for known_name in self.mapping.keys():
            # Skip very different length strings for efficiency
            if abs(len(known_name) - len(name)) > 5:
                continue
                
            # Calculate similarity ratio (1 = exact match, 0 = completely different)
            similarity = Levenshtein.ratio(name, known_name)
            
            if similarity > best_score:
                best_score = similarity
                best_match = known_name
        
        # Return the normalized category if we found a good match
        if best_score >= threshold and best_match:
            return self.mapping[best_match]
        
        return None
    
    def normalize_species_in_json(self, species_data):
        """
        Normalize species data in a JSON object representing species information.
        
        Args:
            species_data (dict): Dictionary or list of dictionaries containing species information
            
        Returns:
            dict: Same structure with additional normalized species name
        """
        if not species_data:
            return species_data
            
        # Handle list of species
        if isinstance(species_data, list):
            for species in species_data:
                if isinstance(species, dict) and "nombre" in species:
                    normalized_name = self.normalize_species_name(species["nombre"])
                    species["nombre_normalizado"] = normalized_name or species["nombre"]
            return species_data
        
        # Handle single species object
        elif isinstance(species_data, dict) and "nombre" in species_data:
            normalized_name = self.normalize_species_name(species_data["nombre"])
            species_data["nombre_normalizado"] = normalized_name or species_data["nombre"]
            return species_data
        
        return species_data
    
    def normalize_indications_species(self, indications_data):
        """
        Normalize species names in indications data.
        
        Args:
            indications_data (list): List of indication dictionaries
            
        Returns:
            list: Updated indications with normalized species names
        """
        if not indications_data or not isinstance(indications_data, list):
            return indications_data
            
        for indication in indications_data:
            if "especie" in indication and isinstance(indication["especie"], dict):
                especie = indication["especie"]
                if "nombre" in especie:
                    normalized_name = self.normalize_species_name(especie["nombre"])
                    especie["nombre_normalizado"] = normalized_name or especie["nombre"]
        
        return indications_data
    
    def normalize_contraindications_species(self, contraindications_data):
        """
        Normalize species names in contraindications data.
        
        Args:
            contraindications_data (list): List of contraindication dictionaries
            
        Returns:
            list: Updated contraindications with normalized species names
        """
        if not contraindications_data or not isinstance(contraindications_data, list):
            return contraindications_data
            
        for contraindication in contraindications_data:
            # Case 1: When contraindication has an "especie" field (standard format)
            if "especie" in contraindication and isinstance(contraindication["especie"], dict):
                especie = contraindication["especie"]
                if "nombre" in especie:
                    normalized_name = self.normalize_species_name(especie["nombre"])
                    especie["nombre_normalizado"] = normalized_name or especie["nombre"]
                    
            # Case 2: When the contraindication itself might be a species name (e.g. FT_1053_ESP format)
            elif "nombre" in contraindication:
                if contraindication["nombre"] is None: # remove (e.g. FT_3740_ESP)
                    contraindications_data.remove(contraindication)
                    continue
                nombre = contraindication["nombre"].strip()
                # Check if this contraindication name might be a species
                # First check direct match in mapping
                nombre_lower = nombre.lower()
                if nombre_lower in self.mapping:
                    # This is a species name - add normalized form
                    contraindication["es_especie"] = True
                    contraindication["nombre_normalizado"] = self.mapping[nombre_lower]
                else:
                    # Try to find a close match
                    normalized_name = self.find_closest_species_match(nombre)
                    if normalized_name:
                        # If we found a close match, it's likely a species
                        contraindication["es_especie"] = True
                        contraindication["nombre_normalizado"] = normalized_name
                    # If no match found, assume it's not a species name
        
        return contraindications_data
    
    def normalize_adverse_reactions_species(self, adverse_reactions_data):
        """
        Normalize species names in adverse reactions data.
        
        Args:
            adverse_reactions_data (list): List of adverse reaction dictionaries
            
        Returns:
            list: Updated adverse reactions with normalized species names
        """
        if not adverse_reactions_data or not isinstance(adverse_reactions_data, list):
            return adverse_reactions_data
            
        for reaction in adverse_reactions_data:
            if "especie" in reaction and isinstance(reaction["especie"], dict):
                especie = reaction["especie"]
                if "nombre" in especie:
                    normalized_name = self.normalize_species_name(especie["nombre"])
                    especie["nombre_normalizado"] = normalized_name or especie["nombre"]
        
        return adverse_reactions_data
    
    def normalize_interactions_species(self, interactions_data):
        """
        Normalize species names in interactions data.
        
        Args:
            interactions_data (list): List of interaction dictionaries
            
        Returns:
            list: Updated interactions with normalized species names
        """
        if not interactions_data or not isinstance(interactions_data, list):
            return interactions_data
            
        for interaction in interactions_data:
            if "especie" in interaction and isinstance(interaction["especie"], dict):
                especie = interaction["especie"]
                if "nombre" in especie:
                    normalized_name = self.normalize_species_name(especie["nombre"])
                    especie["nombre_normalizado"] = normalized_name or especie["nombre"]
        
        return interactions_data
    
    def normalize_withdrawal_period_species(self, withdrawal_periods_data):
        """
        Normalize species names in withdrawal periods data (tiempo_espera).
        
        Args:
            withdrawal_periods_data (list): List of withdrawal period dictionaries
            
        Returns:
            list: Updated withdrawal periods with normalized species names
        """
        if not withdrawal_periods_data or not isinstance(withdrawal_periods_data, list):
            return withdrawal_periods_data
            
        for period in withdrawal_periods_data:
            if "especie" in period and isinstance(period["especie"], dict):
                especie = period["especie"]
                if "nombre" in especie:
                    normalized_name = self.normalize_species_name(especie["nombre"])
                    especie["nombre_normalizado"] = normalized_name or especie["nombre"]
        
        return withdrawal_periods_data
    
    def get_all_unique_species_names(self, json_path):
        """
        Extract all unique species names from the dataset.
        Useful for identifying which species need to be added to the normalization map.
        
        Args:
            json_path (str): Path to the JSON file with medicine data
            
        Returns:
            set: Set of unique species names found in the data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        unique_names = set()
        
        # Extract species names from each medicine entry
        for entry in data.values():
            especies = entry.get("metadata", {}).get("especies", [])
            for especie in especies:
                if "nombre" in especie and especie["nombre"]:
                    unique_names.add(especie["nombre"].strip().lower())
        
        return unique_names
    
    def export_unmapped_species(self, json_path, output_csv):
        """
        Find species names in the dataset that aren't in our normalization map
        and export them to a CSV file.
        
        Args:
            json_path (str): Path to the JSON file with medicine data
            output_csv (str): Path to save the CSV with unmapped species
        """
        # Get all unique species names from the dataset
        unique_species = self.get_all_unique_species_names(json_path)
        
        # Find unmapped species
        unmapped = []
        for species_name in unique_species:
            if species_name not in self.mapping:
                # Try to find a close match
                closest = self.find_closest_species_match(species_name)
                unmapped.append({
                    "original_name": species_name,
                    "closest_match": "None" if closest is None else closest,
                    "suggested_category": ""  # To be filled manually
                })
        
        # Export to CSV
        if unmapped:
            with open(output_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["original_name", "closest_match", "suggested_category"])
                writer.writeheader()
                writer.writerows(unmapped)
            vprint(f"Exported {len(unmapped)} unmapped species to {output_csv}", self.verbose)
        else:
            vprint("All species names are already mapped!", self.verbose)
    
    def import_mappings_from_csv(self, csv_path):
        """
        Import additional mappings from a CSV file.
        Format expected: original_name,closest_match,suggested_category
        
        Args:
            csv_path (str): Path to the CSV file with mappings
            
        Returns:
            int: Number of mappings added
        """
        if not os.path.exists(csv_path):
            vprint(f"CSV file not found: {csv_path}", self.verbose)
            return 0
            
        count = 0
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'original_name' in row and 'suggested_category' in row:
                    # Only add if suggested_category is filled
                    if row['suggested_category'].strip():
                        self.add_mapping(row['original_name'], row['suggested_category'])
                        count += 1
        
        vprint(f"Added {count} new mappings from {csv_path}", self.verbose)
        return count
    
    def group_species_ids(self, json_path, output_json):
        """
        Reads a JSON file containing medicine metadata and groups their registration numbers
        based on normalized species categories.
    
        Args:
            json_path (str): Path to the master_merge.json file
            output_json (str): Path to the output JSON file with species groups
        """
        # Load input data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        grouped = defaultdict(list)  # Dictionary to store species groupings
        missing_normalizations = set()  # Track species that couldn't be normalized
    
        # Iterate over all medicines
        for reg_id, entry in tqdm(data.items(), desc="Processing medicines", disable=not self.verbose):
            especies = entry.get("metadata", {}).get("especies", [])
            for especie in especies:
                nombre = especie.get("nombre", "").strip()
                norm = self.normalize_species_name(nombre)
                
                if norm:
                    grouped[norm].append(reg_id)
                else:
                    missing_normalizations.add(nombre)
    
        # Sort registration numbers and remove duplicates
        grouped_sorted = {
            k: sorted(set(v), key=lambda x: (int(x.split()[0]) if x.split()[0].isdigit() else 999999, x))
            for k, v in grouped.items()
        }
    
        # Save grouped data as JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(grouped_sorted, f, indent=2, ensure_ascii=False)
    
        # Print missing normalizations
        if missing_normalizations:
            vprint(f"\nWarning: Could not normalize {len(missing_normalizations)} species names:", self.verbose)
            for name in sorted(missing_normalizations):
                vprint(f"  - {name}", self.verbose)
    
        vprint(f"\nJSON File saved: {output_json}", self.verbose)


# Script entry point
if __name__ == "__main__":
    
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    species_mapping_path = os.path.join(script_dir, "../../data/priori_resources/species_mapping.json")
    input_path = os.path.join(script_dir, "../../data/posteriori_resources/json_data/master_merge.json")
    output_json_path = os.path.join(script_dir, "../../data/posteriori_resources/json_data/species_groups_registration_numbers.json")
    unmapped_species_csv = os.path.join(script_dir, "../../data/posteriori_resources/unmapped_species.csv")
    # Normalize paths
    species_mapping_path = os.path.abspath(species_mapping_path)
    input_path = os.path.abspath(input_path)
    output_json_path = os.path.abspath(output_json_path)
    unmapped_species_csv = os.path.abspath(unmapped_species_csv)
    # Create a species normalizer with the mapping file
    try:
        normalizer = SpeciesNormalizer(species_mapping_path, verbose=True)
        
        # Export unmapped species for manual review
        normalizer.export_unmapped_species(input_path, unmapped_species_csv)
        
        # After manual review, the assignments can be reimported.
        # normalizer.import_mappings_from_csv(unmapped_species_csv)
        # normalizer.save_mapping("data/posteriori_resources/updated_species_mapping.json")
        
        # Group species IDs
        normalizer.group_species_ids(input_path, output_json_path)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("You need to provide a species mapping file. Example usage:")
        print("normalizer = SpeciesNormalizer('path/to/species_mapping.json')")

