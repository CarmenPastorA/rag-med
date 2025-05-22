

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
from shared.veterinary_utils.utils import get_dict_from_json
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
        doc = get_dict_from_json(json_path)
        
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
        
        # Process ATC codes
        atc = "Códigos ATC:\n" + "\n".join(
            f"- {item.get('codigo', '')}: {item.get('nombre', '')} (Nivel {item.get('nivel', '')})" \
                for item in doc.get("codigos_atc", [])
        )
        
        # Process species - update to use normalized names from especies_cimavet
        species_raw = doc.get('especies_destino', '')
        species_cimavet = doc.get('especies_cimavet', [])
        if species_cimavet:
            target_species = ", ".join([sp.get('nombre_normalizado', sp.get('nombre', '')) for sp in species_cimavet])
        else:
            target_species = species_raw
        
        # Process active ingredients - include both text and structured data
        active_ingredients = doc.get('principios_activos', '')
        active_ingredients_cimavet = doc.get('principios_activos_cimavet', [])
        if active_ingredients_cimavet:
            active_ingredients_detail = "Principios activos detallados:\n" + "\n".join(
                f"- {item.get('nombre', '')}: {item.get('cantidad', '')} {item.get('unidad', '')}" \
                    for item in active_ingredients_cimavet
            )
        else:
            active_ingredients_detail = ""
        
        excipients = doc.get('excipientes', '')
        pharm_form = doc.get('forma_farmaceutica', '')
        dis_conditions = doc.get('condiciones_dispensacion', '')
        admin_conditions = doc.get('condiciones_administracion', '')
        antibiotic = doc.get('antibiotico', False)
        
        # Process administration routes
        admin_routes = doc.get('vias_administracion', [])
        if admin_routes:
            admin_routes_text = "Vías de administración:\n" + "\n".join(
                f"- {route.get('nombre', '')}" for route in admin_routes
            )
        else:
            admin_routes_text = ""
        
        # Process indications by species
        indications = doc.get('indicaciones', [])
        if indications:
            # Group indications by species
            indications_by_species = {}
            for indication in indications:
                especie = indication.get('especie', {})
                especie_name = especie.get('nombre_normalizado', especie.get('nombre', ''))
                if especie_name not in indications_by_species:
                    indications_by_species[especie_name] = []
                indications_by_species[especie_name].append(indication.get('nombre', ''))
            
            # Format indications text
            indications_text = "Indicaciones:\n"
            for especie, inds in indications_by_species.items():
                indications_text += f"Indicaciones para {especie}:\n"
                for ind in inds:
                    indications_text += f"- {ind}\n"
        else:
            indications_text = ""
        
        # Process contraindications - handling multiple formats
        contraindications = doc.get('contraindicaciones', [])
        if contraindications:
            # Separate contraindications by type
            general_contras = []
            species_specific_contras = {}
            contraindicated_species = []
            
            for contra in contraindications:
                # Format 1: General contraindication without species
                if 'especie' not in contra and not contra.get('es_especie', False):
                    general_contras.append(contra.get('nombre', ''))
                # Format 2: Species-specific contraindication
                elif 'especie' in contra:
                    especie = contra.get('especie', {})
                    especie_name = especie.get('nombre_normalizado', especie.get('nombre', ''))
                    if especie_name not in species_specific_contras:
                        species_specific_contras[especie_name] = []
                    species_specific_contras[especie_name].append(contra.get('nombre', ''))
                # Format 3: Contraindication is itself a species
                elif contra.get('es_especie', False):
                    species_name = contra.get('nombre_normalizado', contra.get('nombre', ''))
                    contraindicated_species.append(species_name)
            
            # Format contraindications text
            contraindications_text = "Contraindicaciones:\n"
            
            # General contraindications
            if general_contras:
                for contra in general_contras:
                    contraindications_text += f"- {contra}\n"
            
            # Species-specific contraindications
            for especie, contras in species_specific_contras.items():
                contraindications_text += f"Contraindicaciones para {especie}:\n"
                for contra in contras:
                    contraindications_text += f"- {contra}\n"
            
            # Contraindicated species
            if contraindicated_species:
                contraindications_text += "No usar en las siguientes especies:\n"
                for species in contraindicated_species:
                    contraindications_text += f"- {species}\n"
        else:
            contraindications_text = ""
        
        # Process adverse reactions - handling multiple formats
        adverse_reactions = doc.get('reacciones_adversas', [])
        if adverse_reactions:
            # Group reactions by species
            reactions_by_species = {}
            general_reactions = []
            
            for reaction in adverse_reactions:
                # Format 1: Species-specific reaction
                if 'especie' in reaction:
                    especie = reaction.get('especie', {})
                    especie_name = especie.get('nombre_normalizado', especie.get('nombre', ''))
                    if especie_name not in reactions_by_species:
                        reactions_by_species[especie_name] = []
                    
                    frecuencia = reaction.get('frecuencia', {}).get('nombre', '')
                    nombre = reaction.get('nombre', '')
                    if frecuencia:
                        reaction_text = f"{nombre} ({frecuencia})"
                    else:
                        reaction_text = nombre
                        
                    reactions_by_species[especie_name].append(reaction_text)
                # Format 2: General reaction without species
                else:
                    frecuencia = reaction.get('frecuencia', {}).get('nombre', '')
                    nombre = reaction.get('nombre', '')
                    if frecuencia:
                        reaction_text = f"{nombre} ({frecuencia})"
                    else:
                        reaction_text = nombre
                    general_reactions.append(reaction_text)
            
            # Format reactions text
            adverse_reactions_text = "Reacciones adversas:\n"
            
            # General reactions
            if general_reactions:
                for reaction in general_reactions:
                    adverse_reactions_text += f"- {reaction}\n"
            
            # Species-specific reactions
            for especie, reactions in reactions_by_species.items():
                adverse_reactions_text += f"En {especie}:\n"
                for reaction in reactions:
                    adverse_reactions_text += f"- {reaction}\n"
        else:
            adverse_reactions_text = ""
        
        # Process interactions - handling multiple formats
        interactions = doc.get('interacciones', [])
        if interactions:
            # Group interactions by species
            interactions_by_species = {}
            general_interactions = []
            
            for interaction in interactions:
                # Format 1: Species-specific interaction
                if 'especie' in interaction:
                    especie = interaction.get('especie', {})
                    especie_name = especie.get('nombre_normalizado', especie.get('nombre', ''))
                    if especie_name not in interactions_by_species:
                        interactions_by_species[especie_name] = []
                    interactions_by_species[especie_name].append(interaction.get('nombre', ''))
                # Format 2: General interaction without species
                else:
                    general_interactions.append(interaction.get('nombre', ''))
            
            # Format interactions text
            interactions_text = "Interacciones:\n"
            
            # General interactions
            if general_interactions:
                for interaction in general_interactions:
                    interactions_text += f"- {interaction}\n"
            
            # Species-specific interactions
            for especie, inters in interactions_by_species.items():
                interactions_text += f"En {especie}:\n"
                for inter in inters:
                    interactions_text += f"- {inter}\n"
        else:
            interactions_text = ""
        
        # Process waiting time - handling multiple formats
        waiting_time = doc.get('tiempo_espera', '')
        if waiting_time:
            # Format 1: String
            if isinstance(waiting_time, str):
                waiting_time_text = f"Tiempo de espera: {waiting_time}" if waiting_time else ""
            # Format 2: List of objects
            elif isinstance(waiting_time, list):
                waiting_time_text = "Tiempo de espera:\n"
                
                # Group by species
                waiting_time_by_species = {}
                for wt in waiting_time:
                    especie = wt.get('especie', {})
                    especie_name = especie.get('nombre_normalizado', especie.get('nombre', ''))
                    if especie_name not in waiting_time_by_species:
                        waiting_time_by_species[especie_name] = []
                    
                    tejido = wt.get('tejido', {}).get('nombre', '')
                    cantidad = wt.get('cantidad', '')
                    unidad = wt.get('unidadTiempo', {}).get('nombre', '')
                    wt_text = f"{tejido}: {cantidad} {unidad}"
                    waiting_time_by_species[especie_name].append(wt_text)
                
                # Format waiting time text by species
                for especie, wts in waiting_time_by_species.items():
                    waiting_time_text += f"En {especie}:\n"
                    for wt in wts:
                        waiting_time_text += f"- {wt}\n"
            else:
                waiting_time_text = ""
        else:
            waiting_time_text = ""
        
        # Process presentations
        presentations = doc.get('presentaciones', [])
        if presentations:
            presentations_text = "Presentaciones:\n" + "\n".join(
                f"- {item.get('nombre', '')}" for item in presentations
            )
        else:
            presentations_text = ""
        
        # Complete document as a chunk - now with all new fields
        doc_text = f"Ficha técnica: {med_name}\n"
        doc_text += f"Enlace: {doc_link}\n"
        doc_text += f"Laboratorio titular: {lab_titular}\n"
        doc_text += f"Fecha de primera autorización: {fecha_autorizacion}\n"
        doc_text += f"Especies de destino: {target_species}\n"
        doc_text += f"{atc}\n"
        doc_text += f"Principios activos: {active_ingredients}\n"
        if active_ingredients_detail:
            doc_text += f"{active_ingredients_detail}\n"
        doc_text += f"Excipientes: {excipients}\n"
        doc_text += f"Forma farmacéutica: {pharm_form}\n"
        doc_text += f"Condiciones de dispensación: {dis_conditions}\n"
        doc_text += f"Condiciones de administración: {admin_conditions}\n"
        doc_text += f"Este medicamento{' no' if not antibiotic else ''} es un antibiótico.\n"
        
        # Add new fields to the document text
        if admin_routes_text:
            doc_text += f"{admin_routes_text}\n"
        if indications_text:
            doc_text += f"{indications_text}\n"
        if contraindications_text:
            doc_text += f"{contraindications_text}\n"
        if adverse_reactions_text:
            doc_text += f"{adverse_reactions_text}\n"
        if interactions_text:
            doc_text += f"{interactions_text}\n"
        if waiting_time_text:
            doc_text += f"{waiting_time_text}\n"
        if presentations_text:
            doc_text += f"{presentations_text}\n"
        
        doc_text += f"-----"
        
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
                "text": f"Nombre del Medicamento: {med_name}\n\n{section_text}\n-----",
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
                    "text": f"Nombre del Medicamento: {med_name}\n\n{subsection_text}\n-----",
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





