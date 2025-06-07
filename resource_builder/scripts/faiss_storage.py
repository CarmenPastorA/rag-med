

"""
FAISS storage module for veterinary medicine RAG system.
Uses two-stage retrieval, HierarchicalFaissStorage, 
or only one-stage retrieval, FaissStorage.
Handles document processing, embedding generation, and FAISS index management.

Currently, to use FAISS with Numpy, 
the Numpy version must be lower than 2: pip install 'numpy<2.0.0'
https://github.com/facebookresearch/faiss/issues/3526
"""

import json
import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from tqdm import tqdm
import sys

# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared import dunder_info
from shared.veterinary_utils.utils import get_dict_from_json, regnum2filename
from shared.veterinary_utils.embedding_model import EmbeddingModel
dunder_info.inject_dunder(__name__) # injects the variables

class HierarchicalFaissStorage:
    """
    Hierarchical FAISS storage system for veterinary medicine SmPCs.
    Uses two-stage retrieval: first retrieves relevant documents based on essential info,
    then retrieves specific chunks from those documents.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, embedding_dim: int = 384, 
                 separator: str = "@", store_full_documents: bool = True, verbose: bool = True):
        """
        Initialize the hierarchical FAISS storage system.
        
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
        
        # First stage: Essential info index for document-level retrieval
        self.essential_index = faiss.IndexFlatIP(embedding_dim)
        self.essential_id_to_info = {}  # Maps FAISS ID to document info
        self.essential_cache = {}  # Cache of essential info texts
        
        # Second stage: Detailed chunks index for chunk-level retrieval
        self.chunks_index = faiss.IndexFlatIP(embedding_dim)
        self.chunks_id_to_info = {}  # Maps FAISS ID to chunk info
        
        # Document and chunk caches
        self.document_cache = {}  # Cache of complete documents (optional)
        self.chunks_cache = {}  # Cache of chunks with their text and metadata
    
    def load_essential_info(self, essential_info_path: str, document_id: str) -> str:
        """
        Load essential information for a document from the essential_info directory.
        
        Args:
            essential_info_path: Path to the essential_info directory
            document_id: ID of the document
            
        Returns:
            Essential information text or empty string if not found
        """
        essential_file_path = os.path.join(essential_info_path, regnum2filename(document_id))
        
        if os.path.exists(essential_file_path):
            try:
                with open(essential_file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read essential info for {document_id}: {e}")
                return ""
        else:
            if self.verbose:
                print(f"Warning: Essential info file not found for document {document_id}")
            return ""
    
    def process_document_essential_info(self, json_path: str, essential_info_dir: str) -> Dict[str, Any]:
        """
        Process essential information for a document and prepare it for the first-stage index.
        
        Args:
            json_path: Path to the JSON file of the SmPC
            essential_info_dir: Path to the directory containing essential info files
            
        Returns:
            Dictionary with essential info and metadata
        """
        doc = get_dict_from_json(json_path)
        doc_id = doc.get('document_id', os.path.basename(json_path).split('.')[0])
        
        # Load essential information from file
        essential_text = self.load_essential_info(essential_info_dir, doc_id)
        
        # If no essential info file, create basic essential info from JSON
        if not essential_text:
            med_name = doc.get('nombre_medicamento', '')
            lab_titular = doc.get('laboratorio_titular', '')
            fecha_autorizacion = doc.get('fecha_primera_autorizacion', '')
            
            # Process species
            species_cimavet = doc.get('especies_cimavet', [])
            if species_cimavet:
                target_species = ", ".join([sp.get('nombre_normalizado', sp.get('nombre', '')) for sp in species_cimavet])
            else:
                target_species = doc.get('especies_destino', '')
            
            # Process active ingredients
            active_ingredients = doc.get('principios_activos', '')
            
            essential_text = f"""ID: {doc_id}
Medicamento: {med_name}
Laboratorio: {lab_titular}
Autorización: {fecha_autorizacion}
Especies: {target_species}
Principios activos: {active_ingredients}"""
        
        return {
            "document_id": doc_id,
            "essential_text": essential_text,
            "metadata": {
                "document_id": doc_id,
                "json_path": json_path,
                "nombre_medicamento": doc.get('nombre_medicamento', ''),
                "laboratorio_titular": doc.get('laboratorio_titular', ''),
                "url": doc.get('url', '')
            }
        }
    
    def process_document_chunks(self, json_path: str) -> Dict[str, Dict]:
        """
        Process a JSON document and extract fragments for embeddings (second stage).
        This is the same as the original process_document method.
        
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
    
    def add_document(self, json_path: str, essential_info_dir: str) -> None:
        """
        Process a document and add it to both the essential info index and chunks index.
        
        Args:
            json_path: Path to the JSON file of the technical data sheet
            essential_info_dir: Path to the directory containing essential info files
        """
        # Process essential information for first stage
        essential_info = self.process_document_essential_info(json_path, essential_info_dir)
        
        if essential_info["essential_text"]:
            # Generate embedding for essential info
            essential_embedding = self.embedding_model.get_embeddings(
                [essential_info["essential_text"]], 
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Add to essential index
            essential_faiss_id = self.essential_index.ntotal
            self.essential_index.add(essential_embedding)
            
            # Store mapping for essential info
            self.essential_id_to_info[essential_faiss_id] = {
                "document_id": essential_info["document_id"],
                "essential_text": essential_info["essential_text"],
                "metadata": essential_info["metadata"]
            }
            
            # Cache essential info
            self.essential_cache[essential_info["document_id"]] = essential_info["essential_text"]
        
        # Process document chunks for second stage
        chunks = self.process_document_chunks(json_path)
        
        # Prepare texts and their IDs for chunks
        texts = []
        chunk_ids = []
        
        for chunk_id, chunk_data in chunks.items():
            texts.append(chunk_data["text"])
            chunk_ids.append(chunk_id)
        
        # Generate embeddings for chunks in batches
        embeddings = self.embedding_model.get_embeddings(
            texts, 
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Get current number of vectors in the chunks index
        start_idx = self.chunks_index.ntotal
        
        # Add embeddings to the chunks index
        self.chunks_index.add(embeddings)
        
        # Save mapping of IDs to information for chunks
        for i, chunk_id in enumerate(chunk_ids):
            faiss_id = start_idx + i
            self.chunks_id_to_info[faiss_id] = {
                "chunk_id": chunk_id,
                "metadata": chunks[chunk_id]["metadata"]
            }
    
    def add_documents_from_directory(self, json_directory_path: str, essential_info_directory: str) -> None:
        """
        Process all JSON files in a directory and add them to both indices.
        
        Args:
            json_directory_path: Path to the directory with JSON files
            essential_info_directory: Path to the directory with essential info files
        """
        files_in_dir = os.listdir(json_directory_path)
        json_files = [f for f in files_in_dir if f.endswith('.json')]
        
        for filename in tqdm(json_files, desc="Storing in hierarchical FAISS", disable=not self.verbose):
            json_path = os.path.join(json_directory_path, filename)
            self.add_document(json_path, essential_info_directory)
    
    # ~ def get_relevant_document_ids(self, query: str, top_k: int = 10) -> List[str]:
        # ~ """
        # ~ First stage: Retrieve relevant document IDs based on essential information.
        
        # ~ Args:
            # ~ query: Search query
            # ~ top_k: Number of top documents to retrieve
            
        # ~ Returns:
            # ~ List of relevant document IDs
        # ~ """
        # ~ if self.essential_index.ntotal == 0:
            # ~ return []
        
        # ~ # Generate embedding for the query
        # ~ query_embedding = self.embedding_model.get_embeddings(
            # ~ [query], 
            # ~ batch_size=1,
            # ~ convert_to_numpy=True,
            # ~ normalize_embeddings=True
        # ~ )
        
        # ~ # Search in essential info index
        # ~ scores, indices = self.essential_index.search(query_embedding, min(top_k, self.essential_index.ntotal))
        
        # ~ # Extract document IDs
        # ~ relevant_doc_ids = []
        # ~ for idx in indices[0]:
            # ~ if idx != -1 and idx in self.essential_id_to_info:
                # ~ doc_id = self.essential_id_to_info[idx]["document_id"]
                # ~ relevant_doc_ids.append(doc_id)
        
        # ~ return relevant_doc_ids
    
    # ~ def get_relevant_chunks_from_documents(self, query: str, document_ids: List[str], top_k: int = 5) -> List[Dict]:
        # ~ """
        # ~ Second stage: Retrieve relevant chunks from specified documents.
        
        # ~ Args:
            # ~ query: Search query
            # ~ document_ids: List of document IDs to search within
            # ~ top_k: Number of top chunks to retrieve
            
        # ~ Returns:
            # ~ List of relevant chunks with their information
        # ~ """
        # ~ if self.chunks_index.ntotal == 0 or not document_ids:
            # ~ return []
        
        # ~ # Generate embedding for the query
        # ~ query_embedding = self.embedding_model.get_embeddings(
            # ~ [query], 
            # ~ batch_size=1,
            # ~ convert_to_numpy=True,
            # ~ normalize_embeddings=True
        # ~ )
        
        # ~ # Search in chunks index
        # ~ # We'll search more broadly and then filter by document IDs
        # ~ search_k = min(top_k * len(document_ids) * 2, self.chunks_index.ntotal)
        # ~ scores, indices = self.chunks_index.search(query_embedding, search_k)
        
        # ~ # Filter results by document IDs and collect relevant chunks
        # ~ relevant_chunks = []
        # ~ document_ids_set = set(document_ids)
        
        # ~ for i, idx in enumerate(indices[0]):
            # ~ if idx != -1 and idx in self.chunks_id_to_info:
                # ~ chunk_info = self.chunks_id_to_info[idx]
                # ~ chunk_doc_id = chunk_info["metadata"]["document_id"]
                
                # ~ if chunk_doc_id in document_ids_set:
                    # ~ chunk_id = chunk_info["chunk_id"]
                    # ~ if chunk_id in self.chunks_cache:
                        # ~ relevant_chunks.append({
                            # ~ "chunk_id": chunk_id,
                            # ~ "text": self.chunks_cache[chunk_id]["text"],
                            # ~ "metadata": chunk_info["metadata"],
                            # ~ "score": float(scores[0][i])
                        # ~ })
                        
                        # ~ if len(relevant_chunks) >= top_k:
                            # ~ break
        
        # ~ return relevant_chunks
    
    # ~ def hierarchical_search(self, query: str, top_documents: int = 10, top_chunks: int = 5) -> List[Dict]:
        # ~ """
        # ~ Perform hierarchical search: first retrieve relevant documents, then relevant chunks.
        
        # ~ Args:
            # ~ query: Search query
            # ~ top_documents: Number of top documents to retrieve in first stage
            # ~ top_chunks: Number of top chunks to retrieve in second stage
            
        # ~ Returns:
            # ~ List of relevant chunks with their information
        # ~ """
        # ~ # First stage: Get relevant document IDs
        # ~ relevant_doc_ids = self.get_relevant_document_ids(query, top_documents)
        
        # ~ if not relevant_doc_ids:
            # ~ return []
        
        # ~ # Second stage: Get relevant chunks from those documents
        # ~ relevant_chunks = self.get_relevant_chunks_from_documents(query, relevant_doc_ids, top_chunks)
        
        # ~ return relevant_chunks
    
    def clear_document_cache(self) -> None:
        """
        Clear the document cache to free memory.
        Only call this after all documents have been processed if you don't need rich context.
        """
        self.document_cache = {}
        self.store_full_documents = False
    
    def save_indices(self, essential_index_path: str, essential_mapping_path: str, essential_cache_path: str,
                    chunks_index_path: str, chunks_mapping_path: str, chunks_cache_path: str) -> None:
        """
        Save both FAISS indices and their associated data.
        
        Args:
            essential_index_path: Path to save the essential info index
            essential_mapping_path: Path to save the essential info mapping
            essential_cache_path: Path to save the essential info cache
            chunks_index_path: Path to save the chunks index
            chunks_mapping_path: Path to save the chunks mapping
            chunks_cache_path: Path to save the chunks cache
        """
        # Create directories if they don't exist
        for path in [essential_index_path, chunks_index_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save essential info index and data
        faiss.write_index(self.essential_index, essential_index_path)
        
        with open(essential_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(
                {str(k): v for k, v in self.essential_id_to_info.items()}, 
                f, indent=4, ensure_ascii=False
            )
        
        with open(essential_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.essential_cache, f, indent=4, ensure_ascii=False)
        
        # Save chunks index and data
        faiss.write_index(self.chunks_index, chunks_index_path)
        
        with open(chunks_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(
                {str(k): v for k, v in self.chunks_id_to_info.items()}, 
                f, indent=4, ensure_ascii=False
            )
        
        with open(chunks_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_cache, f, indent=4, ensure_ascii=False)


# Backward compatibility class - wrapper around the original FaissStorage for single-stage retrieval
class FaissStorage:
    """
    Original FAISS storage system for veterinary medicine SmPCs.
    Maintained for backward compatibility.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, embedding_dim: int = 384, 
                 separator: str = "@", store_full_documents: bool = True, verbose: bool = True):
        """
        Initialize the FAISS storage system.
        
        Args:
            embedding_model: Model to generate embeddings with get_embeddings method
            embedding_dim: Dimension of the generated embeddings
            separator: Separator used in chunk identifiers
            store_full_documents: Whether to store full documents in cache
            verbose: Increase output verbosity
        """
        # Use the hierarchical storage but only use the chunks index
        self.hierarchical_storage = HierarchicalFaissStorage(
            embedding_model, embedding_dim, separator, store_full_documents, verbose
        )
        
        # Expose the chunks index as the main index for backward compatibility
        self.index = self.hierarchical_storage.chunks_index
        self.id_to_embedding_info = self.hierarchical_storage.chunks_id_to_info
        self.document_cache = self.hierarchical_storage.document_cache
        self.chunks_cache = self.hierarchical_storage.chunks_cache
    
    def process_document(self, json_path: str) -> Dict[str, Dict]:
        """Process a document and extract fragments for embeddings."""
        return self.hierarchical_storage.process_document_chunks(json_path)
    
    def add_document(self, json_path: str) -> None:
        """Process a document, generate embeddings, and add them to the FAISS index."""
        # Process document chunks only (no essential info processing)
        chunks = self.hierarchical_storage.process_document_chunks(json_path)
        
        # Prepare texts and their IDs
        texts = []
        chunk_ids = []
        
        for chunk_id, chunk_data in chunks.items():
            texts.append(chunk_data["text"])
            chunk_ids.append(chunk_id)
        
        # Generate embeddings in batches
        embeddings = self.hierarchical_storage.embedding_model.get_embeddings(
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
        """Process all JSON files in a directory and add them to the index."""
        files_in_dir = os.listdir(directory_path)
        for filename in tqdm(files_in_dir, desc="Storing in FAISS", disable=not self.hierarchical_storage.verbose):
            if filename.endswith('.json'):
                json_path = os.path.join(directory_path, filename)
                self.add_document(json_path)
    
    def clear_document_cache(self) -> None:
        """Clear the document cache to free memory."""
        self.hierarchical_storage.clear_document_cache()
    
    def save_index(self, index_path: str, mapping_path: str, chunks_path: str = None) -> None:
        """Save the FAISS index and mapping information."""
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
    """Example usage of the hierarchical FAISS storage system"""
    
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    sim_path = os.path.join(script_dir, "../../models/similarity_model")
    essential_info_dir = os.path.join(script_dir, "../../data/posteriori_resources/essential_info")
    jsons_path = os.path.join(script_dir, "../../data/posteriori_resources/processed_json")
    
    # Paths for hierarchical storage
    essential_index_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/essential_index.faiss")
    essential_mapping_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/essential_mapping.json")
    essential_cache_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/essential_cache.json")
    
    chunks_index_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/chunks_index.faiss")
    chunks_mapping_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/chunks_mapping.json")
    chunks_cache_path = os.path.join(script_dir, "../../data/posteriori_resources/faiss_stuff/chunks_cache.json")
    
    # Normalize paths
    sim_path = os.path.abspath(sim_path)
    essential_info_dir = os.path.abspath(essential_info_dir)
    jsons_path = os.path.abspath(jsons_path)
    
    # Create embedding model
    embedding_model = EmbeddingModel(
        sim_path,
        "cuda", #"cpu",
        512
    )
    
    # Create hierarchical FAISS storage
    hierarchical_storage = HierarchicalFaissStorage(
        embedding_model,
        embedding_dim=embedding_model.get_word_embedding_dimension()
    )
    
    # Add documents to both indices
    hierarchical_storage.add_documents_from_directory(jsons_path, essential_info_dir)
    
    # Save both indices
    hierarchical_storage.save_indices(
        essential_index_path, essential_mapping_path, essential_cache_path,
        chunks_index_path, chunks_mapping_path, chunks_cache_path
    )
    
    # Clear document cache to save memory (optional)
    # hierarchical_storage.clear_document_cache()


def example_usage_backward_compatibility():
    """Example usage showing backward compatibility with original FaissStorage"""
    
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    
    # Build paths based on the script directory
    sim_path = os.path.join(script_dir, "../../models/multilingual-e5-large-local")
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
    
    # Create FAISS storage (original interface)
    faiss_storage = FaissStorage(
        embedding_model,
        embedding_dim=embedding_model.get_word_embedding_dimension()
    )
    
    # Add documents (same as before)
    faiss_storage.add_documents_from_directory(jsons_path)
    
    # Save index and chunks cache (same as before)
    faiss_storage.save_index(
        faiss_index_path, 
        mapping_path,
        chunks_path
    )
    
    # Clear document cache to save memory (optional)
    # faiss_storage.clear_document_cache()


if __name__ == "__main__":
    print("Running hierarchical FAISS storage example...")
    example_usage()
    
    print("\n" + "="*50 + "\n")
    
    print("Running backward compatibility example...")
    example_usage_backward_compatibility()
