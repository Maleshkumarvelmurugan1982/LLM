"""
Semantic search engine using sentence transformers and vector databases
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import time

from ..models.schemas import SearchResult, DocumentChunk, ProcessedQuery
from ..models.database import get_database
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)

class EmbeddingEngine:
    """Handle text embeddings using sentence transformers"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self.model = SentenceTransformer(self.settings.embedding_model)
            self.model.to(self.device)
            logger.info(f"Embedding model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings"""
        if not self.model:
            await self.initialize()
        
        try:
            # Batch encode for efficiency
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=len(texts) > 10)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    async def encode_single_text(self, text: str) -> List[float]:
        """Encode a single text into embedding"""
        embeddings = await self.encode_texts([text])
        return embeddings[0]

class SemanticSearchEngine:
    """Main semantic search engine"""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_engine = EmbeddingEngine()
        self.database = None
        
    async def initialize(self):
        """Initialize the search engine"""
        await self.embedding_engine.initialize()
        self.database = await get_database()
        logger.info("Semantic search engine initialized")
    
    async def index_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Index document chunks for search"""
        if not self.database:
            await self.initialize()
        
        try:
            # Extract text content
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = await self.embedding_engine.encode_texts(texts)
            
            # Prepare metadata
            metadatas = []
            ids = []
            for chunk in chunks:
                metadata = {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    **chunk.metadata
                }
                # Filter out None values which ChromaDB doesn't accept
                filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                metadatas.append(filtered_metadata)
                ids.append(chunk.chunk_id)
            
            # Add to vector database
            await self.database.add_documents(
                chunks=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully indexed {len(chunks)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    async def search(self, processed_query: ProcessedQuery, max_results: int = 10, 
                    document_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """Search for relevant document chunks"""
        if not self.database:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_engine.encode_single_text(processed_query.original_query)
            
            # Prepare filters
            where_filter = None
            if document_ids:
                where_filter = {"document_id": {"$in": document_ids}}
            
            # Search in vector database
            start_time = time.time()
            results = await self.database.search(
                query_embedding=query_embedding,
                n_results=max_results,
                where=where_filter
            )
            search_time = time.time() - start_time
            
            # Convert to SearchResult objects
            search_results = []
            if results and results.get('documents'):
                for i in range(len(results['documents'][0])):
                    result = SearchResult(
                        chunk_id=results['ids'][0][i],
                        document_id=results['metadatas'][0][i]['document_id'],
                        text=results['documents'][0][i],
                        similarity_score=float(results['distances'][0][i]) if results.get('distances') else 1.0,
                        metadata=results['metadatas'][0][i]
                    )
                    search_results.append(result)
            
            # Apply additional filtering based on query intent and entities
            filtered_results = await self._filter_results(search_results, processed_query)
            
            logger.info(f"Search completed in {search_time:.2f}s, found {len(filtered_results)} relevant results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _filter_results(self, results: List[SearchResult], processed_query: ProcessedQuery) -> List[SearchResult]:
        """Apply additional filtering based on query context"""
        if not results:
            return results
        
        filtered_results = []
        threshold = self.settings.similarity_threshold
        
        for result in results:
            # Apply similarity threshold
            if result.similarity_score < threshold:
                continue
            
            # Apply intent-based filtering
            if self._is_result_relevant(result, processed_query):
                filtered_results.append(result)
        
        # Sort by relevance score
        filtered_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return filtered_results
    
    def _is_result_relevant(self, result: SearchResult, processed_query: ProcessedQuery) -> bool:
        """Check if result is relevant to the query"""
        intent = processed_query.intent
        text_lower = result.text.lower()
        
        # Intent-specific relevance checks
        if intent == "insurance_claim":
            # Look for insurance-related terms
            insurance_terms = ["policy", "coverage", "claim", "benefit", "eligible", "covered"]
            if not any(term in text_lower for term in insurance_terms):
                return False
        
        elif intent == "medical_query":
            # Look for medical terms
            medical_terms = ["medical", "treatment", "procedure", "surgery", "hospital", "doctor"]
            if not any(term in text_lower for term in medical_terms):
                return False
        
        # Check for entity matches
        entities = processed_query.entities
        
        # Medical procedure matching
        if entities.get("medical_procedure"):
            procedure = entities["medical_procedure"]
            if procedure.get("procedure_name"):
                if procedure["procedure_name"].lower() not in text_lower:
                    # Allow partial matches for medical procedures
                    procedure_words = procedure["procedure_name"].lower().split()
                    if not any(word in text_lower for word in procedure_words):
                        return False
        
        # Location matching (less strict)
        if entities.get("location") and entities["location"].get("city"):
            city = entities["location"]["city"].lower()
            # Location is less critical for insurance policies
            pass
        
        return True
    
    async def get_similar_chunks(self, chunk_id: str, max_results: int = 5) -> List[SearchResult]:
        """Find chunks similar to a given chunk"""
        try:
            # Get the chunk content first
            chunk_results = await self.database.collection.get(ids=[chunk_id])
            if not chunk_results or not chunk_results['documents']:
                return []
            
            chunk_text = chunk_results['documents'][0]
            
            # Create a mock processed query for the chunk
            mock_query = ProcessedQuery(
                original_query=chunk_text,
                entities={},
                intent="general",
                structured_data={},
                keywords=[]
            )
            
            # Search for similar chunks
            results = await self.search(mock_query, max_results + 1)  # +1 to exclude self
            
            # Remove the original chunk from results
            filtered_results = [r for r in results if r.chunk_id != chunk_id]
            
            return filtered_results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to find similar chunks: {e}")
            return []
    
    async def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            await self.database.delete_document(document_id)
            logger.info(f"Deleted chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            return False
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        try:
            stats = await self.database.get_collection_stats()
            stats.update({
                "embedding_model": self.settings.embedding_model,
                "device": self.embedding_engine.device,
                "similarity_threshold": self.settings.similarity_threshold
            })
            return stats
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {}

class ReRanker:
    """Re-rank search results using cross-encoder models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.reranker = None
        
    async def initialize(self):
        """Initialize the re-ranking model"""
        try:
            # Use a cross-encoder for better re-ranking
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Re-ranking model initialized")
        except Exception as e:
            logger.warning(f"Re-ranking model not available: {e}")
            self.reranker = None
    
    async def rerank(self, query: str, results: List[SearchResult], top_k: int = None) -> List[SearchResult]:
        """Re-rank search results"""
        if not self.reranker or not results:
            return results
        
        try:
            # Prepare query-document pairs
            pairs = [(query, result.text) for result in results]
            
            # Get cross-encoder scores
            scores = self.reranker.predict(pairs)
            
            # Update similarity scores and re-sort
            for i, result in enumerate(results):
                result.similarity_score = float(scores[i])
            
            # Sort by new scores
            reranked_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)
            
            if top_k:
                reranked_results = reranked_results[:top_k]
            
            logger.info(f"Re-ranked {len(results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results
