"""
Test cases for the LLM Document Query System
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.query_engine import QueryProcessor
from src.semantic_search import SemanticSearchEngine
from src.decision_engine import DecisionEngine
from src.document_processor import DocumentManager
from src.models.schemas import DocumentType, QueryRequest

class TestQueryProcessor:
    """Test the query processing functionality"""
    
    @pytest.fixture
    async def query_processor(self):
        processor = QueryProcessor()
        await processor.initialize()
        return processor
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, query_processor):
        """Test entity extraction from sample query"""
        query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
        
        processed_query = await query_processor.process_query(query)
        
        # Check if entities were extracted
        entities = processed_query.entities
        assert entities is not None
        
        # Check person entity
        assert entities.get("person") is not None
        person = entities["person"]
        assert person.get("age") == 46
        assert person.get("gender") == "male"
        
        # Check location entity
        assert entities.get("location") is not None
        location = entities["location"]
        assert location.get("city").lower() == "pune"
        
        # Check medical procedure
        assert entities.get("medical_procedure") is not None
        procedure = entities["medical_procedure"]
        assert "knee" in procedure.get("body_part", "").lower()
        
        # Check policy
        assert entities.get("policy") is not None
        policy = entities["policy"]
        assert "3" in policy.get("policy_age", "")
        
        # Check intent classification
        assert processed_query.intent == "insurance_claim"

class TestDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def document_manager(self):
        return DocumentManager()
    
    @pytest.mark.asyncio
    async def test_text_document_processing(self, document_manager):
        """Test processing of text documents"""
        # Use sample document
        sample_doc_path = Path(__file__).parent.parent / "data" / "documents" / "sample_insurance_policy.txt"
        
        if sample_doc_path.exists():
            document_id, chunks = await document_manager.process_document(
                file_path=str(sample_doc_path),
                name="Sample Insurance Policy",
                document_type=DocumentType.TXT
            )
            
            assert document_id is not None
            assert len(chunks) > 0
            
            # Check chunk content
            first_chunk = chunks[0]
            assert first_chunk.document_id == document_id
            assert len(first_chunk.text) > 0
            assert first_chunk.chunk_id is not None

class TestSemanticSearch:
    """Test semantic search functionality"""
    
    @pytest.fixture
    async def search_engine(self):
        engine = SemanticSearchEngine()
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_search_initialization(self, search_engine):
        """Test search engine initialization"""
        assert search_engine.embedding_engine.model is not None
        assert search_engine.database is not None

class TestDecisionEngine:
    """Test decision making functionality"""
    
    @pytest.fixture
    async def decision_engine(self):
        engine = DecisionEngine()
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_rule_engine_initialization(self, decision_engine):
        """Test decision engine initialization"""
        assert decision_engine.rule_engine is not None
        assert decision_engine.ml_engine is not None

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self):
        """Test complete query processing pipeline"""
        # Initialize components
        query_processor = QueryProcessor()
        await query_processor.initialize()
        
        search_engine = SemanticSearchEngine()
        await search_engine.initialize()
        
        decision_engine = DecisionEngine()
        await decision_engine.initialize()
        
        # Process a test query
        test_query = "46M, knee surgery, Pune, 3-month policy"
        
        # Step 1: Process query
        processed_query = await query_processor.process_query(test_query)
        assert processed_query.intent == "insurance_claim"
        
        # Step 2: Search (will be empty without indexed documents)
        search_results = await search_engine.search(processed_query, max_results=10)
        
        # Step 3: Make decision
        response = await decision_engine.make_decision(processed_query, search_results)
        
        assert response.query_id is not None
        assert response.decision is not None
        assert response.justification is not None
        assert response.confidence >= 0.0
        assert response.processing_time > 0.0

# Sample test queries for manual testing
SAMPLE_QUERIES = [
    "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
    "Female patient, 35 years old, hip replacement in Mumbai, 6-month policy",
    "Heart surgery required for 60-year-old man in Delhi, 2-year-old policy",
    "Is cosmetic surgery covered under my health insurance?",
    "What is the waiting period for orthopedic procedures?",
    "Claim for emergency treatment in Bangalore hospital",
    "Pre-existing condition coverage for diabetes treatment"
]

if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        print("Running simple integration test...")
        
        query_processor = QueryProcessor()
        await query_processor.initialize()
        
        test_query = "46M, knee surgery, Pune, 3-month policy"
        processed_query = await query_processor.process_query(test_query)
        
        print(f"Query: {test_query}")
        print(f"Intent: {processed_query.intent}")
        print(f"Entities: {processed_query.entities}")
        print(f"Keywords: {processed_query.keywords}")
        
        print("Test completed successfully!")
    
    asyncio.run(simple_test())
