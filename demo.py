"""
Demo script to showcase the LLM Document Query System functionality
"""

import asyncio
import json
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentManager
from src.query_engine import QueryProcessor
from src.semantic_search import SemanticSearchEngine
from src.decision_engine import DecisionEngine
from src.models.schemas import DocumentType

class SystemDemo:
    """Demonstration of the complete system functionality"""
    
    def __init__(self):
        self.document_manager = DocumentManager()
        self.query_processor = QueryProcessor()
        self.search_engine = SemanticSearchEngine()
        self.decision_engine = DecisionEngine()
        
    async def initialize(self):
        """Initialize all system components"""
        print("🔧 Initializing LLM Document Query System...")
        
        try:
            await self.query_processor.initialize()
            print("✅ Query processor initialized")
            
            await self.search_engine.initialize()
            print("✅ Semantic search engine initialized")
            
            await self.decision_engine.initialize()
            print("✅ Decision engine initialized")
            
            print("🚀 System initialization complete!\n")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    async def process_sample_documents(self):
        """Process sample documents for demonstration"""
        print("📄 Processing sample documents...")
        
        # Sample documents
        sample_docs = [
            {
                "path": "data/documents/sample_insurance_policy.txt",
                "name": "Health Insurance Policy",
                "type": DocumentType.TXT
            },
            {
                "path": "data/documents/claims_processing_manual.txt", 
                "name": "Claims Processing Manual",
                "type": DocumentType.TXT
            }
        ]
        
        total_chunks = 0
        
        for doc_info in sample_docs:
            doc_path = Path(doc_info["path"])
            
            if doc_path.exists():
                try:
                    # Process document
                    document_id, chunks = await self.document_manager.process_document(
                        file_path=str(doc_path),
                        name=doc_info["name"],
                        document_type=doc_info["type"]
                    )
                    
                    # Index for search
                    await self.search_engine.index_documents(chunks)
                    
                    total_chunks += len(chunks)
                    print(f"✅ Processed '{doc_info['name']}': {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"❌ Failed to process '{doc_info['name']}': {e}")
            else:
                print(f"⚠️  Document not found: {doc_path}")
        
        print(f"📊 Total chunks indexed: {total_chunks}\n")
        return total_chunks > 0
    
    async def demonstrate_query_processing(self):
        """Demonstrate query processing with sample queries"""
        print("🔍 Demonstrating query processing...\n")
        
        # Sample queries
        sample_queries = [
            {
                "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
                "description": "Insurance claim evaluation"
            },
            {
                "query": "Is hip surgery covered for a 68-year-old patient?",
                "description": "Coverage inquiry"
            },
            {
                "query": "What is the waiting period for cardiac procedures?",
                "description": "Policy information request"
            },
            {
                "query": "Heart surgery claim for emergency treatment in Mumbai",
                "description": "Emergency claim processing"
            }
        ]
        
        for i, query_info in enumerate(sample_queries, 1):
            print(f"{'='*60}")
            print(f"QUERY {i}: {query_info['description']}")
            print(f"{'='*60}")
            print(f"📝 Query: \"{query_info['query']}\"\n")
            
            try:
                start_time = time.time()
                
                # Step 1: Process query
                print("🔧 Step 1: Processing natural language query...")
                processed_query = await self.query_processor.process_query(query_info["query"])
                
                print(f"   Intent: {processed_query.intent}")
                print(f"   Keywords: {', '.join(processed_query.keywords[:5])}")
                
                # Print extracted entities
                if processed_query.entities:
                    print("   Extracted entities:")
                    for entity_type, entity_data in processed_query.entities.items():
                        if entity_data:
                            print(f"     - {entity_type}: {entity_data}")
                
                # Step 2: Search for relevant information
                print("\n🔍 Step 2: Searching for relevant information...")
                search_results = await self.search_engine.search(
                    processed_query=processed_query,
                    max_results=5
                )
                
                print(f"   Found {len(search_results)} relevant document chunks")
                if search_results:
                    print("   Top matches:")
                    for j, result in enumerate(search_results[:3], 1):
                        print(f"     {j}. Score: {result.similarity_score:.3f} - {result.text[:100]}...")
                
                # Step 3: Make decision
                print("\n⚖️  Step 3: Making decision based on retrieved information...")
                response = await self.decision_engine.make_decision(
                    processed_query=processed_query,
                    search_results=search_results
                )
                
                processing_time = time.time() - start_time
                
                # Display results
                print("\n📋 RESULTS:")
                print(f"   Decision: {response.decision}")
                if response.amount:
                    print(f"   Amount: ₹{response.amount:,.2f}")
                print(f"   Confidence: {response.confidence:.2%}")
                print(f"   Justification: {response.justification}")
                print(f"   Processing Time: {processing_time:.2f} seconds")
                
                if response.supporting_clauses:
                    print(f"\n📖 Supporting Evidence ({len(response.supporting_clauses)} clauses):")
                    for clause in response.supporting_clauses:
                        print(f"   - Document: {clause.document_name}")
                        print(f"     Relevance: {clause.relevance_score:.3f}")
                        print(f"     Text: {clause.text[:150]}...")
                        if clause.page_number:
                            print(f"     Page: {clause.page_number}")
                        print()
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
            print(f"\n{'='*60}\n")
            
            # Add delay between queries for demonstration
            await asyncio.sleep(1)
    
    async def show_system_stats(self):
        """Display system statistics"""
        print("📊 System Statistics:")
        
        try:
            stats = await self.search_engine.get_search_stats()
            
            print(f"   Vector Database:")
            print(f"     - Total chunks: {stats.get('total_chunks', 'N/A')}")
            print(f"     - Embedding model: {stats.get('embedding_model', 'N/A')}")
            print(f"     - Device: {stats.get('device', 'N/A')}")
            print(f"     - Similarity threshold: {stats.get('similarity_threshold', 'N/A')}")
            
        except Exception as e:
            print(f"   Could not retrieve stats: {e}")
        
        print()
    
    async def run_demo(self):
        """Run the complete demonstration"""
        print("🎯 LLM Document Query and Retrieval System Demo")
        print("=" * 50)
        print()
        
        try:
            # Initialize system
            await self.initialize()
            
            # Process sample documents
            docs_processed = await self.process_sample_documents()
            
            if not docs_processed:
                print("⚠️  No documents were processed. Creating minimal demo...")
            
            # Show system stats
            await self.show_system_stats()
            
            # Demonstrate query processing
            await self.demonstrate_query_processing()
            
            print("🎉 Demo completed successfully!")
            print("\nThe system demonstrates:")
            print("✅ Natural language query processing and entity extraction")
            print("✅ Document parsing and semantic indexing")
            print("✅ Intelligent search and information retrieval")
            print("✅ Rule-based decision making with evidence")
            print("✅ Structured JSON responses with explanations")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main function to run the demonstration"""
    demo = SystemDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
