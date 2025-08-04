"""
Query processing engine for natural language understanding
"""

import re
import spacy
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import json

from ..models.schemas import ProcessedQuery, ExtractedEntities, Person, Location, MedicalProcedure, Policy
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)

class EntityExtractor:
    """Extract entities from natural language queries"""
    
    def __init__(self):
        self.settings = get_settings()
        self.nlp = None
        self._medical_keywords = {
            'surgery': ['surgery', 'operation', 'procedure', 'surgical'],
            'body_parts': ['knee', 'hip', 'shoulder', 'back', 'heart', 'brain', 'liver', 'kidney'],
            'medical_conditions': ['diabetes', 'hypertension', 'cancer', 'arthritis', 'fracture']
        }
        
    async def initialize(self):
        """Initialize spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            logger.info("Please install spaCy English model: python -m spacy download en_core_web_sm")
            raise
    
    async def extract_entities(self, query: str) -> ExtractedEntities:
        """Extract all entities from the query"""
        if not self.nlp:
            await self.initialize()
        
        doc = self.nlp(query)
        
        # Extract basic entities
        person = self._extract_person(query, doc)
        location = self._extract_location(query, doc)
        medical_procedure = self._extract_medical_procedure(query, doc)
        policy = self._extract_policy(query, doc)
        amounts = self._extract_amounts(query)
        dates = self._extract_dates(query, doc)
        
        return ExtractedEntities(
            person=person,
            location=location,
            medical_procedure=medical_procedure,
            policy=policy,
            amounts=amounts,
            dates=dates
        )
    
    def _extract_person(self, query: str, doc) -> Optional[Person]:
        """Extract person information"""
        person = Person()
        
        # Extract age
        age_patterns = [
            r'(\d+)[- ]?(?:year[s]?[- ]?old|y/o|yo|yrs?)',
            r'age[:\s]*(\d+)',
            r'(\d+)[- ]?y[ears]*[- ]?old'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    person.age = int(match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract gender
        male_patterns = [r'\bmale\b', r'\bman\b', r'\bM\b', r'\bmr\.?\b']
        female_patterns = [r'\bfemale\b', r'\bwoman\b', r'\bF\b', r'\bmrs?\.?\b', r'\bmiss\b']
        
        for pattern in male_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                person.gender = "male"
                break
        
        if not person.gender:
            for pattern in female_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    person.gender = "female"
                    break
        
        # Extract names from NER
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not person.name:
                person.name = ent.text
                break
        
        # Return None if no person information found
        if not any([person.age, person.gender, person.name]):
            return None
            
        return person
    
    def _extract_location(self, query: str, doc) -> Optional[Location]:
        """Extract location information"""
        location = Location()
        
        # Extract locations from NER
        cities = []
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or location
                cities.append(ent.text)
        
        # Known Indian cities (extend as needed)
        indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'pune', 'hyderabad',
            'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 'bhopal'
        ]
        
        # Check for Indian cities in the query
        query_lower = query.lower()
        for city in indian_cities:
            if city in query_lower:
                location.city = city.title()
                location.country = "India"
                break
        
        # If NER found locations but no Indian city matched, use the first one
        if not location.city and cities:
            location.city = cities[0]
        
        return location if location.city else None
    
    def _extract_medical_procedure(self, query: str, doc) -> Optional[MedicalProcedure]:
        """Extract medical procedure information"""
        procedure = MedicalProcedure()
        query_lower = query.lower()
        
        # Extract procedure types
        for keyword in self._medical_keywords['surgery']:
            if keyword in query_lower:
                procedure.procedure_type = keyword
                procedure.procedure_name = keyword
                break
        
        # Extract body parts
        for body_part in self._medical_keywords['body_parts']:
            if body_part in query_lower:
                procedure.body_part = body_part
                if not procedure.procedure_name and procedure.procedure_type:
                    procedure.procedure_name = f"{body_part} {procedure.procedure_type}"
                elif procedure.procedure_name and procedure.procedure_type:
                    procedure.procedure_name = f"{body_part} {procedure.procedure_type}"
                break
                break
        
        # Specific procedure patterns
        procedure_patterns = [
            (r'knee\s+surgery', 'knee surgery', 'knee'),
            (r'hip\s+replacement', 'hip replacement', 'hip'),
            (r'heart\s+surgery', 'heart surgery', 'heart'),
            (r'appendectomy', 'appendectomy', 'appendix'),
        ]
        
        for pattern, name, body_part in procedure_patterns:
            if re.search(pattern, query_lower):
                procedure.procedure_name = name
                procedure.body_part = body_part
                procedure.procedure_type = 'surgery'
                break
        
        return procedure if any([procedure.procedure_name, procedure.body_part, procedure.procedure_type]) else None
    
    def _extract_policy(self, query: str, doc) -> Optional[Policy]:
        """Extract policy information"""
        policy = Policy()
        
        # Extract policy age/duration
        policy_age_patterns = [
            r'(\d+)[- ]?(?:month[s]?|mo)[- ]?old[- ]?(?:policy|insurance)',
            r'(\d+)[- ]?(?:year[s]?|yr[s]?)[- ]?old[- ]?(?:policy|insurance)',
            r'(?:policy|insurance)[- ]?(?:age|duration)[:\s]*(\d+)[- ]?(?:month[s]?|year[s]?)',
            r'(\d+)[- ]?(?:month[s]?|year[s]?)[- ]?(?:policy|insurance)'
        ]
        
        for pattern in policy_age_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                policy.policy_age = match.group(0)
                break
        
        # Extract policy type
        policy_types = ['health', 'life', 'auto', 'home', 'travel', 'medical']
        for policy_type in policy_types:
            if policy_type in query.lower():
                policy.policy_type = policy_type
                break
        
        # Extract policy number patterns
        policy_number_patterns = [
            r'policy[- ]?(?:number|no\.?|#)[:\s]*([A-Z0-9]+)',
            r'policy[:\s]*([A-Z0-9]{6,})'
        ]
        
        for pattern in policy_number_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                policy.policy_number = match.group(1)
                break
        
        return policy if any([policy.policy_age, policy.policy_type, policy.policy_number]) else None
    
    def _extract_amounts(self, query: str) -> List[float]:
        """Extract monetary amounts"""
        amount_patterns = [
            r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)',  # Indian Rupees
            r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)',  # US Dollars
            r'(?:amount|sum|value|cost|price)[:\s]*₹?\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rupees|dollars|inr|usd)',
            r'(\d+)k',  # Thousands (e.g., 50k)
            r'(\d+)\s*lakh[s]?',  # Indian lakhs
            r'(\d+)\s*crore[s]?'  # Indian crores
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    
                    # Handle Indian number system
                    if 'lakh' in match.group(0).lower():
                        amount *= 100000
                    elif 'crore' in match.group(0).lower():
                        amount *= 10000000
                    elif match.group(0).endswith('k'):
                        amount *= 1000
                    
                    amounts.append(amount)
                except (ValueError, IndexError):
                    continue
        
        return amounts
    
    def _extract_dates(self, query: str, doc) -> List[str]:
        """Extract dates"""
        dates = []
        
        # Extract dates from NER
        for ent in doc.ents:
            if ent.label_ == "DATE":
                dates.append(ent.text)
        
        # Common date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if match.group(0) not in dates:
                    dates.append(match.group(0))
        
        return dates

class QueryProcessor:
    """Main query processor"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.settings = get_settings()
    
    async def initialize(self):
        """Initialize the query processor"""
        await self.entity_extractor.initialize()
    
    async def process_query(self, query: str) -> ProcessedQuery:
        """Process a natural language query"""
        
        # Extract entities
        entities = await self.entity_extractor.extract_entities(query)
        
        # Determine intent
        intent = self._classify_intent(query, entities)
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Structure the data
        structured_data = self._structure_query_data(entities, intent)
        
        return ProcessedQuery(
            original_query=query,
            entities=entities.model_dump(),
            intent=intent,
            structured_data=structured_data,
            keywords=keywords
        )
    
    def _classify_intent(self, query: str, entities: ExtractedEntities) -> str:
        """Classify the intent of the query"""
        query_lower = query.lower()
        
        # Insurance claim keywords
        claim_keywords = ['claim', 'coverage', 'covered', 'eligible', 'reimburse', 'approve', 'pay']
        policy_keywords = ['policy', 'insurance', 'plan', 'premium', 'deductible']
        medical_keywords = ['surgery', 'treatment', 'procedure', 'medical', 'hospital']
        
        # Score different intents
        scores = {
            'insurance_claim': 0,
            'policy_check': 0,
            'medical_query': 0,
            'general': 0
        }
        
        # Check for insurance claim intent
        for keyword in claim_keywords:
            if keyword in query_lower:
                scores['insurance_claim'] += 2
        
        # Check for policy-related intent
        for keyword in policy_keywords:
            if keyword in query_lower:
                scores['policy_check'] += 1
                scores['insurance_claim'] += 1
        
        # Check for medical intent
        for keyword in medical_keywords:
            if keyword in query_lower:
                scores['medical_query'] += 1
                scores['insurance_claim'] += 1
        
        # Boost scores based on entities
        if entities.medical_procedure:
            scores['insurance_claim'] += 2
            scores['medical_query'] += 1
        
        if entities.policy:
            scores['policy_check'] += 2
            scores['insurance_claim'] += 1
        
        if entities.person and entities.person.age:
            scores['insurance_claim'] += 1
        
        # Return the highest scoring intent
        max_intent = max(scores.items(), key=lambda x: x[1])
        return max_intent[0] if max_intent[1] > 0 else 'general'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        # Simple keyword extraction (can be enhanced with TF-IDF, etc.)
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'old', 'year', 'years'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _structure_query_data(self, entities: ExtractedEntities, intent: str) -> Dict[str, Any]:
        """Structure the extracted data for easy processing"""
        structured = {
            'intent': intent,
            'has_person_info': entities.person is not None,
            'has_location_info': entities.location is not None,
            'has_medical_info': entities.medical_procedure is not None,
            'has_policy_info': entities.policy is not None,
            'has_amounts': len(entities.amounts) > 0,
            'has_dates': len(entities.dates) > 0
        }
        
        # Add specific structured fields based on intent
        if intent == 'insurance_claim':
            structured.update({
                'claimant_age': entities.person.age if entities.person else None,
                'claimant_gender': entities.person.gender if entities.person else None,
                'procedure_location': entities.location.city if entities.location else None,
                'procedure_type': entities.medical_procedure.procedure_name if entities.medical_procedure else None,
                'policy_duration': entities.policy.policy_age if entities.policy else None,
                'claim_amounts': entities.amounts
            })
        
        return structured
