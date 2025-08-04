"""
Decision engine for evaluating queries and making decisions based on retrieved information
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import asyncio

from ..models.schemas import (
    ProcessedQuery, SearchResult, QueryResponse, SupportingClause, 
    DecisionType, ExtractedEntities
)
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)

class RuleEngine:
    """Rule-based decision making engine"""
    
    def __init__(self):
        self.settings = get_settings()
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load decision rules (in production, this would come from a database or config file)"""
        return {
            "insurance_claim_rules": {
                "coverage_rules": [
                    {
                        "id": "knee_surgery_coverage",
                        "conditions": {
                            "procedure_types": ["knee surgery", "knee replacement", "knee repair"],
                            "policy_age_minimum": 90,  # days
                            "age_range": [18, 80]
                        },
                        "decision": "approved",
                        "coverage_percentage": 80,
                        "max_amount": 200000,
                        "description": "Knee surgery is covered under orthopedic procedures"
                    },
                    {
                        "id": "hip_surgery_coverage", 
                        "conditions": {
                            "procedure_types": ["hip surgery", "hip replacement", "hip repair"],
                            "policy_age_minimum": 90,
                            "age_range": [25, 75]
                        },
                        "decision": "approved",
                        "coverage_percentage": 85,
                        "max_amount": 250000,
                        "description": "Hip surgery is covered under orthopedic procedures"
                    },
                    {
                        "id": "heart_surgery_coverage",
                        "conditions": {
                            "procedure_types": ["heart surgery", "cardiac surgery", "bypass surgery"],
                            "policy_age_minimum": 180,
                            "age_range": [21, 70]
                        },
                        "decision": "approved", 
                        "coverage_percentage": 90,
                        "max_amount": 500000,
                        "description": "Heart surgery is covered under critical care procedures"
                    }
                ],
                "exclusion_rules": [
                    {
                        "id": "cosmetic_exclusion",
                        "conditions": {
                            "procedure_types": ["cosmetic surgery", "plastic surgery", "aesthetic surgery"]
                        },
                        "decision": "rejected",
                        "description": "Cosmetic procedures are not covered"
                    },
                    {
                        "id": "experimental_exclusion",
                        "conditions": {
                            "procedure_types": ["experimental", "investigational", "clinical trial"]
                        },
                        "decision": "rejected",
                        "description": "Experimental procedures are not covered"
                    }
                ],
                "waiting_period_rules": [
                    {
                        "id": "general_waiting_period",
                        "policy_age_minimum": 30,
                        "description": "General waiting period of 30 days"
                    },
                    {
                        "id": "major_surgery_waiting_period", 
                        "policy_age_minimum": 90,
                        "procedure_types": ["surgery"],
                        "description": "Major surgeries require 90 days waiting period"
                    }
                ]
            }
        }
    
    async def evaluate_insurance_claim(self, processed_query: ProcessedQuery, 
                                      search_results: List[SearchResult]) -> Tuple[DecisionType, float, str, List[SupportingClause]]:
        """Evaluate an insurance claim"""
        
        entities = processed_query.entities
        structured_data = processed_query.structured_data
        
        # Extract key information
        procedure_type = structured_data.get("procedure_type", "").lower()
        claimant_age = structured_data.get("claimant_age")
        policy_duration = self._parse_policy_duration(structured_data.get("policy_duration", ""))
        
        # Check exclusion rules first
        for rule in self.rules["insurance_claim_rules"]["exclusion_rules"]:
            if self._matches_exclusion_rule(rule, procedure_type):
                supporting_clauses = self._find_supporting_clauses(rule, search_results)
                return DecisionType.REJECTED, 0.0, rule["description"], supporting_clauses
        
        # Check coverage rules
        best_match = None
        best_confidence = 0.0
        
        for rule in self.rules["insurance_claim_rules"]["coverage_rules"]:
            confidence = self._calculate_rule_confidence(rule, procedure_type, claimant_age, policy_duration)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = rule
        
        if best_match and best_confidence > 0.5:
            # Check waiting period
            waiting_period_met = self._check_waiting_period(best_match, policy_duration)
            if not waiting_period_met:
                return (
                    DecisionType.REJECTED, 
                    best_confidence,
                    f"Waiting period not met. {best_match['description']}", 
                    self._find_supporting_clauses(best_match, search_results)
                )
            
            # Calculate amount
            amount = self._calculate_coverage_amount(best_match, structured_data.get("claim_amounts", []))
            
            supporting_clauses = self._find_supporting_clauses(best_match, search_results)
            
            return (
                DecisionType.APPROVED,
                best_confidence,
                f"{best_match['description']}. Coverage: {best_match['coverage_percentage']}%",
                supporting_clauses
            ), amount
        
        # Default case - requires review
        supporting_clauses = self._find_general_supporting_clauses(search_results)
        return (
            DecisionType.REQUIRES_REVIEW,
            0.3,
            "Unable to determine coverage automatically. Manual review required.",
            supporting_clauses
        ), 0.0
    
    def _matches_exclusion_rule(self, rule: Dict[str, Any], procedure_type: str) -> bool:
        """Check if procedure matches exclusion rule"""
        procedure_types = rule["conditions"].get("procedure_types", [])
        return any(excluded.lower() in procedure_type for excluded in procedure_types)
    
    def _calculate_rule_confidence(self, rule: Dict[str, Any], procedure_type: str, 
                                  claimant_age: Optional[int], policy_duration: int) -> float:
        """Calculate confidence score for a rule match"""
        confidence = 0.0
        conditions = rule["conditions"]
        
        # Check procedure type match
        procedure_types = conditions.get("procedure_types", [])
        procedure_match = any(proc_type.lower() in procedure_type for proc_type in procedure_types)
        if procedure_match:
            confidence += 0.4
        
        # Check age range
        age_range = conditions.get("age_range", [0, 100])
        if claimant_age and age_range[0] <= claimant_age <= age_range[1]:
            confidence += 0.3
        elif claimant_age is None:
            confidence += 0.1  # Partial credit if age not specified
        
        # Check policy age
        policy_age_min = conditions.get("policy_age_minimum", 0)
        if policy_duration >= policy_age_min:
            confidence += 0.3
        
        return confidence
    
    def _parse_policy_duration(self, policy_age_str: str) -> int:
        """Parse policy duration string to days"""
        if not policy_age_str:
            return 0
        
        policy_age_str = policy_age_str.lower()
        
        # Extract number and unit
        match = re.search(r'(\d+)\s*(month|year|day)', policy_age_str)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            
            if 'month' in unit:
                return number * 30
            elif 'year' in unit:
                return number * 365
            else:  # days
                return number
        
        return 0
    
    def _check_waiting_period(self, rule: Dict[str, Any], policy_duration: int) -> bool:
        """Check if waiting period is satisfied"""
        required_waiting = rule["conditions"].get("policy_age_minimum", 0)
        return policy_duration >= required_waiting
    
    def _calculate_coverage_amount(self, rule: Dict[str, Any], claim_amounts: List[float]) -> float:
        """Calculate coverage amount based on rule"""
        if not claim_amounts:
            return 0.0
        
        max_claim = max(claim_amounts)
        coverage_percentage = rule.get("coverage_percentage", 80) / 100
        max_amount = rule.get("max_amount", float('inf'))
        
        calculated_amount = max_claim * coverage_percentage
        return min(calculated_amount, max_amount)
    
    def _find_supporting_clauses(self, rule: Dict[str, Any], search_results: List[SearchResult]) -> List[SupportingClause]:
        """Find supporting clauses from search results"""
        supporting_clauses = []
        
        # Look for results that support the rule
        for result in search_results[:3]:  # Top 3 results
            if self._is_result_supporting(result, rule):
                clause = SupportingClause(
                    clause_id=f"clause_{result.chunk_id}",
                    document_id=result.document_id,
                    document_name=result.metadata.get("name", "Unknown Document"),
                    text=result.text[:500] + "..." if len(result.text) > 500 else result.text,
                    relevance_score=result.similarity_score,
                    page_number=result.metadata.get("page_number"),
                    section=result.metadata.get("section")
                )
                supporting_clauses.append(clause)
        
        return supporting_clauses
    
    def _find_general_supporting_clauses(self, search_results: List[SearchResult]) -> List[SupportingClause]:
        """Find general supporting clauses when no specific rule matches"""
        supporting_clauses = []
        
        for result in search_results[:2]:  # Top 2 results
            clause = SupportingClause(
                clause_id=f"clause_{result.chunk_id}",
                document_id=result.document_id,
                document_name=result.metadata.get("name", "Unknown Document"),
                text=result.text[:500] + "..." if len(result.text) > 500 else result.text,
                relevance_score=result.similarity_score,
                page_number=result.metadata.get("page_number"),
                section=result.metadata.get("section")
            )
            supporting_clauses.append(clause)
        
        return supporting_clauses
    
    def _is_result_supporting(self, result: SearchResult, rule: Dict[str, Any]) -> bool:
        """Check if search result supports the rule"""
        text_lower = result.text.lower()
        
        # Look for procedure types mentioned in the rule
        procedure_types = rule["conditions"].get("procedure_types", [])
        if any(proc_type.lower() in text_lower for proc_type in procedure_types):
            return True
        
        # Look for coverage-related terms
        coverage_terms = ["covered", "coverage", "benefit", "eligible", "included"]
        if any(term in text_lower for term in coverage_terms):
            return True
        
        return False

class MLDecisionEngine:
    """Machine learning-based decision engine (placeholder for future ML models)"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
    
    async def initialize(self):
        """Initialize ML models"""
        # Placeholder for ML model initialization
        # In a real implementation, you would load trained models here
        logger.info("ML Decision Engine initialized (placeholder)")
    
    async def predict_decision(self, processed_query: ProcessedQuery, 
                             search_results: List[SearchResult]) -> Tuple[DecisionType, float, str]:
        """Predict decision using ML model"""
        # Placeholder implementation
        # In a real scenario, you would:
        # 1. Feature engineering from query and search results
        # 2. Run inference on trained models
        # 3. Return predictions with confidence
        
        return DecisionType.REQUIRES_REVIEW, 0.5, "ML prediction not implemented"

class DecisionEngine:
    """Main decision engine that combines rule-based and ML approaches"""
    
    def __init__(self):
        self.settings = get_settings()
        self.rule_engine = RuleEngine()
        self.ml_engine = MLDecisionEngine()
        
    async def initialize(self):
        """Initialize the decision engine"""
        await self.ml_engine.initialize()
        logger.info("Decision engine initialized")
    
    async def make_decision(self, processed_query: ProcessedQuery, 
                           search_results: List[SearchResult]) -> QueryResponse:
        """Make a decision based on query and search results"""
        
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        
        try:
            # Use rule-based engine for insurance claims
            if processed_query.intent == "insurance_claim":
                decision_result = await self.rule_engine.evaluate_insurance_claim(
                    processed_query, search_results
                )
                
                if len(decision_result) == 2:  # Has amount
                    (decision, confidence, justification, supporting_clauses), amount = decision_result
                else:  # No amount
                    decision, confidence, justification, supporting_clauses = decision_result
                    amount = None
                
            else:
                # For other intents, use general evaluation
                decision, confidence, justification, supporting_clauses = await self._general_evaluation(
                    processed_query, search_results
                )
                amount = None
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            response = QueryResponse(
                query_id=query_id,
                decision=decision,
                amount=amount,
                justification=justification,
                confidence=confidence,
                supporting_clauses=supporting_clauses,
                processing_time=processing_time
            )
            
            logger.info(f"Decision made for query {query_id}: {decision} (confidence: {confidence:.2f})")
            return response
            
        except Exception as e:
            logger.error(f"Failed to make decision: {e}")
            
            # Return error response
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResponse(
                query_id=query_id,
                decision=DecisionType.REQUIRES_REVIEW,
                amount=None,
                justification=f"Error occurred during decision making: {str(e)}",
                confidence=0.0,
                supporting_clauses=[],
                processing_time=processing_time
            )
    
    async def _general_evaluation(self, processed_query: ProcessedQuery, 
                                 search_results: List[SearchResult]) -> Tuple[DecisionType, float, str, List[SupportingClause]]:
        """General evaluation for non-insurance queries"""
        
        if not search_results:
            return (
                DecisionType.REQUIRES_REVIEW,
                0.1,
                "No relevant information found in the documents.",
                []
            )
        
        # Calculate average relevance
        avg_relevance = sum(r.similarity_score for r in search_results) / len(search_results)
        
        # Generate general supporting clauses
        supporting_clauses = []
        for result in search_results[:3]:
            clause = SupportingClause(
                clause_id=f"clause_{result.chunk_id}",
                document_id=result.document_id,
                document_name=result.metadata.get("name", "Unknown Document"),
                text=result.text[:400] + "..." if len(result.text) > 400 else result.text,
                relevance_score=result.similarity_score,
                page_number=result.metadata.get("page_number"),
                section=result.metadata.get("section")
            )
            supporting_clauses.append(clause)
        
        if avg_relevance > 0.8:
            justification = "High confidence match found in the documents."
            decision = DecisionType.APPROVED
            confidence = avg_relevance
        elif avg_relevance > 0.6:
            justification = "Relevant information found, but manual review recommended."
            decision = DecisionType.REQUIRES_REVIEW
            confidence = avg_relevance
        else:
            justification = "Limited relevant information found in the documents."
            decision = DecisionType.REQUIRES_REVIEW
            confidence = avg_relevance
        
        return decision, confidence, justification, supporting_clauses
    
    async def explain_decision(self, response: QueryResponse) -> Dict[str, Any]:
        """Provide detailed explanation of the decision"""
        
        explanation = {
            "query_id": response.query_id,
            "decision_summary": {
                "decision": response.decision,
                "confidence": response.confidence,
                "justification": response.justification
            },
            "evidence": {
                "supporting_clauses_count": len(response.supporting_clauses),
                "supporting_clauses": [
                    {
                        "clause_id": clause.clause_id,
                        "document": clause.document_name,
                        "relevance_score": clause.relevance_score,
                        "text_preview": clause.text[:200] + "..." if len(clause.text) > 200 else clause.text
                    }
                    for clause in response.supporting_clauses
                ]
            },
            "metadata": {
                "processing_time": response.processing_time,
                "timestamp": response.timestamp.isoformat()
            }
        }
        
        if response.amount:
            explanation["decision_summary"]["amount"] = response.amount
        
        return explanation
