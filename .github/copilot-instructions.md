# Copilot Instructions for LLM Document Query System

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview

This is a comprehensive Large Language Model (LLM) system for processing natural language queries and retrieving relevant information from unstructured documents. The system is designed for applications in insurance, legal compliance, HR, and contract management.

## Architecture Components

### Core Modules
- **Document Processor**: Handles PDF, DOCX, TXT, and email parsing
- **Query Engine**: Processes natural language queries and extracts entities
- **Semantic Search**: Uses sentence transformers for document retrieval
- **Decision Engine**: Makes decisions based on rules and retrieved information
- **API Layer**: FastAPI-based REST endpoints

### Technology Stack
- **Framework**: FastAPI with Python 3.9+
- **NLP**: spaCy, transformers, sentence-transformers
- **Vector Database**: ChromaDB for semantic search
- **Document Processing**: PyPDF2, python-docx
- **ML Framework**: PyTorch

## Development Guidelines

### Code Style
- Follow PEP 8 standards
- Use type hints for all function parameters and return values
- Include comprehensive docstrings for classes and methods
- Prefer async/await for I/O operations

### Error Handling
- Use structured logging with loguru
- Implement proper exception handling with specific error types
- Return meaningful error messages in API responses
- Log errors with appropriate context

### Testing
- Write unit tests for all core functionality
- Use pytest with async support
- Include integration tests for end-to-end workflows
- Mock external dependencies in tests

### Performance Considerations
- Batch process embeddings when possible
- Use background tasks for document processing
- Implement caching for frequent queries
- Monitor memory usage with large documents

## Domain-Specific Context

### Insurance Claims Processing
- Focus on medical procedures, policy terms, and eligibility
- Extract entities: age, gender, procedure type, location, policy duration
- Apply business rules for coverage decisions
- Generate audit trails with supporting evidence

### Entity Extraction Patterns
- **Person**: Age patterns (46-year-old, 46M, 46 years), gender indicators
- **Medical**: Procedure names, body parts, medical conditions
- **Location**: City names, especially Indian metro areas
- **Policy**: Policy age, types, numbers
- **Financial**: Currency amounts in INR/USD format

### Decision Logic
- Implement rule-based engines for consistent decisions
- Consider waiting periods, eligibility criteria, and exclusions
- Provide confidence scores and justifications
- Reference specific clauses in source documents

## API Design Patterns

### Request/Response Models
- Use Pydantic models for all API interfaces
- Include comprehensive validation and error handling
- Return structured responses with metadata
- Support batch operations where applicable

### Endpoint Categories
- **Health**: System status and diagnostics
- **Query**: Natural language processing and decisions
- **Documents**: Upload, processing, and management

## Security and Compliance

### Data Handling
- Sanitize all user inputs
- Implement proper file upload validation
- Store sensitive information securely
- Follow data retention policies

### API Security
- Implement authentication and authorization
- Use HTTPS for all communications
- Rate limit API endpoints
- Log security events

## Integration Guidelines

### External Services
- Handle API rate limits gracefully
- Implement retry logic with exponential backoff
- Cache responses when appropriate
- Monitor service dependencies

### Document Processing
- Support incremental document updates
- Handle large files efficiently
- Implement document versioning
- Provide processing status updates

## Monitoring and Observability

### Logging
- Use structured logging with context
- Include correlation IDs for request tracing
- Log performance metrics
- Monitor error rates and patterns

### Metrics
- Track query processing times
- Monitor search accuracy and relevance
- Measure decision confidence distributions
- Alert on system anomalies

## Deployment Considerations

### Environment Configuration
- Use environment variables for configuration
- Support multiple deployment environments
- Implement health checks for containers
- Document infrastructure requirements

### Scalability
- Design for horizontal scaling
- Use async processing for I/O bound operations
- Implement proper resource management
- Monitor system resource usage

## Maintenance Guidelines

### Documentation
- Keep README.md updated with latest features
- Document API changes in OpenAPI specs
- Maintain troubleshooting guides
- Update configuration examples

### Code Maintenance
- Regular dependency updates
- Performance optimization reviews
- Security vulnerability assessments
- Code refactoring for maintainability
