# System Architecture

## Overview

The Fund Performance Analysis System is a full-stack application that combines document processing, vector search, and LLM-powered question answering to provide intelligent fund analysis capabilities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                            │
│                   (Next.js Frontend)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Upload  │  │   Chat   │  │  Funds   │  │   Docs   │  │
│  │   Page   │  │   Page   │  │   Page   │  │   Page   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
┌────────────────────────▼────────────────────────────────────┐
│                   API Layer (FastAPI)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │Documents │  │  Funds   │  │   Chat   │  │ Metrics  │  │
│  │Endpoints │  │Endpoints │  │Endpoints │  │Endpoints │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Service Layer                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Document Processor                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ Docling  │─▶│  Table   │─▶│   SQL    │          │  │
│  │  │  Parser  │  │  Parser  │  │  Insert  │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  │                                                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │   Text   │─▶│Embedding │─▶│  Vector  │          │  │
│  │  │ Chunker  │  │Generator │  │  Store   │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Query Engine (RAG)                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │  Intent  │─▶│  Vector  │─▶│   LLM    │          │  │
│  │  │Classifier│  │  Search  │  │ Response │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  │                                                       │  │
│  │  ┌──────────┐  ┌──────────┐                         │  │
│  │  │ Metrics  │─▶│   SQL    │                         │  │
│  │  │Calculator│  │  Query   │                         │  │
│  │  └──────────┘  └──────────┘                         │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐ ┌────▼─────┐ ┌────────▼────────┐
│   PostgreSQL   │ │  FAISS   │ │     Redis       │
│  (Structured)  │ │ (Vectors)│ │  (Task Queue)   │
└────────────────┘ └──────────┘ └─────────────────┘
```

## Component Details

### 1. Frontend (Next.js)

**Technology Stack:**
- Next.js 14 (App Router)
- React 18
- TailwindCSS
- shadcn/ui components
- TanStack Query (React Query)
- Axios

**Key Features:**
- Server-side rendering for SEO
- Client-side state management
- Real-time updates via polling
- Responsive design

**Pages:**
- `/` - Home page with feature overview
- `/upload` - Document upload interface
- `/chat` - AI-powered Q&A interface
- `/funds` - Fund list and dashboard
- `/funds/[id]` - Fund detail page
- `/documents` - Document management

### 2. Backend (FastAPI)

**Technology Stack:**
- FastAPI (Python 3.11+)
- SQLAlchemy (ORM)
- Pydantic (validation)
- Uvicorn (ASGI server)

**API Structure:**
```
/api
├── /documents
│   ├── POST /upload
│   ├── GET /{id}/status
│   ├── GET /{id}
│   └── DELETE /{id}
├── /funds
│   ├── GET /
│   ├── POST /
│   ├── GET /{id}
│   ├── GET /{id}/transactions
│   └── GET /{id}/metrics
├── /chat
│   ├── POST /query
│   ├── POST /conversations
│   └── GET /conversations/{id}
└── /metrics
    └── GET /funds/{id}/metrics
```

### 3. Document Processing Pipeline

**Flow:**
1. **Upload**: File received via multipart/form-data
2. **Storage**: File saved to disk with timestamp
3. **Database Record**: Document entry created with "pending" status
4. **Background Task**: Async processing starts
5. **Docling Parsing**: PDF structure extracted
6. **Table Extraction**: Financial tables identified and parsed
7. **SQL Storage**: Transactions saved to PostgreSQL
8. **Text Chunking**: Text content split into chunks
9. **Embedding**: Chunks converted to vectors
10. **Vector Storage**: Embeddings saved to FAISS
11. **Status Update**: Document marked as "completed"

**Technologies:**
- **Docling**: IBM's document understanding library
- **FAISS**: Facebook's vector similarity search
- **OpenAI Embeddings**: text-embedding-3-small (1536 dimensions)

### 4. RAG Query Engine

**Query Processing Flow:**
1. **Intent Classification**: Determine query type (calculation, definition, retrieval, general)
2. **Vector Search**: Find relevant document chunks (top-k=5)
3. **Metrics Calculation**: If needed, calculate fund metrics from SQL
4. **Context Assembly**: Combine retrieved docs + metrics + conversation history
5. **LLM Generation**: Generate answer using GPT-4
6. **Response Formatting**: Return answer with sources and metrics

**Prompt Engineering:**
- System prompt defines role and behavior
- Context includes document sources
- Metrics data provided when available
- Conversation history for continuity

### 5. Metrics Calculator

**Supported Metrics:**

**DPI (Distribution to Paid-In)**
```python
PIC = Total Capital Calls - Adjustments
DPI = Total Distributions / PIC
```

**IRR (Internal Rate of Return)**
```python
# Using numpy-financial
cash_flows = [capital_calls (negative), distributions (positive)]
IRR = npf.irr(cash_flows) * 100  # Convert to percentage
```

**PIC (Paid-In Capital)**
```python
PIC = SUM(capital_calls.amount) - SUM(adjustments.amount)
```

### 6. Data Storage

**PostgreSQL Schema:**
- `funds`: Fund master data
- `capital_calls`: Capital call transactions
- `distributions`: Distribution transactions
- `adjustments`: Rebalancing entries
- `documents`: Uploaded document metadata

**pgvector Vector Store:**
- Extension: pgvector PostgreSQL extension
- Dimension: 384 (sentence-transformers/all-MiniLM-L6-v2)
- Index type: ivfflat (approximate nearest neighbor)
- Metadata: Stored in same table with vectors
- Includes: document_id, fund_id, page_number, chunk_index, content

**Redis:**
- Task queue for Celery (future)
- Session storage (future)
- Cache for frequent queries (future)

## Data Flow

### Document Upload Flow
```
User → Frontend → Backend API → File System
                              → PostgreSQL (document record)
                              → Background Task
                                  → Docling Parser
                                  → Table Parser → PostgreSQL (transactions)
                                  → Text Chunker → Embeddings → FAISS
                              → Status Update → PostgreSQL
```

### Chat Query Flow
```
User → Frontend → Backend API → Query Engine
                              → Intent Classifier
                              → Vector Store (similarity search)
                              → Metrics Calculator (if needed)
                              → LLM (GPT-4)
                              → Response Assembly
                              → Frontend (display)
```

## Scalability Considerations

### Current Architecture (MVP)
- Single server deployment
- In-memory conversation storage
- Local FAISS index
- Synchronous processing

### Production Recommendations
1. **Horizontal Scaling**
   - Load balancer for API servers
   - Separate document processing workers
   - Distributed task queue (Celery + Redis)

2. **Database Optimization**
   - Read replicas for PostgreSQL
   - Connection pooling
   - Query optimization and indexing

3. **Vector Store**
   - Migrate to Pinecone or Weaviate for distributed search
   - Implement caching layer
   - Batch embedding generation

4. **Caching**
   - Redis for API responses
   - CDN for frontend assets
   - Metric calculation caching

5. **Monitoring**
   - Application metrics (Prometheus)
   - Error tracking (Sentry)
   - Log aggregation (ELK stack)
   - Performance monitoring (New Relic)

## Security Considerations

### Current Implementation
- No authentication (MVP only)
- CORS enabled for localhost
- File type validation (PDF only)
- File size limits (50MB)

### Production Requirements
1. **Authentication & Authorization**
   - JWT tokens
   - Role-based access control (RBAC)
   - API key management

2. **Data Security**
   - Encryption at rest
   - Encryption in transit (HTTPS)
   - Secure file storage (S3 with encryption)
   - Database encryption

3. **Input Validation**
   - Strict file type checking
   - Malware scanning
   - SQL injection prevention (SQLAlchemy ORM)
   - XSS prevention

4. **Rate Limiting**
   - API rate limits per user
   - Upload frequency limits
   - Query rate limits

## Performance Metrics

### Target Performance
- Document upload: < 5 seconds
- Document parsing: < 2 minutes (for 50-page PDF)
- Chat query response: < 3 seconds
- Metrics calculation: < 1 second
- Page load time: < 2 seconds

### Optimization Strategies
1. **Backend**
   - Async processing for heavy tasks
   - Database query optimization
   - Caching frequent calculations
   - Connection pooling

2. **Frontend**
   - Code splitting
   - Lazy loading
   - Image optimization
   - CDN for static assets

3. **LLM**
   - Prompt caching
   - Streaming responses
   - Smaller models for simple queries
   - Local embeddings for cost reduction

## Deployment

### Development
```bash
docker-compose up
```

### Production Options
1. **Cloud Platforms**
   - AWS: ECS/EKS + RDS + S3
   - GCP: Cloud Run + Cloud SQL + GCS
   - Azure: App Service + Azure SQL + Blob Storage

2. **Containerization**
   - Docker images for backend and frontend
   - Kubernetes for orchestration
   - Helm charts for deployment

3. **CI/CD**
   - GitHub Actions
   - Automated testing
   - Staging environment
   - Blue-green deployment

## Future Enhancements

1. **Multi-tenancy**: Support multiple organizations
2. **Advanced Analytics**: Trend analysis, benchmarking
3. **Excel Export**: Export data to Excel
4. **Email Notifications**: Alert on document processing completion
5. **Audit Logging**: Track all user actions
6. **Advanced Search**: Full-text search across documents
7. **Collaboration**: Share insights and annotations
8. **Mobile App**: Native mobile applications
