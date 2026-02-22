# Dentobase CDSS - AI-Powered Dental Clinical Decision Support System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready backend API for **Dentobase** - an intelligent dental practice management platform with integrated AI-powered clinical decision support. Built for dental practitioners to manage patients while receiving real-time diagnostic assistance through advanced computer vision and retrieval-augmented generation (RAG).

## 🌟 Key Features

### 🦷 Clinical Decision Support
- **Multi-modal AI Analysis**: Analyzes dental radiographs using 9+ vision models (local & cloud)
- **RAG-Enhanced Recommendations**: Clinical guidelines retrieval with LLM-powered treatment suggestions
- **Evidence-Based**: All recommendations cite specific guideline pages for clinical trust
- **Configurable Models**: Switch between GPT-4V, Claude, Groq, Gemini, LLaVA, Gemma3, etc.

### 🏥 Practice Management
- Complete patient records management
- Appointment scheduling
- Clinical notes & diagnoses
- Electronic prescriptions
- Multi-facility support

### 💰 Cost Analysis & Evaluation
- Real-time inference cost tracking
- Deployment scenario recommendations (Tanzania-focused)
- Performance benchmarking dashboard
- Thesis-ready evaluation metrics

### 🔐 Enterprise-Ready
- JWT authentication with refresh tokens
- Role-based access control
- Email verification
- Password reset flows
- PostgreSQL with async SQLAlchemy

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+**
- **UV** (Python package installer - faster than pip)
- **Ollama** (for local AI models) - Optional but recommended

### 1. Clone Repository

```bash
git clone https://github.com/saadatsamjo/Dentobase_Backend.git
cd Dentobase_Backend
```

### 2. Install UV (if not installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Create Virtual Environment & Install Dependencies

```bash
# Create venv and install all dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
# or uv sync
```

### 4. Setup PostgreSQL Database

```bash
# Create database
createdb dentobase

# Or via psql
psql -U postgres
CREATE DATABASE dentobase;
\q
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/dentobase

# Authentication
SECRET_KEY=your-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Email (for verification/password reset)
RESEND_API_KEY=***

# AI Model API Keys (Optional - only if using cloud models)
OPENAI_API_KEY=sk-...
CLAUDE_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...

# Clinical Guidelines PDF Path
PDF_PATH=documents/stg.pdf
```

### 6. Initialize Database

```bash
# Run migrations (if using Alembic)
alembic upgrade head

# Or let FastAPI create tables on startup
python -m app.main
```

### 7. Download Local AI Models (Optional)

For offline/local inference:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download recommended models
ollama pull llava:13b          # Best open-source vision model
ollama pull gemma3:4b          # Lightweight multimodal
ollama pull llama3.1:8b        # LLM for RAG
ollama pull nomic-embed-text   # Embeddings
```

### 8. Run the Application

```bash
# Development (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 9. Access API Documentation

Open your browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📚 API Documentation

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register new user |
| `/api/auth/login` | POST | Login and get tokens |
| `/api/auth/refresh` | POST | Refresh access token |
| `/api/auth/logout` | POST | Logout (revoke tokens) |
| `/api/auth/change-password` | POST | Change password |
| `/api/auth/forgot-password` | POST | Request password reset |
| `/api/auth/reset-password` | POST | Reset password with token |
| `/api/auth/verify-email` | POST | Verify email address |

### Vision Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vision/analyze_image` | POST | Analyze dental X-ray (single model) |
| `/api/vision/test_vision_models` | POST | Test all models on same image |
| `/api/vision/config` | GET | Get current vision config |
| `/api/vision/config` | POST | Update vision model settings |

**Supported Vision Models:**
- Local: LLaVA 7B/13B, Gemma3 4B/12B, LLaVA-Med, BiomedCLIP, Florence-2
- Cloud: GPT-4o, Claude 3.5 Sonnet, Groq Llama 3.2, Gemini 1.5 Flash

### RAG System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rag/answer_question` | POST | Query clinical guidelines |
| `/api/rag/config` | GET/POST | Get/update RAG configuration |
| `/api/rag/upload_document` | POST | Upload PDF guidelines |

**Supported LLMs:**
- Local: Llama 3.1 8B, Mixtral 8x7B, Gemma3 4B
- Cloud: GPT-4o, Claude 3.5 Sonnet, Groq Llama 70B, Gemini Flash

### Clinical Decision Support (CDSS)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cdss/provide_final_recommendation` | POST | Complete CDSS pipeline (vision + RAG) |
| `/api/cdss/recommendation_json` | POST | Generate recommendation from existing findings |
| `/api/cdss/systemconfig` | GET | System health check |

**CDSS Pipeline:**
1. Vision model analyzes radiograph
2. RAG retrieves relevant guidelines
3. LLM fuses findings into clinical recommendation
4. Returns structured output with citations

### Practice Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/create_facility` | POST | Create healthcare facility |
| `/api/system/create_patient` | POST | Register new patient |
| `/api/system/create_appointment` | POST | Schedule appointment |
| `/api/system/create_clinical_note` | POST | Add clinical note |
| `/api/system/create_diagnosis` | POST | Record diagnosis |
| `/api/system/create_encounter` | POST | Create patient encounter |
| `/api/system/create_prescription` | POST | Issue prescription |

### Cost Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cost/vision-comparison` | GET | Compare vision model costs |
| `/api/cost/llm-comparison` | GET | Compare LLM costs |
| `/api/cost/pipeline-estimate` | GET | Estimate CDSS pipeline cost |
| `/api/cost/recommendations` | GET | Deployment recommendations |
| `/api/cost/tanzania-deployment` | GET | Tanzania-specific analysis |

### Evaluation & Benchmarking

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/evaluation/dashboard` | GET | Complete evaluation dashboard |
| `/api/evaluation/comparison` | GET | Model comparison data |
| `/api/evaluation/thesis-table-5-1` | GET | Model specifications table |
| `/api/evaluation/thesis-table-5-2` | GET | Performance comparison table |
| `/api/evaluation/thesis-table-5-3` | GET | Deployment scenarios table |
| `/api/evaluation/thesis-figure-5-1-data` | GET | Pareto frontier plot data |
| `/api/evaluation/latex-tables` | GET | LaTeX-formatted tables |
| `/api/evaluation/markdown-tables` | GET | Markdown-formatted tables |
| `/api/evaluation/export-csv` | GET | Export results to CSV |
| `/api/evaluation/status` | GET | Evaluation system status |
| `/api/evaluation/confidence-explanation` | GET | Confidence score methodology |

### Admin

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/admin/reset-all-configs` | POST | Reset all configurations |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   Vision     │  │     RAG      │  │   Fusion        │ │
│  │   Analysis   │→→│  Knowledge   │→→│   Engine        │ │
│  │              │  │  Retrieval   │  │   (CDSS)        │ │
│  └──────────────┘  └──────────────┘  └─────────────────┘ │
│         ↓                  ↓                    ↓          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │         PostgreSQL Database (Async)                  │ │
│  │  - Patients  - Encounters  - Prescriptions           │ │
│  │  - Facilities - Diagnoses  - Clinical Notes          │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Framework**: FastAPI (async Python web framework)
- **Database**: PostgreSQL 15+ with asyncpg
- **ORM**: SQLAlchemy 2.0 (async)
- **Authentication**: JWT (access + refresh tokens)
- **Vision AI**: 
  - Local: Ollama (LLaVA, Gemma3, etc.)
  - Cloud: OpenAI, Anthropic, Groq, Google Gemini
- **LLM**: LangChain + multiple providers
- **Vector Store**: Chroma (for RAG embeddings)
- **Package Manager**: UV (faster than pip/poetry)

---

## 🔧 Configuration

### Vision Models

Switch between models by POSTing to `/api/vision/config`:

```json
{
  "vision_model_provider": "llava",  // or "groq", "gemini", "gpt4v", "claude"
  "llava_model": "llava:13b"
}
```

### RAG System

Configure LLM and retrieval settings:

```json
{
  "llm_provider": "ollama",  // or "openai", "claude", "groq", "gemini"
  "ollama_llm_model": "llama3.1:8b",
  "retriever_type": "mmr",
  "retrieval_k": 8
}
```

---

## 📊 Deployment Scenarios

Based on real cost/performance analysis:

### Ultra-Low Cost (Rural Clinics)
- **Models**: Gemma3 4B (local)
- **Cost**: $0/year (opex)
- **Hardware**: Mac Mini M4 ($800 capex)
- **Use Case**: Offline rural clinics, no internet

### Best Value (Urban Hospitals)
- **Models**: Gemini 1.5 Flash (cloud)
- **Cost**: $5-7/year @ 50 cases/day
- **Hardware**: Any modern PC
- **Use Case**: Connected clinics, high volume

### Ultra-Fast (Emergency Triage)
- **Models**: Groq Llama 3.2 11B (cloud)
- **Cost**: $18-20/year
- **Speed**: 2-3s inference
- **Use Case**: Speed-critical scenarios

### Hybrid (Recommended for Tanzania)
- **Vision**: Gemma3 4B (local, offline)
- **LLM**: Gemini Flash (cloud, when available)
- **Cost**: $5-10/year
- **Use Case**: Intermittent connectivity

---

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Example API Calls

**Register User:**
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "dentist@example.com",
    "password": "SecurePassword123!",
    "full_name": "Dr. Jane Smith"
  }'
```

**Analyze X-ray:**
```bash
curl -X POST http://localhost:8000/api/vision/analyze_image \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@xray.jpg" \
  -F "context=pain lower right jaw" \
  -F "tooth_number=47"
```

**Get CDSS Recommendation:**
```bash
curl -X POST http://localhost:8000/api/cdss/provide_final_recommendation \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "patient_id=1" \
  -F "chief_complaint=severe toothache" \
  -F "tooth_number=47" \
  -F "image=@periapical.jpg"
```

---

## 📈 Performance Metrics

Real measurements from evaluation system:

| Model | Speed (s) | Cost/Analysis | RAM (GB) | Provider |
|-------|-----------|---------------|----------|----------|
| Gemma 3 4B | 28.7 | $0.00 | 8 | Local |
| LLaVA 13B | 58.7 | $0.00 | 16 | Local |
| Groq Llama 11B | 2.3 | $0.001 | 4 | Cloud |
| Gemini Flash | 4.5 | $0.0003 | 4 | Cloud |
| GPT-4o | 10.2 | $0.025 | 4 | Cloud |

*Data from actual test runs - see `/api/evaluation/dashboard` for live metrics*

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Tanzania Standard Treatment Guidelines** (clinical knowledge source)
- **Ollama** for local model inference
- **LangChain** for RAG orchestration
- **FastAPI** for the excellent web framework

---

## 📧 Contact

- **Website**: [dentobase.com](https://dentobase.com)
- **Email**: support@dentobase.com, samjosaadat@yahoo.com
- **Issues**: [GitHub Issues](https://github.com/saadatsamjo/Dentobase_Backend.git/Issues)

---