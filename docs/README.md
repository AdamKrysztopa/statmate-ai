# StatmateAI Documentation

**Complete documentation for the StatmateAI project.**

---

## 📚 Quick Navigation

### Getting Started
- **[Main README](../README.md)** - Project overview and quick start
- **[Quick Start Guide](QUICK_START.md)** - Fast setup and first API calls (5 minutes)
- **[Backend Setup](BACKEND_SETUP.md)** - Detailed installation and configuration

### Architecture & Design
- **[Architecture Proposal](ARCHITECTURE_PROPOSAL.md)** - System design, technology choices, and future plans
- **[Implementation Guide](IMPLEMENTATION_GUIDE.md)** - Step-by-step guide to what was built and how
- **[Workflow Documentation](WORKFLOW.md)** - LangGraph workflow and statistical decision tree
- **[Service Layer](SERVICE_LAYER.md)** - Business logic layer documentation

### API & Integration
- **[API Reference](API_REFERENCE.md)** - Complete REST API documentation (18 endpoints)
- **Interactive API Docs** - http://localhost:8000/docs (when running)

### Operations
- **[DevOps Plan](DEVOPS_PLAN.md)** - CI/CD, deployment, monitoring, and production setup

---

## 📖 Document Descriptions

### 1. Quick Start Guide
**File**: `QUICK_START.md`  
**Audience**: First-time users  
**Time**: 5 minutes

Get up and running quickly with:
- Minimal setup instructions
- Database initialization
- First API calls
- Testing endpoints

**Start here if you want to**: Try the API immediately

---

### 2. Backend Setup
**File**: `BACKEND_SETUP.md`  
**Audience**: Developers setting up local environment  
**Time**: 15-30 minutes

Comprehensive setup guide including:
- Prerequisites and dependencies
- Step-by-step installation
- Environment configuration
- Database setup
- Testing instructions
- Troubleshooting

**Start here if you want to**: Set up a full development environment

---

### 3. Architecture Proposal
**File**: `ARCHITECTURE_PROPOSAL.md`  
**Audience**: Technical leads, architects, contributors  
**Time**: 30-45 minutes read

Complete system design covering:
- Phase 1: FastAPI + Streamlit (Current)
- Technology stack and rationale
- Database schema and relationships
- API endpoint design
- **Frontend V2**: React/Next.js + TypeScript
- **Mobile Apps**: React Native, Flutter, PWA
- Backend enhancements for V2
- Migration path and comparison table

**Start here if you want to**: Understand the big picture and future plans

---

### 4. Implementation Guide
**File**: `IMPLEMENTATION_GUIDE.md`  
**Audience**: Developers, maintainers, contributors  
**Time**: 45-60 minutes read

Detailed walkthrough of everything implemented:
- Step-by-step implementation details
- Database models explained
- Service layer deep dive
- API routes documentation
- Integration with existing LangGraph workflow
- Frontend implementation
- File structure
- Testing strategies

**Start here if you want to**: Understand how everything works internally

---

### 5. Workflow Documentation
**File**: `WORKFLOW.md`  
**Audience**: Data scientists, statisticians, contributors  
**Time**: 20 minutes read

LangGraph workflow and decision tree:
- Visual workflow graph (Mermaid diagram)
- Node-by-node explanation
- Statistical test selection logic
- Execution flow examples
- State management
- Integration with API
- Extending the workflow
- Troubleshooting

**Start here if you want to**: Understand how statistical tests are selected or add new tests

---

### 6. Service Layer Documentation
**File**: `SERVICE_LAYER.md`  
**Audience**: Backend developers  
**Time**: 30 minutes read

Business logic layer reference:
- Service architecture pattern
- `StorageService` - File I/O operations
- `DatasetService` - Dataset lifecycle
- `AnalysisService` - Statistical analysis execution
- `TaskService` - Scheduled task management
- Method signatures and usage examples
- Best practices
- Testing guidelines

**Start here if you want to**: Add features or modify business logic

---

### 7. API Reference
**File**: `API_REFERENCE.md`  
**Audience**: API consumers, frontend developers, integration partners  
**Time**: Reference document (skim or search as needed)

Complete REST API documentation:
- 18 endpoint descriptions
- Request/response schemas
- Parameter specifications
- Error codes and handling
- Example curl commands
- SDK examples (Python, JavaScript)
- Pagination, rate limiting
- Webhooks (planned)

**Start here if you want to**: Integrate with the API or build a custom frontend

---

### 8. DevOps Plan
**File**: `DEVOPS_PLAN.md`  
**Audience**: DevOps engineers, SREs, production deployment teams  
**Time**: 30-45 minutes read

Production deployment and operations:
- CI/CD pipeline (GitHub Actions)
- Docker containerization
- Environment management (dev/staging/prod)
- Monitoring (Prometheus, Grafana)
- Logging (ELK stack)
- Database management (SQLite → PostgreSQL)
- Security best practices
- Backup and recovery
- Scaling strategies

**Start here if you want to**: Deploy to production or set up CI/CD

---

## 🗂️ Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── DOCUMENTATION_SUMMARY.md     # Documentation overview and stats
├── QUICK_START.md               # 5-minute quick start
├── BACKEND_SETUP.md             # Development environment setup
├── ARCHITECTURE_PROPOSAL.md     # System design and future plans
├── IMPLEMENTATION_GUIDE.md      # What was built and how
├── WORKFLOW.md                  # LangGraph workflow and decision tree
├── SERVICE_LAYER.md             # Business logic documentation
├── API_REFERENCE.md             # REST API complete reference
└── DEVOPS_PLAN.md               # Production deployment guide
```

---

## 📝 Documentation by Use Case

### "I want to try StatmateAI quickly"
1. [Quick Start Guide](QUICK_START.md)
2. Interactive API Docs at http://localhost:8000/docs

### "I want to understand the system design"
1. [Architecture Proposal](ARCHITECTURE_PROPOSAL.md)
2. [Implementation Guide](IMPLEMENTATION_GUIDE.md)

### "I want to develop features"
1. [Backend Setup](BACKEND_SETUP.md)
2. [Implementation Guide](IMPLEMENTATION_GUIDE.md)
3. [Service Layer](SERVICE_LAYER.md)

### "I want to integrate with the API"
1. [API Reference](API_REFERENCE.md)
2. Interactive API Docs at http://localhost:8000/docs

### "I want to deploy to production"
1. [DevOps Plan](DEVOPS_PLAN.md)
2. [Backend Setup](BACKEND_SETUP.md) (for environment variables)

### "I want to contribute"
1. [Architecture Proposal](ARCHITECTURE_PROPOSAL.md) - Understand the vision
2. [Implementation Guide](IMPLEMENTATION_GUIDE.md) - See what exists
3. [Service Layer](SERVICE_LAYER.md) - Understand the patterns
4. [Main README](../README.md) - Contributing section

---

## 🎯 Learning Paths

### Path 1: User/Tester (30 minutes)
```
1. Main README (5 min)
   ↓
2. Quick Start Guide (10 min)
   ↓
3. Try API with curl or UI (15 min)
```

### Path 2: Frontend Developer (2 hours)
```
1. Main README (5 min)
   ↓
2. Backend Setup (20 min) - to run API locally
   ↓
3. API Reference (30 min) - skim all endpoints
   ↓
4. Architecture Proposal (30 min) - Frontend V2 section
   ↓
5. Build custom frontend (remaining time)
```

### Path 3: Backend Developer (4 hours)
```
1. Main README (5 min)
   ↓
2. Architecture Proposal (30 min)
   ↓
3. Backend Setup (30 min)
   ↓
4. Implementation Guide (60 min)
   ↓
5. Service Layer (30 min)
   ↓
6. Code exploration (60 min)
```

### Path 4: DevOps Engineer (3 hours)
```
1. Main README (5 min)
   ↓
2. Architecture Proposal (30 min)
   ↓
3. Backend Setup (20 min)
   ↓
4. DevOps Plan (60 min)
   ↓
5. Setup CI/CD pipeline (60 min)
```

### Path 5: Technical Lead/Architect (3 hours)
```
1. Main README (10 min)
   ↓
2. Architecture Proposal (60 min)
   ↓
3. Implementation Guide (60 min)
   ↓
4. API Reference (20 min) - skim
   ↓
5. DevOps Plan (20 min)
   ↓
6. Service Layer (20 min)
   ↓
7. Code review (30 min)
```

---

## 🔍 Finding Information

### By Topic

**Authentication & Security**
- DevOps Plan → Security Best Practices
- Architecture Proposal → Phase 2 Features

**API Endpoints**
- API Reference → Complete endpoint list
- Interactive Docs → http://localhost:8000/docs

**Database Schema**
- Implementation Guide → Database Schema
- Architecture Proposal → Backend Components

**File Storage**
- Service Layer → StorageService
- Implementation Guide → Integration Points

**Frontend**
- Architecture Proposal → Frontend V1 & V2
- Implementation Guide → Frontend Implementation

**Deployment**
- DevOps Plan → All sections
- Backend Setup → Environment Configuration

**Statistical Tests**
- Main README → Statistical Tests section
- Workflow → Complete workflow graph and test selection logic
- Implementation Guide → Integration with Existing Code

**Task Scheduling**
- Service Layer → TaskService
- API Reference → Scheduled Tasks API

---

## 🆘 Troubleshooting

Having issues? Check these docs:

| Issue                  | Document             | Section                        |
| ---------------------- | -------------------- | ------------------------------ |
| Installation problems  | Backend Setup        | Prerequisites, Installation    |
| API not starting       | Backend Setup        | Running the Application        |
| Database errors        | Backend Setup        | Database Setup                 |
| Import errors          | Implementation Guide | Integration with Existing Code |
| API endpoint errors    | API Reference        | Error Codes Reference          |
| Deployment issues      | DevOps Plan          | Troubleshooting                |
| Environment variables  | Backend Setup        | Environment Configuration      |
| File upload problems   | Service Layer        | StorageService                 |
| Task scheduling issues | Service Layer        | TaskService                    |

---

## 📚 External Resources

### Technology Documentation

**Backend:**
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [SQLAlchemy 2.0 Docs](https://docs.sqlalchemy.org/en/20/)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [APScheduler User Guide](https://apscheduler.readthedocs.io)

**Frontend:**
- [Streamlit Documentation](https://docs.streamlit.io)
- [HTTPX Documentation](https://www.python-httpx.org)

**Statistical Libraries:**
- [SciPy Stats Reference](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)

**AI/LLM:**
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Pydantic AI Documentation](https://ai.pydantic.dev)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

**DevOps:**
- [Docker Documentation](https://docs.docker.com)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Prometheus Documentation](https://prometheus.io/docs)

---

## 📊 Documentation Stats

**Total Documents**: 8 files  
**Total Pages**: ~150 (printed)  
**Total Words**: ~50,000  
**Reading Time**: ~5 hours (complete read)  
**Last Updated**: October 16, 2025

---

## ✍️ Contributing to Documentation

Found an error or want to improve the docs?

1. **Quick fixes**: Open an issue
2. **Larger changes**: Submit a PR
3. **New sections**: Discuss in GitHub Discussions

**Documentation Guidelines:**
- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Link to related sections
- Keep formatting consistent

---

## 📞 Support

- **Questions**: GitHub Discussions
- **Bugs**: GitHub Issues  
- **Email**: support@statmate-ai.com
- **API Issues**: Check API Reference first, then open issue

---

## 🗺️ Documentation Roadmap

### Planned Additions

- [ ] **Tutorial Series**: Step-by-step tutorials for common tasks
- [ ] **Video Walkthroughs**: Screen recordings of setup and usage
- [ ] **API Cookbook**: Common API usage patterns and recipes
- [ ] **Statistical Guide**: When to use which test
- [ ] **Contributor Guide**: Detailed guide for new contributors
- [ ] **Database Migration Guide**: Upgrading schemas
- [ ] **Performance Tuning**: Optimization guide
- [ ] **Security Hardening**: Production security checklist

---

**Happy coding! 🚀**

For the most up-to-date information, always check the [main README](../README.md).

