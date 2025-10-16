# StatmateAI DevOps Plan
## Development, Deployment, and Operations Strategy

---

## 📋 Table of Contents

1. [Development Environment](#development-environment)
2. [CI/CD Pipeline](#cicd-pipeline)
3. [Testing Strategy](#testing-strategy)
4. [Deployment Architecture](#deployment-architecture)
5. [Monitoring & Logging](#monitoring--logging)
6. [Backup & Recovery](#backup--recovery)
7. [Security & Compliance](#security--compliance)
8. [Scaling Strategy](#scaling-strategy)

---

## 🛠️ Development Environment

### Local Setup

#### Prerequisites
```bash
# Required software
- Python 3.11+
- Node.js 18+ (for Frontend V2)
- Docker & Docker Compose
- Git
```

#### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/statmate-ai.git
cd statmate-ai

# Install Python dependencies
pip install uv  # or use pip
uv sync

# Set up environment variables
cp config/.env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, etc.)

# Initialize database
python scripts/init_db.py

# Run backend (Terminal 1)
uvicorn statmate.api.main:app --reload --port 8000

# Run frontend (Terminal 2)
streamlit run statmate/ui/app.py --server.port 8501
```

#### Docker Development
```bash
# Build and run all services
docker-compose -f docker/docker-compose.dev.yml up

# Services available at:
# - API: http://localhost:8000
# - Streamlit UI: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

### Development Tools

#### Code Quality
```bash
# Linting
ruff check statmate/

# Type checking
pyright statmate/

# Format code
ruff format statmate/

# Run all checks
pre-commit run --all-files
```

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

#### 1. Test & Lint (on every push/PR)
```yaml
# .github/workflows/test.yml
name: Test & Lint

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      
      - name: Run linters
        run: |
          ruff check statmate/
          pyright statmate/
      
      - name: Run tests
        run: |
          pytest tests/ --cov=statmate --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

#### 2. Build Docker Images (on main branch)
```yaml
# .github/workflows/build.yml
name: Build Docker Images

on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Build and push API
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.api
          push: true
          tags: |
            yourusername/statmate-api:latest
            yourusername/statmate-api:${{ github.sha }}
      
      - name: Build and push UI
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.ui
          push: true
          tags: |
            yourusername/statmate-ui:latest
            yourusername/statmate-ui:${{ github.sha }}
```

#### 3. Deploy to Staging (on main branch)
```yaml
# .github/workflows/deploy-staging.yml
name: Deploy to Staging

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging server
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.STAGING_HOST }}
          username: ${{ secrets.STAGING_USER }}
          key: ${{ secrets.STAGING_SSH_KEY }}
          script: |
            cd /opt/statmate-ai
            docker-compose pull
            docker-compose up -d
            docker-compose exec -T api alembic upgrade head
```

#### 4. Deploy to Production (on release tag)
```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USER }}
          key: ${{ secrets.PROD_SSH_KEY }}
          script: |
            cd /opt/statmate-ai
            docker-compose pull
            docker-compose up -d
            docker-compose exec -T api alembic upgrade head
      
      - name: Health check
        run: |
          sleep 30
          curl -f https://api.statmate.ai/health || exit 1
```

---

## 🧪 Testing Strategy

### Test Pyramid

```
         /\
        /  \  E2E Tests (5%)
       /    \  - Full workflow tests
      /------\  - Selenium/Playwright
     /        \ Integration Tests (25%)
    /          \ - API endpoint tests
   /            \ - Database tests
  /--------------\ Unit Tests (70%)
 /                \ - Service layer tests
/------------------\ - Statistical tests
```

### Test Structure
```
tests/
├── unit/
│   ├── test_core/
│   │   ├── test_validation.py
│   │   ├── test_config.py
│   │   └── test_exceptions.py
│   ├── test_statistical_core/
│   │   ├── test_normality.py
│   │   ├── test_comparison.py
│   │   └── test_anova.py
│   ├── test_services/
│   │   ├── test_dataset_service.py
│   │   ├── test_analysis_service.py
│   │   └── test_task_service.py
│   └── test_agents/
│       ├── test_normality_agent.py
│       └── test_summarizer_agent.py
├── integration/
│   ├── test_api/
│   │   ├── test_datasets_routes.py
│   │   ├── test_analysis_routes.py
│   │   └── test_tasks_routes.py
│   ├── test_workflow/
│   │   ├── test_statmate_flow.py
│   │   └── test_edge_cases.py
│   └── test_database/
│       ├── test_models.py
│       └── test_migrations.py
├── e2e/
│   ├── test_full_workflow.py
│   ├── test_scheduled_tasks.py
│   └── test_ui_integration.py
├── fixtures/
│   ├── sample_datasets/
│   │   ├── continuous_paired.csv
│   │   ├── continuous_independent.csv
│   │   └── categorical.csv
│   └── conftest.py
└── performance/
    ├── test_large_datasets.py
    └── test_concurrent_requests.py
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test type
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest tests/ --cov=statmate --cov-report=html

# Run performance tests
pytest tests/performance/ --benchmark-only

# Run in parallel
pytest tests/ -n auto
```

### Test Coverage Goals
- **Overall**: 80%+ coverage
- **Core modules**: 90%+ coverage
- **API routes**: 85%+ coverage
- **Services**: 85%+ coverage

---

## 🚀 Deployment Architecture

### Infrastructure Options

#### Option 1: Cloud VPS (DigitalOcean, Linode, Hetzner)

**Recommended for MVP/Small Scale**

```
┌─────────────────────────────────────┐
│         Load Balancer (Nginx)       │
│         SSL/TLS (Let's Encrypt)     │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼────┐
│ FastAPI│      │Streamlit│
│  :8000 │      │  :8501  │
└───┬────┘      └─────────┘
    │
┌───▼────────┐
│  SQLite /  │
│ PostgreSQL │
└────────────┘
```

**Setup Commands:**
```bash
# On VPS
apt update && apt upgrade -y
apt install -y docker.io docker-compose nginx certbot python3-certbot-nginx

# Clone repo
cd /opt
git clone https://github.com/yourusername/statmate-ai.git
cd statmate-ai

# Set up environment
cp config/.env.example .env
nano .env  # Add production secrets

# Start services
docker-compose -f docker/docker-compose.prod.yml up -d

# Configure Nginx
certbot --nginx -d api.statmate.ai -d app.statmate.ai
```

**Nginx Configuration:**
```nginx
# /etc/nginx/sites-available/statmate-api
server {
    listen 80;
    server_name api.statmate.ai;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# /etc/nginx/sites-available/statmate-ui
server {
    listen 80;
    server_name app.statmate.ai;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### Option 2: AWS (Scalable Production)

**Architecture:**
```
Route 53 (DNS)
    │
    ▼
CloudFront (CDN)
    │
    ▼
Application Load Balancer
    │
    ├─────────┬─────────┐
    ▼         ▼         ▼
  ECS       ECS       ECS
(FastAPI) (FastAPI) (Streamlit)
    │
    ▼
RDS PostgreSQL
    │
    ▼
S3 (File Storage)
```

**Services:**
- **ECS Fargate**: Container orchestration
- **RDS PostgreSQL**: Managed database
- **S3**: File storage for datasets/results
- **CloudWatch**: Logging and monitoring
- **ALB**: Load balancing
- **Route 53**: DNS management
- **ACM**: SSL certificates

#### Option 3: Kubernetes (Large Scale)

**Setup:**
```bash
# Install kubectl, helm
curl -LO https://dl.k8s.io/release/v1.28.0/bin/linux/amd64/kubectl
chmod +x kubectl && mv kubectl /usr/local/bin/

# Deploy to cluster
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/ui-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## 📊 Monitoring & Logging

### Application Monitoring

#### 1. Prometheus + Grafana

**Metrics to Track:**
- Request rate (requests/sec)
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- Active analyses running
- Queue length (scheduled tasks)
- Database connections
- Memory usage
- CPU usage

**Setup:**
```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
```

#### 2. Health Check Endpoints

```python
# statmate/api/routes/health.py
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "database": await check_db_connection(),
        "scheduler": check_scheduler_status()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    return generate_latest()
```

### Logging Strategy

#### Structured Logging (JSON format)
```python
# config/logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "data/logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

#### Log Aggregation (ELK Stack or Loki)

**Option A: Elasticsearch + Logstash + Kibana**
```yaml
services:
  elasticsearch:
    image: elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
  
  logstash:
    image: logstash:8.10.0
    volumes:
      - ./logstash/config:/usr/share/logstash/pipeline
  
  kibana:
    image: kibana:8.10.0
    ports:
      - "5601:5601"
```

**Option B: Grafana Loki (Lighter)**
```yaml
services:
  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
  
  promtail:
    image: grafana/promtail:2.9.0
    volumes:
      - /var/log:/var/log
      - ./data/logs:/app/logs
```

### Error Tracking

#### Sentry Integration
```python
# statmate/api/main.py
import sentry_sdk

sentry_sdk.init(
    dsn="https://your-sentry-dsn",
    environment="production",
    traces_sample_rate=0.1,
)
```

---

## 💾 Backup & Recovery

### Database Backup Strategy

#### Automated Daily Backups
```bash
#!/bin/bash
# scripts/backup_db.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/statmate"
DB_FILE="database/statmate.db"

# SQLite backup
sqlite3 $DB_FILE ".backup '$BACKUP_DIR/statmate_$TIMESTAMP.db'"

# Compress
gzip $BACKUP_DIR/statmate_$TIMESTAMP.db

# Upload to S3
aws s3 cp $BACKUP_DIR/statmate_$TIMESTAMP.db.gz s3://statmate-backups/

# Keep only last 30 days locally
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
```

#### Cron Job
```bash
# Add to crontab
0 2 * * * /opt/statmate-ai/scripts/backup_db.sh
```

### File Storage Backup
```bash
#!/bin/bash
# scripts/backup_files.sh

# Sync data directory to S3
aws s3 sync /opt/statmate-ai/data/ s3://statmate-backups/data/ \
    --exclude "*.tmp" \
    --delete
```

### Disaster Recovery Plan

#### Recovery Time Objective (RTO): 2 hours
#### Recovery Point Objective (RPO): 24 hours

**Recovery Steps:**
1. Provision new server
2. Install Docker & dependencies
3. Restore from latest backup
4. Update DNS records
5. Verify functionality

```bash
# scripts/restore.sh
#!/bin/bash

# Download latest backup
aws s3 cp s3://statmate-backups/$(aws s3 ls s3://statmate-backups/ | sort | tail -n 1 | awk '{print $4}') ./

# Restore database
gunzip -c statmate_*.db.gz > database/statmate.db

# Restore files
aws s3 sync s3://statmate-backups/data/ /opt/statmate-ai/data/

# Restart services
docker-compose up -d
```

---

## 🔐 Security & Compliance

### Security Checklist

#### Infrastructure Security
- ✅ Use HTTPS everywhere (TLS 1.3)
- ✅ Configure firewall (UFW/iptables)
- ✅ Disable root SSH login
- ✅ Use SSH keys (no passwords)
- ✅ Keep systems updated (unattended-upgrades)
- ✅ Rate limiting on API endpoints
- ✅ DDoS protection (CloudFlare/AWS Shield)

#### Application Security
- ✅ Input validation on all endpoints
- ✅ Parameterized SQL queries (SQLAlchemy ORM)
- ✅ File upload restrictions (size, type)
- ✅ Secure session management
- ✅ JWT token expiration
- ✅ CORS configuration
- ✅ Secrets in environment variables (not code)
- ✅ Dependency vulnerability scanning

#### Data Security
- ✅ Encrypt data at rest (database encryption)
- ✅ Encrypt data in transit (TLS)
- ✅ Anonymize sensitive data in logs
- ✅ Regular backups
- ✅ Access controls (RBAC if multi-user)

### Compliance (if handling medical data)

#### HIPAA Compliance (US)
- Data encryption (at rest & in transit)
- Access controls & audit logs
- Business Associate Agreements
- Regular security assessments

#### GDPR Compliance (EU)
- Data minimization
- Right to erasure (delete user data)
- Data portability
- Privacy by design

### Security Scanning

```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Run Bandit security linter
        run: |
          pip install bandit
          bandit -r statmate/ -f json -o bandit-report.json
      
      - name: Upload to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

---

## 📈 Scaling Strategy

### Vertical Scaling (Single Server)

**Start with:** 2 CPU / 4GB RAM
**Scale to:** 8 CPU / 16GB RAM

**When to scale:** 
- CPU usage > 70% sustained
- Memory usage > 80%
- Response time > 2s

### Horizontal Scaling (Multiple Servers)

#### Phase 1: Separate Services
```
Server 1: FastAPI
Server 2: Streamlit UI
Server 3: PostgreSQL
```

#### Phase 2: Load Balanced API
```
      Load Balancer
         /    \
    API-1    API-2
         \    /
       PostgreSQL
```

#### Phase 3: Full HA Setup
```
    CloudFront CDN
         │
    Load Balancer
    ┌────┴────┐
    │         │
  API-1     API-2
    │         │
    └────┬────┘
         │
    RDS Primary
         │
    RDS Replica
```

### Caching Strategy

#### Redis for Session/Result Caching
```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

```python
# Cache analysis results
@cache(ttl=3600)  # 1 hour
async def get_analysis_result(analysis_id: str):
    return db.query(Analysis).filter_by(id=analysis_id).first()
```

### Database Optimization

#### Indexes
```python
# Add indexes to frequently queried columns
Index('idx_dataset_upload_timestamp', Dataset.upload_timestamp)
Index('idx_analysis_dataset_id', Analysis.dataset_id)
Index('idx_analysis_status', Analysis.status)
Index('idx_task_next_run', ScheduledTask.next_run)
```

#### Connection Pooling
```python
# config/database.py
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

### Background Job Processing

**For heavy workloads, switch to Celery + Redis:**
```python
# Async task processing
from celery import Celery

app = Celery('statmate', broker='redis://localhost:6379')

@app.task
def run_analysis_task(dataset_id: str, columns: list):
    # Long-running analysis
    result = workflow.run(...)
    return result
```

---

## 📅 Maintenance Schedule

### Daily
- Monitor error logs
- Check system resources
- Review failed tasks

### Weekly
- Review application metrics
- Check disk space
- Update dependencies (security patches)

### Monthly
- Full system backup verification
- Performance optimization review
- Security audit
- Database maintenance (vacuum, analyze)

### Quarterly
- Disaster recovery drill
- Penetration testing
- Dependency updates (major versions)
- Infrastructure cost optimization

---

## 🚨 Incident Response Plan

### Severity Levels

**P0 - Critical** (Response: Immediate)
- Service completely down
- Data breach
- Security vulnerability exploited

**P1 - High** (Response: < 1 hour)
- Major feature broken
- Performance severely degraded
- Scheduled tasks failing

**P2 - Medium** (Response: < 4 hours)
- Minor feature broken
- Non-critical errors
- UI issues

**P3 - Low** (Response: Next business day)
- Cosmetic issues
- Enhancement requests

### On-Call Rotation
- Primary: 24/7 availability
- Secondary: Backup coverage
- Escalation: Manager/CTO

### Incident Response Workflow
1. **Detect**: Monitoring alert
2. **Acknowledge**: On-call responds
3. **Assess**: Determine severity
4. **Mitigate**: Quick fix or rollback
5. **Resolve**: Permanent fix
6. **Document**: Post-mortem report

---

## 📝 Runbooks

### Common Operations

#### Deploy New Version
```bash
ssh production-server
cd /opt/statmate-ai
git pull origin main
docker-compose build
docker-compose up -d
docker-compose exec api alembic upgrade head
```

#### Rollback Deployment
```bash
docker-compose down
git checkout <previous-commit>
docker-compose up -d
```

#### Check Application Logs
```bash
docker-compose logs -f --tail=100 api
docker-compose logs -f --tail=100 ui
```

#### Database Migration
```bash
# Create migration
docker-compose exec api alembic revision --autogenerate -m "description"

# Apply migration
docker-compose exec api alembic upgrade head

# Rollback migration
docker-compose exec api alembic downgrade -1
```

#### Clear Stuck Tasks
```bash
docker-compose exec api python scripts/clear_stuck_tasks.py
```

---

## 🎯 Success Metrics

### Performance KPIs
- API response time < 500ms (p95)
- Analysis completion time < 60s
- Uptime > 99.5%
- Error rate < 1%

### Business KPIs
- Daily active users
- Analyses run per day
- User retention rate
- Average session duration

### Cost Optimization
- Infrastructure cost per user
- Storage cost per dataset
- API cost per analysis

---

**DevOps Plan Version:** 1.0
**Last Updated:** October 2025
**Owner:** DevOps Team

---

## 📞 Contact & Escalation

- **DevOps Lead**: devops@statmate.ai
- **Security Issues**: security@statmate.ai
- **Emergency Hotline**: +1-XXX-XXX-XXXX
- **Status Page**: https://status.statmate.ai

