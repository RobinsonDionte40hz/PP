# EmergentFolds Architecture Refactoring Plan

## Executive Summary

This document outlines the architectural drift from SOLID principles and proposes a refactoring plan to properly separate the **Frontend**, **Backend**, and **Prediction Engine** (tool) into independent, well-isolated environments.

---

## Current State Analysis

### 1. Identified SOLID Violations

#### **Single Responsibility Principle (SRP) Violations**

| Location | Issue |
|----------|-------|
| `backend/requirements.txt` | Mixes backend dependencies with UBF/QCPP dependencies |
| `docker/worker/Dockerfile` | Installs 3 separate requirement files in one image |
| Root `setup.py` | Defines a monolithic package mixing web and science concerns |
| Root directory | Contains 15+ Python scripts that should be in specific modules |

#### **Open/Closed Principle (OCP) Violations**

| Location | Issue |
|----------|-------|
| `prediction_tasks_v2.py` | Direct imports of `ubf_protein` internals instead of interface |
| `backend/app/tasks/` | Hardcoded path manipulation to find `ubf_protein` |

#### **Dependency Inversion Principle (DIP) Violations**

| Location | Issue |
|----------|-------|
| `prediction_runner.py` | Directly imports `src.protein_predictor` (QCPP) |
| `backend/` | Depends on concrete `ubf_protein` classes, not interfaces |
| `docker-compose.yml` | Worker volume-mounts `ubf_protein/` and `src/` directly |

### 2. Environment Coupling Issues

```
Current Architecture (Tightly Coupled):

┌─────────────────────────────────────────────────────────────┐
│                     ROOT PROJECT                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  frontend/  │  │  backend/   │  │  ubf_protein/ + src/│  │
│  │             │  │             │  │                     │  │
│  │  (React)    │──│  (FastAPI)  │──│  (Prediction Engine)│  │
│  │             │  │             │  │                     │  │
│  └─────────────┘  └──────┬──────┘  └──────────┬──────────┘  │
│                          │                     │             │
│                          └─────────────────────┘             │
│                         sys.path hacks + volume mounts       │
└─────────────────────────────────────────────────────────────┘
```

**Problems:**
1. Backend directly imports from `ubf_protein/` using `sys.path.insert()`
2. Docker volumes mount source code across container boundaries
3. Single `.venv` at root tries to satisfy all 3 environments
4. No clear API boundary between backend and prediction engine
5. Root directory cluttered with test scripts that belong in modules

### 3. Dependency Conflicts

| Environment | Python Version | Key Dependencies |
|-------------|----------------|------------------|
| Frontend | N/A (Node.js) | React 19, MUI 7 |
| Backend | 3.8+ | FastAPI, Celery, SQLAlchemy |
| UBF Protein | 3.8+ (PyPy compatible) | **Pure Python only** |
| QCPP (src/) | ≤3.12 | NumPy, SciPy, BioPython |

**Conflict:** Backend `requirements.txt` includes NumPy/SciPy which breaks PyPy compatibility for the engine.

---

## Proposed Architecture

```
Target Architecture (Properly Decoupled):

┌────────────────────────────────────────────────────────────────────────┐
│                          MONOREPO ROOT                                  │
│                                                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │    packages/     │  │    packages/     │  │      packages/       │ │
│  │    frontend/     │  │    backend/      │  │   prediction-engine/ │ │
│  │                  │  │                  │  │                      │ │
│  │  - node_modules/ │  │  - .venv/        │  │  - .venv/            │ │
│  │  - package.json  │  │  - requirements/ │  │  - ubf_protein/      │ │
│  │  - src/          │  │  - app/          │  │  - qcpp/             │ │
│  │                  │  │                  │  │  - setup.py          │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘ │
│           │                     │                       │             │
│           │                     │    REST/gRPC API      │             │
│           │ HTTP                └───────────────────────┘             │
│           │                                                           │
│           └──────────────── Nginx Reverse Proxy ──────────────────────│
└────────────────────────────────────────────────────────────────────────┘
```

---

## Refactoring Plan

### Phase 1: Establish Package Boundaries (Week 1-2)

#### 1.1 Create Package Structure

```
PP/
├── packages/
│   ├── frontend/              # React app (moved from ./frontend)
│   │   ├── .env.example
│   │   ├── package.json
│   │   └── src/
│   │
│   ├── backend/               # FastAPI server (moved from ./backend)
│   │   ├── .venv/             # Isolated virtual environment
│   │   ├── requirements/
│   │   │   ├── base.txt       # Core FastAPI, Celery
│   │   │   ├── dev.txt        # Testing, linting
│   │   │   └── prod.txt       # Production extras
│   │   ├── app/
│   │   └── setup.py           # Backend as installable package
│   │
│   └── prediction-engine/     # Core prediction (from ./ubf_protein + ./src)
│       ├── .venv/             # Isolated virtual environment
│       ├── prediction_engine/
│       │   ├── core/          # From ubf_protein/
│       │   ├── qcpp/          # From src/
│       │   └── api/           # Public interface module
│       │       ├── __init__.py
│       │       ├── runner.py
│       │       └── schemas.py
│       ├── setup.py
│       └── requirements/
│           ├── core.txt       # Pure Python (PyPy compatible)
│           └── qcpp.txt       # NumPy/SciPy (CPython only)
│
├── docker/                    # Container definitions
├── docs/                      # Documentation
├── scripts/                   # Development/deployment scripts
└── docker-compose.yml         # Orchestration
```

#### 1.2 Create Engine API Interface

```python
# packages/prediction-engine/prediction_engine/api/__init__.py
"""
Public API for Prediction Engine.

This is the ONLY module that backend should import.
All other modules are internal implementation details.
"""

from .runner import PredictionRunner, PredictionConfig, PredictionResults
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    ProgressUpdate,
    ScreeningRequest,
    ScreeningResponse,
)

__all__ = [
    'PredictionRunner',
    'PredictionConfig', 
    'PredictionResults',
    'PredictionRequest',
    'PredictionResponse',
    'ProgressUpdate',
    'ScreeningRequest',
    'ScreeningResponse',
]
```

### Phase 2: Decouple Backend from Engine (Week 2-3)

#### 2.1 Backend Should Only Import Public API

```python
# packages/backend/app/tasks/prediction_tasks.py

# BEFORE (current - tight coupling)
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.rmsd_calculator import RMSDCalculator

# AFTER (proposed - interface-only)
from prediction_engine.api import PredictionRunner, PredictionConfig
```

#### 2.2 Install Engine as Package

```python
# packages/backend/setup.py
setup(
    name="emergentfolds-backend",
    install_requires=[
        "fastapi>=0.115.0",
        "celery>=5.4.0",
        # Engine as dependency (not source import)
        "prediction-engine @ file:../prediction-engine",
    ],
)
```

#### 2.3 Clean Backend Requirements

```txt
# packages/backend/requirements/base.txt
# Backend-only dependencies - NO scientific computing

fastapi==0.115.0
uvicorn[standard]==0.32.0
celery==5.4.0
redis==5.2.0
sqlalchemy==2.0.36
alembic==1.13.1
psycopg2-binary==2.9.10
python-socketio==5.11.4
python-multipart==0.0.12
pydantic[email]>=2.0.0
pydantic-settings>=2.0.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.1
httpx==0.27.2
requests==2.32.3
slowapi==0.1.9

# Prediction Engine (installed from local package)
-e ../prediction-engine
```

### Phase 3: Containerization Cleanup (Week 3-4)

#### 3.1 Prediction Engine as Standalone Service

```yaml
# docker-compose.yml (refactored)
services:
  frontend:
    build:
      context: ./packages/frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"

  backend:
    build:
      context: ./packages/backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
    # NO volume mounts for ubf_protein or src

  prediction-worker:
    build:
      context: ./packages/prediction-engine
      dockerfile: Dockerfile.worker
    depends_on:
      - redis
      - backend
    # Self-contained - no external volume mounts
```

#### 3.2 Engine Dockerfile (Self-Contained)

```dockerfile
# packages/prediction-engine/Dockerfile.worker
FROM python:3.11-slim

WORKDIR /app

# Install the prediction engine as a package
COPY . .
RUN pip install -e .[worker]

# Run Celery worker
CMD ["celery", "-A", "prediction_engine.worker", "worker", "--loglevel=info"]
```

### Phase 4: Root Cleanup (Week 4)

#### 4.1 Move Scattered Scripts

| Current Location | New Location | Purpose |
|-----------------|--------------|---------|
| `test_protein.py` | `packages/prediction-engine/cli/test_protein.py` | CLI tool |
| `systematic_protein_testing.py` | `packages/prediction-engine/cli/systematic_test.py` | Batch testing |
| `comprehensive_benchmark.py` | `packages/prediction-engine/benchmarks/` | Performance |
| `cleanup_*.py` | `scripts/maintenance/` | Development |
| `validate_*.py` | `packages/prediction-engine/validation/` | Validation |
| Root `setup.py` | Remove (replaced by package-level setup.py) | - |

#### 4.2 Clean Root Directory

```
PP/
├── packages/           # All application code
├── docker/             # Docker configurations
├── docs/               # Documentation
├── scripts/            # Development/deployment scripts
├── .github/            # GitHub workflows
├── docker-compose.yml
├── docker-compose.prod.yml
├── README.md
├── LICENSE
└── .gitignore
```

---

## Implementation Checklist

### Immediate Actions (This Sprint)

- [ ] Create `packages/` directory structure
- [x] Create `prediction_engine/api/` public interface module (`ubf_protein/api/`)
- [ ] Move `ubf_protein/` to `packages/prediction-engine/prediction_engine/core/`
- [ ] Move `src/` to `packages/prediction-engine/prediction_engine/qcpp/`
- [x] Create `packages/prediction-engine/setup.py` (`ubf_protein/setup.py`)
- [x] Update backend imports to use public API only
- [x] Split `backend/requirements.txt` into base/dev/prod

### Short-term (Next 2 Sprints)

- [ ] Move `frontend/` to `packages/frontend/`
- [ ] Move `backend/` to `packages/backend/`
- [ ] Create isolated `.venv` for each package
- [ ] Update Docker configurations
- [ ] Move root scripts to appropriate packages
- [ ] Update CI/CD workflows

### Long-term (Future)

- [ ] Consider gRPC for engine communication (optional)
- [ ] Add package versioning and changelog
- [ ] Set up internal PyPI for engine distribution
- [ ] Add integration tests for package boundaries

---

## Benefits of Refactoring

### 1. **Independent Development**
- Frontend team works in `packages/frontend/` with Node.js only
- Backend team works in `packages/backend/` with FastAPI
- Science team works in `packages/prediction-engine/` with pure Python

### 2. **Isolated Dependencies**
- No more NumPy in backend requirements
- PyPy-compatible engine without breaking QCPP
- Clear dependency trees for each package

### 3. **Better Testing**
- Unit tests per package
- Integration tests at package boundaries
- No sys.path hacks in test files

### 4. **Easier Deployment**
- Each package builds independently
- Smaller Docker images
- Clear upgrade paths

### 5. **SOLID Compliance**
- **SRP**: Each package has one responsibility
- **OCP**: Add features via API without modifying internals
- **LSP**: All runners implement same interface
- **ISP**: Backend only sees what it needs (`api/` module)
- **DIP**: Backend depends on abstractions, not concrete classes

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing predictions | Keep old task files during transition, feature flag |
| Docker build failures | Test containers in staging environment first |
| Import path confusion | Update all imports systematically with grep |
| CI/CD breakage | Update workflows incrementally |

---

## Current Implementation Status

> **Last Updated**: December 7, 2025 - Phase 3 partial complete

### ✅ Phase 1: Complete

| Task | Status | Notes |
|------|--------|-------|
| Create public API module | ✅ Done | `ubf_protein/api/` with interfaces, schemas, runner |
| Create interfaces | ✅ Done | `IPredictionRunner`, `IScreener`, `IQCPPAdapter` |
| Create data schemas | ✅ Done | `PredictionConfig`, `PredictionResults`, `AggregationRisk` |
| Update backend imports | ✅ Done | All tasks now import from `ubf_protein.api` |
| Split backend requirements | ✅ Done | `base.txt`, `dev.txt`, `prod.txt` |
| Split engine requirements | ✅ Done | `core.txt`, `qcpp.txt`, `worker.txt` |

### ✅ Phase 2: Complete

| Task | Status | Notes |
|------|--------|-------|
| Create setup.py for engine | ✅ Done | `ubf_protein/setup.py` with extras |
| Create CLI entry point | ✅ Done | `ubf_protein/cli.py` with predict-protein command |
| Update worker Dockerfile | ✅ Done | Now uses `pip install -e /packages/ubf_protein[worker]` |
| Update backend Dockerfile | ✅ Done | Now uses `pip install -e /packages/ubf_protein` |
| Update docker-compose.yml | ✅ Done | Updated volume paths to `/packages/` |
| Update docker-compose.prod.yml | ✅ Already OK | Doesn't mount source volumes |
| Update sys.path fallbacks | ✅ Done | Now checks `/packages/ubf_protein` first |

### ✅ Phase 3: Complete

| Task | Status | Notes |
|------|--------|-------|
| Update root test scripts | ✅ Done | `test_protein.py`, `comprehensive_benchmark.py`, `systematic_protein_testing.py` now use public API |
| Add utility functions to API | ✅ Done | `get_optimal_settings()`, `get_quick_test_settings()` |
| Add API integration tests | ✅ Done | `ubf_protein/tests/test_public_api.py` with 29 tests |
| Create CI workflow | ✅ Done | `.github/workflows/tests.yml` for engine, backend, Docker |
| Move to packages/ structure | ⏸️ Deferred | Current structure works fine |

### ✅ Phase 4: Complete (Root Cleanup)

| Task | Status | Notes |
|------|--------|-------|
| Move cleanup scripts | ✅ Done | `scripts/maintenance/` |
| Move benchmark scripts | ✅ Done | `scripts/benchmarks/` |
| Move experiment scripts | ✅ Done | `scripts/experiments/` |
| Move validation scripts | ✅ Done | `scripts/validation/` |
| Move setup/deploy scripts | ✅ Done | `scripts/setup/` |
| Move documentation | ✅ Done | All `.md` guides to `docs/` |
| Handle root setup.py | ✅ Done | Moved to `src/setup.py` (QCPP) |
| Move requirements | ✅ Done | `requirements_qcpp.txt` → `ubf_protein/requirements/` |
| Create test_protein wrapper | ✅ Done | Thin wrapper at root for backward compat |

### Files Created/Modified

**New Files:**
- `ubf_protein/api/__init__.py` - Public API entry point
- `ubf_protein/api/interfaces.py` - Abstract interfaces
- `ubf_protein/api/schemas.py` - Data classes
- `ubf_protein/api/runner.py` - PredictionRunner wrapper + utility functions
- `ubf_protein/api/screening.py` - AggregationScreener wrapper
- `ubf_protein/api/exporters.py` - PDB/JSON/CIF exporters
- `ubf_protein/setup.py` - Package installation
- `ubf_protein/cli.py` - CLI entry point
- `ubf_protein/requirements/core.txt` - Pure Python deps
- `ubf_protein/requirements/qcpp.txt` - NumPy/SciPy deps
- `ubf_protein/requirements/worker.txt` - Full worker deps
- `backend/requirements/base.txt` - Backend core deps
- `backend/requirements/dev.txt` - Development deps
- `backend/requirements/prod.txt` - Production deps

**Modified Files:**
- `backend/app/tasks/prediction_tasks_v2.py` - Uses public API
- `backend/app/tasks/screening_tasks.py` - Uses public API
- `docker/worker/Dockerfile` - Installs package at `/packages/`
- `docker/backend/Dockerfile` - Installs package at `/packages/`
- `docker-compose.yml` - Updated volume paths
- `test_protein.py` - Now imports from `ubf_protein.api`
- `comprehensive_benchmark.py` - Now imports from `ubf_protein.api`
- `systematic_protein_testing.py` - Now imports from `ubf_protein.api`
- `.github/copilot-instructions.md` - Added SOLID principles documentation
- `.github/workflows/tests.yml` - NEW: CI workflow for testing

---

## Questions to Resolve

1. Should prediction engine be a microservice with its own API, or stay as a library?
2. Do we need versioning between backend and engine?
3. Should QCPP be a separate optional package?

---

## Next Steps

1. Review and approve this plan
2. Create GitHub issues for each phase
3. Begin Phase 1 implementation
4. Set up branch protection for `packages/` directory
