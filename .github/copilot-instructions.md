# EmergentFolds - AI Coding Guidelines

## Project Overview

**EmergentFolds** is a protein structure prediction platform for screening, learning, and research.

**Live Site**: https://emergentfolds.com

### What This Platform Does

- **Fast structure prediction** — Submit a sequence, get a 3D structure in seconds to minutes
- **Protein screening** — Rapidly evaluate candidates before expensive experiments
- **Interactive visualization** — Explore structures with 3D viewer, multiple display modes
- **Batch processing** — Campaign mode for systematic multi-protein analysis
- **Aggregation screening** — Identify aggregation-prone regions
- **API access** — REST API for research pipeline integration

### System Architecture

| Component | Directory | Purpose |
|-----------|-----------|---------|
| Web Interface | `frontend/` + `backend/` | Full-stack app for interactive predictions |
| Prediction Engine | `ubf_protein/` | Multi-agent physics-based simulation |
| Quantum Module | `src/` | QCPP quantum coherence integration |

---

## Web Interface (`frontend/` + `backend/`)

### Technology Stack

**Frontend**: React 19, TypeScript, Material-UI 7, Vite, Socket.IO, Zustand, NGL Viewer
**Backend**: FastAPI, Celery, Redis, PostgreSQL, Python-SocketIO, JWT auth
**Infrastructure**: Docker Compose, Nginx (production)

### Key Features
- Multi-step prediction wizard
- Real-time WebSocket monitoring
- 3D protein visualization (NGL Viewer)
- Campaign management for batch predictions
- Work sessions with sharing
- JWT authentication and security

### Documentation
- `docs/SETUP.md` — Installation and configuration
- `docs/USER_GUIDE.md` — Platform features
- `docs/API.md` — REST API and WebSocket reference

---

## Prediction Engine (`ubf_protein/`)

### How It Works

The engine uses multi-agent physics-based simulation:

1. **Multi-agent exploration** — Autonomous agents search conformational space
2. **Energy optimization** — Molecular mechanics force field guides folding
3. **QCPP integration** — Quantum coherence provides physics-based feedback
4. **Collective learning** — Agents share discoveries via memory system

### Architecture (SOLID + Mapless Design)

**Core Principles**:
- SOLID architecture with interface-driven design
- O(1) move generation (no spatial maps or pathfinding)
- Pure Python for PyPy compatibility (2-5x speedup)
- Immutable data models
- Graceful degradation for non-critical failures

**Key Components**:
- `prediction_runner.py` — Unified prediction interface (use this for all predictions)
- `multi_agent_coordinator.py` — Parallel agent exploration
- `energy_function.py` — Molecular mechanics (bonds, angles, dihedral, VDW, electrostatic, H-bond)
- `qcpp_integration.py` — Real-time quantum physics feedback
- `checkpoint.py` — Save/restore exploration state

### Using PredictionRunner

**All predictions must use `PredictionRunner`** — it is the single source of truth.

```python
from ubf_protein.prediction_runner import PredictionRunner, PredictionConfig

config = PredictionConfig(
    sequence="MQIFVKT...",
    agents=10,
    iterations=500,
    qcpp_config="default",  # 'default', 'high_performance', 'high_accuracy', 'none'
)

runner = PredictionRunner(config)
results = runner.run(progress_callback=my_callback)
```

### Running Predictions

```bash
# Single protein
python test_protein.py --sequence MQIFVKT...
python test_protein.py --pdb 1UBQ

# Batch testing
python systematic_protein_testing.py --count 10
```

### Performance Characteristics
- Move evaluation: <2ms typical
- Memory retrieval: <10μs typical
- Small protein (<100 residues): ~30 seconds
- Large protein (200+ residues): 2-5 minutes
- PyPy speedup: 2-5x vs CPython

---

## QCPP System (`src/`)

### Purpose
Physics-based stability analysis using quantum coherence and golden ratio patterns.

### Key Components
- `QuantumCoherenceProteinPredictor` — QCP values, field coherence, THz spectra
- `QCProteinPipeline` — Workflow orchestration

### QCP Formula
QCP scores are calculated using a proprietary formula integrating structural hierarchy, neighbor relationships, and residue properties with golden ratio scaling. The formula produces stability metrics that guide exploration.

### Golden Ratio Integration
- Phi angle: `2 * π / φ ≈ 137.5°`
- Phi harmonics: `[φ⁰, φ¹, φ², φ³, φ⁴]` for frequency calculations
- Distance matching: `3.8 * φ^n` Å

---

## Development Workflow

### Quick Start

```bash
# Docker (recommended)
docker compose up -d

# Or run manually
cd frontend && npm install && npm run dev
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
```

### Running Tests

```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test

# Engine tests
pytest ubf_protein/tests/
```

### File Organization

```
PP/
├── frontend/           # React web application
├── backend/            # FastAPI server + Celery workers
├── ubf_protein/        # Core prediction engine
├── src/                # QCPP quantum coherence module
├── docs/               # Documentation
├── test_protein.py     # CLI for predictions
└── docker-compose.yml  # Container orchestration
```

---

## Coding Conventions

### Python (Backend + Engine)

- Type hints on all public methods
- Frozen dataclasses for models
- Interface-driven design (dependency inversion)
- Graceful error handling (log and continue for non-critical failures)
- Pure Python in `ubf_protein/` (no NumPy for PyPy compatibility)

### TypeScript (Frontend)

- Functional components with hooks
- Zustand for state management
- React Query for server state
- Material-UI for components

### Common Pitfalls

**Engine (`ubf_protein/`)**:
- Don't use NumPy — pure Python for PyPy
- Immutable models — use `replace()` or create new instances
- O(1) moves — never use spatial maps or N² algorithms
- Always use `PredictionRunner` for predictions

**Backend**:
- Use `prediction_tasks_v2.py` (not deprecated `prediction_tasks.py`)
- JWT tokens required for all authenticated endpoints
- WebSocket events for real-time updates

**QCPP**:
- Check `res.has_id('CA')` before accessing CA coordinates
- Handle DSSP failures with fallback SS calculation

---

## Dependencies

### Web Interface
```bash
# Frontend
cd frontend && npm install
# Node.js ≥18

# Backend
cd backend && pip install -r requirements.txt
# Python ≥3.8, Docker for Redis/PostgreSQL
```

### Prediction Engine
```bash
pip install -r ubf_protein/requirements.txt
# Pure Python, PyPy ≥3.8 recommended
```

### QCPP
```bash
pip install -e .
# numpy, scipy, pandas, biopython, matplotlib, scikit-learn
# Python ≤3.12 recommended for BioPython wheels on Windows
```

---

## Configuration Presets

| Preset | Use Case | Speed |
|--------|----------|-------|
| `fast` | Quick screening | ~30 seconds |
| `balanced` | Default predictions | ~1-2 minutes |
| `high_accuracy` | Detailed analysis | ~3-5 minutes |

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/SETUP.md` | Installation and configuration |
| `docs/USER_GUIDE.md` | Platform features |
| `docs/API.md` | REST API reference |
| `docs/CLI_REFERENCE.md` | Command-line tools |
| `ubf_protein/README.md` | Engine documentation |
| `ubf_protein/API.md` | Engine API reference |

---

## Status

- **Web Interface**: Production-ready, live at emergentfolds.com
- **Prediction Engine**: Production-ready, tested on 45+ proteins
- **QCPP Integration**: Complete

**Coming Soon**: Machine learning integration for faster, higher-accuracy predictions
