# EmergentFolds - Quick Reference

## Three Ways to Use the Platform

### 1. 🌐 Live Platform (Recommended)
**https://emergentfolds.com**

- Create an account and start predicting immediately
- Full features: predictions, 3D visualization, real-time monitoring
- Aggregation screening, campaigns, work sessions
- No installation required

**This is the primary way to use the platform.**

---

### 2. 🖥️ Command Line
**For automation and batch processing**

```bash
# Single prediction
python test_protein.py --pdb 1UBQ

# Batch processing
python systematic_protein_testing.py --count 10

# Quick test
python test_protein.py --quick
```

**Documentation**: `docs/CLI_REFERENCE.md`, `QUICKSTART.md`

---

### 3. 🛠️ Local Development
**For developers**

```bash
# Start locally
docker compose up -d
# or: START_ALL.bat (Windows)

# Access at:
# Frontend: http://localhost:3000
# API: http://localhost:8000/docs
```

**Documentation**: `docs/SETUP.md`, `docs/DEVELOPER_GUIDE.md`

---

### 4. 📊 Python API
**For custom integration**

```python
from ubf_protein.prediction_runner import PredictionRunner, PredictionConfig

config = PredictionConfig(
    sequence="MQIFVKT...",
    agents=10,
    iterations=500,
)

runner = PredictionRunner(config)
results = runner.run()
```

**Documentation**: `ubf_protein/README.md`, `ubf_protein/API.md`

---

## 📁 Project Structure at a Glance

```
PP/
├── 🌐 FRONTEND & BACKEND (Web Interface v1.0.0)
│   ├── frontend/          # React + TypeScript UI
│   ├── backend/           # FastAPI + Celery API
│   ├── docker/            # Docker configurations
│   └── docs/              # Web interface documentation
│
├── 🖥️ COMMAND LINE TOOLS
│   ├── test_protein.py                    # Single protein testing
│   ├── systematic_protein_testing.py      # Systematic campaigns
│   └── run_analysis.py                    # QCPP-only analysis
│
├── 📊 CORE SYSTEMS
│   ├── ubf_protein/       # UBF multi-agent system
│   ├── src/               # QCPP quantum coherence
│   └── quantum_coherence_proteins/
│
└── 📚 DOCUMENTATION
    ├── README.md          # Main project overview
    ├── QUICKSTART.md      # Command-line quick start
    ├── RELEASE_NOTES.md   # v1.0.0 web interface release
    └── docs/              # Comprehensive guides
```

---

## ⚡ Quick Start by Use Case

### I want to predict a protein structure with a nice UI
```bash
START_ALL.bat
# Then open http://localhost:3000
```

### I want to test many proteins automatically
```bash
python systematic_protein_testing.py --start 0 --count 10
```

### I want to integrate predictions into my Python code
```python
# See ubf_protein/EXAMPLES.md for code examples
```

### I want to analyze quantum coherence patterns only
```bash
python run_analysis.py
```

---

## 📖 Documentation Quick Links

| What You Need | Where to Find It |
|---------------|------------------|
| Web interface setup | `docs/SETUP.md` |
| Web interface features | `docs/USER_GUIDE.md` |
| REST API reference | `docs/API.md` |
| Command-line usage | `README.md`, `QUICKSTART.md` |
| Python API examples | `ubf_protein/EXAMPLES.md` |
| Troubleshooting | `docs/TROUBLESHOOTING.md` |
| Release notes | `RELEASE_NOTES.md` |
| Developer guide | `docs/DEVELOPER_GUIDE.md` |

---

## 🎯 Feature Comparison

| Feature | Web Interface | Command Line | Python API |
|---------|--------------|--------------|------------|
| Real-time monitoring | ✅ Live charts | ❌ Log files only | ⚠️ Manual polling |
| 3D visualization | ✅ Interactive | ❌ No | ⚠️ External tools |
| Batch predictions | ✅ Campaigns | ✅ Systematic | ✅ Loops |
| History browser | ✅ Built-in | ❌ File system | ⚠️ Manual |
| Easy to use | ✅✅✅ | ✅✅ | ✅ |
| Automation | ⚠️ API calls | ✅✅✅ | ✅✅✅ |
| Custom workflows | ⚠️ Limited | ✅✅ | ✅✅✅ |

---

## 🔧 System Requirements

### Web Interface
- Docker 24+ and Docker Compose 2.20+ (recommended)
- OR Node.js 18+ and Python 3.8+ (manual setup)
- Modern web browser
- 4GB RAM minimum, 8GB recommended

### Command Line
- Python 3.8+ (3.12 recommended)
- BioPython (see installation notes for Windows)
- 2GB RAM minimum

### Python API
- Same as command line
- Optional: PyPy 3.8+ for 2-5x speedup

---

## 🆘 Getting Help

1. **Setup issues**: Check `docs/TROUBLESHOOTING.md`
2. **Usage questions**: See documentation for your chosen interface
3. **Bug reports**: Create issue with reproduction steps
4. **Feature requests**: Describe use case and desired outcome

---

## 📊 System Status (November 21, 2025)

- ✅ **Web Interface**: v1.0.0 PRODUCTION-READY
  - 141/152 tests passing (93%)
  - Full feature set complete
  - Docker deployment ready

- ✅ **UBF System**: RESEARCH-READY
  - 999/1016 tests passing (98.3%)
  - Quantum refinement working
  - 7.5-10Å RMSD typical (research phase)

- ✅ **QCPP System**: OPERATIONAL
  - Experimental validation complete
  - Physics-based stability analysis

---

**Choose your preferred interface and get started!** 🚀

For the most user-friendly experience, we recommend starting with the **Web Interface**.
