# EmergentFolds

**Fast protein structure prediction for screening, learning, and research.**

🌐 **[emergentfolds.com](https://emergentfolds.com)**

---

## What is EmergentFolds?

EmergentFolds is a web platform for predicting protein structures from amino acid sequences. Submit a sequence, get a 3D structure in seconds to minutes, and explore the results with interactive visualization.

The platform is designed for **practical research use**:
- **Screen** candidate proteins quickly before committing to expensive experiments
- **Learn** about protein structure through interactive 3D visualization
- **Test** hypotheses about mutations, domains, and structural features
- **Integrate** predictions into research pipelines via REST API

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Speed** | Predictions complete in 30 seconds to a few minutes |
| **Visualization** | Interactive 3D viewer with multiple display modes |
| **Batch Processing** | Campaign mode for systematic multi-protein screening |
| **Aggregation Screening** | Identify aggregation-prone regions |
| **Collaboration** | Share results via work sessions and export links |
| **API Access** | Full REST API for programmatic integration |

---

## Get Started

### Use the Live Platform

1. Go to **[emergentfolds.com](https://emergentfolds.com)**
2. Create an account
3. Click **New Prediction** → paste your sequence → submit
4. Watch the prediction run, then explore the 3D structure

No installation required.

### Run Locally (Optional)

```bash
git clone <repository-url>
cd PP
docker compose up -d
# Open http://localhost:3000
```

### Command Line

```bash
python test_protein.py --sequence MQIFVKTLTGKTITLEVEPS...
python test_protein.py --pdb 1UBQ
```

---

## Platform Features

### Structure Prediction
- Input: amino acid sequence or FASTA file
- Output: 3D atomic coordinates, energy metrics, quality assessment
- Presets: Fast, Balanced, High-Accuracy

### 3D Visualization
- Interactive rotation, zoom, pan
- Display modes: cartoon, backbone, ball+stick, surface
- Color schemes: secondary structure, residue type, hydrophobicity
- Screenshot and PDB export

### Real-Time Monitoring
- Live energy convergence charts
- Agent exploration status
- Event logging
- Pause/resume/stop controls

### Campaign Management
- Batch submission for multiple proteins
- Statistical analysis across runs
- Checkpoint and resume

### Work Sessions
- Organize predictions into projects
- Share sessions with collaborators
- Export as ZIP archives

### Aggregation Screening
- Identify aggregation-prone regions
- Risk scoring for therapeutic development

---

## How It Works

EmergentFolds uses physics-based simulation with multi-agent optimization:

1. **Multi-agent exploration** — Autonomous agents search conformational space using different strategies
2. **Energy optimization** — Molecular mechanics force field guides folding (bonds, angles, van der Waals, electrostatics, hydrogen bonds)
3. **Quantum coherence guidance** — Physics-based feedback from QCPP integration
4. **Collective learning** — Agents share discoveries to accelerate convergence

The engine is pure Python, optimized for speed, with no heavy external dependencies.

### Coming Soon

- **Machine learning integration** — Faster initial structure generation and hybrid physics+ML refinement

---

## API Access

Integrate predictions into your research pipeline:

```python
import requests

# Submit prediction
response = requests.post(
    "https://emergentfolds.com/api/v1/predictions",
    headers={"Authorization": f"Bearer {token}"},
    json={"sequence": "MQIFVKTLTGKTITLEVEPS...", "config_preset": "balanced"}
)
prediction_id = response.json()["id"]

# Get results
results = requests.get(
    f"https://emergentfolds.com/api/v1/predictions/{prediction_id}/results",
    headers={"Authorization": f"Bearer {token}"}
)
```

Full API documentation: [docs/API.md](docs/API.md)

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Setup Guide](docs/SETUP.md) | Local installation and configuration |
| [User Guide](docs/USER_GUIDE.md) | Platform features and workflows |
| [API Reference](docs/API.md) | REST API documentation |
| [CLI Reference](docs/CLI_REFERENCE.md) | Command-line tools |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues |

---

## Technical Stack

**Web Interface**
- Frontend: React 19, TypeScript, Material-UI, Vite
- Backend: FastAPI, Celery, Redis, PostgreSQL
- Real-time: Socket.IO
- Visualization: NGL Viewer

**Prediction Engine**
- Core: UBF multi-agent system
- Physics: QCPP quantum coherence integration
- Performance: <2ms per move evaluation

---

## Project Structure

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

## Production Deployment

The live site **[emergentfolds.com](https://emergentfolds.com)** runs on:

| Component | Details |
|-----------|---------|
| **Hosting** | Hostinger VPS |
| **Server** | `/opt/PP` |
| **Compose File** | `docker-compose.prod.yml` |
| **SSL/HTTPS** | Nginx with Let's Encrypt |
| **Database** | PostgreSQL 15 |
| **Cache/Queue** | Redis 7 |

### Quick Commands (on VPS)

```bash
# SSH into server
ssh root@your-vps-ip

# Navigate to project
cd /opt/PP

# View running containers
docker ps -a

# Restart all services
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d

# View logs
docker logs pp_backend --tail 100
docker logs pp_nginx --tail 100
docker logs pp_worker --tail 100

# Check service health
docker exec pp_redis redis-cli ping
```

### Troubleshooting Production

| Issue | Solution |
|-------|----------|
| **502 Bad Gateway** | Restart nginx: `docker restart pp_nginx` |
| **Can't login** | Check Redis: `docker logs pp_redis --tail 50` |
| **Slow dashboard** | Check backend: `docker logs pp_backend --tail 100` |
| **Container network issues** | Full restart: `docker compose -f docker-compose.prod.yml down && docker compose -f docker-compose.prod.yml up -d` |

Full deployment guide: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## Local Development

### Prerequisites
- Docker 24.0+ with Docker Compose
- Node.js 18+ (frontend development)
- Python 3.8–3.12 (backend development)

### Setup

```bash
# Docker (recommended)
docker compose up -d

# Or run manually
cd frontend && npm install && npm run dev
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
```

### Tests

```bash
pytest backend/tests/           # Backend
npm test --prefix frontend      # Frontend
pytest ubf_protein/tests/       # Engine
```

---

## Contributing

Questions or contributions? Check the [documentation](docs/) and [open issues](https://github.com/your-repo/issues).

---

## License

See [LICENSE](LICENSE).

---

**Live Platform**: [emergentfolds.com](https://emergentfolds.com)

*December 2025*
