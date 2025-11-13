# Developer Guide - Protein Prediction Platform

This guide provides technical information for developers working on the Protein Prediction Platform.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Backend Development](#backend-development)
- [Frontend Development](#frontend-development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Architecture Overview

The platform follows a modern microservices architecture with the following components:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │────▶│  PP System  │
│  (React)    │     │  (FastAPI)  │     │   (Python)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                    │                    
       │                    ▼                    
       │            ┌─────────────┐              
       │            │   Redis     │              
       │            │  (Queue)    │              
       │            └─────────────┘              
       │                    │                    
       └────────────────────┼────────────────────┐
                            ▼                    │
                    ┌─────────────┐      ┌──────▼──────┐
                    │   Celery    │      │  Socket.IO  │
                    │  (Worker)   │      │(WebSocket)  │
                    └─────────────┘      └─────────────┘
```

### Component Responsibilities

**Frontend (React + TypeScript)**
- User interface and interaction
- Real-time updates via WebSocket
- State management with Zustand
- Data fetching with React Query

**Backend (FastAPI + Python)**
- REST API endpoints
- Request validation
- Business logic
- WebSocket server
- Database operations

**Worker (Celery + Python)**
- Asynchronous task processing
- Long-running predictions
- Background jobs

**Redis**
- Message broker for Celery
- Caching layer
- Session storage

**PP System**
- Core prediction algorithms
- QCPP integration
- UBF multi-agent system

## Technology Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8-3.12 | Core language |
| FastAPI | 0.104+ | Web framework |
| Celery | 5.3+ | Task queue |
| Redis | 7.0+ | Message broker & cache |
| SQLAlchemy | 2.0+ | ORM (optional) |
| Pydantic | 2.0+ | Data validation |
| Socket.IO | 5.0+ | WebSocket |
| pytest | 8.0+ | Testing |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19.2+ | UI framework |
| TypeScript | 5.9+ | Type safety |
| Vite | Latest | Build tool |
| MUI | 7.0+ | Component library |
| React Router | 7.0+ | Routing |
| Zustand | 5.0+ | State management |
| React Query | 5.0+ | Data fetching |
| Socket.IO Client | 4.0+ | WebSocket |
| Recharts | 2.0+ | Charts |
| NGL Viewer | 2.0+ | 3D visualization |
| Vitest | 4.0+ | Testing |

### Infrastructure

- **Docker**: Containerization
- **Nginx**: Reverse proxy (production)
- **PostgreSQL**: Database (optional)

## Project Structure

```
PP/
├── backend/                    # Backend application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app
│   │   ├── config.py          # Configuration
│   │   ├── api/               # API endpoints
│   │   │   ├── predictions.py
│   │   │   ├── campaigns.py
│   │   │   └── results.py
│   │   ├── models/            # Database models
│   │   │   ├── prediction.py
│   │   │   └── campaign.py
│   │   ├── schemas/           # Pydantic schemas
│   │   │   ├── prediction.py
│   │   │   └── campaign.py
│   │   ├── services/          # Business logic
│   │   │   ├── prediction_service.py
│   │   │   └── campaign_service.py
│   │   ├── tasks/             # Celery tasks
│   │   │   └── prediction_tasks.py
│   │   ├── integrations/      # PP system integration
│   │   │   ├── pp_wrapper.py
│   │   │   ├── result_parser.py
│   │   │   ├── config_mapper.py
│   │   │   └── file_manager.py
│   │   ├── websocket/         # WebSocket handlers
│   │   │   ├── socket_manager.py
│   │   │   └── events.py
│   │   └── utils/             # Utilities
│   ├── tests/                 # Backend tests
│   ├── requirements.txt
│   ├── pytest.ini
│   └── celery_app.py
├── frontend/                   # Frontend application
│   ├── src/
│   │   ├── main.tsx           # Entry point
│   │   ├── App.tsx            # Root component
│   │   ├── components/        # React components
│   │   │   ├── layout/
│   │   │   ├── common/
│   │   │   ├── dashboard/
│   │   │   ├── prediction/
│   │   │   ├── monitoring/
│   │   │   ├── results/
│   │   │   ├── visualization/
│   │   │   ├── campaign/
│   │   │   ├── history/
│   │   │   └── settings/
│   │   ├── pages/             # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── PredictionForm.tsx
│   │   │   ├── LiveMonitoring.tsx
│   │   │   ├── ResultsAnalysis.tsx
│   │   │   └── ...
│   │   ├── services/          # API services
│   │   │   ├── api.ts
│   │   │   ├── predictionService.ts
│   │   │   └── websocketService.ts
│   │   ├── hooks/             # Custom hooks
│   │   │   ├── usePredictions.ts
│   │   │   └── useWebSocket.ts
│   │   ├── store/             # Zustand stores
│   │   │   ├── index.ts
│   │   │   ├── uiStore.ts
│   │   │   └── predictionStore.ts
│   │   ├── types/             # TypeScript types
│   │   │   └── api.ts
│   │   ├── utils/             # Utilities
│   │   └── theme/             # MUI theme
│   ├── __tests__/             # Frontend tests
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── vitest.config.ts
├── docker/                     # Docker configurations
│   ├── frontend/
│   ├── backend/
│   └── worker/
├── docs/                       # Documentation
├── ubf_protein/               # PP system
└── docker-compose.yml
```

## Backend Development

### Setting Up Development Environment

See [Setup Guide](SETUP.md) for detailed instructions.

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Running the Backend

```bash
# Development server with hot reload
uvicorn app.main:app --reload --port 8000

# With specific host
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Starting Celery Worker

```bash
celery -A app.celery_app:celery_app worker --loglevel=info

# Windows (requires --pool=solo)
celery -A app.celery_app:celery_app worker --loglevel=info --pool=solo
```

### API Development

#### Creating a New Endpoint

1. **Define Schema** (`app/schemas/your_model.py`):
```python
from pydantic import BaseModel, Field

class YourRequestSchema(BaseModel):
    field1: str = Field(..., description="Description")
    field2: int = Field(default=100, ge=1, le=1000)

class YourResponseSchema(BaseModel):
    id: str
    status: str
    data: dict
```

2. **Create Endpoint** (`app/api/your_endpoint.py`):
```python
from fastapi import APIRouter, HTTPException, status
from app.schemas.your_model import YourRequestSchema, YourResponseSchema

router = APIRouter(prefix="/your-endpoint", tags=["Your Feature"])

@router.post("/", response_model=YourResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_item(request: YourRequestSchema):
    # Implementation
    return YourResponseSchema(
        id="item_123",
        status="created",
        data={"key": "value"}
    )

@router.get("/{item_id}", response_model=YourResponseSchema)
async def get_item(item_id: str):
    # Implementation
    pass
```

3. **Register Router** (`app/main.py`):
```python
from app.api import your_endpoint

app.include_router(your_endpoint.router, prefix="/api")
```

#### Error Handling

Use consistent error responses:

```python
from fastapi import HTTPException, status

# Not found
raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail={"error": "NOT_FOUND", "message": "Item not found"}
)

# Validation error
raise HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail={
        "error": "VALIDATION_ERROR",
        "message": "Invalid input",
        "details": {"field": "sequence", "reason": "Invalid format"}
    }
)
```

### Creating Celery Tasks

1. **Define Task** (`app/tasks/your_tasks.py`):
```python
from app.celery_app import celery_app
from celery import Task

class YourTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        # Cleanup or notifications
        pass
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        # Error handling
        pass

@celery_app.task(bind=True, base=YourTask, name='tasks.your_task')
def your_task(self, param1: str, param2: int):
    try:
        # Long-running work
        result = do_work(param1, param2)
        return result
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
```

2. **Call Task**:
```python
from app.tasks.your_tasks import your_task

# Async execution
task = your_task.delay("value1", 100)
task_id = task.id

# Get result later
result = your_task.AsyncResult(task_id)
if result.ready():
    data = result.get()
```

### WebSocket Development

#### Emitting Events

```python
from app.websocket.socket_manager import socket_manager

# Emit to specific room
await socket_manager.emit_to_room(
    room=prediction_id,
    event='progress_update',
    data={
        'prediction_id': prediction_id,
        'iteration': 100,
        'metrics': {...}
    }
)

# Broadcast to all
await socket_manager.broadcast(
    event='system_status',
    data={'status': 'healthy'}
)
```

### Database Operations

If using SQLAlchemy:

```python
from app.models.prediction import Prediction
from app.database import get_db

async def create_prediction(db: Session, data: dict):
    prediction = Prediction(**data)
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

async def get_prediction(db: Session, prediction_id: str):
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()
```

### Configuration

Configuration is managed in `app/config.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "PP Platform"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = None
    
    class Config:
        env_file = ".env"

settings = Settings()
```

Access settings:

```python
from app.config import settings
print(settings.redis_url)
```

## Frontend Development

### Setting Up Development Environment

```bash
cd frontend
npm install
npm run dev
```

### Project Configuration

#### Vite Configuration (`vite.config.ts`)

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/socket.io': {
        target: 'http://localhost:8000',
        ws: true,
      },
    },
  },
});
```

### Creating Components

#### Component Structure

```typescript
// src/components/feature/MyComponent.tsx
import React from 'react';
import { Box, Typography } from '@mui/material';

interface MyComponentProps {
  title: string;
  onAction: () => void;
}

export const MyComponent: React.FC<MyComponentProps> = ({ title, onAction }) => {
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5">{title}</Typography>
      <button onClick={onAction}>Click Me</button>
    </Box>
  );
};
```

#### Using MUI Components

```typescript
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Button,
  Grid
} from '@mui/material';

export const ExampleComponent = () => {
  return (
    <Card>
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6">Title</Typography>
            <Typography variant="body2">Content</Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Button variant="contained" color="primary">
              Action
            </Button>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};
```

### State Management

#### Creating a Store

```typescript
// src/store/myStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface MyState {
  items: string[];
  addItem: (item: string) => void;
  removeItem: (index: number) => void;
}

export const useMyStore = create<MyState>()(
  persist(
    (set) => ({
      items: [],
      addItem: (item) => set((state) => ({
        items: [...state.items, item]
      })),
      removeItem: (index) => set((state) => ({
        items: state.items.filter((_, i) => i !== index)
      })),
    }),
    {
      name: 'my-store',
    }
  )
);
```

#### Using the Store

```typescript
import { useMyStore } from '@/store/myStore';

export const MyComponent = () => {
  const items = useMyStore((state) => state.items);
  const addItem = useMyStore((state) => state.addItem);
  
  return (
    <div>
      {items.map((item, i) => <div key={i}>{item}</div>)}
      <button onClick={() => addItem('New')}>Add</button>
    </div>
  );
};
```

### API Integration

#### Creating an API Service

```typescript
// src/services/myService.ts
import { api } from './api';

export interface MyData {
  id: string;
  name: string;
}

export const myService = {
  getAll: () => api.get<MyData[]>('/my-endpoint'),
  
  getById: (id: string) => api.get<MyData>(`/my-endpoint/${id}`),
  
  create: (data: Partial<MyData>) => 
    api.post<MyData>('/my-endpoint', data),
  
  update: (id: string, data: Partial<MyData>) => 
    api.put<MyData>(`/my-endpoint/${id}`, data),
  
  delete: (id: string) => api.delete(`/my-endpoint/${id}`),
};
```

#### Using React Query

```typescript
// src/hooks/useMyData.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { myService } from '@/services/myService';

export const useMyData = () => {
  return useQuery({
    queryKey: ['myData'],
    queryFn: () => myService.getAll(),
  });
};

export const useCreateMyData = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: myService.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['myData'] });
    },
  });
};
```

#### Using in Component

```typescript
import { useMyData, useCreateMyData } from '@/hooks/useMyData';

export const MyComponent = () => {
  const { data, isLoading, error } = useMyData();
  const createMutation = useCreateMyData();
  
  const handleCreate = () => {
    createMutation.mutate({ name: 'New Item' });
  };
  
  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  
  return (
    <div>
      {data?.map(item => <div key={item.id}>{item.name}</div>)}
      <button onClick={handleCreate}>Create</button>
    </div>
  );
};
```

### WebSocket Integration

```typescript
// src/hooks/useWebSocket.ts
import { useEffect } from 'use';
import { websocketService } from '@/services/websocketService';

export const useWebSocket = (predictionId: string, onUpdate: (data: any) => void) => {
  useEffect(() => {
    websocketService.connect();
    websocketService.joinRoom(predictionId);
    
    const handler = (data: any) => {
      if (data.prediction_id === predictionId) {
        onUpdate(data);
      }
    };
    
    websocketService.on('progress_update', handler);
    
    return () => {
      websocketService.off('progress_update', handler);
      websocketService.leaveRoom(predictionId);
    };
  }, [predictionId, onUpdate]);
};
```

### Routing

```typescript
// src/routes/index.tsx
import { createBrowserRouter } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import { Dashboard } from '@/pages/Dashboard';
import { PredictionForm } from '@/pages/PredictionForm';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { path: '/', element: <Dashboard /> },
      { path: '/predict', element: <PredictionForm /> },
      // ...more routes
    ],
  },
]);
```

## Testing

### Backend Testing

Run backend tests:

```bash
cd backend
pytest                          # Run all tests
pytest tests/test_api.py       # Run specific file
pytest -v                       # Verbose
pytest --cov=app               # With coverage
```

#### Writing Tests

```python
# tests/test_my_feature.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_item():
    response = client.post('/api/items', json={
        'name': 'Test Item',
        'value': 100
    })
    assert response.status_code == 201
    data = response.json()
    assert data['name'] == 'Test Item'
    assert 'id' in data

def test_get_item_not_found():
    response = client.get('/api/items/nonexistent')
    assert response.status_code == 404
```

### Frontend Testing

Run frontend tests:

```bash
cd frontend
npm test                    # Run tests in watch mode
npm test -- --run          # Run once
npm run test:coverage      # With coverage
```

#### Writing Component Tests

```typescript
// src/__tests__/MyComponent.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { MyComponent } from '@/components/MyComponent';

describe('MyComponent', () => {
  it('renders title', () => {
    render(<MyComponent title="Test Title" onAction={() => {}} />);
    expect(screen.getByText('Test Title')).toBeInTheDocument();
  });
  
  it('calls onAction when button clicked', () => {
    const mockAction = vi.fn();
    render(<MyComponent title="Test" onAction={mockAction} />);
    
    fireEvent.click(screen.getByRole('button'));
    expect(mockAction).toHaveBeenCalled();
  });
});
```

## Deployment

### Docker Deployment

```bash
# Build all services
docker-compose build

# Start in production mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Considerations

1. **Environment Variables**: Use production values
2. **SSL/TLS**: Configure HTTPS with Let's Encrypt
3. **Reverse Proxy**: Use Nginx for routing
4. **Database**: Use PostgreSQL for persistence
5. **Monitoring**: Add logging and monitoring
6. **Backups**: Configure automated backups
7. **Scaling**: Use Docker Swarm or Kubernetes

### Performance Optimization

**Backend**:
- Enable Redis caching
- Use connection pooling
- Implement rate limiting
- Optimize database queries

**Frontend**:
- Code splitting with lazy loading
- Bundle optimization
- Image optimization
- Enable gzip compression

## Contributing

### Code Style

**Backend**:
- Follow PEP 8
- Use type hints
- Write docstrings
- Format with `black`

**Frontend**:
- Follow Airbnb style guide
- Use TypeScript strictly
- Write JSDoc comments
- Format with Prettier

### Git Workflow

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and commit: `git commit -m "feat: add my feature"`
3. Push branch: `git push origin feature/my-feature`
4. Create pull request
5. Wait for review and CI checks
6. Merge to main

### Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructuring
- `test:` Tests
- `chore:` Maintenance

Example: `feat: add RMSD calculation to results page`

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [MUI Documentation](https://mui.com/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [Docker Documentation](https://docs.docker.com/)
