# GGUF Forge - Agent Instructions

This document provides guidelines for agents working on GGUF Forge, a FastAPI application that converts HuggingFace models to GGUF format.

## Essential Commands

**Running the Application**
```bash
python app_gguf.py
```

**Docker (Recommended for Production)**
```bash
docker-compose up -d
docker-compose logs -f
```

**Installation**
```bash
pip install -r requirements.txt
```

**Database Health Check**
```bash
curl http://localhost:8000/api/health
```

**Testing**
No automated test suite exists. Manual testing via the web UI at http://localhost:8000 is recommended.

## Code Style Guidelines

### Language & Framework
- Python 3.10+
- FastAPI for web framework
- Async/await patterns for all I/O operations
- Type hints using `from typing import Optional, List, Dict, Any`

### Imports
```python
# Standard library imports first
import os
import asyncio
from pathlib import Path

# Third-party imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports
from database import get_db_connection
from models import ProcessRequest
```

### Naming Conventions
- **Variables/Functions**: `snake_case` (e.g., `get_db_connection`, `model_id`)
- **Classes**: `PascalCase` (e.g., `ModelWorkflow`, `DatabaseRow`)
- **Constants**: `UPPER_CASE` (e.g., `QUANTS`, `PARALLEL_QUANT_JOBS`)
- **Private methods**: `_prefix` (e.g., `_adapt_params`, `_reconnect`)

### Database Access
Always use async database operations:
```python
conn = await get_db_connection()
await conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
model = await conn.fetchone()
await conn.commit()
await conn.close()
```

**Important**: Use `?` placeholders for all SQL queries to prevent injection. The database abstraction automatically handles parameter binding for both SQLite and MSSQL.

### Error Handling
```python
try:
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    model = await conn.fetchone()
except Exception as e:
    logger.error(f"Failed to fetch model: {e}")
    raise HTTPException(status_code=404, detail="Model not found")
finally:
    await conn.close()
```

Use `logger` for logging errors: `logger.error()` for errors, `logger.warning()` for warnings, `logger.info()` for info.

### Pydantic Models
Define API request/response models in `models.py`:
```python
class ProcessRequest(BaseModel):
    model_id: str
    quants: Optional[List[str]] = None
```

### FastAPI Routes
- Organize routes in `routes/` directory by functionality
- Use `APIRouter(prefix="/api")` for endpoints
- Configure dependencies via module-level `configure()` function
- Example pattern from `routes/models.py`:
```python
router = APIRouter(prefix="/api")

def configure(admin_dependency):
    global _require_admin_func
    _require_admin_func = admin_dependency

@router.post("/models/process")
async def process_model(req: ProcessRequest, background_tasks: BackgroundTasks, user=Depends(get_admin)):
    # Implementation
```

### Background Tasks
Use FastAPI `BackgroundTasks` for async operations:
```python
@router.post("/models/process")
async def process_model(req: ProcessRequest, background_tasks: BackgroundTasks):
    workflow = ModelWorkflow(model_id, hf_repo_id)
    background_tasks.add_task(workflow.run_pipeline)
    return {"status": "started", "id": model_id}
```

### Environment Variables
Always access via `os.getenv()` with defaults:
```python
HF_TOKEN = os.getenv("HF_TOKEN", "")
PARALLEL_QUANT_JOBS = int(os.getenv("PARALLEL_QUANT_JOBS", "2"))
```

Never commit `.env` files. Use `.env.example` as template.

### Database Compatibility
This app supports both SQLite and MSSQL. The `database.py` module abstracts differences. Write queries using SQLite syntax - the adapter converts to MSSQL automatically.

### WebSocket Updates
Use `websocket_manager.py` for real-time updates:
```python
from websocket_manager import manager as ws_manager
await ws_manager.broadcast_model_update(model_id, {"status": "quantizing"})
```

### File Paths
Use `pathlib.Path` for all file operations:
```python
from pathlib import Path
CACHE_DIR = BASE_DIR / ".cache"
model_dir = CACHE_DIR / "models"
model_dir.mkdir(parents=True, exist_ok=True)
```

### Logging
```python
import logging
logger = logging.getLogger("GGUF_Forge")
logger.info("Application starting")
logger.error("Failed to connect to database")
```

### Adding New Features
1. Define Pydantic models in `models.py`
2. Create route file in `routes/` or add to existing
3. Configure dependencies in `app_gguf.py`
4. Update WebSocket channels if needed
5. Add database migrations using ALTER TABLE with try/except

### Commit Messages
Follow format: `add feature description`, `fix bug description`, `refactor: description`

## Architecture Notes

- **App Entry**: `app_gguf.py` - FastAPI app with lifespan management
- **Database**: `database.py` - Abstract base class with SQLite/MSSQL implementations
- **Workflow**: `workflow.py` - `ModelWorkflow` class manages conversion pipeline
- **Managers**: `managers.py` - `LlamaCppManager` and `HuggingFaceManager`
- **Security**: `security.py` - Rate limiting, bot detection, spam protection
- **Routes**: `routes/auth.py`, `routes/models.py`, `routes/requests.py`, `routes/tickets.py`

## Key Patterns

- **Async Context**: Use `async with` for database connections
- **Connection Pooling**: MSSQL uses `aioodbc` pool; SQLite creates new connections
- **Migration Pattern**: Try ALTER TABLE, pass on error (column exists)
- **Validation**: Use Pydantic for request validation; validate HF repo via `validate_hf_repo_sync()`
- **Parallel Processing**: Quantizations run in parallel (configurable via `PARALLEL_QUANT_JOBS`)
