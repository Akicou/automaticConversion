"""
Model conversion routes for GGUF Forge.
"""
import os
import uuid
import json
import asyncio

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request

from database import get_db_connection
from models import ProcessRequest
from workflow import ModelWorkflow, running_workflows, get_quants_list
from managers import HuggingFaceManager

router = APIRouter(prefix="/api")

_require_admin_func = None


def configure(admin_dependency):
    global _require_admin_func
    _require_admin_func = admin_dependency


async def get_admin(request: Request):
    if _require_admin_func:
        return await _require_admin_func(request)
    raise HTTPException(status_code=500, detail="Admin dependency not configured")


def validate_hf_repo_sync(repo_id: str) -> tuple[bool, str]:
    """Validate that the input is a valid HuggingFace model repo (synchronous version for admin endpoint).

    Returns (is_valid, error_message).
    """
    from huggingface_hub import HfApi

    repo_id = repo_id.strip()

    if not repo_id:
        return False, "Repository ID cannot be empty"

    if '/' not in repo_id:
        return False, (
            "Invalid repository format. Please use 'owner/repo-name' format, "
            "e.g., 'meta-llama/Llama-2-7b-hf'. "
            "Note: Discussion titles, search queries, or partial names are not accepted."
        )

    try:
        api = HfApi()
        repo_info_obj = api.repo_info(repo_id, repo_type="model")

        model_extensions = ('.safetensors', '.bin', '.pt', '.gguf', '.pth')
        model_files = [f for f in repo_info_obj.siblings if f.rfilename.lower().endswith(model_extensions)]

        if not model_files:
            return False, (
                f"Repository '{repo_id}' exists but doesn't contain any quantizable model files "
                "(.safetensors, .bin, .pt, .gguf, .pth)."
            )

        return True, f"Valid model repository with {len(model_files)} model files"

    except Exception as e:
        error_str = str(e).lower()
        if "404" in error_str or "not found" in error_str:
            return False, (
                f"Repository '{repo_id}' not found. "
                "Please check the repository name and ensure it exists on HuggingFace."
            )
        elif "private" in error_str or "access denied" in error_str:
            return False, (
                f"Repository '{repo_id}' is private or not accessible. "
                "Only public model repositories can be requested."
            )
        else:
            return False, f"Failed to access repository: {e}"


@router.get("/hf/search")
async def search_hf(q: str):
    mgr = HuggingFaceManager(token=os.getenv("HF_TOKEN"))
    try:
        return await mgr.search_models(q)
    except Exception as e:
        return []


def validate_quants(quants: list) -> list:
    available = get_quants_list()
    if not quants:
        return []
    return [q for q in quants if q in available]


@router.post("/models/process")
async def process_model(req: ProcessRequest, background_tasks: BackgroundTasks, user = Depends(get_admin)):
    is_valid, msg = validate_hf_repo_sync(req.model_id)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    conn = await get_db_connection()
    await conn.execute("SELECT * FROM models WHERE hf_repo_id = ?", (req.model_id,))
    existing = await conn.fetchone()
    
    if existing and existing['status'] in ['pending', 'downloading', 'converting', 'quantizing', 'initializing']:
         await conn.close()
         raise HTTPException(status_code=400, detail="Model already processing")
    
    # Validate and process quants
    available_quants = get_quants_list()
    if req.quants:
        quants_to_run = validate_quants(req.quants)
        if not quants_to_run:
            await conn.close()
            raise HTTPException(status_code=400, detail="No valid quants specified")
    else:
        quants_to_run = available_quants
    
    new_id = str(uuid.uuid4())
    
    # Delete existing record first (if any) for MSSQL compatibility
    # This replaces "INSERT OR REPLACE" which is SQLite-specific
    if existing:
        await conn.execute("DELETE FROM models WHERE hf_repo_id = ?", (req.model_id,))
    
    quants_msg = ', '.join(quants_to_run) if len(quants_to_run) < len(available_quants) else 'all quants'
    await conn.execute(
        "INSERT INTO models (id, hf_repo_id, status, progress, log, error_details) VALUES (?, ?, ?, ?, ?, ?)",
        (new_id, req.model_id, "pending", 0, f"Queued... Quants: {quants_msg}", "")
    )
    await conn.commit()
    await conn.close()
    
    workflow = ModelWorkflow(new_id, req.model_id, quants_to_run=quants_to_run)
    background_tasks.add_task(workflow.run_pipeline)
    
    return {"status": "started", "id": new_id, "quants": quants_to_run}


@router.get("/status/all")
async def get_all_status():
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 50")
    models = await conn.fetchall()
    await conn.close()
    return [m.to_dict() for m in models]


@router.get("/status/model/{model_id}")
async def get_model_status(model_id: str):
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM models WHERE hf_repo_id = ?", (model_id,))
    model = await conn.fetchone()
    await conn.close()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.to_dict()


@router.delete("/models/{model_id}")
async def delete_model_record(model_id: str, user = Depends(get_admin)):
    """Admin only: Delete a model record from the database."""
    conn = await get_db_connection()
    await conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    await conn.commit()
    await conn.close()
    return {"status": "deleted"}


@router.post("/models/{model_id}/terminate")
async def terminate_model(model_id: str, user = Depends(get_admin)):
    """Admin only: Terminate a running job."""
    # Check if this workflow is currently running
    if model_id not in running_workflows:
        # Check if the job exists in DB
        conn = await get_db_connection()
        await conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        model = await conn.fetchone()
        await conn.close()
        
        if not model:
            raise HTTPException(status_code=404, detail="Job not found")
        if model['status'] in ['complete', 'error', 'terminated', 'interrupted']:
            raise HTTPException(status_code=400, detail=f"Job already finished with status: {model['status']}")
        
        # Job exists but not in running_workflows - mark as terminated directly
        conn = await get_db_connection()
        await conn.execute(
            "UPDATE models SET status = ?, error_details = ? WHERE id = ?",
            ("terminated", "Terminated by administrator (job was not actively running)", model_id)
        )
        await conn.commit()
        await conn.close()
        return {"status": "terminated", "message": "Job marked as terminated"}
    
    # Terminate the running workflow
    workflow = running_workflows[model_id]
    await workflow.terminate()
    
    return {"status": "terminating", "message": "Termination signal sent. Job will stop shortly."}


@router.post("/models/{model_id}/continue")
async def continue_model(model_id: str, background_tasks: BackgroundTasks, user = Depends(get_admin)):
    """Admin only: Continue an interrupted job."""
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    model = await conn.fetchone()
    
    if not model:
        await conn.close()
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Only allow continuing interrupted jobs
    if model['status'] != 'interrupted':
        await conn.close()
        raise HTTPException(status_code=400, detail=f"Can only continue interrupted jobs. Current status: {model['status']}")
    
    # Check if job is already running
    if model_id in running_workflows:
        await conn.close()
        raise HTTPException(status_code=400, detail="Job is already running")
    
    # Get the list of completed quants
    completed_quants = []
    if model.get('completed_quants'):
        try:
            completed_quants = json.loads(model['completed_quants'])
        except (json.JSONDecodeError, TypeError):
            completed_quants = []
    
    # Calculate remaining quants
    all_quants = get_quants_list()
    remaining_quants = [q for q in all_quants if q not in completed_quants]
    
    if not remaining_quants:
        await conn.close()
        raise HTTPException(status_code=400, detail="All quants already completed for this job")
    
    # Reset status and clear error
    await conn.execute(
        "UPDATE models SET status = ?, error_details = ?, log = ? WHERE id = ?",
        ("pending", "", model['log'] + "\n\n━━━ RESUMING JOB ━━━\n", model_id)
    )
    await conn.commit()
    await conn.close()
    
    # Start the workflow in resume mode
    workflow = ModelWorkflow(
        model_id, 
        model['hf_repo_id'], 
        resume_mode=True,
        completed_quants=completed_quants
    )
    background_tasks.add_task(workflow.run_pipeline)
    
    return {
        "status": "continued", 
        "message": f"Job resumed. {len(completed_quants)} quants already done, {len(remaining_quants)} remaining.",
        "completed": completed_quants,
        "remaining": remaining_quants
    }
