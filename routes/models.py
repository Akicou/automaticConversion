"""
Model conversion routes for GGUF Forge.
"""
import os
import uuid
import json
import asyncio

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request

from pathlib import Path

from database import get_db_connection
from models import ProcessRequest, ContinueRequestBody, LocalProcessRequest
from workflow import (
    ModelWorkflow,
    running_workflows,
    get_quants_list,
    get_model_queue,
    get_cache_dir,
    scan_local_models,
)
from managers import HuggingFaceManager, LlamaCppManager

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
    
    if existing and existing['status'] in ['pending', 'downloading', 'converting', 'quantizing', 'uploading', 'initializing']:
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

    # Get admin options
    ignore_space_check = req.ignore_space_check if hasattr(req, 'ignore_space_check') else False
    enable_shard_merging = req.enable_shard_merging if hasattr(req, 'enable_shard_merging') else True

    # Build log message
    log_msg = f"Queued... Quants: {quants_msg}"
    if ignore_space_check:
        log_msg += "\n⚠ Admin override: Space check disabled"
    if not enable_shard_merging:
        log_msg += "\n⚠ Admin override: Shard merging disabled"

    await conn.execute(
        "INSERT INTO models (id, hf_repo_id, status, progress, log, error_details, quants_to_run, ignore_space_check, enable_shard_merging) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (new_id, req.model_id, "pending", 0, log_msg, "", json.dumps(quants_to_run), 1 if ignore_space_check else 0, 1 if enable_shard_merging else 0)
    )
    await conn.commit()
    await conn.close()

    workflow = ModelWorkflow(
        new_id,
        req.model_id,
        quants_to_run=quants_to_run,
        force_llama_update=req.force_llama_update if hasattr(req, 'force_llama_update') else False,
        ignore_space_check=ignore_space_check,
        enable_shard_merging=enable_shard_merging
    )
    
    # Add to queue instead of running directly
    queue = get_model_queue()
    if queue:
        await queue.add(workflow)
    else:
        # Fallback to direct execution if queue not initialized
        background_tasks.add_task(workflow.run_pipeline)
    
    return {"status": "queued", "id": new_id, "quants": quants_to_run}


@router.get("/models/local")
async def list_local_models(user = Depends(get_admin)):
    """Admin only: scan CACHE_DIR for already-downloaded HuggingFace models."""
    cache_dir = get_cache_dir()
    if cache_dir is None:
        return {"cache_dir": None, "models": []}
    models_found = scan_local_models(cache_dir)
    return {"cache_dir": str(cache_dir), "models": models_found}


@router.post("/models/process-local")
async def process_local_model(req: LocalProcessRequest, background_tasks: BackgroundTasks, user = Depends(get_admin)):
    """Admin only: quantize an already-downloaded model from disk.

    The source safetensors directory is preserved across the workflow.
    """
    cache_dir = get_cache_dir()
    if cache_dir is None:
        raise HTTPException(status_code=500, detail="CACHE_DIR not configured")

    # Path-traversal guard: resolve and require it to be inside CACHE_DIR.
    try:
        resolved = Path(req.path).resolve()
        cache_resolved = Path(cache_dir).resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}")

    try:
        resolved.relative_to(cache_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path must be inside the configured models directory")

    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {resolved}")
    if not (resolved / "config.json").is_file():
        raise HTTPException(status_code=400, detail="Missing config.json — not a HuggingFace model directory")
    if not any(resolved.glob("*.safetensors")):
        raise HTTPException(status_code=400, detail="No *.safetensors files found in directory")

    repo_id = (req.repo_id or "").strip()
    if not repo_id or "/" not in repo_id:
        raise HTTPException(status_code=400, detail="repo_id must be in 'owner/name' form (use 'local/<name>' for flat layouts)")

    # Validate and process quants
    available_quants = get_quants_list()
    if req.quants:
        quants_to_run = validate_quants(req.quants)
        if not quants_to_run:
            raise HTTPException(status_code=400, detail="No valid quants specified")
    else:
        quants_to_run = available_quants

    conn = await get_db_connection()
    await conn.execute("SELECT * FROM models WHERE hf_repo_id = ?", (repo_id,))
    existing = await conn.fetchone()

    if existing and existing['status'] in ['pending', 'downloading', 'converting', 'quantizing', 'uploading', 'initializing']:
        await conn.close()
        raise HTTPException(status_code=400, detail="Model already processing")

    if existing:
        await conn.execute("DELETE FROM models WHERE hf_repo_id = ?", (repo_id,))

    new_id = str(uuid.uuid4())
    quants_msg = ', '.join(quants_to_run) if len(quants_to_run) < len(available_quants) else 'all quants'
    keep_local = bool(req.keep_local_only)
    log_msg = f"Queued (local source). Quants: {quants_msg}. Keep local only: {keep_local}"
    if req.ignore_space_check:
        log_msg += "\n⚠ Admin override: Space check disabled"
    if not req.enable_shard_merging:
        log_msg += "\n⚠ Admin override: Shard merging disabled"

    await conn.execute(
        "INSERT INTO models (id, hf_repo_id, status, progress, log, error_details, quants_to_run, "
        "ignore_space_check, enable_shard_merging, local_source_path, keep_local_only) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            new_id,
            repo_id,
            "pending",
            0,
            log_msg,
            "",
            json.dumps(quants_to_run),
            1 if req.ignore_space_check else 0,
            1 if req.enable_shard_merging else 0,
            str(resolved),
            1 if keep_local else 0,
        )
    )
    await conn.commit()
    await conn.close()

    workflow = ModelWorkflow(
        new_id,
        repo_id,
        quants_to_run=quants_to_run,
        ignore_space_check=bool(req.ignore_space_check),
        enable_shard_merging=bool(req.enable_shard_merging),
        local_source_path=str(resolved),
        keep_local_only=keep_local,
    )

    queue = get_model_queue()
    if queue:
        await queue.add(workflow)
    else:
        background_tasks.add_task(workflow.run_pipeline)

    return {"status": "queued", "id": new_id, "quants": quants_to_run, "keep_local_only": keep_local}


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
async def continue_model(model_id: str, background_tasks: BackgroundTasks, body: ContinueRequestBody = None, user = Depends(get_admin)):
    """Admin only: Continue an interrupted job."""
    import logging
    logger = logging.getLogger("GGUF_Forge")
    logger.info(f"Continue request for {model_id}: body={body}, ignore_space_check={body.ignore_space_check if body else 'NO BODY'}")

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

    # Get the original quants_to_run if stored, otherwise use all quants
    quants_to_run = get_quants_list()
    if model.get('quants_to_run'):
        try:
            stored_quants = json.loads(model['quants_to_run'])
            if stored_quants:
                quants_to_run = stored_quants
        except (json.JSONDecodeError, TypeError):
            pass

    # Calculate remaining quants based on the original quants_to_run
    remaining_quants = [q for q in quants_to_run if q not in completed_quants]

    if not remaining_quants:
        await conn.close()
        raise HTTPException(status_code=400, detail="All quants already completed for this job")

    # Get ignore_space_check from body or stored value
    ignore_space_check = False
    if body and body.ignore_space_check:
        ignore_space_check = True
    elif model.get('ignore_space_check'):
        ignore_space_check = bool(model['ignore_space_check'])

    logger.info(f"ignore_space_check determined: {ignore_space_check} (from body: {body.ignore_space_check if body else None}, from db: {model.get('ignore_space_check')})")

    # Get enable_shard_merging from stored value (default to True)
    enable_shard_merging = True
    if model.get('enable_shard_merging') is not None:
        enable_shard_merging = bool(model['enable_shard_merging'])

    # Get requester info
    requested_by = model.get('requested_by')

    # Build log message
    log_msg = model['log'] + "\n\n━━━ RESUMING JOB ━━━\n"
    if ignore_space_check:
        log_msg += "⚠ Admin override: Space check disabled\n"

    # Reset status and clear error
    await conn.execute(
        "UPDATE models SET status = ?, error_details = ?, log = ?, ignore_space_check = ? WHERE id = ?",
        ("pending", "", log_msg, 1 if ignore_space_check else 0, model_id)
    )
    await conn.commit()
    await conn.close()

    # Start the workflow in resume mode
    workflow = ModelWorkflow(
        model_id,
        model['hf_repo_id'],
        resume_mode=True,
        completed_quants=completed_quants,
        quants_to_run=quants_to_run,
        ignore_space_check=ignore_space_check,
        enable_shard_merging=enable_shard_merging,
        requested_by=requested_by
    )

    # Add to queue instead of running directly
    queue = get_model_queue()
    if queue:
        await queue.add(workflow)
    else:
        # Fallback to direct execution if queue not initialized
        background_tasks.add_task(workflow.run_pipeline)

    return {
        "status": "queued",
        "message": f"Job resumed and added to queue. {len(completed_quants)} quants already done, {len(remaining_quants)} remaining.",
        "completed": completed_quants,
        "remaining": remaining_quants,
        "ignore_space_check": ignore_space_check
    }


@router.get("/queue/status")
async def get_queue_status(user = Depends(get_admin)):
    """Admin only: Get current model queue status.
    
    Returns information about currently processing model and waiting models.
    """
    queue = get_model_queue()
    if not queue:
        return {
            "enabled": False,
            "message": "Queue system not initialized"
        }
    
    status = queue.get_status()
    
    # Enrich with model details from database
    conn = await get_db_connection()
    try:
        result = {
            "enabled": True,
            "current_model": None,
            "queue": []
        }
        
        # Get current model details
        if status["current_model_id"]:
            await conn.execute("SELECT * FROM models WHERE id = ?", (status["current_model_id"],))
            current = await conn.fetchone()
            if current:
                result["current_model"] = {
                    "id": current["id"],
                    "hf_repo_id": current["hf_repo_id"],
                    "status": current["status"],
                    "progress": current["progress"]
                }
        
        # Get queued models details
        for item in status["queue"]:
            await conn.execute("SELECT * FROM models WHERE id = ?", (item["model_id"],))
            model = await conn.fetchone()
            if model:
                result["queue"].append({
                    "position": item["position"],
                    "id": model["id"],
                    "hf_repo_id": model["hf_repo_id"],
                    "status": model["status"]
                })
        
        return result
    finally:
        await conn.close()


@router.post("/llama-cpp/update")
async def force_update_llama_cpp(force: bool = False, rebuild: bool = False, user = Depends(get_admin)):
    """Admin only: Update llama.cpp repository to latest commit.
    
    Args:
        force: If True, forcefully reset to latest remote commit (discards local changes).
               If False, perform normal git pull.
        rebuild: If True, rebuild llama.cpp after updating.
    
    Returns:
        Status of the update operation including current commit hash.
    """
    try:
        # Re-read config so any admin UI change takes effect before we touch git
        import managers
        await managers.refresh_llama_config()

        # Update the repository
        await LlamaCppManager.clone_repo(force=force)

        # Get current commit info
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            "git", "log", "-1", "--format=%H|%s|%an|%ar",
            cwd=managers.LLAMA_CPP_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        commit_info = {}
        if proc.returncode == 0:
            parts = stdout.decode().strip().split('|')
            if len(parts) >= 4:
                commit_info = {
                    "hash": parts[0][:8],  # Short hash
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3]
                }
        
        # Optionally rebuild
        rebuild_status = "not_requested"
        if rebuild:
            rebuild_status = "building"
            await LlamaCppManager.build()
            rebuild_status = "complete"
        
        return {
            "status": "success",
            "update_type": "force_reset" if force else "pull",
            "commit": commit_info,
            "rebuild": rebuild_status,
            "message": f"llama.cpp successfully {'force updated' if force else 'updated'}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update llama.cpp: {str(e)}"
        )
