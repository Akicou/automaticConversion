"""
Model conversion workflow for GGUF Forge.
"""
import os
import sys
import json
import re
import shutil
import asyncio
import traceback
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime


from huggingface_hub import HfApi, snapshot_download, create_repo, hf_hub_download
from huggingface_hub.utils import tqdm as hf_tqdm

from database import get_db_connection
from managers import LlamaCppManager, get_app_version
from websocket_manager import broadcast_model_update, broadcast_transfer_progress

# These will be set by main app
CACHE_DIR = None
LLAMA_CPP_DIR = None
QUANTS = None
PARALLEL_QUANT_JOBS = None

# Global registry for running workflows (for termination support)
running_workflows: dict = {}  # model_id -> ModelWorkflow instance

# Global model queue instance
model_queue = None


def set_workflow_config(cache_dir: Path, llama_cpp_dir: Path, quants: list, parallel_jobs: int):
    """Set configuration for workflow module."""
    global CACHE_DIR, LLAMA_CPP_DIR, QUANTS, PARALLEL_QUANT_JOBS
    CACHE_DIR = cache_dir
    LLAMA_CPP_DIR = llama_cpp_dir
    QUANTS = quants
    PARALLEL_QUANT_JOBS = parallel_jobs


def get_quants_list():
    """Get the list of quants to process."""
    return QUANTS


async def get_quant_priority_order():
    """Get the priority order for quants from database. Returns default if not configured."""
    from database import get_db_connection

    try:
        conn = await get_db_connection()
        await conn.execute("SELECT priority_order FROM quant_priority WHERE id = 1")
        row = await conn.fetchone()
        await conn.close()

        if row and row.get('priority_order'):
            try:
                custom_order = json.loads(row['priority_order'])
                # Validate and filter
                valid_order = [q for q in custom_order if q in QUANTS]
                # Add missing quants at the end
                missing = [q for q in QUANTS if q not in valid_order]
                return valid_order + missing
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception:
        pass

    return list(QUANTS)


class ModelQueue:
    """Queue system to process one model at a time.
    
    Ensures only one model workflow runs at any given time, preventing
    resource contention from multiple simultaneous downloads/quantizations.
    """
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.current_workflow = None
        self.worker_task = None
        self._queue_list = []  # Track queue for status reporting
        
    async def add(self, workflow: "ModelWorkflow"):
        """Add a workflow to the queue."""
        import logging
        logger = logging.getLogger("GGUF_Forge")
        
        self._queue_list.append({
            "model_id": workflow.model_id,
            "hf_repo_id": workflow.hf_repo_id,
            "added_at": asyncio.get_event_loop().time()
        })
        
        queue_position = len(self._queue_list)
        logger.info(f"Model {workflow.hf_repo_id} added to queue (position {queue_position})")
        
        # Update model status in database
        from database import get_db_connection
        conn = await get_db_connection()
        try:
            if queue_position == 1 and self.current_workflow is None:
                # First in queue and nothing processing - will start immediately
                await conn.execute(
                    "UPDATE models SET log = ? WHERE id = ?",
                    (f"In queue (position 1) - starting immediately...\nQuants: {', '.join(workflow.quants_to_run) if hasattr(workflow, 'quants_to_run') and workflow.quants_to_run else 'all'}", workflow.model_id)
                )
            else:
                await conn.execute(
                    "UPDATE models SET log = ? WHERE id = ?",
                    (f"In queue (position {queue_position}) - waiting for other models to complete...\nQuants: {', '.join(workflow.quants_to_run) if hasattr(workflow, 'quants_to_run') and workflow.quants_to_run else 'all'}", workflow.model_id)
                )
            await conn.commit()
        finally:
            await conn.close()
        
        await self.queue.put(workflow)
        
        # Broadcast queue update via WebSocket
        from websocket_manager import manager as ws_manager
        await ws_manager.broadcast("models", {
            "type": "queue_update",
            "queue_size": self.queue.qsize(),
            "current_model": self.current_workflow.model_id if self.current_workflow else None
        })
        
    def start_worker(self):
        """Start the background worker that processes the queue."""
        import logging
        logger = logging.getLogger("GGUF_Forge")
        logger.info("Starting model queue worker...")
        self.worker_task = asyncio.create_task(self._worker())
        
    async def _worker(self):
        """Background worker that processes workflows one at a time."""
        import logging
        logger = logging.getLogger("GGUF_Forge")
        logger.info("Model queue worker started")
        
        while True:
            try:
                # Wait for next workflow in queue
                workflow = await self.queue.get()
                self.current_workflow = workflow
                
                # Remove from tracking list
                self._queue_list = [item for item in self._queue_list if item["model_id"] != workflow.model_id]
                
                # Update queue positions for waiting models
                await self._update_queue_positions()
                
                logger.info(f"Processing model: {workflow.hf_repo_id} (ID: {workflow.model_id})")
                
                try:
                    # Run the workflow pipeline
                    await workflow.run_pipeline()
                    logger.info(f"Model {workflow.hf_repo_id} completed successfully")
                except Exception as e:
                    logger.error(f"Model {workflow.hf_repo_id} failed: {e}")
                finally:
                    self.current_workflow = None
                    self.queue.task_done()
                    
                    # Broadcast queue update
                    from websocket_manager import manager as ws_manager
                    await ws_manager.broadcast("models", {
                        "type": "queue_update",
                        "queue_size": self.queue.qsize(),
                        "current_model": None
                    })
                    
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                self.current_workflow = None
                
    async def _update_queue_positions(self):
        """Update database with current queue positions for waiting models."""
        from database import get_db_connection
        
        position = 1
        for item in self._queue_list:
            try:
                conn = await get_db_connection()
                await conn.execute(
                    "UPDATE models SET log = REPLACE(log, 'position ' || ?, 'position ' || ?) WHERE id = ?",
                    (str(position + 1), str(position), item["model_id"])
                )
                await conn.commit()
                await conn.close()
                position += 1
            except Exception:
                pass  # Non-critical, continue
                
    def get_status(self):
        """Get current queue status."""
        return {
            "current_model_id": self.current_workflow.model_id if self.current_workflow else None,
            "current_hf_repo": self.current_workflow.hf_repo_id if self.current_workflow else None,
            "waiting_count": self.queue.qsize(),
            "queue": [
                {
                    "model_id": item["model_id"],
                    "hf_repo_id": item["hf_repo_id"],
                    "position": idx + 1
                }
                for idx, item in enumerate(self._queue_list)
            ]
        }
        
    async def clear(self):
        """Clear the queue (admin function)."""
        import logging
        logger = logging.getLogger("GGUF_Forge")
        
        # Clear the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        self._queue_list.clear()
        logger.info("Queue cleared")


def get_model_queue():
    """Get the global model queue instance."""
    global model_queue
    return model_queue


def set_model_queue(queue: ModelQueue):
    """Set the global model queue instance."""
    global model_queue
    model_queue = queue


class ModelWorkflow:
    def __init__(self, model_id: str, hf_repo_id: str, resume_mode: bool = False,
                 completed_quants: Optional[List[str]] = None, quants_to_run: Optional[List[str]] = None,
                 ignore_space_check: bool = False, force_llama_update: bool = False,
                 enable_shard_merging: bool = True, requested_by: Optional[str] = None):
        self.model_id = model_id
        self.hf_repo_id = hf_repo_id
        self.log_buffer = []
        self.model_dir = None
        self.fp16_path = None
        self.quant_paths = []
        # Time tracking
        self.start_time = None
        self.step_times = {}  # step_name -> (start, end)
        self.quant_times = []  # list of (q_type, duration_seconds)
        # Transfer progress tracking
        self.transfer_files = {}  # filename -> {"progress": 0, "size": "", "speed": ""}
        # Termination support
        self.terminated = False
        self.running_processes: List[asyncio.subprocess.Process] = []
        # Resume support
        self.resume_mode = resume_mode
        self.completed_quants: List[str] = completed_quants or []  # Quants that have been uploaded already
        # Custom quants - if specified, only these quants will be processed
        self.quants_to_run: List[str] = quants_to_run if quants_to_run else QUANTS
        # For tracking the HF repo (needed for resume)
        self.new_repo_id = None
        self.hf_token = None
        self.api = None
        # Admin override - skip conservative disk space checks
        self.ignore_space_check = ignore_space_check
        # Force llama.cpp update flag
        self.force_llama_update = force_llama_update
        # Admin override - enable/disable shard merging
        self.enable_shard_merging = enable_shard_merging
        # Who requested this conversion (HuggingFace username)
        self.requested_by = requested_by
    
    async def terminate(self):
        """Request termination of this workflow."""
        self.terminated = True
        await self.log("⚠ TERMINATION REQUESTED - Stopping workflow...")
        # Kill any running processes
        for proc in list(self.running_processes):
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    
    def check_terminated(self):
        """Check if terminated and raise exception if so."""
        if self.terminated:
            raise Exception("Workflow terminated by admin")

    async def log(self, message: str):
        print(f"[{self.hf_repo_id}] {message}")
        self.log_buffer.append(message)
        # Keep last 8k chars for better visibility in UI
        await self._update_db(log="\n".join(self.log_buffer)[-8000:])

    async def progress(self, percent: int):
        await self._update_db(progress=percent)

    async def status(self, status_msg: str):
        await self._update_db(status=status_msg)

    async def _update_db(self, **kwargs):
        conn = await get_db_connection()
        try:
            updates = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [self.model_id]
            await conn.execute(f"UPDATE models SET {updates} WHERE id = ?", values)
            await conn.commit()
            
            # Fetch updated model data and broadcast via WebSocket
            await conn.execute("SELECT * FROM models WHERE id = ?", (self.model_id,))
            model_data = await conn.fetchone()
            if model_data:
                await broadcast_model_update(model_data.to_dict())
        finally:
            await conn.close()
    
    async def save_completed_quant(self, q_type: str):
        """Save a completed quant to the database for resume capability."""
        if q_type not in self.completed_quants:
            self.completed_quants.append(q_type)
        await self._update_db(completed_quants=json.dumps(self.completed_quants))
    
    async def cleanup_safetensors(self):
        """Remove downloaded safetensors model directory to free up space."""
        if self.model_dir and Path(self.model_dir).exists():
            await self.log("  Cleaning up safetensors model to free disk space...")
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, lambda: shutil.rmtree(self.model_dir, ignore_errors=True))
                await self.log("  ✓ Safetensors model cleaned up")
                self.model_dir = None
            except Exception as e:
                await self.log(f"  ⚠ Failed to cleanup safetensors: {e}")

    def _get_gguf_shards(self, base_path: Path) -> List[Tuple[int, int, Path]]:
        """Find sharded GGUF files matching a base output path."""
        stem = base_path.stem
        if not stem:
            return []
        
        pattern = re.compile(rf"^{re.escape(stem)}-(\d{{5}})-of-(\d{{5}})\.gguf$")
        shard_sets = {}
        
        for file_path in base_path.parent.glob(f"{stem}-?????-of-?????.gguf"):
            match = pattern.match(file_path.name)
            if not match:
                continue
            idx = int(match.group(1))
            total = int(match.group(2))
            shard_sets.setdefault(total, []).append((idx, file_path))
        
        if not shard_sets:
            return []
        
        # Prefer the shard set with the most parts (handles stale leftovers).
        total = max(shard_sets.keys(), key=lambda t: len(shard_sets[t]))
        shards = shard_sets[total]
        shards.sort(key=lambda s: s[0])
        return [(idx, total, path) for idx, path in shards]
    
    async def _cleanup_gguf_shards(self, shard_paths: List[Path], q_type: str):
        """Delete shard files after successful merge."""
        loop = asyncio.get_event_loop()
        for shard_path in shard_paths:
            try:
                await loop.run_in_executor(None, lambda p=shard_path: p.unlink(missing_ok=True))
            except Exception as e:
                await self.log(f"      ⚠ {q_type} Failed to delete shard {shard_path.name}: {e}")
    
    async def ensure_unsharded_gguf(self, q_path: Path, q_type: str) -> Optional[Path]:
        """Merge sharded GGUF output into a single file when needed."""
        shards = self._get_gguf_shards(q_path)
        if not shards:
            if q_path.exists():
                return q_path
            await self.log(f"      ⚠ {q_type} Output file missing: {q_path.name}")
            return None

        total = shards[0][1]
        shard_paths = [path for _, _, path in shards]
        shard_indices = {idx for idx, _, _ in shards}
        missing = [i for i in range(1, total + 1) if i not in shard_indices]

        if missing:
            preview = ", ".join(f"{i:05d}" for i in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            await self.log(f"      ⚠ {q_type} Shard set incomplete (missing {preview}{suffix})")
            return None

        # Check if shard merging is disabled
        if not self.enable_shard_merging:
            await self.log(f"      ℹ {q_type} Shard merging disabled by admin - keeping sharded output")
            # Return the first shard path as the output
            return shard_paths[0]

        if q_path.exists():
            try:
                base_mtime = q_path.stat().st_mtime
                latest_shard_mtime = max(p.stat().st_mtime for p in shard_paths)
                if latest_shard_mtime < base_mtime:
                    await self.log(f"      ℹ {q_type} Shards are older than merged output - skipping merge")
                    return q_path
            except Exception:
                pass

        await self.log(f"      ℹ {q_type} Output is sharded ({total} parts). Merging...")
        
        try:
            gguf_split_bin = LlamaCppManager.get_gguf_split_path()
        except FileNotFoundError as e:
            await self.log(f"      ⚠ {q_type} Merge tool not found: {e}")
            return None
        
        merge_output = q_path
        if merge_output.exists():
            merge_output = q_path.with_suffix(".merged.gguf")
        if merge_output.exists():
            try:
                merge_output.unlink(missing_ok=True)
            except Exception:
                pass
        
        process = await asyncio.create_subprocess_exec(
            str(gguf_split_bin), "--merge", str(shard_paths[0]), str(merge_output),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self.running_processes.append(process)
        stdout, stderr = await process.communicate()
        try:
            self.running_processes.remove(process)
        except ValueError:
            pass
        
        if process.returncode != 0:
            error_output = (stderr.decode().strip() or stdout.decode().strip() or "Unknown error")
            await self.log(f"      ⚠ {q_type} Shard merge failed: {error_output[:200]}")
            return None
        
        if merge_output != q_path:
            try:
                q_path.unlink(missing_ok=True)
            except Exception as e:
                await self.log(f"      ⚠ {q_type} Failed to remove old output: {e}")
                return None
            try:
                merge_output.replace(q_path)
            except Exception as e:
                await self.log(f"      ⚠ {q_type} Failed to finalize merged file: {e}")
                return None
        
        await self.log(f"      ✓ {q_type} Shards merged into {q_path.name}")
        await self._cleanup_gguf_shards(shard_paths, q_type)
        return q_path

    async def upload_status_readme(self, quant_base_name: str, uploaded_files: List[str]):
        """Upload a temporary README with current conversion status."""
        if not (self.hf_token and self.new_repo_id and self.api):
            return

        try:
            app_version = await get_app_version()
            updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            completed_display = ", ".join(uploaded_files) if uploaded_files else "None yet"
            remaining = [q for q in self.quants_to_run if q not in uploaded_files]
            remaining_display = ", ".join(remaining) if remaining else "None"
            progress = f"{len(uploaded_files)}/{len(self.quants_to_run)}"

            # Build requester section
            requester_section = ""
            if self.requested_by:
                requester_section = f"\n- Requested by: [@{self.requested_by}](https://huggingface.co/{self.requested_by})"

            readme_content = f"""---
tags:
- gguf
- llama.cpp
- quantization
base_model: {self.hf_repo_id}
---

# {quant_base_name}-GGUF

This repository is being generated by GGUF Forge and will update as quants finish.

## Status
- Job ID: `{self.model_id}`{requester_section}
- Stage: Quantizing
- Updated: {updated_at}
- Progress: {progress}
- Completed quants: {completed_display}
- Remaining quants: {remaining_display}

## Ollama Support
Full Ollama support is provided by merging any sharded GGUF output into a single file after quantization.

---
*This README is temporary and will be replaced when conversion completes.*
*Converted automatically by [GGUF Forge](https://gguforge.com) {app_version}*

"""
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.api.upload_file(
                    path_or_fileobj=readme_content.encode('utf-8'),
                    path_in_repo="README.md",
                    repo_id=self.new_repo_id,
                    repo_type="model"
                )
            )
            await self.log("  ✓ Status README uploaded")
        except Exception as e:
            await self.log(f"  ⚠ Status README upload failed: {e}")

    async def notify_requester(self, quant_base_name: str, uploaded_files: List[str]):
        """Create a HuggingFace discussion to notify the requester that their model is ready."""
        if not (self.hf_token and self.new_repo_id and self.api and self.requested_by):
            return

        try:
            await self.log(f"  Notifying @{self.requested_by} via HuggingFace...")

            # Create a discussion on the new GGUF repo to notify the user
            discussion_title = f"🎉 Your GGUF conversion is complete!"
            discussion_body = f"""Hey @{self.requested_by}! 👋

Great news! Your requested GGUF conversion is now complete!

**Model**: [`{self.hf_repo_id}`](https://huggingface.co/{self.hf_repo_id})
**GGUF Repo**: [`{self.new_repo_id}`](https://huggingface.co/{self.new_repo_id})

## Available Quantizations
{chr(10).join([f'- **{q}**' for q in uploaded_files])}

## What's Next?
- Download your preferred quantization from the [Files tab](https://huggingface.co/{self.new_repo_id}/tree/main)
- Use with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.ai/), or any GGUF-compatible inference engine
- Star the repo if you find it useful! ⭐

---
*This notification was sent automatically by [GGUF Forge](https://gguforge.com)*
"""

            loop = asyncio.get_event_loop()
            from huggingface_hub import HfApi
            sync_api = HfApi(token=self.hf_token)

            await loop.run_in_executor(
                None,
                lambda: sync_api.create_discussion(
                    repo_id=self.new_repo_id,
                    repo_type="model",
                    title=discussion_title,
                    description=discussion_body
                )
            )
            await self.log(f"  ✓ Notification sent to @{self.requested_by}")
        except Exception as e:
            # Non-fatal - don't fail the whole job if notification fails
            await self.log(f"  ⚠ Could not notify requester: {str(e)[:100]}")

    def start_step(self, step_name: str):
        """Start timing a step."""
        import time
        self.step_times[step_name] = {"start": time.time(), "end": None}
    
    def end_step(self, step_name: str):
        """End timing a step."""
        import time
        if step_name in self.step_times:
            self.step_times[step_name]["end"] = time.time()
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_timing_summary(self) -> dict:
        """Get timing summary for the job."""
        import time
        summary = {
            "total_time": 0,
            "avg_quant_time": 0,
            "step_times": {}
        }
        
        if self.start_time:
            summary["total_time"] = time.time() - self.start_time
        
        for step, times in self.step_times.items():
            if times["start"] and times["end"]:
                duration = times["end"] - times["start"]
                summary["step_times"][step] = duration
        
        if self.quant_times:
            avg_time = sum(t for _, t in self.quant_times) / len(self.quant_times)
            summary["avg_quant_time"] = avg_time
        
        return summary
    
    async def update_transfer_progress(self, filename: str, progress: int, size: str = "", speed: str = "", transfer_type: str = "download"):
        """Update and broadcast transfer progress for a file."""
        self.transfer_files[filename] = {
            "name": filename,
            "progress": progress,
            "size": size,
            "speed": speed
        }
        
        # Broadcast the current transfer state
        files_list = list(self.transfer_files.values())
        await broadcast_transfer_progress(self.model_id, transfer_type, files_list)
    
    def clear_transfer_progress(self):
        """Clear transfer progress tracking."""
        self.transfer_files = {}
        return

    async def check_disk_space(self, required_gb: float):
        loop = asyncio.get_event_loop()
        total, used, free = await loop.run_in_executor(None, shutil.disk_usage, CACHE_DIR)
        free_gb = free / (2**30)

        import logging
        logger = logging.getLogger("GGUF_Forge")
        logger.info(f"check_disk_space called: ignore_space_check={self.ignore_space_check}, required={required_gb:.1f}GB, available={free_gb:.1f}GB")

        if self.ignore_space_check:
            await self.log(f"  ⚠ Space check BYPASSED by admin (Available: {free_gb:.1f}GB)")
            await self.log(f"  ⚠ Original requirement was: {required_gb:.1f}GB")
            await self.log(f"  ℹ Sequential processing requires much less space than conservative estimate")
            return
        
        await self.log(f"  Disk space check: Need {required_gb:.1f}GB, Available {free_gb:.1f}GB")
        if free_gb < required_gb:
            raise Exception(f"Insufficient disk space. Required: {required_gb:.1f}GB, Available: {free_gb:.1f}GB")
        await self.log(f"  ✓ Sufficient disk space")

    async def get_model_size_gb(self) -> float:
        """Get model size from HuggingFace API in GB."""
        try:
            hf_token = os.getenv("HF_TOKEN")
            api = HfApi(token=hf_token)
            
            # Run blocking API call in executor
            loop = asyncio.get_event_loop()
            model_info = await loop.run_in_executor(
                None,
                lambda: api.model_info(self.hf_repo_id, files_metadata=True)
            )
            
            total_bytes = 0
            if model_info.siblings:
                for sibling in model_info.siblings:
                    if hasattr(sibling, 'size') and sibling.size:
                        total_bytes += sibling.size
            
            size_gb = total_bytes / (2**30)
            return size_gb
        except Exception as e:
            await self.log(f"  ⚠ Could not fetch model size: {e}")
            return 10.0  # Default fallback

    async def cleanup(self):
        """Remove all downloaded and generated files."""
        await self.log("Starting cleanup...")
        loop = asyncio.get_event_loop()
        try:
            # Remove downloaded model directory
            if self.model_dir and Path(self.model_dir).exists():
                await self.log(f"Removing downloaded model: {self.model_dir}")
                await loop.run_in_executor(None, lambda: shutil.rmtree(self.model_dir, ignore_errors=True))
            
            # Remove FP16 file
            if self.fp16_path and self.fp16_path.exists():
                await self.log(f"Removing FP16 file: {self.fp16_path}")
                await loop.run_in_executor(None, lambda: self.fp16_path.unlink(missing_ok=True))
            
            # Remove all quantized files
            for q_path in self.quant_paths:
                if q_path.exists():
                    await self.log(f"Removing quant file: {q_path}")
                    await loop.run_in_executor(None, lambda p=q_path: p.unlink(missing_ok=True))
            
            await self.log("Cleanup completed.")
        except Exception as e:
            await self.log(f"Cleanup error (non-fatal): {e}")

    async def run_pipeline(self):
        import time
        import multiprocessing
        error_details = ""
        try:
            # Register in global registry for termination support
            running_workflows[self.model_id] = self
            
            self.start_time = time.time()
            await self.status("initializing")
            await self.progress(0)
            await self.log("━━━ GGUF Forge Pipeline Started ━━━")
            await self.log(f"Job ID: {self.model_id}")
            await self.log(f"Model: {self.hf_repo_id}")
            await self.log(f"Version: {await get_app_version()}")
            await self.log("")
            
            # 1. Setup Llama
            self.check_terminated()
            self.start_step("setup")
            await self.log("▶ STEP 1: Setting up llama.cpp...")
            await self.log("  Checking llama.cpp installation...")
            if self.force_llama_update:
                await self.log("  Force update enabled - will fetch latest llama.cpp commit...")
            await LlamaCppManager.clone_repo(force=self.force_llama_update)
            self.check_terminated()
            await self.log("  Building llama.cpp (this may take a while)...")
            await LlamaCppManager.build()
            quantize_bin = LlamaCppManager.get_quantize_path()
            await self.log(f"  ✓ llama-quantize ready: {quantize_bin.name}")
            self.end_step("setup")
            await self.progress(10)
            await self.log("")

            # Check if FP16 file already exists (crash recovery)
            self.fp16_path = CACHE_DIR / f"{self.hf_repo_id.replace('/', '-')}-f16.gguf"
            fp16_exists = self.fp16_path.exists() and self.fp16_path.stat().st_size > 0

            # 2. Download (skip if FP16 already exists)
            self.check_terminated()
            if fp16_exists and self.resume_mode:
                await self.log("▶ STEP 2: Download SKIPPED (FP16 file exists from previous run)")
                await self.log(f"  ✓ Using existing FP16 file: {self.fp16_path.name}")
                await self.log("")
            else:
                self.start_step("download")
                await self.status("downloading")
                await self.log("▶ STEP 2: Downloading model from HuggingFace...")
                await self.log(f"  Source: https://huggingface.co/{self.hf_repo_id}")

                # Get actual model size and calculate required space
                model_size_gb = await self.get_model_size_gb()
                await self.log(f"  Model size: {model_size_gb:.2f}GB")
                required_gb = max(5.0, model_size_gb * 3)
                await self.check_disk_space(required_gb)

                # Clear any previous transfer progress
                self.clear_transfer_progress()

                # Get list of files to download
                api = HfApi()
                loop = asyncio.get_event_loop()
                try:
                    repo_files = await loop.run_in_executor(
                        None,
                        lambda: api.list_repo_files(self.hf_repo_id)
                    )
                    # Filter for model files (safetensors, bin, json, etc.)
                    download_files = [f for f in repo_files if any(f.endswith(ext) for ext in
                        ['.safetensors', '.bin', '.pt', '.pth', '.json', '.txt', '.model', '.tiktoken', '.py'])]

                    await self.log(f"  Found {len(download_files)} files to download")

                    # Download files with progress tracking
                    local_dir = CACHE_DIR / self.hf_repo_id
                    local_dir.mkdir(parents=True, exist_ok=True)

                    total_files = len(download_files)
                    for idx, filename in enumerate(download_files):
                        self.check_terminated()
                        short_name = filename.split('/')[-1] if '/' in filename else filename

                        # Initialize progress for this file
                        await self.update_transfer_progress(short_name, 0, "", "Starting...", "download")

                        # Download file in thread pool
                        try:
                            await loop.run_in_executor(
                                None,
                                lambda f=filename: hf_hub_download(
                                    repo_id=self.hf_repo_id,
                                    filename=f,
                                    local_dir=local_dir,
                                    local_dir_use_symlinks=False
                                )
                            )
                            # Mark as complete
                            await self.update_transfer_progress(short_name, 100, "", "Complete", "download")
                        except Exception as e:
                            await self.log(f"  ⚠ Failed to download {short_name}: {e}")
                            await self.update_transfer_progress(short_name, -1, "", "Failed", "download")

                        # Update overall progress (10-30% for download step)
                        step_progress = 10 + int((idx + 1) / total_files * 20)
                        await self.progress(step_progress)

                    self.model_dir = str(local_dir)

                except Exception as e:
                    # Fallback to snapshot_download if file listing fails
                    await self.log(f"  Using batch download...")
                    self.model_dir = await loop.run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id=self.hf_repo_id,
                            local_dir=CACHE_DIR / self.hf_repo_id,
                            local_dir_use_symlinks=False
                        )
                    )

                # Clear download progress display
                self.clear_transfer_progress()
                await broadcast_transfer_progress(self.model_id, "download", [])

                await self.log(f"  ✓ Downloaded to {self.model_dir}")
                self.end_step("download")
                await self.progress(30)
                await self.log("")

            # 3. Convert to FP16 (skip if FP16 already exists)
            self.check_terminated()
            if fp16_exists and self.resume_mode:
                await self.log("▶ STEP 3: Conversion SKIPPED (FP16 file exists from previous run)")
                await self.log(f"  ✓ Using existing FP16 file: {self.fp16_path.name}")
                await self.log("")
            else:
                self.start_step("convert")
                await self.status("converting")
                await self.log("▶ STEP 3: Converting to GGUF format (FP16)...")
                convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

                cmd = [sys.executable, str(convert_script), str(self.model_dir), "--outfile", str(self.fp16_path), "--outtype", "f16"]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                self.running_processes.append(process)

                async for line in process.stdout:
                    decoded = line.decode().strip()
                    if decoded:
                        await self.log(f"  {decoded}")

                returncode = await process.wait()
                try:
                    self.running_processes.remove(process)
                except ValueError:
                    pass

                if returncode != 0:
                    raise Exception("Conversion to GGUF failed. Check logs for details.")

                await self.log(f"  ✓ FP16 conversion complete: {self.fp16_path.name}")
                self.end_step("convert")
                await self.progress(50)

                # Clean up safetensors immediately - only the GGUF file is needed for quantization
                await self.cleanup_safetensors()
                await self.log("")

            # 4. Quantize and Upload (each quant is uploaded immediately after creation, then deleted)
            self.check_terminated()
            self.start_step("quantize")
            await self.status("quantizing")
            await self.log("▶ STEP 4: Quantizing and uploading each format...")
            quant_base_name = self.hf_repo_id.split("/")[-1]
            self.hf_token = os.getenv("HF_TOKEN")
            
            # Get current user's HuggingFace username to create repo under their account
            self.api = HfApi(token=self.hf_token)
            
            if self.hf_token:
                try:
                    loop = asyncio.get_event_loop()
                    user_info = await loop.run_in_executor(None, self.api.whoami)
                    hf_username = user_info.get("name") or user_info.get("user")
                    self.new_repo_id = f"{hf_username}/{quant_base_name}-GGUF"
                    await self.log(f"  Target repo: {self.new_repo_id}")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: create_repo(self.new_repo_id, repo_type="model", token=self.hf_token, exist_ok=True)
                    )
                    await self.log(f"  ✓ Repo ready: https://huggingface.co/{self.new_repo_id}")
                except Exception as e:
                    await self.log(f"  ⚠ Could not create repo: {e}")
                    self.new_repo_id = None
            else:
                await self.log("  ⚠ No HF_TOKEN set - files will be quantized but not uploaded")

            await self.log("")
            uploaded_files = []  # List of quant types that were uploaded

            # Determine which quants to process (use custom list if set, skip already completed ones)
            quants_to_process = [q for q in self.quants_to_run if q not in self.completed_quants]

            # Sort quants by priority order
            priority_order = await get_quant_priority_order()
            priority_map = {q: i for i, q in enumerate(priority_order)}
            quants_to_process.sort(key=lambda q: priority_map.get(q, 999))

            await self.log(f"  Quant priority order: {', '.join(quants_to_process)}")

            if self.resume_mode and self.completed_quants:
                await self.log(f"  📋 Resume mode: {len(self.completed_quants)} quants already completed")
                await self.log(f"     Already done: {', '.join(self.completed_quants)}")
                await self.log(f"     Remaining: {len(quants_to_process)} quants to process")
                uploaded_files = list(self.completed_quants)  # Count already uploaded as successful
                await self.log("")
            elif len(self.quants_to_run) < len(QUANTS):
                # User requested specific quants
                await self.log(f"  📋 Custom quants requested: {', '.join(self.quants_to_run)}")
                await self.log("")
            
            if self.hf_token and self.new_repo_id:
                await self.upload_status_readme(quant_base_name, uploaded_files)
            
            total_quants = len(self.quants_to_run)
            completed_count = len(self.completed_quants)
            
            # Detect CPU cores
            total_cores = multiprocessing.cpu_count()
            # Calculate threads per job
            num_parallel = max(1, min(PARALLEL_QUANT_JOBS, len(quants_to_process)))
            threads_per_job = max(1, total_cores // num_parallel)
            
            await self.log(f"  CPU cores: {total_cores} total")
            await self.log(f"  Parallel jobs: {num_parallel}")
            await self.log(f"  Threads per job: {threads_per_job}")
            await self.log(f"  Mode: Parallel quantize ({num_parallel} at a time)")
            await self.log("")
            
            # Semaphore to limit parallel quantization jobs
            semaphore = asyncio.Semaphore(num_parallel)
            
            async def process_single_quant(q_type: str, overall_idx: int):
                async with semaphore:
                    self.check_terminated()
                    await self.log(f"  [{overall_idx}/{total_quants}] Starting {q_type}...")
                    
                    q_path = CACHE_DIR / f"{quant_base_name}.{q_type}.gguf"
                    quant_start = time.time()
                    
                    try:
                        # === QUANTIZE ===
                        env = os.environ.copy()
                        if quantize_bin and quantize_bin.parent:
                            current_ld = env.get('LD_LIBRARY_PATH', '')
                            env['LD_LIBRARY_PATH'] = f"{quantize_bin.parent}:{current_ld}"
                        
                        # Apply threads constraint
                        env['OMP_NUM_THREADS'] = str(threads_per_job)
                        env['MKL_NUM_THREADS'] = str(threads_per_job)
                        env['OPENBLAS_NUM_THREADS'] = str(threads_per_job)
                        
                        process = await asyncio.create_subprocess_exec(
                            str(quantize_bin), str(self.fp16_path), str(q_path), q_type,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env
                        )
                        self.running_processes.append(process)
                        stdout, stderr = await process.communicate()
                        try:
                            self.running_processes.remove(process)
                        except ValueError:
                            pass
                        
                        quant_duration = time.time() - quant_start
                        
                        if process.returncode != 0:
                            await self.log(f"      ⚠ {q_type} quantization failed: {stderr.decode()[:200]}")
                            return
                        
                        self.quant_times.append((q_type, quant_duration))
                        await self.log(f"      ✓ {q_type} Quantized ({self.format_duration(quant_duration)})")
                        
                        # Ensure output is a single GGUF file (merge shards if needed)
                        merged_path = await self.ensure_unsharded_gguf(q_path, q_type)
                        if not merged_path:
                            return
                        q_path = merged_path
                        
                        # === UPLOAD ===
                        if self.hf_token and self.new_repo_id:
                            self.check_terminated()
                            
                            filename = f"{quant_base_name}.{q_type}.gguf"
                            file_size = q_path.stat().st_size if q_path.exists() else 0
                            size_str = f"{file_size / (1024**3):.2f}GB" if file_size > 0 else ""
                            
                            await self.update_transfer_progress(filename, 0, size_str, "Uploading...", "upload")
                            
                            try:
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(
                                    None,
                                    lambda: self.api.upload_file(
                                        path_or_fileobj=q_path,
                                        path_in_repo=filename,
                                        repo_id=self.new_repo_id,
                                        repo_type="model"
                                    )
                                )
                                
                                await self.update_transfer_progress(filename, 100, size_str, "Complete", "upload")
                                await self.log(f"      ✓ {q_type} Uploaded to HuggingFace")
                                uploaded_files.append(q_type)
                                
                                # Save progress to DB for resume capability
                                await self.save_completed_quant(q_type)
                                
                            except Exception as e:
                                await self.update_transfer_progress(filename, -1, size_str, "Failed", "upload")
                                await self.log(f"      ⚠ {q_type} Upload failed: {e}")
                                # Don't delete the file if upload failed - keep for potential manual recovery or retry
                                return
                        else:
                            await self.log(f"      ℹ {q_type} Skipping upload (no HF token)")
                            uploaded_files.append(q_type)
                        
                        # === DELETE QUANT FILE ===
                        try:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, lambda: q_path.unlink(missing_ok=True))
                        except Exception as e:
                            await self.log(f"      ⚠ {q_type} Failed to delete: {e}")
                        
                        # Clear transfer progress for this file
                        self.transfer_files.pop(f"{quant_base_name}.{q_type}.gguf", None)
                        await broadcast_transfer_progress(self.model_id, "upload", list(self.transfer_files.values()))
                        
                    except Exception as e:
                        await self.log(f"      ⚠ {q_type} error: {e}")
                    
                    # Update overall progress
                    current_completed = len(uploaded_files)
                    step_progress = 50 + int(current_completed / total_quants * 40)
                    await self.progress(step_progress)

            # Create tasks for all quants
            tasks = []
            for q_type in quants_to_process:
                overall_idx = self.quants_to_run.index(q_type) + 1
                tasks.append(process_single_quant(q_type, overall_idx))
            
            # Run all tasks concurrently with semaphore limit
            if tasks:
                await asyncio.gather(*tasks)
            
            self.end_step("quantize")
            await self.log("")
            await self.log(f"  ✓ Completed {len(uploaded_files)}/{total_quants} quants")
            
            await self.progress(90)
            
            await self.log("")

            # 5. Readme
            if self.hf_token and uploaded_files and self.new_repo_id:
                await self.log("▶ STEP 5: Generating README...")
                
                # Get app version (async)
                app_version = await get_app_version()
                
                # Get timing summary
                timing = self.get_timing_summary()
                total_time_str = self.format_duration(timing["total_time"])
                avg_quant_str = self.format_duration(timing["avg_quant_time"]) if timing["avg_quant_time"] > 0 else "N/A"
                
                # Build timing details
                timing_details = []
                if "download" in timing["step_times"]:
                    timing_details.append(f"- Download: {self.format_duration(timing['step_times']['download'])}")
                if "convert" in timing["step_times"]:
                    timing_details.append(f"- FP16 Conversion: {self.format_duration(timing['step_times']['convert'])}")
                if "quantize" in timing["step_times"]:
                    timing_details.append(f"- Quantization: {self.format_duration(timing['step_times']['quantize'])}")
                
                timing_section = "\n".join(timing_details)

                # Build requester section
                requester_section = ""
                if self.requested_by:
                    requester_section = f"""
## 🙏 Requested By

This conversion was requested by [@{self.requested_by}](https://huggingface.co/{self.requested_by}).
"""

                readme_content = f"""---
tags:
- gguf
- llama.cpp
- quantization
base_model: {self.hf_repo_id}
---

# {quant_base_name}-GGUF

This model was converted to GGUF format from [`{self.hf_repo_id}`](https://huggingface.co/{self.hf_repo_id}) using GGUF Forge.
{requester_section}
## Quants
The following quants are available:
{', '.join(uploaded_files)}

## Ollama Support
Full Ollama support is provided by merging any sharded GGUF output into a single file after quantization.

## Conversion Stats

| Metric | Value |
|--------|-------|
| Job ID | `{self.model_id}` |
| GGUF Forge Version | {app_version} |
| Total Time | {total_time_str} |
| Avg Time per Quant | {avg_quant_str} |

### Step Breakdown
{timing_section}

## 🚀 Convert Your Own Models

**Want to convert more models to GGUF?**

👉 **[gguforge.com](https://gguforge.com)** — Free hosted GGUF conversion service. Login with HuggingFace and request conversions instantly!

## Links

 - 🌐 **Free Hosted Service**: [gguforge.com](https://gguforge.com)
 - 🛠️ Self-host GGUF Forge: [GitHub](https://github.com/Akicuo/automaticConversion)
 - 📦 llama.cpp (quantization engine): [GitHub](https://github.com/ggerganov/llama.cpp)
 - 💬 Community & Support: [Discord](https://discord.gg/4vafUgVX3a)


---
*Converted automatically by [GGUF Forge](https://gguforge.com) {app_version}*

"""
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.api.upload_file(
                        path_or_fileobj=readme_content.encode('utf-8'),
                        path_in_repo="README.md",
                        repo_id=self.new_repo_id,
                        repo_type="model"
                    )
                )
                await self.log(f"  ✓ README uploaded")
                await self.log("")

                # Notify the requester via HuggingFace discussion
                if self.requested_by and self.new_repo_id:
                    await self.notify_requester(quant_base_name, uploaded_files)

            # Log timing summary
            timing = self.get_timing_summary()
            await self.status("complete")
            await self.progress(100)
            await self.log("━━━ Pipeline Complete ━━━")
            await self.log(f"✓ Successfully converted {self.hf_repo_id}")
            await self.log(f"✓ Job ID: {self.model_id}")
            await self.log(f"✓ Total Time: {self.format_duration(timing['total_time'])}")
            if timing["avg_quant_time"] > 0:
                await self.log(f"✓ Avg Time per Quant: {self.format_duration(timing['avg_quant_time'])}")
            if self.new_repo_id:
                await self.log(f"✓ Uploaded to: https://huggingface.co/{self.new_repo_id}")
            await self._update_db(completed_at=datetime.now())

        except Exception as e:
            error_details = traceback.format_exc()
            await self.log("")
            if self.terminated:
                await self.log("━━━ Pipeline Terminated ━━━")
                await self.log("⚠ Job was terminated by administrator")
                await self._update_db(error_details="Terminated by administrator", status="terminated")
            else:
                await self.log("━━━ Pipeline Failed ━━━")
                await self.log(f"✗ ERROR: {str(e)}")
                await self._update_db(error_details=error_details, status="error")
            import logging
            logging.getLogger("GGUF_Forge").exception("Pipeline failed")
        
        finally:
            # Remove from global registry
            running_workflows.pop(self.model_id, None)
            
            # Always cleanup files
            await self.log("")
            await self.log("▶ Cleanup...")
            await self.cleanup()
