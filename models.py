"""
Pydantic models for API requests/responses.
"""
from typing import Optional, List
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class ProcessRequest(BaseModel):
    model_id: str
    quants: Optional[List[str]] = None  # If None, uses all quants
    force_llama_update: Optional[bool] = False  # If True, forcefully update llama.cpp to latest commit
    ignore_space_check: Optional[bool] = False  # If True, bypass conservative disk space checks
    enable_shard_merging: Optional[bool] = True  # If True, merge GGUF shards into single file for Ollama compatibility
    convert_outtype: Optional[str] = None  # Fork-specific compact GGUF outtype (e.g. "q8_0", "iq2_xxs"). When set, the converter produces a single direct output and llama-quantize is skipped.


class ModelRequestSubmit(BaseModel):
    hf_repo_id: str
    requested_quants: Optional[List[str]] = None  # e.g., ["Q4_K_M", "Q8_0"] - None means all quants


class ApproveRequestBody(BaseModel):
    """Admin can optionally modify quant selection when approving."""
    approved_quants: Optional[List[str]] = None  # If None, uses requested_quants or all quants
    ignore_space_check: Optional[bool] = False  # If True, bypass conservative disk space checks
    enable_shard_merging: Optional[bool] = True  # If True, merge GGUF shards into single file for Ollama compatibility


class RejectRequest(BaseModel):
    reason: Optional[str] = ""


class TicketMessage(BaseModel):
    message: str


class CreateTicketRequest(BaseModel):
    request_id: int
    initial_message: Optional[str] = ""


class ContinueRequestBody(BaseModel):
    """Admin options when continuing an interrupted job."""
    ignore_space_check: Optional[bool] = False  # If True, bypass conservative disk space checks


class LocalProcessRequest(BaseModel):
    """Admin: quantize an already-downloaded model from disk."""
    path: str  # Absolute path to the model directory (must be inside CACHE_DIR)
    repo_id: str  # Used as the synthetic hf_repo_id for tracking (e.g., "org/repo" or "local/name")
    quants: Optional[List[str]] = None
    keep_local_only: Optional[bool] = False  # If True, skip HF upload and write GGUFs into <path>/gguf/
    ignore_space_check: Optional[bool] = False
    enable_shard_merging: Optional[bool] = True
    convert_outtype: Optional[str] = None  # Fork-specific compact GGUF outtype; when set, skips llama-quantize.


class LlamaCppSourceConfig(BaseModel):
    """Admin: configure which llama.cpp fork/folder GGUF Forge uses."""
    repo: str  # Git URL, e.g. https://github.com/<owner>/llama.cpp
    dir: str   # Absolute or relative folder path for the local checkout
    outtypes: Optional[List[str]] = None  # Fork-specific compact --outtype values (e.g. ["iq2_xxs", "q8_0"]).
