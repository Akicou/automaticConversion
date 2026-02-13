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
