"""
Request system routes for GGUF Forge.
"""
import uuid
import json

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks

from database import get_db_connection
from models import ModelRequestSubmit, RejectRequest, ApproveRequestBody
from workflow import ModelWorkflow, get_quants_list
from websocket_manager import broadcast_requests_update, broadcast_my_requests_update

router = APIRouter(prefix="/api/requests")

# Dependencies - will be set by main app
_require_admin_func = None
_get_current_user_func = None
_spam_protection = None


def configure(admin_dep, user_dep, spam_prot):
    """Configure routes with dependencies."""
    global _require_admin_func, _get_current_user_func, _spam_protection
    _require_admin_func = admin_dep
    _get_current_user_func = user_dep
    _spam_protection = spam_prot


async def get_admin(request: Request):
    """Dependency that uses the configured admin check."""
    if _require_admin_func:
        return await _require_admin_func(request)
    raise HTTPException(status_code=500, detail="Admin dependency not configured")


def validate_quants(quants: list) -> list:
    """Validate quant names against available quants. Returns valid quants only."""
    available = get_quants_list()
    if not quants:
        return []
    return [q for q in quants if q in available]


@router.post("/submit")
async def submit_request(req: ModelRequestSubmit, request: Request):
    """Submit a request for a model to be converted (requires login)."""
    user = await _get_current_user_func(request)
    
    # Require login for submissions (anti-spam)
    if not user:
        raise HTTPException(status_code=401, detail="Login required to submit requests")
    
    requester = user['username']
    
    # Check spam protection - pending limit
    allowed, reason = await _spam_protection.check_pending_limit(requester)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    # Check spam protection - hourly rate
    allowed, reason = await _spam_protection.can_submit(requester)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    # Validate requested quants
    requested_quants = validate_quants(req.requested_quants) if req.requested_quants else []
    requested_quants_json = json.dumps(requested_quants) if requested_quants else ""
    
    conn = await get_db_connection()
    # Check if already requested
    await conn.execute("SELECT * FROM requests WHERE hf_repo_id = ? AND status = 'pending'", (req.hf_repo_id,))
    existing = await conn.fetchone()
    if existing:
        await conn.close()
        return {"status": "already_requested", "message": "This model has already been requested."}
    
    await conn.execute(
        "INSERT INTO requests (hf_repo_id, requested_by, status, requested_quants) VALUES (?, ?, ?, ?)",
        (req.hf_repo_id, requester, "pending", requested_quants_json)
    )
    await conn.commit()
    await conn.close()
    
    # Record submission for rate limiting
    await _spam_protection.record_submission(requester)
    
    # Broadcast update via WebSocket
    await broadcast_requests_update()
    await broadcast_my_requests_update()
    
    quant_msg = f" with quants: {', '.join(requested_quants)}" if requested_quants else " (all quants)"
    return {"status": "submitted", "message": f"Your request has been submitted for admin review{quant_msg}."}


@router.get("/all")
async def get_all_requests(user = Depends(get_admin)):
    """Admin only: View all pending requests."""
    conn = await get_db_connection()
    # Server-side filtering: only fetch pending requests for better performance
    await conn.execute("SELECT * FROM requests WHERE status = 'pending' ORDER BY created_at DESC")
    requests = await conn.fetchall()
    await conn.close()
    
    # Parse quant JSON for each request
    result = []
    for r in requests:
        r_dict = r.to_dict()
        # Parse requested_quants JSON if present
        if r_dict.get('requested_quants'):
            try:
                r_dict['requested_quants'] = json.loads(r_dict['requested_quants'])
            except:
                r_dict['requested_quants'] = []
        else:
            r_dict['requested_quants'] = []
        # Parse approved_quants JSON if present
        if r_dict.get('approved_quants'):
            try:
                r_dict['approved_quants'] = json.loads(r_dict['approved_quants'])
            except:
                r_dict['approved_quants'] = []
        else:
            r_dict['approved_quants'] = []
        result.append(r_dict)
    
    return result


@router.post("/{request_id}/approve")
async def approve_request(
    request_id: int, 
    background_tasks: BackgroundTasks, 
    body: ApproveRequestBody = None,
    user = Depends(get_admin)
):
    """Admin only: Approve a request and start conversion.
    
    Admin can optionally modify the quant selection via body.approved_quants.
    If not provided, uses the user's requested_quants or all quants if empty.
    """
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,))
    req = await conn.fetchone()
    
    if not req:
        await conn.close()
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Determine which quants to use
    available_quants = get_quants_list()
    
    # Priority: admin override > user request > all quants
    if body and body.approved_quants:
        # Admin explicitly set quants
        approved_quants = validate_quants(body.approved_quants)
        if not approved_quants:
            await conn.close()
            raise HTTPException(status_code=400, detail="No valid quants specified")
    else:
        # Use user's requested quants if available
        user_quants_json = req.get('requested_quants', '')
        if user_quants_json:
            try:
                user_quants = json.loads(user_quants_json)
                approved_quants = validate_quants(user_quants)
            except:
                approved_quants = []
        else:
            approved_quants = []
        
        # If no specific quants requested, use all
        if not approved_quants:
            approved_quants = available_quants
    
    approved_quants_json = json.dumps(approved_quants)
    
    # Get ignore_space_check flag from request body (admin only)
    ignore_space_check = body.ignore_space_check if body and body.ignore_space_check else False
    
    # Update request status and approved quants
    await conn.execute(
        "UPDATE requests SET status = 'approved', approved_quants = ? WHERE id = ?", 
        (approved_quants_json, request_id)
    )
    await conn.commit()
    
    # Start the conversion
    hf_repo_id = req['hf_repo_id']
    new_id = str(uuid.uuid4())
    
    # Build initial log message
    log_msg = f"Queued from approved request... Quants: {', '.join(approved_quants)}"
    if ignore_space_check:
        log_msg += "\n⚠ Admin override: Space check disabled"
    
    # Get requestor before closing connection
    requested_by = req.get('requested_by', '')
    
    await conn.execute(
        "INSERT INTO models (id, hf_repo_id, status, progress, log, error_details, ignore_space_check, requested_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (new_id, hf_repo_id, "pending", 0, log_msg, "", 1 if ignore_space_check else 0, requested_by)
    )
    await conn.commit()
    await conn.close()
    
    # Pass approved quants, space check flag, and requestor to workflow
    workflow = ModelWorkflow(new_id, hf_repo_id, quants_to_run=approved_quants, ignore_space_check=ignore_space_check, requested_by=requested_by)
    background_tasks.add_task(workflow.run_pipeline)
    
    # Broadcast update via WebSocket
    await broadcast_requests_update()
    await broadcast_my_requests_update()
    
    return {"status": "approved", "model_id": new_id, "quants": approved_quants}


@router.post("/{request_id}/reject")
async def reject_request(request_id: int, body: RejectRequest = None, user = Depends(get_admin)):
    """Admin only: Reject a request with optional reason."""
    reason = body.reason if body else ""
    conn = await get_db_connection()
    await conn.execute("UPDATE requests SET status = 'rejected', decline_reason = ? WHERE id = ?", (reason, request_id))
    await conn.commit()
    await conn.close()
    
    # Broadcast update via WebSocket
    await broadcast_requests_update()
    await broadcast_my_requests_update()
    
    return {"status": "rejected"}


@router.get("/my")
async def get_my_requests(request: Request):
    """Get current user's request history."""
    user = await _get_current_user_func(request)
    if not user:
        return []
    
    username = user['username']
    conn = await get_db_connection()
    await conn.execute(
        "SELECT * FROM requests WHERE requested_by = ? ORDER BY created_at DESC",
        (username,)
    )
    requests = await conn.fetchall()
    await conn.close()
    
    # Parse quant JSON for each request
    result = []
    for r in requests:
        r_dict = r.to_dict()
        if r_dict.get('requested_quants'):
            try:
                r_dict['requested_quants'] = json.loads(r_dict['requested_quants'])
            except:
                r_dict['requested_quants'] = []
        else:
            r_dict['requested_quants'] = []
        if r_dict.get('approved_quants'):
            try:
                r_dict['approved_quants'] = json.loads(r_dict['approved_quants'])
            except:
                r_dict['approved_quants'] = []
        else:
            r_dict['approved_quants'] = []
        result.append(r_dict)
    
    return result
