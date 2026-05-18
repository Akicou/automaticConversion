"""
User Settings Routes
Handles user preferences and settings management
"""
import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List

from database import get_db_connection, get_app_config, set_app_config
from models import LlamaCppSourceConfig

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Admin dependency - will be set by main app
_require_admin_func = None


def configure(admin_dep):
    """Configure routes with dependencies."""
    global _require_admin_func
    _require_admin_func = admin_dep


async def get_admin(request: Request):
    """Dependency that uses the configured admin check."""
    if _require_admin_func:
        return await _require_admin_func(request)
    raise HTTPException(status_code=500, detail="Admin dependency not configured")


class SettingsUpdate(BaseModel):
    default_quants: Optional[List[str]] = None
    theme: Optional[str] = None  # 'dark', 'light', or 'auto'
    notifications_enabled: Optional[bool] = None
    auto_refresh_interval: Optional[int] = None


@router.get("/user/{username}")
async def get_user_settings(username: str):
    """Get settings for a specific user."""
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM user_preferences WHERE hf_username = ?", (username,))
    settings = await conn.fetchone()
    await conn.close()

    if not settings:
        # Return default settings
        return {
            "hf_username": username,
            "default_quants": None,
            "theme": "dark",
            "notifications_enabled": True,
            "auto_refresh_interval": 30
        }

    # Parse JSON fields
    result = dict(settings)
    if result.get('default_quants'):
        import json
        try:
            result['default_quants'] = json.loads(result['default_quants'])
        except:
            result['default_quants'] = None

    return result


@router.post("/user/{username}")
async def update_user_settings(username: str, settings: SettingsUpdate):
    """Update settings for a specific user."""
    conn = await get_db_connection()

    # Check if settings exist
    await conn.execute("SELECT * FROM user_preferences WHERE hf_username = ?", (username,))
    existing = await conn.fetchone()

    # Prepare values
    default_quants_json = None
    if settings.default_quants is not None:
        import json
        default_quants_json = json.dumps(settings.default_quants)

    if existing:
        # Update existing settings
        update_fields = []
        update_values = []

        if settings.default_quants is not None:
            update_fields.append("default_quants = ?")
            update_values.append(default_quants_json)
        if settings.theme is not None:
            update_fields.append("theme = ?")
            update_values.append(settings.theme)
        if settings.notifications_enabled is not None:
            update_fields.append("notifications_enabled = ?")
            update_values.append(1 if settings.notifications_enabled else 0)
        if settings.auto_refresh_interval is not None:
            update_fields.append("auto_refresh_interval = ?")
            update_values.append(settings.auto_refresh_interval)

        if update_fields:
            update_values.append(username)
            query = f"UPDATE user_preferences SET {', '.join(update_fields)} WHERE hf_username = ?"
            await conn.execute(query, update_values)
    else:
        # Create new settings
        await conn.execute(
            """INSERT INTO user_preferences (hf_username, default_quants, theme, notifications_enabled, auto_refresh_interval)
            VALUES (?, ?, ?, ?, ?)""",
            (
                username,
                default_quants_json,
                settings.theme if settings.theme is not None else 'dark',
                1 if settings.notifications_enabled else 0 if settings.notifications_enabled is not None else 1,
                settings.auto_refresh_interval if settings.auto_refresh_interval is not None else 30
            )
        )

    await conn.commit()
    await conn.close()

    return {"status": "updated"}


@router.get("/quants")
async def get_available_quants():
    """Get list of available quantization types."""
    from workflow import get_quants_list
    return {"quants": get_quants_list()}


class QuantPriorityUpdate(BaseModel):
    priority_order: List[str]  # Ordered list of quant types (e.g., ["Q4_K_M", "Q5_K_M", ...])


@router.get("/quant-priority")
async def get_quant_priority():
    """Get the current quant priority order. Returns default order if not configured."""
    from workflow import get_quants_list

    conn = await get_db_connection()
    await conn.execute("SELECT priority_order FROM quant_priority WHERE id = 1")
    row = await conn.fetchone()
    await conn.close()

    default_quants = get_quants_list()

    if row and row.get('priority_order'):
        try:
            custom_order = json.loads(row['priority_order'])
            # Validate that all quants in custom order are valid
            valid_order = [q for q in custom_order if q in default_quants]
            # Add any missing quants at the end
            missing = [q for q in default_quants if q not in valid_order]
            return {
                "priority_order": valid_order + missing,
                "is_custom": len(valid_order) > 0,
                "default_order": default_quants
            }
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "priority_order": default_quants,
        "is_custom": False,
        "default_order": default_quants
    }


@router.post("/quant-priority")
async def set_quant_priority(request: Request, body: QuantPriorityUpdate):
    """Admin only: Set the quant priority order."""
    await get_admin(request)

    from workflow import get_quants_list

    available_quants = get_quants_list()

    # Validate all quants in the priority order
    for q in body.priority_order:
        if q not in available_quants:
            raise HTTPException(status_code=400, detail=f"Invalid quant type: {q}")

    # Check for duplicates
    if len(body.priority_order) != len(set(body.priority_order)):
        raise HTTPException(status_code=400, detail="Duplicate quant types in priority order")

    conn = await get_db_connection()
    await conn.execute(
        "UPDATE quant_priority SET priority_order = ?, updated_at = ? WHERE id = 1",
        (json.dumps(body.priority_order), datetime.now())
    )
    await conn.commit()
    await conn.close()

    return {
        "status": "updated",
        "priority_order": body.priority_order
    }


@router.post("/quant-priority/reset")
async def reset_quant_priority(request: Request):
    """Admin only: Reset quant priority to default order."""
    await get_admin(request)

    conn = await get_db_connection()
    await conn.execute(
        "UPDATE quant_priority SET priority_order = '', updated_at = ? WHERE id = 1",
        (datetime.now(),)
    )
    await conn.commit()
    await conn.close()

    from workflow import get_quants_list
    return {
        "status": "reset",
        "priority_order": get_quants_list()
    }


# ---------------- llama.cpp source configuration (admin only) ----------------

def _resolve_dir(raw: str) -> Path:
    """Resolve a user-supplied dir string against the app base dir for relative paths."""
    import managers
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    base = managers.BASE_DIR or Path.cwd()
    return (base / p).resolve()


async def _validate_source(repo: str, raw_dir: str, outtypes: Optional[List[str]] = None) -> dict:
    """Validate a candidate (repo, dir [, outtypes]) combo. Does not mutate anything."""
    import managers

    repo = (repo or "").strip()
    raw_dir = (raw_dir or "").strip()

    if not repo:
        return {"valid": False, "conflict": False, "reason": "Repo URL is required."}
    if not raw_dir:
        return {"valid": False, "conflict": False, "reason": "Folder path is required."}

    raw_outtypes = list(outtypes or [])
    normalized_outtypes = managers.normalize_outtypes(raw_outtypes)
    # Reject entries whose shape is invalid (anything that survived stripping but didn't normalize).
    bad = [v for v in raw_outtypes
           if isinstance(v, str) and v.strip() and v.strip().lower() not in normalized_outtypes]
    if bad:
        return {
            "valid": False,
            "conflict": False,
            "reason": f"Invalid outtype(s): {', '.join(bad)}. Use lowercase alphanumeric/underscore tokens (e.g. q8_0, iq2_xxs).",
        }

    resolved = _resolve_dir(raw_dir)

    # If the target exists and looks like a git checkout, compare origins.
    if (resolved / ".git").exists():
        current = await managers.get_current_origin(resolved)
        if current and managers._normalize_repo_url(current) != managers._normalize_repo_url(repo):
            return {
                "valid": False,
                "conflict": True,
                "reason": f"Folder is already a clone of '{current}', which does not match '{repo}'.",
                "current_origin": current,
                "resolved_dir": str(resolved),
            }
        return {
            "valid": True,
            "conflict": False,
            "reason": "Existing clone matches the configured repo.",
            "current_origin": current,
            "resolved_dir": str(resolved),
            "outtypes": normalized_outtypes,
        }

    # Doesn't exist or isn't a git repo — make sure we can create/write the parent.
    parent = resolved if resolved.exists() else resolved.parent
    if not parent.exists():
        return {
            "valid": False,
            "conflict": False,
            "reason": f"Parent folder '{parent}' does not exist.",
            "resolved_dir": str(resolved),
        }

    return {
        "valid": True,
        "conflict": False,
        "reason": "Folder is available — will clone on first use.",
        "current_origin": None,
        "resolved_dir": str(resolved),
        "outtypes": normalized_outtypes,
    }


@router.get("/admin/llama-cpp")
async def get_llama_cpp_source(request: Request):
    """Admin only: return the current llama.cpp repo + folder configuration."""
    await get_admin(request)
    import managers
    current_origin = await managers.get_current_origin()
    return {
        "repo": managers.LLAMA_CPP_REPO,
        "dir": str(managers.LLAMA_CPP_DIR) if managers.LLAMA_CPP_DIR else "",
        "current_origin": current_origin,
        "default_repo": managers.DEFAULT_LLAMA_CPP_REPO,
        "installed": (managers.LLAMA_CPP_DIR / "CMakeLists.txt").exists() if managers.LLAMA_CPP_DIR else False,
        "outtypes": list(managers.LLAMA_CPP_OUTTYPES),
    }


@router.get("/llama-cpp-outtypes")
async def get_llama_cpp_outtypes_public():
    """Public read of the configured fork outtypes — used to populate the conversion form."""
    import managers
    return {"outtypes": list(managers.LLAMA_CPP_OUTTYPES)}


@router.post("/admin/llama-cpp/validate")
async def validate_llama_cpp_source(request: Request, body: LlamaCppSourceConfig):
    """Admin only: dry-run validation of a (repo, dir [, outtypes]) combo without saving."""
    await get_admin(request)
    return await _validate_source(body.repo, body.dir, body.outtypes)


@router.post("/admin/llama-cpp")
async def set_llama_cpp_source(request: Request, body: LlamaCppSourceConfig):
    """Admin only: persist new llama.cpp repo + folder + outtypes and reload config."""
    await get_admin(request)
    import managers

    result = await _validate_source(body.repo, body.dir, body.outtypes)
    if not result.get("valid"):
        raise HTTPException(status_code=400, detail=result.get("reason", "Invalid configuration"))

    normalized_outtypes = result.get("outtypes") or []
    await set_app_config("llama_cpp_repo", body.repo.strip())
    await set_app_config("llama_cpp_dir", body.dir.strip())
    await set_app_config("llama_cpp_outtypes", json.dumps(normalized_outtypes))
    await managers.refresh_llama_config()

    return {
        "status": "updated",
        "repo": managers.LLAMA_CPP_REPO,
        "dir": str(managers.LLAMA_CPP_DIR),
        "resolved_dir": result.get("resolved_dir"),
        "outtypes": list(managers.LLAMA_CPP_OUTTYPES),
    }
