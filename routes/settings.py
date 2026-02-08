"""
User Settings Routes
Handles user preferences and settings management
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List

from database import get_db_connection

router = APIRouter(prefix="/api/settings", tags=["settings"])


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
