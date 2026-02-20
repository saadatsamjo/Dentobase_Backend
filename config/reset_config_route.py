# config/reset_config_route.py
from fastapi import APIRouter, Depends
from app.users.auth_dependencies import get_current_admin
from app.users.user_models.user_model import User

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/reset-all-configs")
async def reset_all_configs(
    admin: User = Depends(get_current_admin)
):
    """
    Reset all system configs to file defaults.
    Admin-only operation.
    """
    from config.ragconfig import rag_settings
    from config.visionconfig import vision_settings
    
    # Reload settings from files
    rag_settings.__init__()
    vision_settings.__init__()
    
    return {
        "message": "All configs reset to defaults",
        "reset_by": admin.email
    }