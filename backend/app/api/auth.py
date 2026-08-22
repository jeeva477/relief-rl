from pydantic import BaseModel, EmailStr
from fastapi import APIRouter
from backend.app.auth import authenticate_admin

router = APIRouter(prefix="/api/auth", tags=["auth"])

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/admin/login")
def admin_login(payload: LoginRequest):
    return {"access_token": authenticate_admin(payload.email, payload.password), "token_type": "bearer", "role": "ADMIN"}
