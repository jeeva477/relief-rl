"""Minimal JWT authentication for the admin surface.

Set ADMIN_EMAIL, ADMIN_PASSWORD and AUTH_SECRET in production. Passwords are
never stored in the browser; the frontend receives a short-lived signed token.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import base64
import json
from fastapi import Header, HTTPException


def _secret() -> str:
    return os.getenv("AUTH_SECRET", "change-me-in-production")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign(value: str) -> str:
    return _b64(hmac.new(_secret().encode(), value.encode(), hashlib.sha256).digest())


def create_admin_token(email: str) -> str:
    payload = {"sub": email, "role": "ADMIN", "exp": int((datetime.now(timezone.utc) + timedelta(hours=2)).timestamp())}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    return f"{body}.{_sign(body)}"


def verify_admin_token(authorization: str | None) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Admin authentication required.")
    token = authorization[7:].strip()
    try:
        body, signature = token.split(".", 1)
        if not hmac.compare_digest(signature, _sign(body)):
            raise ValueError
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if payload.get("role") != "ADMIN" or int(payload.get("exp", 0)) < int(datetime.now(timezone.utc).timestamp()):
            raise ValueError
        return payload
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired admin token.") from exc


def require_admin(authorization: str | None = Header(default=None)) -> dict:
    return verify_admin_token(authorization)


def authenticate_admin(email: str, password: str) -> str:
    expected_email = os.getenv("ADMIN_EMAIL", "admin@relief-rl.local")
    expected_password = os.getenv("ADMIN_PASSWORD", "change-me-in-production")
    if not hmac.compare_digest(email, expected_email) or not hmac.compare_digest(password, expected_password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials.")
    return create_admin_token(email)
