import os
import hmac
import hashlib
import secrets
import httpx
from typing import Optional

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_ADMIN_REDIRECT_URI = os.environ.get("GOOGLE_ADMIN_REDIRECT_URI", "")
THC_MASTER_EMAIL = os.environ.get("THC_MASTER_EMAIL", "")
THC_ALLOWED_EMAILS = os.environ.get("THC_ALLOWED_EMAILS", "")
THC_SESSION_SECRET = os.environ.get("THC_SESSION_SECRET", "")


def generate_state() -> str:
    return secrets.token_urlsafe(32)


def is_authorized_email(email: str) -> bool:
    if not email:
        return False
    email_lower = email.lower().strip()
    if THC_MASTER_EMAIL and email_lower == THC_MASTER_EMAIL.lower().strip():
        return True
    if THC_ALLOWED_EMAILS:
        allowed = [e.strip().lower() for e in THC_ALLOWED_EMAILS.split(",")]
        if email_lower in allowed:
            return True
    return False


def generate_api_key(email: str) -> str:
    return hmac.new(
        THC_SESSION_SECRET.encode(),
        email.lower().strip().encode(),
        hashlib.sha256
    ).hexdigest()


def verify_api_key(key: str) -> Optional[str]:
    if not key:
        return None
    emails_to_check = []
    if THC_MASTER_EMAIL:
        emails_to_check.append(THC_MASTER_EMAIL.lower().strip())
    if THC_ALLOWED_EMAILS:
        emails_to_check.extend([e.strip().lower() for e in THC_ALLOWED_EMAILS.split(",")])
    
    for email in emails_to_check:
        expected_key = generate_api_key(email)
        if hmac.compare_digest(key, expected_key):
            if is_authorized_email(email):
                return email
    return None


def build_google_auth_url(state: str) -> str:
    from urllib.parse import urlencode
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_ADMIN_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account",
        "state": state
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


async def exchange_code_for_email(code: str) -> Optional[dict]:
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_ADMIN_REDIRECT_URI,
        "grant_type": "authorization_code",
        "code": code
    }
    
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(token_url, data=token_data)
        if token_resp.status_code != 200:
            return None
        token_json = token_resp.json()
        access_token = token_json.get("access_token")
        if not access_token:
            return None
        
        userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        user_resp = await client.get(userinfo_url, headers=headers)
        if user_resp.status_code != 200:
            return None
        user_json = user_resp.json()
        
        email = user_json.get("email", "")
        if user_json.get("email_verified") and is_authorized_email(email):
            return {
                "email": email,
                "name": user_json.get("name", "")
            }
        return None