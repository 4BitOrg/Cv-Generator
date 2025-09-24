# app.py
"""
FastAPI app for CV Generator project.
- Handles Gumroad webhook -> creates user (via Supabase Admin API) + inserts transaction
- CRUD for documents (CV / CoverLetter) tied to auth.users (Supabase)
- Templates listing
- Jobs CRUD
- Subscription & refund handling
- AI CV generation using OpenAI API (PDF/Word export supported)
"""

import os
import json
import uuid
from openai import OpenAI
import requests
from typing import Optional
from fastapi import FastAPI, Header, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage

# For export
from fpdf import FPDF
from docx import Document

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 0)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")

# Supabase endpoints
# keep as-is; many Supabase installs accept POST to /auth/v1/admin or /auth/v1/admin/users depending on config.
# Your project previously worked with this value; if you hit an error, try swapping to "/auth/v1/admin" or "/auth/v1/admin/users".
SUPABASE_AUTH_ADMIN = f"{SUPABASE_URL}/auth/v1/admin/users"
SUPABASE_AUTH_USER = f"{SUPABASE_URL}/auth/v1/user"
SUPABASE_REST = f"{SUPABASE_URL}/rest/v1"

# Request headers
ADMIN_HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}
ANON_HEADERS = {
    "apikey": SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}

# Init FastAPI
app = FastAPI(title="CV Generator Backend (FastAPI + Supabase)")

# Init OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Models
# ----------------------------
class DocumentIn(BaseModel):
    template_id: Optional[str]
    type: str
    content_json: dict

class JobIn(BaseModel):
    title: str
    description: Optional[str] = None

class CVRequest(BaseModel):
    name: str
    skills: str
    experience: str
    education: str
    projects: str

# ----------------------------
# Helper functions for export
# ----------------------------
def save_cv_as_pdf(cv_text, filename="cv.pdf"):
    """Save CV text as PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in cv_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)
    return filename

def save_cv_as_docx(cv_text, filename="cv.docx"):
    """Save CV text as Word file."""
    doc = Document()
    for line in cv_text.split("\n"):
        doc.add_paragraph(line)
    doc.save(filename)
    return filename

# ----------------------------
# Supabase helper functions
# ----------------------------
def supabase_select(table: str, eq: Optional[dict] = None):
    """Select rows with optional filters."""
    url = f"{SUPABASE_REST}/{table}"
    params = {}
    if eq:
        for k, v in eq.items():
            params[f"{k}"] = f"eq.{v}"
    resp = requests.get(url, headers=ADMIN_HEADERS, params=params)
    if not resp.ok:
        raise HTTPException(status_code=500, detail=f"Supabase select error: {resp.text}")
    try:
        return resp.json()
    except Exception:
        return []

def supabase_insert(table: str, row: dict):
    """Insert row into Supabase table."""
    url = f"{SUPABASE_REST}/{table}"
    resp = requests.post(url, headers=ADMIN_HEADERS, data=json.dumps(row))
    if not resp.ok:
        raise HTTPException(status_code=500, detail=f"Supabase insert error: {resp.text}")
    # Supabase sometimes returns empty body on success -> protect against JSON decode error
    if resp.text:
        try:
            return resp.json()
        except Exception:
            return {"ok": True}
    return {"ok": True}

def supabase_update(table: str, id_col: str, id_val: str, payload: dict):
    """Update row in Supabase."""
    url = f"{SUPABASE_REST}/{table}?{id_col}=eq.{id_val}"
    resp = requests.patch(url, headers=ADMIN_HEADERS, data=json.dumps(payload))
    if not resp.ok:
        raise HTTPException(status_code=500, detail=f"Supabase update error: {resp.text}")
    if resp.text:
        try:
            return resp.json()
        except Exception:
            return {"ok": True}
    return {"ok": True}

def supabase_delete(table: str, id_col: str, id_val: str):
    """Delete row in Supabase."""
    url = f"{SUPABASE_REST}/{table}?{id_col}=eq.{id_val}"
    resp = requests.delete(url, headers=ADMIN_HEADERS)
    if not resp.ok:
        raise HTTPException(status_code=500, detail=f"Supabase delete error: {resp.text}")
    if resp.text:
        try:
            return resp.json()
        except Exception:
            return {"ok": True}
    return {"ok": True}

def create_supabase_user(email: str, password: Optional[str] = None, email_confirm: bool = True):
    """
    Create user in Supabase Auth (admin only).
    - If password is not provided, generate a temporary password and return it (so caller can email it).
    - Returns dict: {"id": <uuid or None>, "password": <password or None>}
      If user already exists, returns {"id": <existing id>, "password": None}
    """
    # generate a temporary password if none provided
    generated_password = password
    if not generated_password:
        # use a readable random string
        generated_password = uuid.uuid4().hex[:12]

    body = {"email": email, "password": generated_password}
    if email_confirm:
        body["email_confirm"] = True

    resp = requests.post(SUPABASE_AUTH_ADMIN, headers=ADMIN_HEADERS, data=json.dumps(body))

    if not resp.ok:
        # try to parse response safely
        txt = resp.text or ""
        try:
            j = resp.json()
        except Exception:
            j = {"raw": txt}

        # If user already exists, fetch the user id and return password=None
        text_join = json.dumps(j) if isinstance(j, dict) else str(j)
        if "User already registered" in text_join or "already exists" in text_join or resp.status_code in (409,):
            # fetch existing user id via admin REST users table (service role)
            q_url = f"{SUPABASE_REST}/users?email=eq.{email}"
            r = requests.get(q_url, headers=ADMIN_HEADERS)
            if r.ok:
                try:
                    users = r.json()
                    if users:
                        return {"id": users[0].get("id"), "password": None}
                except Exception:
                    pass
            # fallback: return id None but no password
            return {"id": None, "password": None}
        # otherwise raise meaningful error
        raise HTTPException(status_code=500, detail=f"Supabase create user error: {j}")

    # success - parse response safely
    try:
        data = resp.json()
        user_id = data.get("id") or data.get("user", {}).get("id")
    except Exception:
        # if for some reason JSON missing, try to fetch user
        q_url = f"{SUPABASE_REST}/users?email=eq.{email}"
        r = requests.get(q_url, headers=ADMIN_HEADERS)
        if r.ok:
            try:
                users = r.json()
                user_id = users[0].get("id") if users else None
            except Exception:
                user_id = None
        else:
            user_id = None

    return {"id": user_id, "password": generated_password if user_id else None}

def get_user_from_token(token: str):
    """Validate access token with Supabase Auth."""
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization token")
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(SUPABASE_AUTH_USER, headers=headers)
    if resp.status_code == 200:
        try:
            return resp.json()
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid or expired token (no JSON)")
    raise HTTPException(status_code=401, detail="Invalid or expired token")

def send_welcome_email(to_email: str, password: Optional[str] = None):
    """Send welcome email via SMTP."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS or SMTP_PORT == 0:
        return False
    msg = EmailMessage()
    msg["Subject"] = "Welcome to CV Generator"
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    body = f"Hello!\n\nYour account is ready.\nEmail: {to_email}\n"
    if password:
        body += (
            "A temporary password has been generated for you. Please login and change it.\n\n"
            f"Temporary password: {password}\n\n"
        )
    body += "Login: <your-frontend-url>/login\n\nBest regards,\nCV Generator Team"
    msg.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.login(SMTP_USER, SMTP_PASS)
        smtp.send_message(msg)
    return True

# ----------------------------
# Access control
# ----------------------------
def check_user_access(user_id: str):
    """Check if user has valid transaction or subscription."""
    rows = supabase_select("transactions", eq={"user_id": user_id})
    if not rows:
        raise HTTPException(status_code=403, detail="No valid purchase found")

    last_tx = sorted(rows, key=lambda r: r.get("purchased_at", ""), reverse=True)[0]

    if last_tx.get("status") == "refunded":
        raise HTTPException(status_code=403, detail="Purchase refunded, access revoked")

    # subscription checks simplified (no recurrence column in schema)
    return True

def require_user(authorization: Optional[str] = Header(None)):
    """Extract user from Authorization header (Bearer token)."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    token = authorization.split(" ", 1)[1]
    user = get_user_from_token(token)
    return user

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/webhook/gumroad")
async def gumroad_webhook(payload: dict):
    """Receive Gumroad webhook, create Supabase user, insert transaction."""
    email = payload.get("email")
    sale_id = payload.get("sale_id")
    product_id = payload.get("product_id")
    price_cents = payload.get("price", 0)
    refunded = payload.get("refunded", False)

    if not email or not sale_id:
        raise HTTPException(status_code=400, detail="Missing email or sale_id")

    # create user and possibly receive a generated temporary password
    created = create_supabase_user(email=email, password=None, email_confirm=True)
    user_id = created.get("id")
    temp_password = created.get("password")  # will be None if user already existed

    # if create_supabase_user couldn't return id, try to fetch by email
    if not user_id:
        q_url = f"{SUPABASE_REST}/users?email=eq.{email}"
        resp = requests.get(q_url, headers=ADMIN_HEADERS)
        if resp.ok and resp.text:
            try:
                arr = resp.json()
                if arr:
                    user_id = arr[0].get("id")
            except Exception:
                pass

    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to ensure user")

    # Avoid duplicates
    existing = supabase_select("transactions", eq={"gumroad_tx_id": sale_id})
    if existing:
        return {"ok": True, "msg": "Duplicate ignored"}

    tx_row = {
        "user_id": user_id,
        "gumroad_tx_id": sale_id,
        "amount": float(price_cents) / 100.0,
        "currency": "USD",
        "status": "refunded" if refunded else "success",
        "purchased_at": datetime.utcnow().isoformat(),
    }
    supabase_insert("transactions", tx_row)

    # create a profile row if you want (optional) - only if profiles table exists
    try:
        profile_row = {"id": user_id, "name": email.split("@")[0]}
        supabase_insert("profiles", profile_row)
    except Exception:
        # non-fatal if profiles table doesn't exist or insert fails
        pass

    try:
        # send welcome email; include temp_password if it was generated
        send_welcome_email(email, password=temp_password)
    except Exception as ex:
        print("Warning: welcome email failed:", ex)

    return {"ok": True, "user_id": user_id}

@app.post("/generate-cv")
def generate_cv(
    data: CVRequest,
    format: str = Query("pdf", description="Choose 'pdf' or 'docx'"),
    user=Depends(require_user)
):
    """
    Generate ATS-friendly CV using OpenAI API.
    Save it in Supabase and return file (PDF or Word).
    """
    check_user_access(user.get("id"))

    # Build prompt for OpenAI
    prompt = f"""
    Generate a professional ATS-friendly CV based on this data:
    Name: {data.name}
    Skills: {data.skills}
    Experience: {data.experience}
    Education: {data.education}
    Projects: {data.projects}
    Format it in plain text with clear sections (Summary, Skills, Experience, Education, Projects).
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    cv_text = response.choices[0].message.content

    # Save CV to Supabase documents table
    row = {
        "user_id": user.get("id"),
        "type": "CV",
        "template_id": None,
        "content_json": {"text": cv_text},
        "created_at": datetime.utcnow().isoformat(),
    }
    supabase_insert("documents", row)

    # Export file
    if format.lower() == "docx":
        filepath = save_cv_as_docx(cv_text, "cv.docx")
        return FileResponse(
            filepath,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="cv.docx"
        )
    else:
        filepath = save_cv_as_pdf(cv_text, "cv.pdf")
        return FileResponse(
            filepath,
            media_type="application/pdf",
            filename="cv.pdf"
        )

@app.get("/templates")
def list_templates():
    """List all templates from Supabase."""
    data = supabase_select("templates")
    return {"templates": data}

@app.post("/documents")
def create_document(payload: DocumentIn, user=Depends(require_user)):
    """Create a new document (CV or CoverLetter)."""
    user_id = user.get("id")
    check_user_access(user_id)
    row = {
        "user_id": user_id,
        "template_id": payload.template_id,
        "type": payload.type,
        "content_json": payload.content_json,
        "created_at": datetime.utcnow().isoformat(),
    }
    res = supabase_insert("documents", row)
    return res

@app.get("/documents")
def get_documents(user=Depends(require_user)):
    """List documents for the current user."""
    user_id = user.get("id")
    check_user_access(user_id)
    rows = supabase_select("documents", eq={"user_id": user_id})
    return {"documents": rows}

@app.get("/documents/{doc_id}")
def get_document(doc_id: str, user=Depends(require_user)):
    """Get single document by ID."""
    rows = supabase_select("documents", eq={"id": doc_id})
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = rows[0]
    if doc.get("user_id") != user.get("id"):
        raise HTTPException(status_code=403, detail="Forbidden")
    check_user_access(user.get("id"))
    return doc

@app.put("/documents/{doc_id}")
def update_document(doc_id: str, payload: DocumentIn, user=Depends(require_user)):
    """Update existing document."""
    rows = supabase_select("documents", eq={"id": doc_id})
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = rows[0]
    if doc.get("user_id") != user.get("id"):
        raise HTTPException(status_code=403, detail="Forbidden")
    check_user_access(user.get("id"))
    update_payload = {
        "template_id": payload.template_id,
        "type": payload.type,
        "content_json": payload.content_json,
    }
    res = supabase_update("documents", "id", doc_id, update_payload)
    return res

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user=Depends(require_user)):
    """Delete document by ID."""
    rows = supabase_select("documents", eq={"id": doc_id})
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = rows[0]
    if doc.get("user_id") != user.get("id"):
        raise HTTPException(status_code=403, detail="Forbidden")
    check_user_access(user.get("id"))
    supabase_delete("documents", "id", doc_id)
    return {"deleted": True}

@app.post("/jobs")
def create_job(payload: JobIn, user=Depends(require_user)):
    """Create a new job posting for the current user."""
    user_id = user.get("id")
    check_user_access(user_id)
    row = {
        "user_id": user_id,
        "title": payload.title,
        "description": payload.description,
        "created_at": datetime.utcnow().isoformat(),
    }
    res = supabase_insert("jobs", row)
    return res

@app.get("/jobs")
def list_jobs(user=Depends(require_user)):
    """List jobs for the current user."""
    user_id = user.get("id")
    check_user_access(user_id)
    rows = supabase_select("jobs", eq={"user_id": user_id})
    return {"jobs": rows}

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, user=Depends(require_user)):
    """Delete a job posting."""
    rows = supabase_select("jobs", eq={"id": job_id}")
    if not rows:
        raise HTTPException(status_code=404, detail="Job not found")
    job = rows[0]
    if job.get("user_id") != user.get("id"):
        raise HTTPException(status_code=403, detail="Forbidden")
    check_user_access(user.get("id"))
    supabase_delete("jobs", "id", job_id)
    return {"deleted": True}

@app.get("/admin/transactions")
def admin_transactions(x_service_role: Optional[str] = Header(None)):
    """Admin endpoint: list all transactions."""
    if x_service_role != SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    rows = supabase_select("transactions")
    return {"transactions": rows}
