"""
NeuraScan ADHD Detection — FastAPI Backend v2
=============================================
Endpoints:
  POST /auth/register          — Create account + send welcome email
  POST /auth/login             — Login, returns JWT
  POST /auth/forgot-password   — Send OTP to email
  POST /auth/verify-otp        — Verify OTP + reset password
  POST /assess/submit          — Submit answers + response times → ML score
  GET  /assess/history         — Get user's past assessments
  GET  /assess/{id}            — Get single assessment result
  GET  /health                 — Health check

Run with:
  uvicorn main:app --reload --port 8000
"""

import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import numpy as np
import uuid, hashlib, hmac, base64, json, time, random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from supabase import create_client, Client
from ml_model import ADHDModel

# ── Config ────────────────────────────────────────────────────
SUPABASE_URL       = os.getenv("SUPABASE_URL")
SUPABASE_KEY       = os.getenv("SUPABASE_KEY")
JWT_SECRET         = os.getenv("JWT_SECRET_KEY")
GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="NeuraScan API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # "http://localhost:3000",
        # "https://frontend-adhd-rarijit03s-projects.vercel.app",
      "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase ──────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── ML Model ──────────────────────────────────────────────────
model = ADHDModel()
model.load_or_train()

# ── OTP store (in-memory, production should use Redis) ────────
otp_store = {}  # {email: {"otp": "123456", "expires": timestamp, "full_name": str}}

# ── Auth helpers ──────────────────────────────────────────────
security = HTTPBearer()

def make_token(user_id: str) -> str:
    header  = base64.urlsafe_b64encode(json.dumps({"alg":"HS256"}).encode()).decode()
    payload = base64.urlsafe_b64encode(json.dumps({
        "sub": user_id,
        "exp": (datetime.utcnow() + timedelta(days=7)).isoformat()
    }).encode()).decode()
    sig = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{payload}.{sig}"

def verify_token(token: str) -> str:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid token")
    header, payload, sig = parts
    expected = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=401, detail="Invalid token signature")
    data = json.loads(base64.urlsafe_b64decode(payload + "==").decode())
    if datetime.fromisoformat(data["exp"]) < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Token expired")
    return data["sub"]

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    return verify_token(creds.credentials)

def hash_password(pw: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), b"neurascan_salt", 260000).hex()

def mask_id(user_id: str) -> str:
    return f"USR-{user_id[:8].upper()}***"

# ── Email sender ──────────────────────────────────────────────
def send_email(to_email: str, subject: str, html_body: str):
    """Send HTML email via Gmail SMTP."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"NeuraScan <{GMAIL_ADDRESS}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())

# def welcome_email_html(full_name: str) -> str:
#     return f"""
# <!DOCTYPE html>
# <html>
# <head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
# <body style="margin:0;padding:0;background:#0d0f14;font-family:'Segoe UI',Arial,sans-serif">
#   <table width="100%" cellpadding="0" cellspacing="0" style="background:#0d0f14;padding:40px 0">
#     <tr><td align="center">
#       <table width="560" cellpadding="0" cellspacing="0" style="background:#13161e;border-radius:16px;overflow:hidden;border:1px solid rgba(139,92,246,0.2)">

#         <!-- Header -->
#         <tr><td style="background:linear-gradient(135deg,#8b5cf6,#06b6d4);padding:32px 40px;text-align:center">
#           <table cellpadding="0" cellspacing="0" style="margin:0 auto">
#             <tr>
#               <td style="background:rgba(255,255,255,0.15);border-radius:12px;padding:10px 14px;display:inline-block">
#                 <span style="font-size:24px;font-weight:900;color:#fff;font-family:Georgia,serif">N</span>
#               </td>
#               <td style="padding-left:12px">
#                 <div style="font-size:22px;font-weight:900;color:#fff;letter-spacing:-0.5px">NeuraScan</div>
#                 <div style="font-size:11px;color:rgba(255,255,255,0.75);letter-spacing:2px;text-transform:uppercase">ADHD Intelligence Platform</div>
#               </td>
#             </tr>
#           </table>
#         </td></tr>

#         <!-- Welcome content -->
#         <tr><td style="padding:40px">
#           <h1 style="font-size:26px;color:#eceef5;margin:0 0 8px;font-weight:700">Welcome, {full_name}! 🎉</h1>
#           <p style="font-size:15px;color:#8892b0;margin:0 0 28px;line-height:1.6">Your NeuraScan account has been created successfully. You now have access to our AI-powered ADHD screening platform.</p>

#           <!-- Feature list -->
#           <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px">
#             {''.join([f"""
#             <tr>
#               <td style="padding:12px 16px;background:#161b26;border-radius:10px;margin-bottom:8px;display:block;border:1px solid rgba(139,92,246,0.15)">
#                 <span style="font-size:18px">{icon}</span>
#                 <span style="font-size:14px;color:#eceef5;font-weight:600;margin-left:10px">{title}</span>
#                 <span style="font-size:12px;color:#5a6480;display:block;margin-left:30px;margin-top:2px">{desc}</span>
#               </td>
#             </tr>
#             <tr><td style="height:8px"></td></tr>
            
#             """ for icon, title, desc in [
#                 ("🧠", "Stacked ML Ensemble", "GBM + RF trained on ADHD-200 clinical data"),
#                 ("👁️", "MediaPipe Eye Tracking", "Real-time blink and movement biometrics"),
#                 ("📋", "Clinical PDF Reports", "Structured 2-page screening reports"),
#                 ("📊", "Assessment History", "Track your results over time"),
#             ]])}
#           </table>

#           <!-- CTA -->
#           <table width="100%" cellpadding="0" cellspacing="0">
#             <tr><td align="center" style="padding:8px 0">
#               <a href="http://localhost:3000" style="display:inline-block;background:linear-gradient(135deg,#8b5cf6,#06b6d4);color:#fff;text-decoration:none;padding:14px 36px;border-radius:12px;font-size:15px;font-weight:700">Start Your Assessment →</a>
#             </td></tr>
#           </table>
#         </td></tr>

#         <!-- Privacy note -->
#         <tr><td style="background:#0d0f14;padding:20px 40px;border-top:1px solid rgba(255,255,255,0.06)">
#           <p style="font-size:12px;color:#5a6480;margin:0;text-align:center;line-height:1.6">
#             🔒 Your data is handled with HIPAA-conscious security. No video is stored or transmitted.<br>
#             NeuraScan is a screening tool only — not a clinical diagnosis.
#           </p>
#         </td></tr>

#         <!-- Footer -->
#         <tr><td style="padding:16px 40px;text-align:center">
#           <p style="font-size:11px;color:#3a4060;margin:0">© 2026 NeuraScan · ADHD Screening Platform</p>
#         </td></tr>

#       </table>
#     </td></tr>
#   </table>
# </body>
# </html>
# """
def welcome_email_html(full_name: str) -> str:
    features = [
        ("🧠", "Stacked ML Ensemble", "GBM + RF trained on ADHD-200 clinical data"),
        ("👁️", "MediaPipe Eye Tracking", "Real-time blink and movement biometrics"),
        ("📋", "Clinical PDF Reports", "Structured 2-page screening reports"),
        ("📊", "Assessment History", "Track your results over time"),
    ]
    features_html = ""
    for icon, title, desc in features:
        features_html += f"""
        <tr>
          <td style="padding:10px 14px;background:#161b26;border-radius:10px;border:1px solid rgba(139,92,246,0.15);display:block;margin-bottom:8px">
            <span style="font-size:18px">{icon}</span>
            <span style="font-size:14px;color:#eceef5;font-weight:600;margin-left:10px">{title}</span>
            <div style="font-size:12px;color:#5a6480;margin-left:30px;margin-top:2px">{desc}</div>
          </td>
        </tr>
        <tr><td style="height:8px"></td></tr>
        """

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#0d0f14;font-family:'Segoe UI',Arial,sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0d0f14;padding:40px 0">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background:#13161e;border-radius:16px;overflow:hidden;border:1px solid rgba(139,92,246,0.2)">
        <tr><td style="background:linear-gradient(135deg,#8b5cf6,#06b6d4);padding:32px 40px;text-align:center">
          <div style="font-size:22px;font-weight:900;color:#fff">NeuraScan</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.75);letter-spacing:2px;text-transform:uppercase;margin-top:4px">ADHD Intelligence Platform</div>
        </td></tr>
        <tr><td style="padding:40px">
          <h1 style="font-size:26px;color:#eceef5;margin:0 0 8px;font-weight:700">Welcome, {full_name}! 🎉</h1>
          <p style="font-size:15px;color:#8892b0;margin:0 0 28px;line-height:1.6">Your NeuraScan account has been created. You now have access to our AI-powered ADHD screening platform.</p>
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px">
            {features_html}
          </table>
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr><td align="center">
              <a href="http://localhost:3000" style="display:inline-block;background:linear-gradient(135deg,#8b5cf6,#06b6d4);color:#fff;text-decoration:none;padding:14px 36px;border-radius:12px;font-size:15px;font-weight:700">Start Your Assessment →</a>
            </td></tr>
          </table>
        </td></tr>
        <tr><td style="background:#0d0f14;padding:20px 40px;border-top:1px solid rgba(255,255,255,0.06)">
          <p style="font-size:12px;color:#5a6480;margin:0;text-align:center;line-height:1.6">
            🔒 HIPAA-conscious security. No video stored or transmitted.<br>
            NeuraScan is a screening tool only — not a clinical diagnosis.
          </p>
        </td></tr>
        <tr><td style="padding:16px 40px;text-align:center">
          <p style="font-size:11px;color:#3a4060;margin:0">© 2026 NeuraScan · ADHD Screening Platform</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""

def otp_email_html(full_name: str, otp: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0d0f14;font-family:'Segoe UI',Arial,sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0d0f14;padding:40px 0">
    <tr><td align="center">
      <table width="520" cellpadding="0" cellspacing="0" style="background:#13161e;border-radius:16px;overflow:hidden;border:1px solid rgba(139,92,246,0.2)">

        <!-- Header -->
        <tr><td style="background:linear-gradient(135deg,#8b5cf6,#06b6d4);padding:24px 40px;text-align:center">
          <div style="font-size:20px;font-weight:900;color:#fff">NeuraScan</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.75);letter-spacing:2px;text-transform:uppercase;margin-top:4px">Password Reset</div>
        </td></tr>

        <!-- Content -->
        <tr><td style="padding:40px;text-align:center">
          <p style="font-size:16px;color:#eceef5;margin:0 0 6px;font-weight:600">Hi {full_name},</p>
          <p style="font-size:14px;color:#8892b0;margin:0 0 32px;line-height:1.6">You requested a password reset. Enter the OTP below to continue.<br>This code expires in <strong style="color:#eceef5">10 minutes</strong>.</p>

          <!-- OTP Box -->
          <div style="background:#0d0f14;border:2px solid #8b5cf6;border-radius:14px;padding:28px 20px;display:inline-block;margin-bottom:28px">
            <div style="font-size:13px;color:#8892b0;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">Your One-Time Password</div>
            <div style="font-size:48px;font-weight:900;color:#8b5cf6;letter-spacing:16px;font-family:'Courier New',monospace">{otp}</div>
          </div>

          <p style="font-size:13px;color:#5a6480;margin:0;line-height:1.6">
            If you didn't request this, you can safely ignore this email.<br>
            Your password will not change unless you use this OTP.
          </p>
        </td></tr>

        <!-- Footer -->
        <tr><td style="background:#0d0f14;padding:16px 40px;text-align:center;border-top:1px solid rgba(255,255,255,0.06)">
          <p style="font-size:11px;color:#3a4060;margin:0">© 2026 NeuraScan · This is an automated message</p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
"""

# ── Schemas ───────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class AssessmentRequest(BaseModel):
    answers: dict
    response_times: List[float]
    session_id: str

# ── Auth Routes ───────────────────────────────────────────────
@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest):
    # Check existing
    existing = supabase.table("users").select("id").eq("email", req.email).execute()
    if existing.data:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(uuid.uuid4())
    hashed  = hash_password(req.password)

    supabase.table("users").insert({
        "id":            user_id,
        "email":         req.email,
        "full_name":     req.full_name,
        "password_hash": hashed,
        "created_at":    datetime.utcnow().isoformat(),
    }).execute()

    # Send welcome email (non-blocking — don't fail registration if email fails)
    try:
        send_email(
            to_email  = req.email,
            subject   = f"Welcome to NeuraScan, {req.full_name.split()[0]}! 🧠",
            html_body = welcome_email_html(req.full_name.split()[0])
        )
        print(f"✅ Welcome email sent to {req.email}")
    except Exception as e:
        print(f"⚠️ Welcome email failed (non-critical): {e}")

    token = make_token(user_id)
    return {
        "token": token,
        "user":  {"id": mask_id(user_id), "email": req.email, "full_name": req.full_name}
    }

@app.post("/auth/login")
def login(req: LoginRequest):
    result = supabase.table("users").select("*").eq("email", req.email).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user   = result.data[0]
    hashed = hash_password(req.password)
    if not hmac.compare_digest(user["password_hash"], hashed):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = make_token(user["id"])
    return {
        "token": token,
        "user":  {"id": mask_id(user["id"]), "email": user["email"], "full_name": user["full_name"]}
    }

@app.post("/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    # Look up user
    result = supabase.table("users").select("id,full_name,email").eq("email", req.email).execute()

    # Always return success to prevent email enumeration attacks
    if not result.data:
        return {"message": "If that email exists, an OTP has been sent."}

    user      = result.data[0]
    full_name = user.get("full_name", "User").split()[0]
    otp       = str(random.randint(100000, 999999))

    otp_store[req.email] = {
        "otp":       otp,
        "expires":   time.time() + 600,  # 10 minutes
        "full_name": full_name,
    }

    try:
        send_email(
            to_email  = req.email,
            subject   = "NeuraScan — Your Password Reset OTP",
            html_body = otp_email_html(full_name, otp)
        )
        print(f"✅ OTP email sent to {req.email} (OTP: {otp})")
        return {"message": "OTP sent to your email."}
    except Exception as e:
        print(f"❌ OTP email failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send OTP email: {str(e)}")

@app.post("/auth/verify-otp")
def verify_otp(req: VerifyOTPRequest):
    stored = otp_store.get(req.email)

    if not stored:
        raise HTTPException(status_code=400, detail="No OTP found for this email. Please request a new one.")

    if time.time() > stored["expires"]:
        del otp_store[req.email]
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")

    if stored["otp"] != req.otp.strip():
        raise HTTPException(status_code=400, detail="Invalid OTP. Please check and try again.")

    # Update password
    new_hash = hash_password(req.new_password)
    update_result = supabase.table("users").update({"password_hash": new_hash}).eq("email", req.email).execute()

    if not update_result.data:
        raise HTTPException(status_code=500, detail="Failed to update password. Please try again.")

    del otp_store[req.email]
    print(f"✅ Password reset successful for {req.email}")
    return {"message": "Password updated successfully!"}

# ── Assessment Routes ─────────────────────────────────────────
@app.post("/assess/submit")
def submit_assessment(req: AssessmentRequest, user_id: str = Depends(get_current_user)):
    if len(req.answers) < 9:
        raise HTTPException(status_code=400, detail="Expected at least 9 answers")

    # Build full 18-question feature vector
    q_scores = [req.answers.get(str(i), 0) for i in range(1, 19)]
    rt_clean = [t for t in req.response_times if 200 < t < 30000]
    avg_rt   = float(np.mean(rt_clean)) if rt_clean else 5000.0
    rt_std   = float(np.std(rt_clean))  if rt_clean else 0.0

    features = q_scores + [avg_rt, rt_std]
    result   = model.predict(features)

    # Domain subscores using actual answered questions
    inatt_ids   = [1,2,3,4,7,8,9,10,11]
    hyper_ids   = [5,6,12,13,14,15,16,17,18]
    inatt_score = float(np.mean([req.answers.get(str(i), 0) for i in inatt_ids]) / 4)
    hyper_score = float(np.mean([req.answers.get(str(i), 0) for i in hyper_ids]) / 4)

    assessment_id = str(uuid.uuid4())

    supabase.table("assessments").insert({
        "id":             assessment_id,
        "user_id":        user_id,
        "session_id":     req.session_id,
        "answers":        json.dumps(req.answers),
        "response_times": json.dumps(req.response_times),
        "final_score":    result["final_score"],
        "q_score":        result["q_score"],
        "t_score":        result["t_score"],
        "inatt_score":    inatt_score,
        "hyper_score":    hyper_score,
        "severity":       result["severity"],
        "avg_rt_ms":      avg_rt,
        "created_at":     datetime.utcnow().isoformat(),
    }).execute()

    return {
        "assessment_id": assessment_id,
        **result,
        "inatt_score": inatt_score,
        "hyper_score": hyper_score,
        "avg_rt_ms":   avg_rt,
    }

@app.get("/assess/history")
def get_history(user_id: str = Depends(get_current_user)):
    result = supabase.table("assessments") \
        .select("id,final_score,severity,inatt_score,hyper_score,created_at") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    return {"assessments": result.data}

@app.get("/assess/{assessment_id}")
def get_assessment(assessment_id: str, user_id: str = Depends(get_current_user)):
    result = supabase.table("assessments") \
        .select("*") \
        .eq("id", assessment_id) \
        .eq("user_id", user_id) \
        .execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return result.data[0]

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "model": model.model_type}
