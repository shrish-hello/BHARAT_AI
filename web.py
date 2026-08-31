from flask import Flask, request, jsonify, send_from_directory, session
from google import genai
import os
import sqlite3
import hashlib
import secrets
import time
from datetime import datetime, timedelta, date
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bharat_ai.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    secrets.token_hex(32)
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

TRIAL_DAYS = 7
DAILY_LIMIT = 20
MEMBERSHIP_PRICE = 1000

uploaded_text = ""


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            grade TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            trial_start TEXT NOT NULL,
            membership TEXT DEFAULT 'trial',
            membership_expires TEXT,
            daily_count INTEGER DEFAULT 0,
            daily_date TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# --------------------------------------------------
# PASSWORD HELPERS
# --------------------------------------------------

def hash_password(password):
    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120000
    )

    return (
        salt.hex() +
        ":" +
        password_hash.hex()
    )


def verify_password(password, stored_hash):
    try:
        salt_hex, hash_hex = stored_hash.split(":")

        salt = bytes.fromhex(salt_hex)

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            120000
        )

        return secrets.compare_digest(
            password_hash.hex(),
            hash_hex
        )

    except Exception:
        return False


# --------------------------------------------------
# USER HELPERS
# --------------------------------------------------

def get_current_user():
    user_id = session.get("user_id")

    if not user_id:
        return None

    conn = get_db()

    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    conn.close()

    return user


def reset_daily_count_if_needed(user):
    today = date.today().isoformat()

    if user["daily_date"] != today:
        conn = get_db()

        conn.execute(
            """
            UPDATE users
            SET daily_count = 0,
                daily_date = ?
            WHERE id = ?
            """,
            (today, user["id"])
        )

        conn.commit()
        conn.close()

        return 0

    return user["daily_count"] or 0


def trial_status(user):
    try:
        start = datetime.fromisoformat(user["trial_start"])

        expiry = start + timedelta(days=TRIAL_DAYS)

        now = datetime.utcnow()

        if now < expiry:
            remaining = expiry - now

            return {
                "active": True,
                "days_remaining": max(
                    1,
                    remaining.days + (1 if remaining.seconds else 0)
                )
            }

        return {
            "active": False,
            "days_remaining": 0
        }

    except Exception:
        return {
            "active": False,
            "days_remaining": 0
        }


def access_status(user):
    if not user:
        return {
            "logged_in": False,
            "allowed": False
        }

    membership = user["membership"] or "trial"

    if membership == "lifetime":
        return {
            "logged_in": True,
            "allowed": True,
            "membership": "lifetime",
            "daily_limit": DAILY_LIMIT,
            "daily_used": reset_daily_count_if_needed(user),
            "daily_remaining": max(
                0,
                DAILY_LIMIT - reset_daily_count_if_needed(user)
            )
        }

    trial = trial_status(user)

    if not trial["active"]:
        return {
            "logged_in": True,
            "allowed": False,
            "membership": "expired",
            "daily_limit": DAILY_LIMIT,
            "daily_used": reset_daily_count_if_needed(user),
            "daily_remaining": 0
        }

    used = reset_daily_count_if_needed(user)

    return {
        "logged_in": True,
        "allowed": used < DAILY_LIMIT,
        "membership": "trial",
        "trial_days_remaining": trial["days_remaining"],
        "daily_limit": DAILY_LIMIT,
        "daily_used": used,
        "daily_remaining": max(
            0,
            DAILY_LIMIT - used
        )
    }


# --------------------------------------------------
# FILE READER
# --------------------------------------------------

def read_file(path):
    name = path.lower()

    if name.endswith(".txt"):
        try:
            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                return f.read()
        except Exception:
            return ""

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)

            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        except Exception:
            return ""

    if name.endswith(".docx"):
        try:
            from docx import Document

            doc = Document(path)

            return "\n".join(
                p.text
                for p in doc.paragraphs
            )

        except Exception:
            return ""

    return ""


# --------------------------------------------------
# GEMINI
# --------------------------------------------------

def ask_gemini(prompt):
    last_error = None

    models_to_try = [
        MODEL,
        "gemini-3.7-flash",
        "gemini-2.5-flash"
    ]

    tried = set()

    for model in models_to_try:
        if model in tried:
            continue

        tried.add(model)

        for attempt in range(3):
            try:
                print(
                    f"[BHARAT AI] Trying {model}, attempt {attempt + 1}",
                    flush=True
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                answer = getattr(response, "text", None)

                if answer and answer.strip():
                    print(
                        f"[BHARAT AI] Response received from {model}",
                        flush=True
                    )

                    return answer.strip()

                last_error = Exception(
                    "Gemini returned an empty response."
                )

            except Exception as e:
                last_error = e

                print(
                    f"[BHARAT AI] {model} failed: {str(e)}",
                    flush=True
                )

                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))

    raise last_error or Exception(
        "All Gemini models failed."
    )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# --------------------------------------------------
# REGISTER
# --------------------------------------------------

@app.route("/register", methods=["POST"])
@app.route("/signup", methods=["POST"])
@app.route("/create-account", methods=["POST"])
def register():
    try:
        data = request.get_json(
            silent=True
        ) or request.form.to_dict()

        name = str(
            data.get("name") or
            data.get("student_name") or
            ""
        ).strip()

        email = str(
            data.get("email") or
            ""
        ).strip().lower()

        grade = str(
            data.get("grade") or
            ""
        ).strip()

        password = str(
            data.get("password") or
            ""
        )

        guardian_confirmed = data.get(
            "guardian_confirmed",
            data.get(
                "parent_confirmed",
                data.get("parentGuardianConfirmed", False)
            )
        )

        if isinstance(
            guardian_confirmed,
            str
        ):
            guardian_confirmed = guardian_confirmed.lower() in [
                "true",
                "1",
                "yes",
                "on"
            ]

        if not name:
            return jsonify({
                "success": False,
                "message": "Please enter the student name."
            }), 400

        if not email or "@" not in email:
            return jsonify({
                "success": False,
                "message": "Please enter a valid email address."
            }), 400

        if len(password) < 6:
            return jsonify({
                "success": False,
                "message": "Password must be at least 6 characters."
            }), 400

        if not guardian_confirmed:
            return jsonify({
                "success": False,
                "message": "A parent or guardian must confirm this student account."
            }), 400

        conn = get_db()

        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing:
            conn.close()

            return jsonify({
                "success": False,
                "message": "An account with this email already exists."
            }), 409

        now = datetime.utcnow().isoformat()

        password_hash = hash_password(password)

        cursor = conn.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                grade,
                password_hash,
                created_at,
                trial_start,
                membership,
                daily_count,
                daily_date
            )
            VALUES (?, ?, ?, ?, ?, ?, 'trial', 0, ?)
            """,
            (
                name,
                email,
                grade,
                password_hash,
                now,
                now,
                date.today().isoformat()
            )
        )

        user_id = cursor.lastrowid

        conn.commit()
        conn.close()

        session["user_id"] = user_id

        print(
            f"[BHARAT AI] New account created: {email}",
            flush=True
        )

        return jsonify({
            "success": True,
            "message": "Account created successfully. Your 7-day free trial has started.",
            "user": {
                "id": user_id,
                "name": name,
                "email": email,
                "grade": grade
            },
            "trial_days": TRIAL_DAYS,
            "daily_limit": DAILY_LIMIT
        })

    except Exception as e:
        print(
            f"[REGISTER ERROR] {str(e)}",
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Unable to create the account right now."
        }), 500


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json(
            silent=True
        ) or request.form.to_dict()

        email = str(
            data.get("email") or
            ""
        ).strip().lower()

        password = str(
            data.get("password") or
            ""
        )

        if not email or not password:
            return jsonify({
                "success": False,
                "message": "Email and password are required."
            }), 400

        conn = get_db()

        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        conn.close()

        if not user or not verify_password(
            password,
            user["password_hash"]
        ):
            return jsonify({
                "success": False,
                "message": "Incorrect email or password."
            }), 401

        session["user_id"] = user["id"]

        status = access_status(user)

        print(
            f"[BHARAT AI] Login successful: {email}",
            flush=True
        )

        return jsonify({
            "success": True,
            "message": "Login successful.",
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "grade": user["grade"]
            },
            "status": status
        })

    except Exception as e:
        print(
            f"[LOGIN ERROR] {str(e)}",
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Unable to log in right now."
        }), 500


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()

    return jsonify({
        "success": True,
        "message": "Logged out successfully."
    })


# --------------------------------------------------
# CURRENT USER
# --------------------------------------------------

@app.route("/me", methods=["GET"])
@app.route("/session", methods=["GET"])
@app.route("/auth/me", methods=["GET"])
def me():
    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "logged_in": False
        }), 401

    return jsonify({
        "success": True,
        "logged_in": True,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "grade": user["grade"]
        },
        "status": access_status(user)
    })


# --------------------------------------------------
# ACCOUNT STATUS
# --------------------------------------------------

@app.route("/account", methods=["GET"])
@app.route("/account/status", methods=["GET"])
def account_status():
    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "logged_in": False
        }), 401

    return jsonify({
        "success": True,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "grade": user["grade"]
        },
        "status": access_status(user),
        "membership_price": MEMBERSHIP_PRICE
    })


# --------------------------------------------------
# ASK BHARAT AI
# --------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask():
    global uploaded_text

    try:
        user = get_current_user()

        if not user:
            return jsonify({
                "success": False,
                "answer": "Please log in before asking BHARAT AI a question."
            }), 401

        status = access_status(user)

        if not status.get("allowed"):
            if status.get("membership") == "expired":
                return jsonify({
                    "success": False,
                    "answer": "Your 7-day free trial has ended. Please ask a parent or guardian to review the membership options."
                }), 403

            return jsonify({
                "success": False,
                "answer": "You have reached today's question limit. Please try again tomorrow."
            }), 429

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get("question") or
            data.get("prompt") or
            ""
        ).strip()

        if not question:
            return jsonify({
                "success": False,
                "answer": "Please ask me a question."
            }), 400

        prompt = f"""
You are BHARAT AI, a friendly educational AI tutor for students.

Student:
Name: {user["name"]}
Grade: {user["grade"] or "Not specified"}

Your job is to:
- Explain concepts clearly.
- Use language suitable for the student's grade.
- Teach instead of simply giving unexplained answers.
- Give step-by-step explanations when useful.
- Support English, Hindi and Hinglish.
- Be accurate and honest.
- If you are uncertain, say so.
- Keep answers reasonably concise unless the student asks for detail.
- Never claim to have performed an action you did not perform.

Student question:
{question}
"""

        if uploaded_text:
            prompt += f"""

The student uploaded study material.

Use it when relevant.

--- STUDY MATERIAL ---
{uploaded_text[:50000]}
--- END STUDY MATERIAL ---

If the answer is not present in the uploaded material, clearly say that it is not found there and then answer using your general knowledge.
"""

        print(
            f"[BHARAT AI] Question from {user['email']}: {question}",
            flush=True
        )

        answer = ask_gemini(prompt)

        conn = get_db()

        today = date.today().isoformat()

        conn.execute(
            """
            UPDATE users
            SET daily_count =
                CASE
                    WHEN daily_date = ?
                    THEN daily_count + 1
                    ELSE 1
                END,
                daily_date = ?
            WHERE id = ?
            """,
            (
                today,
                today,
                user["id"]
            )
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:
        print(
            f"[ASK ERROR] {str(e)}",
            flush=True
        )

        return jsonify({
            "success": False,
            "answer": "BHARAT AI is temporarily unable to answer. Please try again in a moment."
        }), 500


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_text

    try:
        user = get_current_user()

        if not user:
            return jsonify({
                "success": False,
                "message": "Please log in first."
            }), 401

        if "file" not in request.files:
            return jsonify({
                "success": False,
                "message": "No file selected."
            }), 400

        file = request.files["file"]

        if not file.filename:
            return jsonify({
                "success": False,
                "message": "No file selected."
            }), 400

        filename = secure_filename(
            file.filename
        )

        if not filename:
            return jsonify({
                "success": False,
                "message": "Invalid file name."
            }), 400

        allowed = (
            ".txt",
            ".pdf",
            ".docx"
        )

        if not filename.lower().endswith(
            allowed
        ):
            return jsonify({
                "success": False,
                "message": "Supported files are TXT, PDF and DOCX."
            }), 400

        path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(path)

        text = read_file(path)

        if not text.strip():
            return jsonify({
                "success": False,
                "message": "The file was uploaded, but I couldn't read its text."
            }), 400

        uploaded_text = text

        print(
            f"[BHARAT AI] File uploaded: {filename}",
            flush=True
        )

        return jsonify({
            "success": True,
            "message": f"{filename} uploaded successfully. You can now ask questions about it."
        })

    except Exception as e:
        print(
            f"[UPLOAD ERROR] {str(e)}",
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Unable to process the uploaded file."
        }), 500


# --------------------------------------------------
# CLEAR STUDY MATERIAL
# --------------------------------------------------

@app.route("/clear", methods=["POST"])
def clear():
    global uploaded_text

    uploaded_text = ""

    return jsonify({
        "success": True,
        "message": "Study material cleared."
    })


# --------------------------------------------------
# MEMBERSHIP INFORMATION
# --------------------------------------------------

@app.route("/membership", methods=["GET"])
def membership():
    user = get_current_user()

    return jsonify({
        "success": True,
        "price": MEMBERSHIP_PRICE,
        "currency": "INR",
        "name": "BHARAT AI Lifetime Membership",
        "description": "Lifetime access subject to the applicable usage limits.",
        "logged_in": bool(user)
    })


# --------------------------------------------------
# MEMBERSHIP STATUS
# --------------------------------------------------

@app.route("/membership/status", methods=["GET"])
def membership_status():
    user = get_current_user()

    if not user:
        return jsonify({
            "success": False,
            "logged_in": False
        }), 401

    return jsonify({
        "success": True,
        "membership": user["membership"] or "trial",
        "price": MEMBERSHIP_PRICE,
        "daily_limit": DAILY_LIMIT,
        "status": access_status(user)
    })


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "BHARAT AI is running",
        "model": MODEL,
        "database": "connected"
    })


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Route not found."
    }), 404


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "success": False,
        "error": "File is too large. Maximum size is 10 MB."
    }), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


# --------------------------------------------------
# START
# --------------------------------------------------

if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        "==========================================",
        flush=True
    )

    print(
        "BHARAT AI SERVER STARTING",
        flush=True
    )

    print(
        f"Model: {MODEL}",
        flush=True
    )

    print(
        f"Daily limit: {DAILY_LIMIT}",
        flush=True
    )

    print(
        f"Trial: {TRIAL_DAYS} days",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
