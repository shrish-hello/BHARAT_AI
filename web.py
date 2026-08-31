from flask import Flask, request, jsonify, send_from_directory, session
from google import genai
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import time
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bharat_ai.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-in-render")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

PRIMARY_MODEL = "gemini-3.7-flash"
FALLBACK_MODEL = "gemini-2.5-flash"

TRIAL_DAYS = 7
TRIAL_DAILY_LIMIT = 10
MEMBER_DAILY_LIMIT = 50

uploaded_material = {}


def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            membership TEXT NOT NULL DEFAULT 'trial'
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            user_id INTEGER NOT NULL,
            usage_date TEXT NOT NULL,
            questions INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, usage_date)
        )
    """)

    connection.commit()
    connection.close()


init_db()


def today_string():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_user():
    user_id = session.get("user_id")

    if not user_id:
        return None

    connection = get_db()

    user = connection.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    connection.close()

    return user


def trial_status(user):
    created_at = datetime.fromisoformat(user["created_at"])

    now = datetime.now(timezone.utc)

    elapsed = now - created_at

    days_used = elapsed.days + 1

    remaining = max(
        0,
        TRIAL_DAYS - elapsed.days
    )

    active = elapsed < timedelta(days=TRIAL_DAYS)

    return active, days_used, remaining


def get_daily_usage(user_id):
    connection = get_db()

    row = connection.execute(
        """
        SELECT questions
        FROM usage
        WHERE user_id = ? AND usage_date = ?
        """,
        (user_id, today_string())
    ).fetchone()

    connection.close()

    if not row:
        return 0

    return row["questions"]


def increment_usage(user_id):
    connection = get_db()

    connection.execute(
        """
        INSERT INTO usage(user_id, usage_date, questions)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, usage_date)
        DO UPDATE SET questions = questions + 1
        """,
        (user_id, today_string())
    )

    connection.commit()
    connection.close()


def get_limit(user):
    if user["membership"] == "lifetime":
        return MEMBER_DAILY_LIMIT

    active, _, _ = trial_status(user)

    if active:
        return TRIAL_DAILY_LIMIT

    return 0


def read_file(path):
    lower = path.lower()

    if lower.endswith(".txt"):
        try:
            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:
                return file.read()
        except Exception as error:
            print("TXT ERROR:", repr(error), flush=True)
            return ""

    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)

            text = []

            for page in reader.pages:
                text.append(
                    page.extract_text() or ""
                )

            return "\n".join(text)

        except Exception as error:
            print("PDF ERROR:", repr(error), flush=True)
            return ""

    if lower.endswith(".docx"):
        try:
            from docx import Document

            document = Document(path)

            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        except Exception as error:
            print("DOCX ERROR:", repr(error), flush=True)
            return ""

    return ""


def retryable(error):
    message = str(error).lower()

    words = [
        "503",
        "429",
        "500",
        "unavailable",
        "overloaded",
        "high demand",
        "temporarily",
        "rate limit",
        "resource exhausted",
        "timeout",
        "timed out"
    ]

    return any(
        word in message
        for word in words
    )


def ask_gemini(prompt):
    models = [
        PRIMARY_MODEL,
        FALLBACK_MODEL
    ]

    last_error = None

    for model in models:

        for attempt in range(2):

            try:

                print(
                    f"Gemini model={model} "
                    f"attempt={attempt + 1}",
                    flush=True
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                answer = getattr(
                    response,
                    "text",
                    None
                )

                if answer and answer.strip():

                    print(
                        f"Gemini success: {model}",
                        flush=True
                    )

                    return answer.strip()

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            except Exception as error:

                last_error = error

                print(
                    "GEMINI ERROR:",
                    repr(error),
                    flush=True
                )

                if retryable(error):

                    if attempt == 0:
                        time.sleep(2)

                    continue

                break

    raise RuntimeError(
        f"Gemini unavailable: {last_error}"
    )


@app.route("/")
def home():
    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.route("/signup", methods=["POST"])
def signup():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        name = str(
            data.get("name", "")
        ).strip()

        email = str(
            data.get("email", "")
        ).strip().lower()

        password = str(
            data.get("password", "")
        )

        consent = bool(
            data.get("parent_guardian_confirmed", False)
        )

        if not name:
            return jsonify({
                "error": "Please enter your name."
            }), 400

        if not email or "@" not in email:
            return jsonify({
                "error": "Please enter a valid email address."
            }), 400

        if len(password) < 8:
            return jsonify({
                "error": "Password must be at least 8 characters."
            }), 400

        if not consent:
            return jsonify({
                "error": "A parent/guardian confirmation is required before creating a student account."
            }), 400

        connection = get_db()

        existing = connection.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing:
            connection.close()

            return jsonify({
                "error": "An account with this email already exists."
            }), 409

        password_hash = generate_password_hash(
            password
        )

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        cursor = connection.execute(
            """
            INSERT INTO users
            (name, email, password_hash, created_at, membership)
            VALUES (?, ?, ?, ?, 'trial')
            """,
            (
                name,
                email,
                password_hash,
                created_at
            )
        )

        user_id = cursor.lastrowid

        connection.commit()
        connection.close()

        session["user_id"] = user_id

        return jsonify({
            "success": True
        })

    except Exception as error:

        print(
            "SIGNUP ERROR:",
            repr(error),
            flush=True
        )

        return jsonify({
            "error": "Could not create the account."
        }), 500


@app.route("/login", methods=["POST"])
def login():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        email = str(
            data.get("email", "")
        ).strip().lower()

        password = str(
            data.get("password", "")
        )

        connection = get_db()

        user = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        connection.close()

        if not user or not check_password_hash(
            user["password_hash"],
            password
        ):

            return jsonify({
                "error": "Incorrect email or password."
            }), 401

        session["user_id"] = user["id"]

        return jsonify({
            "success": True
        })

    except Exception as error:

        print(
            "LOGIN ERROR:",
            repr(error),
            flush=True
        )

        return jsonify({
            "error": "Could not log in."
        }), 500


@app.route("/logout", methods=["POST"])
def logout():

    session.clear()

    return jsonify({
        "success": True
    })


@app.route("/me")
def me():

    user = get_user()

    if not user:

        return jsonify({
            "logged_in": False
        })

    active_trial, days_used, days_remaining = trial_status(
        user
    )

    usage = get_daily_usage(
        user["id"]
    )

    limit = get_limit(user)

    if user["membership"] == "lifetime":

        plan = "lifetime"

    elif active_trial:

        plan = "trial"

    else:

        plan = "expired"

    return jsonify({
        "logged_in": True,
        "name": user["name"],
        "email": user["email"],
        "plan": plan,
        "days_used": days_used,
        "days_remaining": days_remaining,
        "usage": usage,
        "daily_limit": limit
    })


@app.route("/ask", methods=["POST"])
def ask():

    user = get_user()

    if not user:

        return jsonify({
            "answer": "Please log in before asking BHARAT AI a question."
        }), 401

    limit = get_limit(user)

    usage = get_daily_usage(
        user["id"]
    )

    if limit == 0:

        return jsonify({
            "answer": (
                "Your 7-day free trial has ended. "
                "You can continue using BHARAT AI after "
                "activating Lifetime Membership."
            ),
            "limit_reached": True
        }), 200

    if usage >= limit:

        return jsonify({
            "answer": (
                f"You have reached your daily limit of "
                f"{limit} questions. "
                "Your limit will reset tomorrow."
            ),
            "limit_reached": True
        }), 200

    try:

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get("question", "")
        ).strip()

        if not question:

            return jsonify({
                "answer": "Please ask me a question."
            }), 200

        material = uploaded_material.get(
            user["id"],
            ""
        )

        prompt = f"""
You are BHARAT AI, a friendly AI tutor for students.

Explain concepts clearly and accurately.

Use simple language appropriate for the student's level.

Support:
- English
- Hindi
- Hinglish

For school questions:
- Explain step by step when useful.
- Give examples.
- Help the student understand the concept.
- Do not unnecessarily make answers extremely long.
- Do not pretend to know something you don't know.

Student question:

{question}
"""

        if material:

            prompt += f"""

The student has uploaded study material.

Use it when relevant.

--- STUDY MATERIAL ---
{material[:40000]}
--- END STUDY MATERIAL ---

If the answer is not in the uploaded material,
clearly say that and then use general knowledge.
"""

        answer = ask_gemini(
            prompt
        )

        increment_usage(
            user["id"]
        )

        new_usage = get_daily_usage(
            user["id"]
        )

        return jsonify({
            "answer": answer,
            "usage": new_usage,
            "daily_limit": limit
        }), 200

    except Exception as error:

        print(
            "ASK ERROR:",
            repr(error),
            flush=True
        )

        return jsonify({
            "answer": (
                "BHARAT AI could not get a response right now. "
                "Please try again in a moment."
            )
        }), 500


@app.route("/upload", methods=["POST"])
def upload():

    user = get_user()

    if not user:

        return jsonify({
            "message": "Please log in first."
        }), 401

    try:

        if "file" not in request.files:

            return jsonify({
                "message": "No file selected."
            }), 400

        file = request.files["file"]

        if not file.filename:

            return jsonify({
                "message": "No file selected."
            }), 400

        filename = secure_filename(
            file.filename
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in [
            ".txt",
            ".pdf",
            ".docx"
        ]:

            return jsonify({
                "message": "Only TXT, PDF and DOCX files are supported."
            }), 400

        user_folder = os.path.join(
            UPLOAD_FOLDER,
            str(user["id"])
        )

        os.makedirs(
            user_folder,
            exist_ok=True
        )

        path = os.path.join(
            user_folder,
            filename
        )

        file.save(path)

        text = read_file(
            path
        )

        if not text.strip():

            return jsonify({
                "message": "The file was uploaded, but no readable text was found."
            }), 400

        uploaded_material[user["id"]] = text[:40000]

        return jsonify({
            "message": (
                f"{filename} uploaded successfully. "
                "You can now ask questions about it."
            )
        }), 200

    except Exception as error:

        print(
            "UPLOAD ERROR:",
            repr(error),
            flush=True
        )

        return jsonify({
            "message": "Could not process the file."
        }), 500


@app.route("/clear", methods=["POST"])
def clear():

    user = get_user()

    if user:
        uploaded_material.pop(
            user["id"],
            None
        )

    return jsonify({
        "message": "Study material cleared."
    })


@app.route("/health")
def health():

    return jsonify({
        "status": "BHARAT AI is running"
    })


@app.route("/membership")
def membership():

    user = get_user()

    if not user:

        return jsonify({
            "error": "Login required."
        }), 401

    return jsonify({
        "price": 1000,
        "currency": "INR",
        "type": "one_time",
        "name": "BHARAT AI Lifetime Membership",
        "daily_limit": MEMBER_DAILY_LIMIT,
        "payment_ready": False
    })


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
