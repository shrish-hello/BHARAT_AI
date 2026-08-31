from flask import Flask, request, jsonify, send_from_directory, session
from google import genai
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime, timedelta
import os
import time
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "bharat-ai-change-this-secret")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE = os.path.join(BASE_DIR, "bharat_ai.db")

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in Render Environment Variables.")

client = genai.Client(api_key=API_KEY)

MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash"
]

TRIAL_DAYS = 7
DAILY_LIMIT = 20


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            grade TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL,
            questions_today INTEGER DEFAULT 0,
            last_question_date TEXT,
            lifetime INTEGER DEFAULT 0
        )
    """)

    connection.commit()
    connection.close()


init_db()


def reset_daily_usage(student):
    today = datetime.utcnow().date().isoformat()

    if student["last_question_date"] != today:
        connection = get_db()

        connection.execute(
            """
            UPDATE students
            SET questions_today = 0,
                last_question_date = ?
            WHERE id = ?
            """,
            (today, student["id"])
        )

        connection.commit()
        connection.close()

        return 0

    return student["questions_today"]


def get_current_student():
    student_id = session.get("student_id")

    if not student_id:
        return None

    connection = get_db()

    student = connection.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    ).fetchone()

    connection.close()

    return student


def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        student = get_current_student()

        if not student:
            return jsonify({
                "answer": "Please log in before asking BHARAT AI a question."
            }), 401

        return function(*args, **kwargs)

    return wrapper


def read_file(path):
    name = path.lower()

    if name.endswith(".txt"):
        try:
            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:
                return file.read()
        except Exception:
            return ""

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)

            pages = []

            for page in reader.pages:
                pages.append(page.extract_text() or "")

            return "\n".join(pages)

        except Exception:
            return ""

    if name.endswith(".docx"):
        try:
            from docx import Document

            document = Document(path)

            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        except Exception:
            return ""

    return ""


def ask_gemini(prompt):
    last_error = None

    for model in MODELS:

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                text = getattr(response, "text", None)

                if text and text.strip():
                    return text.strip()

                last_error = Exception(
                    f"{model} returned an empty response."
                )

            except Exception as error:

                last_error = error

                error_text = str(error).lower()

                retryable = any(
                    phrase in error_text
                    for phrase in [
                        "503",
                        "429",
                        "unavailable",
                        "overloaded",
                        "high demand",
                        "rate limit",
                        "resource exhausted",
                        "temporarily unavailable",
                        "internal server error",
                        "deadline exceeded",
                        "timeout"
                    ]
                )

                if retryable and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                break

    raise last_error or Exception(
        "All Gemini models failed."
    )


@app.route("/")
def home():
    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "BHARAT AI is running",
        "service": "online"
    })


@app.route("/register", methods=["POST"])
def register():

    try:

        data = request.get_json(silent=True) or {}

        name = str(data.get("name", "")).strip()
        email = str(data.get("email", "")).strip().lower()
        grade = str(data.get("grade", "")).strip()
        password = str(data.get("password", ""))

        guardian_approved = bool(
            data.get("guardian_approved", False)
        )

        if not name:
            return jsonify({
                "message": "Please enter the student's name."
            }), 400

        if not email or "@" not in email:
            return jsonify({
                "message": "Please enter a valid email address."
            }), 400

        if not grade:
            return jsonify({
                "message": "Please select a class."
            }), 400

        if len(password) < 6:
            return jsonify({
                "message": "Password must be at least 6 characters."
            }), 400

        if not guardian_approved:
            return jsonify({
                "message": "A parent or guardian must approve this student account."
            }), 400

        connection = get_db()

        existing = connection.execute(
            "SELECT id FROM students WHERE email = ?",
            (email,)
        ).fetchone()

        if existing:
            connection.close()

            return jsonify({
                "message": "An account with this email already exists."
            }), 409

        created_at = datetime.utcnow().isoformat()

        connection.execute(
            """
            INSERT INTO students
            (
                name,
                email,
                grade,
                password,
                created_at,
                questions_today,
                last_question_date,
                lifetime
            )
            VALUES (?, ?, ?, ?, ?, 0, ?, 0)
            """,
            (
                name,
                email,
                grade,
                password,
                created_at,
                datetime.utcnow().date().isoformat()
            )
        )

        connection.commit()

        student = connection.execute(
            "SELECT * FROM students WHERE email = ?",
            (email,)
        ).fetchone()

        connection.close()

        session["student_id"] = student["id"]

        return jsonify({
            "message": "Account created successfully. Your 7-day free trial has started.",
            "student": {
                "name": student["name"],
                "email": student["email"],
                "grade": student["grade"]
            }
        })

    except Exception as error:

        return jsonify({
            "message": f"Registration error: {str(error)}"
        }), 500


@app.route("/login", methods=["POST"])
def login():

    try:

        data = request.get_json(silent=True) or {}

        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))

        if not email or not password:
            return jsonify({
                "message": "Please enter your email and password."
            }), 400

        connection = get_db()

        student = connection.execute(
            """
            SELECT *
            FROM students
            WHERE email = ?
            AND password = ?
            """,
            (email, password)
        ).fetchone()

        connection.close()

        if not student:
            return jsonify({
                "message": "Incorrect email or password."
            }), 401

        session["student_id"] = student["id"]

        return jsonify({
            "message": "Login successful.",
            "student": {
                "name": student["name"],
                "email": student["email"],
                "grade": student["grade"]
            }
        })

    except Exception as error:

        return jsonify({
            "message": f"Login error: {str(error)}"
        }), 500


@app.route("/logout", methods=["POST"])
def logout():

    session.clear()

    return jsonify({
        "message": "Logged out successfully."
    })


@app.route("/me")
def me():

    student = get_current_student()

    if not student:
        return jsonify({
            "logged_in": False
        })

    created = datetime.fromisoformat(
        student["created_at"]
    )

    trial_end = created + timedelta(days=TRIAL_DAYS)

    now = datetime.utcnow()

    trial_active = now < trial_end

    reset_daily_usage(student)

    connection = get_db()

    updated_student = connection.execute(
        "SELECT * FROM students WHERE id = ?",
        (student["id"],)
    ).fetchone()

    connection.close()

    return jsonify({
        "logged_in": True,
        "student": {
            "name": updated_student["name"],
            "email": updated_student["email"],
            "grade": updated_student["grade"]
        },
        "trial": {
            "active": trial_active,
            "days_left": max(
                0,
                (trial_end.date() - now.date()).days
            )
        },
        "usage": {
            "questions_today": updated_student["questions_today"],
            "daily_limit": DAILY_LIMIT
        },
        "lifetime": bool(updated_student["lifetime"])
    })


@app.route("/ask", methods=["POST"])
@login_required
def ask():

    student = get_current_student()

    try:

        data = request.get_json(silent=True) or {}

        question = str(
            data.get("question", "")
        ).strip()

        if not question:

            return jsonify({
                "answer": "Please ask me a question."
            }), 400

        created = datetime.fromisoformat(
            student["created_at"]
        )

        trial_end = created + timedelta(
            days=TRIAL_DAYS
        )

        now = datetime.utcnow()

        is_trial_active = now < trial_end

        if not is_trial_active and not student["lifetime"]:

            return jsonify({
                "answer": "Your 7-day free trial has ended. Please activate BHARAT AI Lifetime Membership to continue."
            }), 403

        questions_today = reset_daily_usage(student)

        if questions_today >= DAILY_LIMIT and not student["lifetime"]:

            return jsonify({
                "answer": f"You have reached today's limit of {DAILY_LIMIT} questions. Please try again tomorrow."
            }), 429

        prompt = f"""
You are BHARAT AI, a friendly AI tutor for students.

Student name:
{student["name"]}

Class:
{student["grade"]}

Your job is to:
- Explain concepts clearly.
- Use simple language suitable for the student's class.
- Teach instead of only giving an answer.
- Give step-by-step explanations when useful.
- Support English, Hindi and Hinglish.
- Be accurate and honest.
- Never invent facts.
- If you are unsure, clearly say that you are unsure.
- Keep explanations appropriate for students.

Student question:

{question}
"""

        uploaded_text = session.get("uploaded_text", "")

        if uploaded_text:

            prompt += f"""

The student has uploaded study material.

Use it when relevant.

--- STUDY MATERIAL ---

{uploaded_text[:50000]}

--- END STUDY MATERIAL ---

If the answer is not present in the study material,
say that clearly and then answer using your general knowledge.
"""

        answer = ask_gemini(prompt)

        connection = get_db()

        today = datetime.utcnow().date().isoformat()

        connection.execute(
            """
            UPDATE students
            SET questions_today = questions_today + 1,
                last_question_date = ?
            WHERE id = ?
            """,
            (today, student["id"])
        )

        connection.commit()
        connection.close()

        return jsonify({
            "answer": answer
        })

    except Exception as error:

        error_text = str(error)

        print(
            "BHARAT AI Gemini error:",
            error_text,
            flush=True
        )

        return jsonify({
            "answer": "BHARAT AI is temporarily unable to answer. Please try again in a moment."
        }), 500


@app.route("/upload", methods=["POST"])
@login_required
def upload():

    try:

        if "file" not in request.files:

            return jsonify({
                "message": "No file selected."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "message": "No file selected."
            }), 400

        filename = secure_filename(
            file.filename
        )

        if not filename:

            return jsonify({
                "message": "Invalid file name."
            }), 400

        allowed_extensions = (
            ".txt",
            ".pdf",
            ".docx"
        )

        if not filename.lower().endswith(
            allowed_extensions
        ):

            return jsonify({
                "message": "Only TXT, PDF and DOCX files are supported."
            }), 400

        path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(path)

        text = read_file(path)

        if not text:

            return jsonify({
                "message": "The file was uploaded, but I couldn't read its text."
            }), 400

        session["uploaded_text"] = text[:50000]

        return jsonify({
            "message": f"{filename} uploaded successfully. You can now ask questions about it."
        })

    except Exception as error:

        return jsonify({
            "message": f"Upload error: {str(error)}"
        }), 500


@app.route("/clear", methods=["POST"])
@login_required
def clear():

    session.pop("uploaded_text", None)

    return jsonify({
        "message": "Study material cleared."
    })


@app.route("/membership", methods=["POST"])
@login_required
def membership():

    return jsonify({
        "message": "Lifetime membership payment setup is not connected yet.",
        "price": 1000
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
