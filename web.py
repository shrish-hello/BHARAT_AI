from flask import Flask, request, jsonify, send_from_directory, session, Response, stream_with_context
from google import genai
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime, timedelta
import time
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATABASE = os.path.join(BASE_DIR, "bharat_ai.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "bharat-ai-secret-key-change-this"
)

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set in Render Environment Variables."
    )

client = genai.Client(api_key=API_KEY)

MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash"
]

TRIAL_DAYS = 7
DAILY_LIMIT = 20


def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db()

    db.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            grade TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL,
            guardian_confirmed INTEGER DEFAULT 0,
            questions_today INTEGER DEFAULT 0,
            last_question_date TEXT,
            lifetime INTEGER DEFAULT 0
        )
    """)

    db.commit()
    db.close()


init_db()


def get_student():
    student_id = session.get("student_id")

    if not student_id:
        return None

    db = get_db()

    student = db.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    ).fetchone()

    db.close()

    return student


def reset_daily_questions(student):
    today = datetime.utcnow().date().isoformat()

    if student["last_question_date"] != today:
        db = get_db()

        db.execute(
            """
            UPDATE students
            SET questions_today = 0,
                last_question_date = ?
            WHERE id = ?
            """,
            (today, student["id"])
        )

        db.commit()
        db.close()

        return 0

    return student["questions_today"]


def get_status(student):
    created_at = datetime.fromisoformat(
        student["created_at"]
    )

    trial_end = created_at + timedelta(
        days=TRIAL_DAYS
    )

    now = datetime.utcnow()

    trial_active = now < trial_end

    trial_days_remaining = max(
        0,
        (trial_end.date() - now.date()).days
    )

    questions_today = reset_daily_questions(student)

    return {
        "membership": (
            "lifetime"
            if student["lifetime"]
            else (
                "trial"
                if trial_active
                else "expired"
            )
        ),
        "trial_days_remaining": trial_days_remaining,
        "daily_limit": DAILY_LIMIT,
        "daily_used": questions_today,
        "daily_remaining": max(
            0,
            DAILY_LIMIT - questions_today
        )
    }


def build_prompt(student, question):
    prompt = f"""
You are BHARAT AI, a fast, friendly and intelligent AI tutor for students.

Student name:
{student["name"]}

Student class:
{student["grade"]}

Rules:
- Answer quickly and clearly.
- Use simple language suitable for the student's class.
- Explain concepts instead of only giving the final answer.
- Use step-by-step explanations when useful.
- Support English, Hindi and Hinglish.
- Be accurate and honest.
- Do not invent information.
- If you are unsure, say so clearly.
- Keep answers appropriate for students.
- Avoid unnecessary repetition.

Student question:

{question}
"""

    uploaded_text = session.get(
        "uploaded_text",
        ""
    )

    if uploaded_text:
        prompt += f"""

The student has uploaded study material.

Use this material when it is relevant.

--- STUDY MATERIAL ---

{uploaded_text[:50000]}

--- END STUDY MATERIAL ---

If the answer is not in the uploaded material,
you may answer using your general knowledge.
"""

    return prompt


def get_model_stream(prompt):
    last_error = None

    for model in MODELS:

        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=prompt
            )

            return model, stream

        except Exception as error:

            last_error = error

            error_text = str(error).lower()

            retryable = any(
                phrase in error_text
                for phrase in [
                    "429",
                    "500",
                    "503",
                    "unavailable",
                    "overloaded",
                    "rate limit",
                    "resource exhausted",
                    "temporarily unavailable",
                    "internal server error",
                    "timeout",
                    "deadline exceeded"
                ]
            )

            if retryable:
                time.sleep(0.5)
                continue

    raise Exception(
        str(last_error)
        if last_error
        else "All Gemini models failed."
    )


def generate_normal_answer(prompt):
    last_error = None

    for model in MODELS:

        for attempt in range(2):

            try:

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
                    return answer.strip()

            except Exception as error:

                last_error = error

                error_text = str(error).lower()

                retryable = any(
                    phrase in error_text
                    for phrase in [
                        "429",
                        "500",
                        "503",
                        "unavailable",
                        "overloaded",
                        "rate limit",
                        "resource exhausted",
                        "temporarily unavailable",
                        "internal server error",
                        "timeout",
                        "deadline exceeded"
                    ]
                )

                if retryable and attempt == 0:
                    time.sleep(0.5)
                    continue

                break

    raise Exception(
        str(last_error)
        if last_error
        else "All Gemini models failed."
    )


def read_file(path):
    lower_name = path.lower()

    if lower_name.endswith(".txt"):

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

    if lower_name.endswith(".pdf"):

        try:
            from pypdf import PdfReader

            reader = PdfReader(path)

            text = []

            for page in reader.pages:
                text.append(
                    page.extract_text() or ""
                )

            return "\n".join(text)

        except Exception:
            return ""

    if lower_name.endswith(".docx"):

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
        "streaming": True
    })


@app.route("/register", methods=["POST"])
def register():

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

        grade = str(
            data.get("grade", "")
        ).strip()

        password = str(
            data.get("password", "")
        )

        guardian_confirmed = (
            data.get("guardian_confirmed")
            is True
        )

        if not name:
            return jsonify({
                "success": False,
                "message": "Please enter the student's name."
            }), 400

        if not email or "@" not in email:
            return jsonify({
                "success": False,
                "message": "Please enter a valid email address."
            }), 400

        if not grade:
            return jsonify({
                "success": False,
                "message": "Please select a class."
            }), 400

        if len(password) < 6:
            return jsonify({
                "success": False,
                "message": "Password must be at least 6 characters."
            }), 400

        if not guardian_confirmed:
            return jsonify({
                "success": False,
                "message": "A parent or guardian must approve this student account."
            }), 400

        db = get_db()

        existing = db.execute(
            """
            SELECT id
            FROM students
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        if existing:
            db.close()

            return jsonify({
                "success": False,
                "message": "An account with this email already exists."
            }), 409

        created_at = datetime.utcnow().isoformat()

        db.execute(
            """
            INSERT INTO students
            (
                name,
                email,
                grade,
                password,
                created_at,
                guardian_confirmed,
                questions_today,
                last_question_date,
                lifetime
            )
            VALUES (?, ?, ?, ?, ?, 1, 0, ?, 0)
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

        db.commit()

        student = db.execute(
            """
            SELECT *
            FROM students
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        db.close()

        session["student_id"] = student["id"]

        return jsonify({
            "success": True,
            "message": "Account created successfully.",
            "user": {
                "name": student["name"],
                "email": student["email"],
                "grade": student["grade"]
            },
            "trial_days": TRIAL_DAYS,
            "daily_limit": DAILY_LIMIT
        })

    except Exception as error:

        print(
            "REGISTER ERROR:",
            str(error),
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Unable to create the account."
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

        if not email or not password:
            return jsonify({
                "success": False,
                "message": "Please enter your email and password."
            }), 400

        db = get_db()

        student = db.execute(
            """
            SELECT *
            FROM students
            WHERE email = ?
            AND password = ?
            """,
            (
                email,
                password
            )
        ).fetchone()

        db.close()

        if not student:
            return jsonify({
                "success": False,
                "message": "Incorrect email or password."
            }), 401

        session["student_id"] = student["id"]

        return jsonify({
            "success": True,
            "message": "Login successful.",
            "user": {
                "name": student["name"],
                "email": student["email"],
                "grade": student["grade"]
            },
            "status": get_status(student)
        })

    except Exception as error:

        print(
            "LOGIN ERROR:",
            str(error),
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Unable to log in."
        }), 500


@app.route("/logout", methods=["POST"])
def logout():

    session.clear()

    return jsonify({
        "success": True,
        "message": "Logged out successfully."
    })


@app.route("/me")
def me():

    student = get_student()

    if not student:
        return jsonify({
            "success": True,
            "logged_in": False
        })

    return jsonify({
        "success": True,
        "logged_in": True,
        "user": {
            "name": student["name"],
            "email": student["email"],
            "grade": student["grade"]
        },
        "status": get_status(student)
    })


@app.route("/account/status")
def account_status():

    student = get_student()

    if not student:
        return jsonify({
            "success": False,
            "message": "Not logged in."
        }), 401

    return jsonify({
        "success": True,
        "status": get_status(student)
    })


@app.route("/ask", methods=["POST"])
def ask():

    student = get_student()

    if not student:
        return jsonify({
            "success": False,
            "answer": "Please log in before asking BHARAT AI a question."
        }), 401

    try:

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get("question", "")
        ).strip()

        if not question:
            return jsonify({
                "success": False,
                "answer": "Please ask me a question."
            }), 400

        status = get_status(student)

        if (
            status["membership"] == "expired"
            and not student["lifetime"]
        ):
            return jsonify({
                "success": False,
                "answer": "Your 7-day free trial has ended. Please activate BHARAT AI Lifetime Membership to continue."
            }), 403

        if (
            status["daily_remaining"] <= 0
            and not student["lifetime"]
        ):
            return jsonify({
                "success": False,
                "answer": f"You have reached today's limit of {DAILY_LIMIT} questions. Please try again tomorrow."
            }), 429

        prompt = build_prompt(
            student,
            question
        )

        answer = generate_normal_answer(
            prompt
        )

        db = get_db()

        db.execute(
            """
            UPDATE students
            SET questions_today = questions_today + 1,
                last_question_date = ?
            WHERE id = ?
            """,
            (
                datetime.utcnow().date().isoformat(),
                student["id"]
            )
        )

        db.commit()
        db.close()

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as error:

        print(
            "BHARAT AI ERROR:",
            str(error),
            flush=True
        )

        return jsonify({
            "success": False,
            "answer": "BHARAT AI is currently temporarily unavailable to answer. Please try again in a moment."
        }), 500


@app.route("/ask-stream", methods=["POST"])
def ask_stream():

    student = get_student()

    if not student:

        return jsonify({
            "success": False,
            "answer": "Please log in before asking BHARAT AI a question."
        }), 401

    try:

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get("question", "")
        ).strip()

        if not question:

            return jsonify({
                "success": False,
                "answer": "Please ask me a question."
            }), 400

        status = get_status(student)

        if (
            status["membership"] == "expired"
            and not student["lifetime"]
        ):

            return jsonify({
                "success": False,
                "answer": "Your 7-day free trial has ended."
            }), 403

        if (
            status["daily_remaining"] <= 0
            and not student["lifetime"]
        ):

            return jsonify({
                "success": False,
                "answer": f"You have reached today's limit of {DAILY_LIMIT} questions. Please try again tomorrow."
            }), 429

        prompt = build_prompt(
            student,
            question
        )

    except Exception as error:

        return jsonify({
            "success": False,
            "answer": str(error)
        }), 500


    @stream_with_context
    def generate():

        full_answer = ""

        try:

            model, stream = get_model_stream(
                prompt
            )

            yield "data: " + json.dumps({
                "type": "start",
                "model": model
            }) + "\n\n"

            for chunk in stream:

                text = getattr(
                    chunk,
                    "text",
                    None
                )

                if text:

                    full_answer += text

                    yield "data: " + json.dumps({
                        "type": "text",
                        "text": text
                    }) + "\n\n"

            if full_answer.strip():

                db = get_db()

                db.execute(
                    """
                    UPDATE students
                    SET questions_today = questions_today + 1,
                        last_question_date = ?
                    WHERE id = ?
                    """,
                    (
                        datetime.utcnow().date().isoformat(),
                        student["id"]
                    )
                )

                db.commit()
                db.close()

            yield "data: " + json.dumps({
                "type": "done"
            }) + "\n\n"

        except Exception as error:

            print(
                "STREAMING ERROR:",
                str(error),
                flush=True
            )

            yield "data: " + json.dumps({
                "type": "error",
                "message": "BHARAT AI is temporarily unavailable. Please try again."
            }) + "\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.route("/upload", methods=["POST"])
def upload():

    student = get_student()

    if not student:

        return jsonify({
            "success": False,
            "message": "Please log in first."
        }), 401

    try:

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

        if not filename.lower().endswith(
            (".txt", ".pdf", ".docx")
        ):

            return jsonify({
                "success": False,
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
                "success": False,
                "message": "The file was uploaded, but I couldn't read its text."
            }), 400

        session["uploaded_text"] = text[:50000]

        return jsonify({
            "success": True,
            "message": f"{filename} uploaded successfully. You can now ask questions about it."
        })

    except Exception as error:

        print(
            "UPLOAD ERROR:",
            str(error),
            flush=True
        )

        return jsonify({
            "success": False,
            "message": "Unable to upload the file."
        }), 500


@app.route("/clear", methods=["POST"])
def clear():

    session.pop(
        "uploaded_text",
        None
    )

    return jsonify({
        "success": True,
        "message": "Study material cleared."
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
