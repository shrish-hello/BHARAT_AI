from flask import Flask, request, jsonify, send_from_directory
from google import genai
import os
import time
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash"
]

uploaded_text = ""


def read_file(path):
    name = path.lower()

    if name.endswith(".txt"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print("TXT READ ERROR:", repr(e), flush=True)
            return ""

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = []

            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)

            return "\n".join(pages)
        except Exception as e:
            print("PDF READ ERROR:", repr(e), flush=True)
            return ""

    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print("DOCX READ ERROR:", repr(e), flush=True)
            return ""

    return ""


def is_retryable_error(error):
    message = str(error).lower()

    retry_words = [
        "503",
        "unavailable",
        "overloaded",
        "high demand",
        "temporarily",
        "429",
        "rate limit",
        "resource exhausted",
        "deadline",
        "timeout",
        "timed out",
        "internal error",
        "500"
    ]

    return any(word in message for word in retry_words)


def is_auth_error(error):
    message = str(error).lower()

    auth_words = [
        "401",
        "403",
        "unauthorized",
        "permission denied",
        "api key",
        "invalid api key",
        "authentication"
    ]

    return any(word in message for word in auth_words)


def generate_answer(prompt):
    last_error = None

    for model in MODELS:
        for attempt in range(2):
            try:
                print(
                    f"Trying Gemini model: {model}, attempt: {attempt + 1}",
                    flush=True
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                answer = getattr(response, "text", None)

                if answer and answer.strip():
                    print(
                        f"Gemini success using {model}",
                        flush=True
                    )
                    return answer.strip()

                raise RuntimeError(
                    f"{model} returned an empty response."
                )

            except Exception as e:
                last_error = e

                print(
                    f"Gemini error with {model}: {repr(e)}",
                    flush=True
                )

                if is_auth_error(e):
                    raise RuntimeError(
                        "Gemini API authentication failed. "
                        "Please check GEMINI_API_KEY in Render Environment."
                    )

                if is_retryable_error(e):
                    if attempt == 0:
                        time.sleep(2)
                    else:
                        time.sleep(1)
                    continue

                break

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}"
    )


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    global uploaded_text

    try:
        data = request.get_json(silent=True) or {}

        question = str(
            data.get("question", "")
        ).strip()

        if not question:
            return jsonify({
                "answer": "Please ask me a question."
            }), 200

        prompt = f"""
You are BHARAT AI, an intelligent and friendly AI tutor designed for students.

Your responsibilities:

- Explain concepts clearly.
- Use language suitable for a Class 6 student when the question is school-related.
- Give step-by-step explanations when useful.
- Help students understand concepts instead of only giving answers.
- Support English, Hindi and Hinglish.
- Keep answers organized and easy to read.
- Use examples when they help.
- Be accurate and honest.
- Never pretend to know something you do not know.
- If the student asks a school question, explain it at an appropriate level.

Student question:

{question}
"""

        if uploaded_text:
            material = uploaded_text[:40000]

            prompt += f"""

The student has uploaded study material.

Use it when relevant.

--- STUDY MATERIAL ---
{material}
--- END STUDY MATERIAL ---

If the answer is available in the study material, prioritize it.

If the answer is not present in the study material,
say that it is not found in the uploaded material and
then answer using your general knowledge.
"""

        answer = generate_answer(prompt)

        return jsonify({
            "answer": answer
        }), 200

    except Exception as e:
        print(
            "ASK ERROR:",
            repr(e),
            flush=True
        )

        error_text = str(e)

        return jsonify({
            "answer": (
                "BHARAT AI could not get a response from Gemini right now.\n\n"
                f"Reason: {error_text}"
            )
        }), 200


@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_text

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

        filename = secure_filename(file.filename)

        if not filename:
            return jsonify({
                "message": "Invalid file name."
            }), 400

        allowed_extensions = {
            ".txt",
            ".pdf",
            ".docx"
        }

        extension = os.path.splitext(filename)[1].lower()

        if extension not in allowed_extensions:
            return jsonify({
                "message": "Supported files: TXT, PDF and DOCX."
            }), 400

        path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(path)

        text = read_file(path)

        if not text.strip():
            return jsonify({
                "message": (
                    "The file was uploaded, "
                    "but I couldn't extract readable text from it."
                )
            }), 400

        uploaded_text = text[:40000]

        return jsonify({
            "message": (
                f"{filename} uploaded successfully. "
                "You can now ask questions about it."
            )
        }), 200

    except Exception as e:
        print(
            "UPLOAD ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "message": f"Upload error: {str(e)}"
        }), 500


@app.route("/clear", methods=["POST"])
def clear():
    global uploaded_text

    uploaded_text = ""

    return jsonify({
        "message": "Study material cleared."
    }), 200


@app.route("/health")
def health():
    return jsonify({
        "status": "BHARAT AI is running",
        "gemini": "configured",
        "models": MODELS
    }), 200


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

