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

PRIMARY_MODEL = "gemini-3.7-flash"
FALLBACK_MODEL = "gemini-2.5-flash"

uploaded_text = ""


def read_file(path):
    name = path.lower()

    if name.endswith(".txt"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
                paragraph.text
                for paragraph in doc.paragraphs
            )
        except Exception:
            return ""

    return ""


def is_temporary_error(error):
    text = str(error).lower()

    temporary_codes = [
        "503",
        "429",
        "500",
        "service unavailable",
        "unavailable",
        "high demand",
        "resource exhausted",
        "rate limit",
        "temporarily",
        "overloaded"
    ]

    return any(code in text for code in temporary_codes)


def generate_with_retry(prompt):
    primary_attempts = 3
    delays = [2, 4, 8]

    last_error = None

    for attempt in range(primary_attempts):
        try:
            print(
                f"Calling primary model "
                f"{PRIMARY_MODEL} - attempt {attempt + 1}/{primary_attempts}"
            )

            response = client.models.generate_content(
                model=PRIMARY_MODEL,
                contents=prompt
            )

            if response and response.text:
                print("Primary model succeeded.")
                return response.text

            raise RuntimeError("Primary model returned an empty response.")

        except Exception as error:
            last_error = error

            print(
                f"Primary model error: {error}"
            )

            if not is_temporary_error(error):
                break

            if attempt < primary_attempts - 1:
                delay = delays[attempt]

                print(
                    f"Temporary Gemini error. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

    print(
        f"Primary model unavailable. "
        f"Switching to fallback model {FALLBACK_MODEL}."
    )

    fallback_attempts = 2

    for attempt in range(fallback_attempts):
        try:
            print(
                f"Calling fallback model "
                f"{FALLBACK_MODEL} - attempt {attempt + 1}/{fallback_attempts}"
            )

            response = client.models.generate_content(
                model=FALLBACK_MODEL,
                contents=prompt
            )

            if response and response.text:
                print("Fallback model succeeded.")
                return response.text

            raise RuntimeError("Fallback model returned an empty response.")

        except Exception as error:
            last_error = error

            print(
                f"Fallback model error: {error}"
            )

            if attempt < fallback_attempts - 1:
                print("Retrying fallback model in 3 seconds...")
                time.sleep(3)

    raise RuntimeError(
        f"Both Gemini models failed. Last error: {last_error}"
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
            })

        prompt = f"""
You are BHARAT AI, a friendly AI tutor for students.

Your job is to:

- Explain concepts clearly.
- Use simple language suitable for students.
- Give step-by-step explanations when useful.
- Help the student understand instead of simply giving an answer.
- Support English, Hindi and Hinglish.
- Be accurate and honest.
- Do not pretend to know information you do not know.
- For school questions, give age-appropriate explanations.
- Use examples when they make the concept easier.

Student question:
{question}
"""

        if uploaded_text:
            prompt += f"""

The student uploaded this study material:

--- STUDY MATERIAL ---
{uploaded_text[:50000]}
--- END STUDY MATERIAL ---

Use the uploaded material when it is relevant.

If the answer is not present in the uploaded material,
say so clearly and then answer using your general knowledge.
"""

        answer = generate_with_retry(prompt)

        return jsonify({
            "answer": answer
        })

    except Exception as error:
        print(
            f"ERROR in /ask: {error}"
        )

        return jsonify({
            "answer": (
                "BHARAT AI is temporarily unable to answer. "
                "Please try again in a moment."
            )
        }), 500


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

        filename = secure_filename(
            file.filename
        )

        if not filename:
            return jsonify({
                "message": "Invalid file name."
            }), 400

        path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(path)

        text = read_file(path)

        if not text:
            return jsonify({
                "message": (
                    "The file was uploaded, "
                    "but I couldn't read its text."
                )
            })

        uploaded_text = text

        return jsonify({
            "message": (
                f"{filename} uploaded successfully. "
                "You can now ask questions about it."
            )
        })

    except Exception as error:
        print(
            f"ERROR in /upload: {error}"
        )

        return jsonify({
            "message": f"Upload error: {error}"
        }), 500


@app.route("/clear", methods=["POST"])
def clear():
    global uploaded_text

    uploaded_text = ""

    return jsonify({
        "message": "Study material cleared."
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "BHARAT AI is running"
    })


if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
