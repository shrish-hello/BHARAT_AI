rom flask import Flask, request, jsonify, send_from_directory
import os
import json
import urllib.request
import urllib.error
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

MODEL = "gemini-3.7-flash"

MAX_FILE_TEXT = 12000
MAX_QUESTION = 4000
MAX_OUTPUT_TOKENS = 1200

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

            parts = []

            for page in reader.pages:
                text = page.extract_text() or ""

                if text:
                    parts.append(text)

                if sum(len(x) for x in parts) >= MAX_FILE_TEXT:
                    break

            return "\n".join(parts)[:MAX_FILE_TEXT]

        except Exception:
            return ""

    if name.endswith(".docx"):
        try:
            from docx import Document

            doc = Document(path)

            parts = []

            for paragraph in doc.paragraphs:
                if paragraph.text:
                    parts.append(paragraph.text)

                if sum(len(x) for x in parts) >= MAX_FILE_TEXT:
                    break

            return "\n".join(parts)[:MAX_FILE_TEXT]

        except Exception:
            return ""

    return ""


def ask_gemini(prompt):
    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS
        }
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))

        candidates = result.get("candidates", [])

        if not candidates:
            return "I couldn't generate an answer right now."

        candidate = candidates[0]

        content = candidate.get("content", {})
        parts = content.get("parts", [])

        answer_parts = []

        for part in parts:
            text = part.get("text")

            if text:
                answer_parts.append(text)

        answer = "".join(answer_parts).strip()

        if not answer:
            return "I couldn't generate an answer right now."

        return answer

    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            error_body = ""

        return f"Gemini API error {e.code}: {error_body[:1000]}"

    except urllib.error.URLError as e:
        return f"Connection error while contacting Gemini: {str(e)}"

    except Exception as e:
        return f"Gemini error: {str(e)}"


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

        question = question[:MAX_QUESTION]

        if not question:
            return jsonify({
                "answer": "Please ask me a question."
            })

        prompt = """
You are BHARAT AI, a friendly AI tutor designed to help students.

Rules:
- Explain things clearly and simply.
- Give step-by-step explanations when useful.
- Help the student understand instead of only giving an answer.
- Support English, Hindi and Hinglish.
- Be respectful and encouraging.
- Do not invent facts.
- If you are unsure, say so.
- Keep answers reasonably concise unless the student asks for more detail.

Student question:
""" + question

        if uploaded_text:
            material = uploaded_text[:MAX_FILE_TEXT]

            prompt += """

The student has uploaded study material.

Use it when it is relevant.

--- STUDY MATERIAL ---
""" + material + """
--- END STUDY MATERIAL ---

If the requested information is not in the study material, you may answer from general knowledge.
"""

        answer = ask_gemini(prompt)

        if answer.startswith("Gemini API error"):
            return jsonify({
                "answer": answer
            }), 502

        if answer.startswith("Connection error"):
            return jsonify({
                "answer": answer
            }), 502

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "answer": f"BHARAT AI server error: {str(e)}"
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

        path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(path)

        text = read_file(path)

        if not text:
            return jsonify({
                "message": "The file was uploaded, but I couldn't read its text."
            }), 400

        uploaded_text = text[:MAX_FILE_TEXT]

        return jsonify({
            "message": (
                f"{filename} uploaded successfully. "
                "You can now ask questions about it."
            )
        })

    except Exception as e:
        return jsonify({
            "message": f"Upload error: {str(e)}"
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
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
