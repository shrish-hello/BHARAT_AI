from flask import Flask, render_template, request, jsonify
from google import genai
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.6-flash"

uploaded_text = ""

def read_file(path):
    global uploaded_text

    name = path.lower()

    if name.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    return ""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Please ask me a question."})

        prompt = f"""
You are BHARAT AI, a friendly AI tutor for students.

Your job is to:
- Explain concepts clearly.
- Use simple language suitable for students.
- Give step-by-step explanations when useful.
- Help the student understand rather than simply giving an answer.
- Support English, Hindi and Hinglish.
- Never pretend to know information that you do not know.

Student question:
{question}
"""

        if uploaded_text:
            prompt += f"""

The student uploaded this study material:

--- STUDY MATERIAL ---
{uploaded_text[:50000]}
--- END STUDY MATERIAL ---

Use the uploaded material when it is relevant to the student's question.
If the answer is not present in the uploaded material, clearly say that and then answer using your general knowledge.
"""

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return jsonify({"answer": response.text})

    except Exception as e:
        return jsonify({"answer": f"BHARAT AI error: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_text

    if "file" not in request.files:
        return jsonify({"message": "No file selected."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"message": "No file selected."}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    text = read_file(path)

    if not text:
        return jsonify({
            "message": "The file was uploaded, but I couldn't read its text."
        })

    uploaded_text = text

    return jsonify({
        "message": f"{filename} uploaded successfully. You can now ask questions about it."
    })

@app.route("/clear", methods=["POST"])
def clear():
    global uploaded_text
    uploaded_text = ""
    return jsonify({"message": "Study material cleared."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
