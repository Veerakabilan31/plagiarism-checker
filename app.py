# ================================
# 📌 IMPORTS
# ================================
from flask import Flask, render_template, request
import os
import re
from sentence_transformers import SentenceTransformer, util
from docx import Document
import PyPDF2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# ================================
# 📌 LOAD MODEL
# ================================
model = SentenceTransformer('all-MiniLM-L6-v2')


# ================================
# 📌 FILE READING
# ================================

def read_txt(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except:
        return ""


def read_docx(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return ""


def read_pdf(path):
    text = ""
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
    except:
        return ""

    # Reject scanned PDF
    if len(text.strip()) < 50:
        return None

    return text


def extract_text(path):
    if path.endswith(".txt"):
        return read_txt(path)
    elif path.endswith(".docx"):
        return read_docx(path)
    elif path.endswith(".pdf"):
        return read_pdf(path)
    return ""


# ================================
# 📌 SENTENCE SPLIT
# ================================
def split_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ================================
# 📌 PLAGIARISM CHECK
# ================================
def check_similarity(file1, file2):

    text1 = extract_text(file1)
    text2 = extract_text(file2)

    if text1 is None or text2 is None:
        return 0, [("❌ Scanned PDF detected. Upload text-based PDF.", 0)]

    if not text1.strip() or not text2.strip():
        return 0, [("❌ One file is empty or unreadable", 0)]

    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    if not s1 or not s2:
        return 0, [("❌ Not enough valid sentences", 0)]

    emb1 = model.encode(s1, convert_to_tensor=True)
    emb2 = model.encode(s2, convert_to_tensor=True)

    sim_matrix = util.cos_sim(emb1, emb2)

    matches = []
    total_score = 0
    count = 0

    for i in range(len(s1)):
        best_score = sim_matrix[i].max().item()

        if best_score > 0.75:
            matches.append((s1[i], round(best_score, 2)))
            total_score += best_score
            count += 1

    final_score = (total_score / count) * 100 if count else 0

    return round(final_score, 2), matches


# ================================
# 📌 ROUTES
# ================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check', methods=['POST'])
def check():
    try:
        file1 = request.files['file1']
        file2 = request.files['file2']

        if not file1 or not file2:
            return render_template('result.html',
                                   score=0,
                                   matches=[("❌ Upload both files", 0)])

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)

        file1.save(path1)
        file2.save(path2)

        score, matches = check_similarity(path1, path2)

        return render_template('result.html', score=score, matches=matches)

    except Exception as e:
        return render_template('result.html',
                               score=0,
                               matches=[(f"❌ Error: {str(e)}", 0)])


# ================================
# 📌 RUN (FOR DEPLOYMENT)
# ================================
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)