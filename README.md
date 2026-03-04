# QuestionnaireAI

Built for the Almabase GTM Engineering Internship assignment.

The idea is simple — companies receive vendor assessments, security audits, compliance questionnaires all the time and answering them manually is painful. This tool lets you upload your internal docs once, then feed in any questionnaire and get AI-generated answers with citations back.

---

## How it works

You create a project, upload your reference documents (PDFs, DOCX, TXT — whatever you have), then upload the questionnaire. The tool parses out the numbered questions, runs TF-IDF retrieval to find the most relevant chunks from your docs for each question, and sends that context to the AI to generate a grounded answer. Everything references which document it came from.

After generation you can:
- Review all answers on one page with confidence scores
- Edit any answer inline
- Regenerate only the ones you want to redo
- Export the whole thing as a DOCX

---

## Fictional company

**NovaMed Health** — healthcare SaaS for hospital patient intake and HIPAA-compliant data sharing. The `sample_data/` folder has three policy docs (security, privacy, infrastructure) and a sample vendor assessment questionnaire you can use to test the full flow end to end.

---

## Stack

- **Backend:** Flask + SQLite
- **Auth:** session-based, SHA-256 hashed passwords
- **File parsing:** pypdf, python-docx
- **Retrieval:** TF-IDF + cosine similarity (scikit-learn)
- **AI:** OpenRouter API (`google/gemini-2.5-flash`) via direct HTTP — no SDK
- **Export:** python-docx
- **Frontend:** Jinja2 + vanilla JS

---

## Setup

```bash
pip install flask pypdf python-docx scikit-learn werkzeug
```

Add your OpenRouter API key to the `.env` file:

```
OPENROUTER_API_KEY=your-key-here
```

Run it:

```bash
python app.py
```

Open `http://localhost:5000`, sign up, create a project and you're good to go.

### Quick demo

1. Create a project
2. Upload the three `.txt` files from `sample_data/` as reference docs
3. Upload `vendor_assessment_questionnaire.txt` as the questionnaire
4. Hit Generate — takes a few seconds
5. Review answers, edit anything that's off, export as DOCX

---

## Why TF-IDF instead of embeddings

Compliance documents have a lot of specific terminology and keywords that map pretty directly to what questions are asking. TF-IDF cosine similarity handles that well and doesn't need an external vector DB or embedding API. For a tool like this where keyword overlap is high, it's good enough and keeps the setup dead simple.

The tradeoff is semantic matching — something like "breach notification SLA" not matching "72-hour PHI notification window" perfectly. With more time I'd swap in embeddings for that.

---

## Notes / assumptions

- Questions need to be numbered (`1.`, `2)`, `Q1.`, etc.) for the parser to pick them up
- Confidence scores come from the model's own assessment, not retrieval scores
- SHA-256 for passwords is fine for a demo — production would use bcrypt

---

## What I'd do with more time

- Async generation with real-time progress (right now the page just waits)
- Embedding-based retrieval for better semantic matching
- Run history diffing (compare two runs on the same questionnaire)
- Excel/CSV questionnaire input support
- Team sharing on projects

---
