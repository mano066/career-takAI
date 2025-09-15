# Career TakAI

**Career TakAI** is a **RAG-based (Retrieval-Augmented Generation) personal resume knowledge base chatbot**.
It allows users to interact with a personal career knowledge base, including resumes, CVs, certifications, and other documents, to get **accurate, context-aware answers** about professional experience, skills, and achievements.

Think of it as a personal AI assistant for your career!

---

## 🚀 Features

* **Knowledge Base Chat:** Ask questions about your career, experience, and skills.
* **RAG-Powered:** Combines document retrieval with AI generation for precise answers.
* **Personalized:** Uses your own resume, CV, and other documents.
* **Unknown Questions Logging:** Automatically records questions the AI cannot answer.
* **Contact Recording:** Optionally collects user contact info for networking or follow-ups.

---

## 🛠️ Tech Stack

* **Python 3.11+** – Backend logic
* **Groq AI / LLM** – RAG-powered chatbot
* **LangChain / Groq tools** – Tool integration for logging unknown questions and user info
* **Gradio** – Web-based chat interface
* **PDF / Text parsing** – Ingest documents from multiple formats

---

## 📂 Project Structure

```
career-takAI/
│
├─ manova_ai_assistant/     # Core AI assistant code
├─ me/                      # Personal documents (PDFs, text, etc.)
├─ knowledge_base/          # Resume, certificates, and other knowledge files
├─ main.py                  # Entry point for running the chatbot
├─ requirements.txt         # Python dependencies
├─ .env                     # Environment variables (API keys)
└─ README.md                # Project documentation
```

---

## ⚡ How it Works

1. **Document Ingestion:** Load your PDFs, text files, and images.
2. **Vectorization & Retrieval:** AI reads your knowledge base to retrieve relevant context.
3. **Chat with AI:** Ask questions, and the AI answers based on your uploaded knowledge.
4. **Tool Integration:** Records unknown questions and optionally records user contact info.

---

## 💻 Getting Started

### Clone the repository

```bash
git clone https://github.com/mano066/career-takAI.git
cd career-takAI
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file with the following keys:

```env
PUSHOVER_TOKEN=your_pushover_api_token
PUSHOVER_USER=your_pushover_user_key
```

> Used to send notifications for unknown questions and recorded contacts.

### Run the chatbot

```bash
python main.py
```

* Open your browser at `http://localhost:7857`.
* Chat with your AI assistant about your career and skills.

---

## 🖥️ Features in the Web UI

* Chat interface with professional responses
* Contact buttons (Email, LinkedIn, Phone)
* Optional contact form to submit user info
* Example questions for guidance
* Dark mode and animations

---

## 🔮 Future Improvements

* Support for multiple users and separate knowledge bases
* Web-based hosting with authentication
* Enhanced support for images, PDFs, and other document formats
* Context memory for long conversations

