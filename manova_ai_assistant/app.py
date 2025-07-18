import os
import json
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
import gradio as gr
from groq import Groq

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Additional context or notes about the conversation"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

class Me:
    def __init__(self):
        self.groq = Groq()
        self.name = "Manova"
        self.knowledge_base_text = ""
        self.image_paths = []
        self._load_documents()

    def _load_documents(self):
        """Load documents from multiple formats and paths"""
        doc_paths = {
            "pdf": [
                "me/cv.pdf",
                "knowledge_base/ManovaLebaku Moses.pdf"
            ],
            "text": [
                "me/summary.txt"
            ],
            "images": [
                "knowledge_base/certifications/WhatsApp Image 2024-04-18 at 2.59.18 PM (4).jpeg"
            ]
        }

        self.knowledge_base_text = ""  # Reset before loading

        # Load PDF text
        for pdf_path in doc_paths["pdf"]:
            if os.path.exists(pdf_path):
                print(f"Loading PDF: {pdf_path}")
                try:
                    reader = PdfReader(pdf_path)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            self.knowledge_base_text += "\n" + text
                            print(f"Loaded page {i+1} of {pdf_path} with {len(text)} chars")
                        else:
                            print(f"No text found on page {i+1} of {pdf_path}")
                except Exception as e:
                    print(f"Failed to read PDF {pdf_path}: {e}")
            else:
                print(f"PDF not found: {pdf_path}")

        # Load text files
        for txt_path in doc_paths["text"]:
            if os.path.exists(txt_path):
                print(f"Loading text file: {txt_path}")
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        self.knowledge_base_text += "\n" + content
                        print(f"Loaded text file {txt_path} with {len(content)} chars")
                except Exception as e:
                    print(f"Failed to read text file {txt_path}: {e}")
            else:
                print(f"Text file not found: {txt_path}")

        # Store image paths (just existence check)
        self.image_paths = [p for p in doc_paths["images"] if os.path.exists(p)]
        print(f"Loaded {len(self.image_paths)} image paths")

        print(f"Total knowledge base length: {len(self.knowledge_base_text)} characters")

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        return f"""You are acting as {self.name}, a professional AI assistant representing {self.name} on their personal website.
Your responsibilities include:
- Answering questions ONLY based on the knowledge base below. Do NOT answer if the information is not contained in the knowledge base.
- If you do NOT know the answer from the knowledge base, respond: "I don't know" and record the question.
- Engage professionally and politely.
- Use tools to record unknown questions and user details.
- NEVER make up information or guess beyond the knowledge base.

Knowledge Base Content:
{self.knowledge_base_text}

Guidelines:
- Be professional, friendly, and helpful.
- Stay strictly on topic about {self.name}'s professional background."""

    def chat(self, message, history):
        # Clean history keys to avoid Groq errors
        history = [{k: v for k, v in item.items() if k not in ('metadata', 'options')} for item in history]
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        recorded_unknown = False
        while not done:
            response = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=messages, 
                tools=tools,
                temperature=0.7
            )
            if response.choices[0].finish_reason == "tool_calls":
                message_obj = response.choices[0].message
                tool_calls = message_obj.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message_obj)
                messages.extend(results)
            else:
                done = True

        answer = response.choices[0].message.content

        # Check if answer indicates lack of knowledge, then record unknown question
        if any(phrase in answer.lower() for phrase in ["i don't know", "cannot answer", "not in my knowledge base", "no information"]):
            if not recorded_unknown:
                record_unknown_question(question=message)
                recorded_unknown = True

        return answer

def submit_contact(name, email):
    if email:
        record_user_details(email=email, name=name or "Name not provided", notes="Submitted via contact form")
        return gr.update(value="Thank you! Contact info recorded.")
    else:
        return gr.update(value="❗ Please enter your email.")

custom_css = """
.gradio-container {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    max-width: 800px;
    margin: 0 auto;
}
.dark .gradio-container {
    background: #1a1a1a;
}
.chatbot {
    min-height: 500px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.dark .chatbot {
    background: #2d2d2d;
}
.textbox textarea {
    min-height: 80px !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
.dark .textbox textarea {
    background: #333;
    color: white;
}
.button {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
}
.dark .button {
    background: #4f46e5 !important;
}
.contact-buttons {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    justify-content: center;
}
.contact-button {
    padding: 8px 16px;
    border-radius: 20px;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 14px;
    transition: all 0.3s ease;
}
.contact-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.email-button {
    background: #4285F4;
    color: white !important;
}
.linkedin-button {
    background: #0077B5;
    color: white !important;
}
.phone-button {
    background: #34B7F1;
    color: white !important;
}
.chat-input-container {
    display: flex;
    gap: 10px;
    align-items: center;
}
.contact-icon {
    font-size: 20px;
}
.accordion-header {
    font-weight: bold !important;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.user-message {
    animation: fadeIn 0.3s ease forwards;
}
.bot-message {
    animation: fadeIn 0.3s ease forwards 0.1s;
}
"""

# ... (keep all previous imports and setup code)

if __name__ == "__main__":
    me = Me()

    with gr.Blocks(
        title=f"Chat with {me.name}",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        # Header with contact buttons (keep your existing header code)
        gr.Markdown(f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1>Chat with {me.name}</h1>
            <div class='contact-buttons'>
                <a href='mailto:manomathew080@gmail.com' class='contact-button email-button'>
                    <span class='contact-icon'>✉️</span> Email
                </a>
                <a href='https://in.linkedin.com/in/manova-m-509145157' target='_blank' class='contact-button linkedin-button'>
                    <span class='contact-icon'>🔗</span> LinkedIn
                </a>
                <a href='tel:+886-958334626' class='contact-button phone-button'>
                    <span class='contact-icon'>📞</span> Call
                </a>
            </div>
        </div>
        <p style='text-align: center; margin-bottom: 20px;'>
            Ask me about my professional background, skills, and experience.
            I'll do my best to answer your questions about my career and work.
        </p>
        """)

        chatbot = gr.Chatbot(
            bubble_full_width=False,
            show_copy_button=True,
            avatar_images=("user.png", "bot.png"),
            height=500
        )

        # Input area with textbox and button
        with gr.Row():
            textbox = gr.Textbox(
                placeholder="Type your message here...",
                container=False,
                autofocus=True,
                lines=2,
                scale=7
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("📩 Contact Information (optional)", open=False):
            with gr.Row():
                name_input = gr.Textbox(label="Your Name", lines=1)
                email_input = gr.Textbox(label="Your Email", lines=1)
            contact_submit_btn = gr.Button("Send Contact Info", variant="secondary")

        gr.Examples(
            examples=[
                "What's your professional background?",
                "What skills do you have?",
                "Can you tell me about your work experience?",
                "How can I contact you for potential opportunities?"
            ],
            inputs=textbox,
            label="💡 Example Questions"
        )

        def respond(message, chat_history, name, email):
            if not message.strip():  # Don't respond to empty messages
                return chat_history
                
            groq_history = []
            for user_msg, bot_msg in chat_history:
                groq_history.append({"role": "user", "content": user_msg})
                if bot_msg:  # Only add non-empty bot messages
                    groq_history.append({"role": "assistant", "content": bot_msg})

            response = me.chat(message, groq_history)
            chat_history.append((message, response))

            if email:
                record_user_details(email=email, name=name or "Name not provided", notes=f"From chat: {message}")

            return chat_history

        # Handle both Enter key and button click
        textbox.submit(
            fn=respond,
            inputs=[textbox, chatbot, name_input, email_input],
            outputs=[chatbot],
            queue=False
        ).then(
            lambda: "",
            outputs=[textbox]
        )
        
        submit_btn.click(
            fn=respond,
            inputs=[textbox, chatbot, name_input, email_input],
            outputs=[chatbot],
            queue=False
        ).then(
            lambda: "",
            outputs=[textbox]
        )

        contact_submit_btn.click(
            submit_contact,
            inputs=[name_input, email_input],
            outputs=[email_input]
        )

    demo.launch(
    )