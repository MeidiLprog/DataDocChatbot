# app.py
import time
import gradio as gr
from dotenv import load_dotenv
from rag_answers import ask  # your Pinecone + Groq pipeline

load_dotenv()

TITLE = "Data Engineer SQL / Pandas / ML at your service"

def stream_answer(message: str, history: list):
    """
    ChatInterface sends (message, history).
    We only need 'message', but must accept both to avoid the arity error.
    """
    msg = (message or "").strip()
    if not msg:
        yield "Ask a question."
        return
    try:
        full = ask(msg)
    except Exception as e:
        yield f"Error: {e}"
        return

    # typing effect
    step = max(2, len(full) // 120)
    for i in range(0, len(full), step):
        yield full[: i + step]
        time.sleep(0.02)

demo = gr.ChatInterface(
    fn=stream_answer,                       # <-- now (message, history) is accepted
    title=TITLE,
    description="Ask about SQL / Pandas / ML. Answers are grounded in your indexed documents and include citations.",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    textbox=gr.Textbox(placeholder="e.g., How do I do a SELECT in SQL?", autofocus=True),
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
    css="""
    .gradio-container {max-width: 900px !important; margin: auto;}
    """,
)

if __name__ == "__main__":
    # If localhost is blocked by a proxy, keep share=True to get a public link.
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True, show_error=True)
