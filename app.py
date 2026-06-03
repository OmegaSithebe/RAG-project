import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(context):
    result = "## 📚 Relevant Context\n\n"

    for doc in context:
        source = doc.metadata.get("source", "Unknown")

        result += f"**Source:** {source}\n\n"
        result += f"{doc.page_content}\n\n"
        result += "---\n\n"

    return result


def chat(history):

    last_message = history[-1]["content"]

    answer, context = answer_question(
        last_message,
        history[:-1],
    )

    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return history, format_context(context)


def put_message_in_chatbot(message, history):
    history = history or []

    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return "", history


def main():

    theme = gr.themes.Soft(
        font=["Inter", "system-ui", "sans-serif"]
    )

    with gr.Blocks(
        title="Insurellm Expert Assistant"
    ) as ui:

        gr.Markdown(
            """
# 🏢 Insurellm Expert Assistant

Ask me anything about Insurellm!
"""
        )

        with gr.Row():

            with gr.Column(scale=1):

                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                )

                message = gr.Textbox(
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):

                context_markdown = gr.Markdown(
                    value="*Retrieved context will appear here*",
                    container=True,
                )

        (
            message.submit(
                put_message_in_chatbot,
                inputs=[message, chatbot],
                outputs=[message, chatbot],
            )
            .then(
                chat,
                inputs=chatbot,
                outputs=[chatbot, context_markdown],
            )
        )

    ui.launch(
        inbrowser=True,
    )


if __name__ == "__main__":
    main()