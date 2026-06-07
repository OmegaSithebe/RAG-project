import gradio as gr
from dotenv import load_dotenv
from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(docs):
    """Format retrieved context for display."""
    result = ""
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        result += f"📄 Source: {source}\n\n{doc.page_content}\n\n{'-'*50}\n\n"
    return result # The format_context function is designed to take a list of retrieved context documents and format them into a readable string for display in the user interface. It iterates through each document in the list, extracting the source information from the document's metadata (defaulting to "Unknown" if no source is provided) and appending it to the result string along with the page content of the document. Each document's content is separated by a line of dashes for better readability. The final formatted string is returned, which can then be displayed in the context box of the chatbot interface, allowing users to see the relevant information that was retrieved based on their query.


def chat(message, history):
    """Handle user input and return assistant response."""
    history = history or []  # ensure it's always a list

    # Pass history into answer_question
    answer, docs = answer_question(message, history)

    # Update history after getting the answer
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, format_context(docs) # The chat function is responsible for handling user input and generating the assistant's response. It takes the user's message and the conversation history as input. The function first ensures that the history is always a list, even if it is initially None. It then calls the answer_question function, passing both the user's message and the conversation history to generate an answer and retrieve relevant context documents. After receiving the answer, the function updates the conversation history by appending the user's message and the assistant's response as separate entries. Finally, it returns an empty string (to clear the input box), the updated history, and a formatted string of the retrieved context documents for display in the user interface. This allows the chatbot to maintain a coherent conversation while also providing users with relevant information from the knowledge base.




def main():
    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="Conversation", height=500)
                message = gr.Textbox(placeholder="Ask a question...", show_label=False)
            with gr.Column():
                context_box = gr.Textbox(label="Retrieved Context", lines=25)

        message.submit(chat, inputs=[message, chatbot], outputs=[message, chatbot, context_box])
    ui.launch(inbrowser=True) # The main function sets up the Gradio user interface for the Insurellm Expert Assistant chatbot. It creates a Blocks container with a title and a Markdown header to introduce the chatbot. The interface is organized into two columns: the left column contains a Chatbot component for displaying the conversation and a Textbox for user input, while the right column contains another Textbox for displaying the retrieved context. The message Textbox is configured to trigger the chat function when the user submits a query, passing the user's message and the current conversation history as inputs, and updating the message box, chatbot display, and context box with the outputs from the chat function. Finally, the UI is launched in the browser for users to interact with the chatbot.
 

if __name__ == "__main__":
    main() # This block ensures that the main function is called only when the script is executed directly (python app2.py). It prevents the main function from running if the file is imported as a module in another script. When executed, it will set up and launch the Gradio user interface for the Insurellm Expert Assistant chatbot, allowing users to interact with it through their web browser.

