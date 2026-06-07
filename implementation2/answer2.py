from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document   #The code snippet is responsible for setting up the environment and defining the necessary components for a Retrieval-Augmented Generation (RAG) chatbot. It imports the required libraries, loads environment variables, initializes the embeddings model, and sets up the vectorstore and retriever for fetching relevant context documents. The SYSTEM_PROMPT variable defines the system message that will be used to guide the chatbot's responses based on the retrieved context. The resolve_query function is designed to enhance the user's query by looking back at the conversation history to find any relevant information that may have been mentioned in previous user turns. The fetch_context function retrieves relevant documents from the vectorstore based on the user's query, and the answer_question function generates a response using the language model while incorporating both the retrieved context and a lightweight conversation history.

load_dotenv(override=True) # Load environment variables from a .env file, overriding existing ones if necessary. This allows the script to access any required configuration or credentials that may be stored in the .env file, such as API keys for the OpenAI service. By setting override=True, it ensures that any existing environment variables with the same names will be replaced by those defined in the .env file, which can be useful for testing or development purposes where you want to ensure that specific values are used.

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")  # Define the path for the vector database, which will be used to store the embeddings generated from the documents in the knowledge base. The path is set to a folder named "vector_db" located in the parent directory of the current file's directory. This allows the script to easily access and manage the vector database where the embeddings are stored, enabling efficient retrieval of relevant context when answering user queries. By using Path from the pathlib library, it ensures that the file paths are constructed in a way that is compatible across different operating systems.

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(temperature=0, model_name=MODEL)   #These three lines initialize the components necessary for the RAG chatbot. The embeddings variable is set up using the OpenAIEmbeddings class with the specified model "text-embedding-3-large", which will be used to generate embeddings for the documents in the knowledge base. The vectorstore variable is initialized as a Chroma vectorstore, pointing to the specified DB_NAME directory and using the defined embeddings function. This vectorstore will store the embeddings and allow for efficient retrieval of relevant documents based on user queries. Finally, the retriever variable is created by calling the as_retriever method on the vectorstore, with search_kwargs specifying that it should return the top 5 relevant documents (k=5) when a query is made. The llm variable is initialized as a ChatOpenAI instance with a temperature of 0 (for deterministic responses) and using the specified model defined in the MODEL variable. This language model will be responsible for generating responses based on the retrieved context and user queries.

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing Insurellm.
Use the retrieved context to answer questions.
If you don’t know, say so.
Context:
{context}
""" # The SYSTEM_PROMPT variable defines a template for the system message that will be used to guide the chatbot's responses. It sets the tone and role of the assistant as a knowledgeable and friendly representative of Insurellm. The prompt instructs the assistant to use the retrieved context to answer questions and to admit when it does not know the answer. The {context} placeholder will be replaced with the actual retrieved context documents when generating responses, ensuring that the assistant has access to relevant information to provide accurate and informed answers to user queries.

def resolve_query(query: str, history: list):
    # Look back for the last user turn mentioning a name
    last_user = None
    if history:
        for turn in reversed(history):
            if turn["role"] == "user":
                last_user = turn["content"]
                break
    if last_user:
        return f"{last_user}\n{query}"
    return query # The resolve_query function is designed to enhance the user's query by looking back at the conversation history to find any relevant information that may have been mentioned in previous user turns. It iterates through the conversation history in reverse order, searching for the last user turn (where the role is "user") and retrieves its content. If a previous user turn is found, it combines that content with the current query, effectively providing additional context to the language model when generating a response. If no previous user turn is found, it simply returns the original query. This approach allows the chatbot to maintain continuity in the conversation and provide more informed responses based on the user's previous inputs.



def fetch_context(query: str) -> list[Document]:
    """Retrieve relevant context documents."""
    return retriever.invoke(query) # The fetch_context function is responsible for retrieving relevant context documents based on the user's query. It takes the query as input and uses the retriever (which is set up to search the vectorstore) to invoke a search for relevant documents. The retriever will return a list of Document objects that are deemed relevant to the query, which can then be used to provide context for generating responses in the RAG chatbot. This function allows the chatbot to access and utilize information from the knowledge base when answering user questions, improving the accuracy and relevance of its responses.


def answer_question(query: str, history=None):
    """
    Generate an answer with context injection AND lightweight history.
    History is a list of {"role": "user"/"assistant", "content": "..."}.
    """
    resolved_query = resolve_query(query, history or [])
    docs = fetch_context(resolved_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Start with system message
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    # Add past conversation turns (lightweight memory)
    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
    print(history)
    # Finally add the new user query
    messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)
    return response.content, docs # The answer_question function is responsible for generating a response to a user's query by incorporating both the retrieved context and a lightweight conversation history. It first resolves the query using the resolve_query function, which may enhance the query based on previous user turns in the conversation history. Then, it fetches relevant context documents using the fetch_context function and combines their content into a single string. The function constructs a list of messages starting with a system message that includes the retrieved context. It then adds past conversation turns from the history, distinguishing between user and assistant messages. Finally, it appends the new user query as a HumanMessage. The complete list of messages is then passed to the language model (llm) to generate a response, which is returned along with the retrieved context documents. First the prompt in triple quotes is used to define the system message that will guide the assistant's responses. Then the resolved_query is obtained by calling the resolve_query function, which may enhance the query based on previous user turns in the conversation history. The fetch_context function is called to retrieve relevant context documents based on the resolved query, and their content is combined into a single string. A list of messages is constructed, starting with a SystemMessage that includes the retrieved context. The function then iterates through the conversation history, adding past user and assistant messages to the list of messages. Finally, it appends the new user query as a HumanMessage. The complete list of messages is passed to the language model (llm) to generate a response, which is returned along with the retrieved context documents.
