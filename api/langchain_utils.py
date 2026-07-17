from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from chroma_utils import vectorstore


# -----------------------------
# Retriever
# -----------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# -----------------------------
# Contextualize Question Prompt
# -----------------------------
contextualize_q_system_prompt = """
Given a chat history and the latest user question,
which might reference context in the chat history,
formulate a standalone question.

Do NOT answer the question.
Only rewrite it if needed.
Otherwise return it unchanged.
"""


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)



# -----------------------------
# Question Answer Prompt
# -----------------------------
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not available in the context,
reply:
"I couldn't find that information in the uploaded documents."

Context:
{context}
""",
        ),

        MessagesPlaceholder(variable_name="chat_history"),

        ("human", "{input}"),
    ]
)



# -----------------------------
# Create RAG Chain
# -----------------------------
def get_rag_chain(model="llama-3.1-8b-instant"):

    llm = ChatGroq(
        model=model,
        temperature=0,
    )


    # Convert follow-up questions into standalone questions
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )


    # Answer generation chain
    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )


    # Complete RAG pipeline
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )


    return rag_chain