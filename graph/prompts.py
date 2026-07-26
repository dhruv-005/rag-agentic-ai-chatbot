ANSWER_GENERATION_PROMPT = """You are an expert assistant on
Agentic AI. Answer the question below using the context provided.

Give a detailed and complete answer. Always reference page numbers
when available. Use all relevant information from the context.

Context:
{context}

Question: {question}

Answer:"""


SELF_CHECK_PROMPT = """Is this answer supported by the context?

Context: {context}
Question: {question}
Answer: {answer}

Reply yes or no:"""


NO_ANSWER_RESPONSE = (
    "This specific topic was not found in the knowledge base. "
    "Please ask about Agentic AI definition, components, "
    "use cases, risks, or future outlook."
)
