# all prompt templates live here so they are easy to tweak


ANSWER_GENERATION_PROMPT = """You are a helpful assistant that answers
questions strictly based on the context provided below.

Rules you must follow:
- Only use information from the context below to answer
- If the context does not contain the answer say exactly:
  "I don't have enough information in the knowledge base to answer this."
- Always mention which page the information came from
- Keep your answer clear and to the point
- Do not make up or guess any information not in the context

Context from the Agentic AI eBook:
{context}

Question: {question}

Answer:"""


SELF_CHECK_PROMPT = """Look at this question, context, and answer carefully.

Question: {question}

Context:
{context}

Answer given: {answer}

Is every part of the answer fully supported by the context above?
Reply with only one word: yes or no"""


NO_ANSWER_RESPONSE = (
    "I don't have enough information in the knowledge base to answer "
    "this question. The retrieved content was not relevant enough to "
    "give you a reliable answer. Please try rephrasing your question "
    "or ask something more specific about Agentic AI."
)
