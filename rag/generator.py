import os

def build_prompt(question, results):
    context = "\n\n".join(
        f"[Source: {r['source']}, page/section: {r['page']}]\n{r['text']}"
        for r in results
    )

    return f"""You are a document-grounded knowledge assistant.

Answer the question using ONLY the supplied context.
Do not invent facts. If the context does not contain enough
information, say that you could not find the answer in the
provided documents.

Always cite the relevant source names and page/section numbers.

CONTEXT:
{context}

QUESTION:
{question}
"""

def generate_answer(question, results, api_key=None, model=None):
    if not results:
        return "I could not find relevant information in the indexed documents."

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        # Useful local fallback: return the strongest retrieved evidence
        # rather than pretending an LLM generated an answer.
        best = results[0]
        return (
            "No OpenAI API key is configured, so the application is showing "
            "the strongest retrieved evidence instead of generating an LLM answer.\n\n"
            f"**Relevant evidence:** {best['text']}\n\n"
            f"**Source:** {best['source']}, page/section {best['page']}"
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=build_prompt(question, results),
    )
    return response.output_text
