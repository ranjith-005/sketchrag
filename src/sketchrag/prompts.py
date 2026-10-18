"""Prompts shared by every system, so only the context differs between systems.
Wording is adapted from LongBench's dataset2prompt for each task family."""

SYSTEM = (
    "You answer questions about a long document that arrived as a stream. You only see the "
    "context provided, which may be a compressed memory of the stream rather than the full text. "
    "Follow the answer-format instruction exactly."
)

INSTRUCTION = {
    "qa": "Answer with the shortest possible phrase (a name, date, number or a few words; for a yes/no "
          "question answer only yes or no). If the context does not contain the answer, give your best guess. "
          "Do not explain and do not output any other words.",
    "exact": "Answer with the exact value only. Do not output any other words.",
    "summ": "Write a concise summary that answers the query (or summarises the report if no query is given).",
    "retrieval": "The question is an abstract. Answer with the paragraph it comes from, in the format "
                 "'Paragraph N'. Do not output any other words.",
}

MAX_TOKENS = {"qa": 128, "exact": 64, "summ": 600, "retrieval": 32}


def user_prompt(kind: str, context: str, question: str) -> str:
    q = question.strip() or "Summarise the report."
    return f"<context>\n{context}\n</context>\n\nQuestion: {q}\n\n{INSTRUCTION[kind]}"
