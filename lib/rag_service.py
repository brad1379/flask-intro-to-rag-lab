from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "the",
    "their",
    "to",
    "use",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> set[str]:
    """Convert text into a set of searchable lowercase tokens."""
    words = re.findall(r"[a-z0-9']+", text.lower())

    tokens = set()
    for word in words:
        word = word.strip("'")
        if len(word) <= 1:
            continue
        if word in STOPWORDS:
            continue
        tokens.add(word)

    return tokens


def document_search_text(document: dict[str, Any]) -> str:
    """Combine searchable document fields into one text value."""
    title = document.get("title", "")
    category = document.get("category", "")
    tags = " ".join(document.get("tags", []))
    text = document.get("text", "")

    return f"{title} {category} {tags} {text}"


def score_document(query: str, document: dict[str, Any]) -> dict[str, Any]:
    """Score a document using keyword overlap."""
    query_tokens = tokenize(query)
    document_tokens = tokenize(document_search_text(document))
    title_tokens = tokenize(document.get("title", ""))

    matched_terms = sorted(query_tokens & document_tokens)
    score = len(matched_terms)

    # Give a small boost when the query words show up in the title
    for token in query_tokens:
        if token in title_tokens:
            score += 0.5

    return {
        "document": document,
        "score": score,
        "matched_terms": matched_terms,
    }


def retrieve_context(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 2,
    minimum_score: float = 1.0,
) -> list[dict[str, Any]]:
    """Select the most relevant documents for the query."""
    matches = []
    for document in documents:
        result = score_document(query, document)
        if result["score"] >= minimum_score:
            matches.append(result)

    matches.sort(key=lambda match: match["score"], reverse=True)

    return matches[:limit]


def format_context(context_matches: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context block for the prompt."""
    if not context_matches:
        return "No relevant context was found in the approved company documents."

    blocks = []
    for match in context_matches:
        document = match["document"]
        block = (
            f"Source ID: {document['id']}\n"
            f"Title: {document['title']}\n"
            f"Category: {document['category']}\n"
            f"Content: {document['text']}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, context_matches: list[dict[str, Any]]) -> str:
    """Build a structured prompt with instructions, context, question, and requirements."""
    context = format_context(context_matches)

    prompt = f"""Instructions:
You are a helpful internal assistant for company employees and developers.
Use only the context provided below to answer the question.
Do not invent policies, numbers, or details that are not supported by the context.
If the context does not contain the answer, say that the approved documents do not have enough information.

Context:
{context}

Question:
{query}

Response requirements:
- Answer clearly in a few sentences.
- Only include information found in the context.
- Mention the Source ID of any document you used.
"""
    return prompt


def source_metadata(match: dict[str, Any]) -> dict[str, str]:
    """Return source information that is safe to expose in the API response."""
    document = match["document"]
    return {
        "id": document["id"],
        "title": document["title"],
    }
