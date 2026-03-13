import string

def allowed_letters(n: int) -> str:
    if n < 2:
        raise ValueError("Need at least 2 choices.")
    if n > 26:
        raise ValueError("Max 26 choices supported (A-Z).")
    return ", ".join(string.ascii_uppercase[:n])

def format_choices(choices: list[str]) -> str:
    letters = string.ascii_uppercase
    return "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])

def build_prompt(question: str, previous_answer: str, choices: list[str]) -> str:
    letters = allowed_letters(len(choices))
    choices_str = format_choices(choices)

    return f"""Question: {question}
Existing Answer: {previous_answer if previous_answer else "(empty)"}

Possible Choices:
{choices_str}

Task:
1) Evaluate the previous answer from 0 to 100 by aggregating:
   - clarity (0-30)
   - completeness (0-30)
   - accuracy (0-40)
   Empty responses should receive the minimum score.

2) Treat the previous answer only as a candidate response, not as ground truth.
   Critically assess it and use only any parts that are logically valid or helpful.
   Then answer the question independently in your own words.
   Do not simply paraphrase or lightly edit the previous answer.
   If the previous answer is flawed, incomplete, or misleading, correct it.
   If it is strong, you may agree with it, but only after independent reasoning.
   Keep your response concise — maximum 990 tokens. No LaTeX formatting.

3) Based on your independent final response, identify the single best letter from the possible choices.

IMPORTANT:
- Maintain independence of judgment.
- Do not assume the previous answer is correct.
- Do not preserve the previous answer unless it is justified by your own reasoning.
- Return ONLY valid JSON on a SINGLE LINE
- No newlines inside the response field
- No markdown formatting like ** or *
- No bullet points or numbered lists in the response

Target Schema:
{{"score": <number>, "response": "<max 80 words, plain text, no newlines>", "letter": "<one of: {letters}>"}}

Example Output:
{{"score": 85, "response": "The capital of France is Paris because it has been the political center of France since the 12th century.", "letter": "C"}}

"""
