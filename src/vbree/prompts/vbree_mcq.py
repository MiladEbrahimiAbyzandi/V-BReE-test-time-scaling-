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
1) Rate the answer from 0 to 100 by aggregating:
   - clarity (0-30)
   - completeness (0-30)
   - accuracy (0-40)
   Empty responses should receive the minimum score.
2) Refine the answer for maximum clarity, completeness, and accuracy.
   Remove filler and omit feedback or references to the original version.
   If no improvements are possible, provide the text verbatim.
3) Based on your updated response, identify the single letter from the possible choices.

IMPORTANT:
Return ONLY valid JSON, with exactly these keys:
- "score" (number 0-100)
- "response" (string)
- "letter" (one of: {letters})

Example:
{{"score": 85, "response": "....", "letter": "C"}}
"""