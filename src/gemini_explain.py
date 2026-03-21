from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv


def _get_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        return genai
    except Exception:
        return None


def _get_model_name() -> str:
    load_dotenv()
    # Default: Gemini Flash model (user requirement)
    return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def explain_disease_with_gemini(
    *,
    predicted_class: str,
    confidence_score: float,
    model_accuracy_percent: float,
    explain_ctx: dict[str, Any],
    user_image_bytes: Optional[bytes] = None,
) -> str:
    """
    Gemini is used ONLY for explanation and symptom validation.
    It MUST NOT re-classify or override the QSVM prediction.
    """
    genai = _get_client()
    if genai is None:
        return (
            "Gemini explanation is unavailable (missing `GEMINI_API_KEY` or dependency issue). "
            "Set `GEMINI_API_KEY` in your environment to enable explanations."
        )

    system_guardrails = (
        "You are assisting a tomato leaf disease detection app.\n"
        "IMPORTANT RULES:\n"
        "- DO NOT perform classification or suggest a different class.\n"
        "- Treat the predicted class as fixed.\n"
        "- Provide symptom-based explanation and practical guidance.\n"
        "- If the prediction confidence is low, say so and recommend retaking the photo.\n"
        "- Use concise, farmer-friendly language.\n"
    )

    prompt = (
        f"{system_guardrails}\n"
        f"Fixed model output (do not change): {predicted_class}\n"
        f"Model confidence score (0-1, heuristic): {confidence_score:.3f}\n"
        f"Model test accuracy (%): {model_accuracy_percent if model_accuracy_percent==model_accuracy_percent else 'unknown'}\n"
        f"Image/feature summary (for reasoning, not classification):\n{explain_ctx}\n\n"
        "Tasks:\n"
        "1) Explain the disease: what it is and typical causes.\n"
        "2) List key visible symptoms the user should check for.\n"
        "3) Give immediate actions (cultural + sanitation) and preventive steps.\n"
        "4) If applicable, mention when to consult local agri extension / consider lab confirmation.\n"
    )

    try:
        model = genai.GenerativeModel(_get_model_name())
        parts = [prompt]
        if user_image_bytes:
            parts.append({"mime_type": "image/jpeg", "data": user_image_bytes})
        resp = model.generate_content(parts)
        text = getattr(resp, "text", None)
        return text.strip() if text else "Gemini returned no explanation text."
    except Exception as e:
        return f"Gemini explanation failed: {e}"

