"""Default fallback messages used throughout the agent, per supported language."""

from app.schemas.language import Language

_WORKFLOW_ERROR_MSGS: dict[Language, str] = {
    Language.english: "I encountered an issue processing your request. Please try again in a moment.",
    Language.polish: "Napotkałem problem z przetworzeniem Twojego zapytania. Spróbuj ponownie za chwilę.",
    Language.german: (
        "Bei der Verarbeitung Ihrer Anfrage ist ein Problem aufgetreten. Bitte versuchen Sie es gleich erneut."
    ),
    Language.spanish: "Encontré un problema al procesar tu solicitud. Por favor, inténtalo de nuevo en un momento.",
}

_GUARDRAILS_REFUSAL_MSGS: dict[Language, str] = {
    Language.english: (
        "I'm not able to help with that request. "
        "I can assist you with questions about your health data, activity, sleep, and recovery metrics."
    ),
    Language.polish: (
        "Nie mogę pomóc z tym zapytaniem. "
        "Mogę odpowiadać na pytania dotyczące Twoich danych zdrowotnych, aktywności, snu i wskaźników regeneracji."
    ),
    Language.german: (
        "Ich kann bei dieser Anfrage nicht helfen. "
        "Ich kann Ihnen bei Fragen zu Ihren Gesundheitsdaten, Aktivitäten, Schlaf- und Erholungswerten helfen."
    ),
    Language.spanish: (
        "No puedo ayudar con esa solicitud. "
        "Puedo ayudarte con preguntas sobre tus datos de salud, actividad, sueño y métricas de recuperación."
    ),
}


def get_workflow_error_msg(language: Language | None = None) -> str:
    return _WORKFLOW_ERROR_MSGS.get(language or Language.english, _WORKFLOW_ERROR_MSGS[Language.english])


def get_guardrails_refusal_msg(language: Language | None = None) -> str:
    return _GUARDRAILS_REFUSAL_MSGS.get(language or Language.english, _GUARDRAILS_REFUSAL_MSGS[Language.english])
