import re
from typing import Dict, List


# ============================================================
# SEMANTIC UNDERSTANDING - STAGE 1
# ============================================================
# Input:
#     Recognized ISL sign sequence
#     Example: "I WANT WATER"
#
# Output:
#     Structured semantic information:
#       - normalized sequence
#       - intent
#       - subject
#       - action
#       - object
#       - entities
#       - natural sentence
#
# NOTE:
# This is the first semantic-understanding prototype.
# It is intentionally rule-based so that we can verify the
# interface before connecting a VLM/LLM.
# ============================================================


# ============================================================
# BASIC VOCABULARY
# ============================================================

PRONOUNS = {
    "I",
    "YOU",
    "WE",
}

ACTION_WORDS = {
    "WANT",
    "NEED",
    "GIVE",
    "TAKE",
    "GO",
    "COME",
    "EAT",
    "DRINK",
    "SLEEP",
    "KNOW",
    "UNDERSTAND",
    "ASK",
    "EXPLAIN",
    "HELP",
    "WAIT",
    "STOP",
    "WORK",
    "WELCOME",
    "THANK_YOU",
    "SORRY",
    "PLEASE",
}

OBJECT_WORDS = {
    "HELP",
    "WATER",
    "FOOD",
    "MONEY",
    "MEDICINE",
    "DOCTOR",
    "NURSE",
    "HOSPITAL",
    "AMBULANCE",
    "HOME",
    "SCHOOL",
    "FRIEND",
    "FAMILY",
    "MOTHER",
    "FATHER",
    "CHILD",
    "STUDENT",
    "PAIN",
    "DISEASE",
    "DANGER",
    "ACCIDENT",
    "EMERGENCY",
}

QUESTION_WORDS = {
    "ASK",
    "KNOW",
    "UNDERSTAND",
    "EXPLAIN",
}

EMERGENCY_WORDS = {
    "EMERGENCY",
    "DANGER",
    "ACCIDENT",
    "AMBULANCE",
    "HOSPITAL",
    "PAIN",
    "SICK",
    "DISEASE",
}

SOCIAL_WORDS = {
    "HELLO",
    "WELCOME",
    "THANK_YOU",
    "SORRY",
    "PLEASE",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_sequence(text: str) -> List[str]:
    """Convert recognized sign text into clean tokens."""

    if not text:
        return []

    text = text.upper().strip()

    # Allow both spaces and commas between recognized signs.
    text = text.replace(",", " ")

    tokens = [
        token.strip()
        for token in re.split(r"\s+", text)
        if token.strip()
    ]

    return tokens


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(tokens: List[str]) -> str:
    """Detect a high-level intent from the sign sequence."""

    if not tokens:
        return "UNKNOWN"

    token_set = set(tokens)

    # Emergency / safety has priority.
    if token_set.intersection(EMERGENCY_WORDS):
        return "EMERGENCY"

    # Questions / information seeking.
    if token_set.intersection(QUESTION_WORDS):
        return "QUESTION"

    # Requests.
    if "WANT" in token_set or "NEED" in token_set:
        return "REQUEST"

    # Actions / commands.
    if "STOP" in token_set:
        return "COMMAND"

    if "GO" in token_set or "COME" in token_set:
        return "ACTION"

    # Social communication.
    if token_set.intersection(SOCIAL_WORDS):
        return "SOCIAL"

    return "STATEMENT"


# ============================================================
# SUBJECT / ACTION / OBJECT EXTRACTION
# ============================================================

def extract_semantic_roles(tokens: List[str]) -> Dict[str, str]:
    """
    Extract simple semantic roles from an ISL sign sequence.

    Priority:
    1. I / YOU / WE are explicit subjects.
    2. If there is no pronoun, a topic before the action can be
       treated as the subject.
    3. The first object after the action becomes the object.

    Examples:
        I WANT WATER
        YOU NEED HELP
        FAMILY NEED FOOD
    """

    subject = None
    action = None
    object_value = None
    action_index = None

    # Find the first action.
    for index, token in enumerate(tokens):
        if token in ACTION_WORDS:
            action = token
            action_index = index
            break

    # Explicit pronoun has highest priority.
    for token in tokens:
        if token in PRONOUNS:
            subject = token
            break

    # No pronoun: use a topic/object before the action as subject.
    if subject is None and action_index is not None:
        for token in tokens[:action_index]:
            if token in OBJECT_WORDS and token != action:
                subject = token
                break

    # Prefer an object after the action.
    if action_index is not None:
        for token in tokens[action_index + 1:]:
            if token in OBJECT_WORDS and token != action:
                object_value = token
                break

    # Fallback to another object token.
    if object_value is None:
        for token in tokens:
            if token in OBJECT_WORDS and token != subject and token != action:
                object_value = token
                break

    return {
        "subject": subject,
        "action": action,
        "object": object_value,
    }


# ============================================================
# NATURAL SENTENCE GENERATION
# ============================================================

def generate_sentence(
    tokens: List[str],
    intent: str,
    roles: Dict[str, str]
) -> str:
    """Convert a recognized sign sequence into a simple sentence."""

    if not tokens:
        return ""

    subject = roles["subject"]
    action = roles["action"]
    object_value = roles["object"]

    # ----------------------------
    # Common request patterns
    # ----------------------------

    if intent == "REQUEST":

        if subject == "I" and action == "WANT" and object_value:
            return f"I want {object_value.lower()}."

        if subject == "I" and action == "NEED" and object_value:
            return f"I need {object_value.lower()}."

        # Preserve the recognized subject.
        if subject == "I" and action == "WANT" and object_value:
            return f"I want {object_value.replace('_', ' ').lower()}."

        if subject == "I" and action == "NEED" and object_value:
            return f"I need {object_value.replace('_', ' ').lower()}."

        if subject == "YOU" and action == "WANT" and object_value:
            return f"You want {object_value.replace('_', ' ').lower()}."

        if subject == "YOU" and action == "NEED" and object_value:
            return f"You need {object_value.replace('_', ' ').lower()}."

        if subject == "WE" and action == "WANT" and object_value:
            return f"We want {object_value.replace('_', ' ').lower()}."

        if subject == "WE" and action == "NEED" and object_value:
            return f"We need {object_value.replace('_', ' ').lower()}."

        if subject and subject not in PRONOUNS and action == "WANT" and object_value:
            return (
                f"{subject.replace('_', ' ').title()} wants "
                f"{object_value.replace('_', ' ').lower()}."
            )

        if subject and subject not in PRONOUNS and action == "NEED" and object_value:
            return (
                f"{subject.replace('_', ' ').title()} needs "
                f"{object_value.replace('_', ' ').lower()}."
            )

        if action == "WANT" and object_value:
            return f"I want {object_value.replace('_', ' ').lower()}."

        if action == "NEED" and object_value:
            return f"I need {object_value.replace('_', ' ').lower()}."

    # ----------------------------
    # Emergency
    # ----------------------------

    if intent == "EMERGENCY":

        if "AMBULANCE" in tokens:
            return "I need an ambulance."

        if "HOSPITAL" in tokens:
            return "I need a hospital."

        if "HELP" in tokens:
            return "I need help."

        if "PAIN" in tokens:
            return "I am in pain."

        if "SICK" in tokens:
            return "I am sick."

        if "DANGER" in tokens:
            return "There is danger."

        if "ACCIDENT" in tokens:
            return "There has been an accident."

        return "This is an emergency."

    # ----------------------------
    # Questions
    # ----------------------------

    if intent == "QUESTION":

        if "DOCTOR" in tokens:
            return "Where is the doctor?"

        if "HOSPITAL" in tokens:
            return "Where is the hospital?"

        if "MEDICINE" in tokens:
            return "I need information about the medicine."

        return "I have a question."

    # ----------------------------
    # Commands
    # ----------------------------

    if intent == "COMMAND":

        if "STOP" in tokens:
            return "Please stop."

    # ----------------------------
    # Social
    # ----------------------------

    if intent == "SOCIAL":

        if "HELLO" in tokens:
            return "Hello."

        if "WELCOME" in tokens:
            return "Welcome."

        if "THANK_YOU" in tokens:
            return "Thank you."

        if "SORRY" in tokens:
            return "Sorry."

        if "PLEASE" in tokens:
            return "Please."

    # ----------------------------
    # Simple action sentences
    # ----------------------------

    if subject == "I" and action == "GO":
        return "I want to go."

    if subject == "I" and action == "COME":
        return "I will come."

    if subject == "I" and action == "EAT":
        return "I want to eat."

    if subject == "I" and action == "DRINK":
        return "I want to drink."

    if subject == "I" and action == "SLEEP":
        return "I want to sleep."

    # ----------------------------
    # Fallback
    # ----------------------------

    return " ".join(token.replace("_", " ").lower() for token in tokens).capitalize() + "."


# ============================================================
# MAIN SEMANTIC FUNCTION
# ============================================================

def understand(sequence_text: str) -> Dict:

    tokens = normalize_sequence(sequence_text)

    intent = detect_intent(tokens)

    roles = extract_semantic_roles(tokens)

    sentence = generate_sentence(
        tokens,
        intent,
        roles
    )

    return {
        "sign_sequence": " ".join(tokens),
        "tokens": tokens,
        "intent": intent,
        "subject": roles["subject"],
        "action": roles["action"],
        "object": roles["object"],
        "natural_sentence": sentence,
    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SEMANTIC UNDERSTANDING - STAGE 1")
    print("=" * 70)

    test_sequences = [
        "I WANT WATER",
        "I NEED DOCTOR",
        "YOU NEED HELP",
        "WE NEED WATER",
        "FAMILY NEED FOOD",
        "I WANT FOOD",
        "EMERGENCY",
        "I NEED AMBULANCE",
        "I PAIN",
        "HELLO",
        "THANK_YOU",
        "I WANT MEDICINE",
    ]

    for sequence in test_sequences:

        result = understand(sequence)

        print()
        print("Input        :", result["sign_sequence"])
        print("Intent       :", result["intent"])
        print("Subject      :", result["subject"])
        print("Action       :", result["action"])
        print("Object       :", result["object"])
        print("Sentence     :", result["natural_sentence"])

    print()
    print("=" * 70)
    print("Semantic understanding test completed.")
    print("=" * 70)
