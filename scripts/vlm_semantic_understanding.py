import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# ISL SEMANTIC UNDERSTANDING
# Qwen2.5-3B-Instruct
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ------------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are the semantic understanding module of an Indian Sign Language
(ISL) communication assistant.

The input is an ISL GLOSS SEQUENCE produced by a sign-recognition model.
The tokens are sign labels, NOT ordinary English grammar.

Your task is to convert the recognized ISL gloss sequence into a
conservative semantic representation.

IMPORTANT RULES
---------------

1. Treat the recognized sign sequence as authoritative.

2. Preserve the meaning of all recognized signs.

3. ISL gloss order may differ from normal English grammar.

4. Do not invent signs, people, objects, locations, events, or facts.

5. Do not add information that is not supported by the recognized signs.

6. Do not change the intent unless the recognized signs support it.

7. Use only these intent categories:
   REQUEST
   QUESTION
   COMMAND
   EMERGENCY
   SOCIAL
   STATEMENT
   UNKNOWN

8. If a sequence is incomplete or ambiguous, interpret it conservatively.

9. Return ONLY valid JSON.

KNOWN PROJECT PATTERNS
----------------------

I WANT WATER
REQUEST
subject = I
action = WANT
object = WATER
natural_sentence = "I want water."

YOU NEED HELP
REQUEST
subject = YOU
action = NEED
object = HELP
natural_sentence = "You need help."

WE NEED WATER
REQUEST
subject = WE
action = NEED
object = WATER
natural_sentence = "We need water."

FAMILY NEED FOOD
REQUEST
subject = FAMILY
action = NEED
object = FOOD
natural_sentence = "Family needs food."

I WANT FOOD
REQUEST
subject = I
action = WANT
object = FOOD
natural_sentence = "I want food."

I NEED MEDICINE
REQUEST
subject = I
action = NEED
object = MEDICINE
natural_sentence = "I need medicine."

I NEED AMBULANCE
EMERGENCY
subject = I
action = NEED
object = AMBULANCE
natural_sentence = "I need an ambulance."

I PAIN
EMERGENCY
subject = I
action = EXPERIENCE
object = PAIN
natural_sentence = "I am in pain."

I PAIN HELP
EMERGENCY
subject = I
action = EXPERIENCE
object = PAIN HELP
natural_sentence = "I am in pain and need help."

EMERGENCY
EMERGENCY
subject = null
action = EMERGENCY
object = null
natural_sentence = "This is an emergency."

HELLO
SOCIAL
subject = null
action = GREET
object = null
natural_sentence = "Hello."

THANK_YOU
SOCIAL
subject = null
action = THANK
object = YOU
natural_sentence = "Thank you."

HELP
REQUEST
subject = null
action = HELP
object = null
natural_sentence = "I need help."

OUTPUT FORMAT
-------------

{
  "intent": "...",
  "subject": "...",
  "action": "...",
  "object": "...",
  "natural_sentence": "..."
}

Use null when a field is genuinely absent.
"""

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("=" * 70)
print("Loading ISL semantic language model...")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if torch.cuda.is_available():

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32
    )

    model.to("cpu")

model.eval()

print("Model loaded successfully.")
print(
    "Device:",
    "CUDA" if torch.cuda.is_available() else "CPU"
)

# ------------------------------------------------------------
# DETERMINISTIC PROJECT CASES
# ------------------------------------------------------------
#
# These are handled before the language model because they are
# known project-specific ISL gloss patterns. This prevents the
# language model from changing an already-defined interpretation.
#
# ------------------------------------------------------------

SPECIAL_CASES = {

    "HELLO": {
        "intent": "SOCIAL",
        "subject": None,
        "action": "GREET",
        "object": None,
        "natural_sentence": "Hello."
    },

    "THANK_YOU": {
        "intent": "SOCIAL",
        "subject": None,
        "action": "THANK",
        "object": "YOU",
        "natural_sentence": "Thank you."
    },

    "I PAIN": {
        "intent": "EMERGENCY",
        "subject": "I",
        "action": "EXPERIENCE",
        "object": "PAIN",
        "natural_sentence": "I am in pain."
    },

    "I PAIN HELP": {
        "intent": "EMERGENCY",
        "subject": "I",
        "action": "EXPERIENCE",
        "object": "PAIN HELP",
        "natural_sentence": "I am in pain and need help."
    },

    "I NEED AMBULANCE": {
        "intent": "EMERGENCY",
        "subject": "I",
        "action": "NEED",
        "object": "AMBULANCE",
        "natural_sentence": "I need an ambulance."
    },

    "EMERGENCY": {
        "intent": "EMERGENCY",
        "subject": None,
        "action": "EMERGENCY",
        "object": None,
        "natural_sentence": "This is an emergency."
    },

    "HELP": {
        "intent": "REQUEST",
        "subject": None,
        "action": "HELP",
        "object": None,
        "natural_sentence": "I need help."
    }
}

# ------------------------------------------------------------
# JSON EXTRACTION
# ------------------------------------------------------------

def extract_json(response):
    """
    Extract JSON even if the model accidentally adds surrounding text.
    """

    response = response.strip()

    # First try the complete response.
    try:
        return json.loads(response)

    except json.JSONDecodeError:
        pass

    # Fallback: locate the first JSON object.
    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1 or end <= start:

        raise ValueError(
            "Model did not return valid JSON:\n"
            + response
        )

    return json.loads(
        response[start:end + 1]
    )

# ------------------------------------------------------------
# RESULT VALIDATION
# ------------------------------------------------------------

ALLOWED_INTENTS = {
    "REQUEST",
    "QUESTION",
    "COMMAND",
    "EMERGENCY",
    "SOCIAL",
    "STATEMENT",
    "UNKNOWN"
}


def validate_result(result):

    required_keys = [
        "intent",
        "subject",
        "action",
        "object",
        "natural_sentence"
    ]

    # Ensure every expected field exists.
    for key in required_keys:

        if key not in result:
            result[key] = None

    # Prevent unexpected intent values.
    intent = result["intent"]

    if intent not in ALLOWED_INTENTS:

        result["intent"] = "UNKNOWN"

    # Natural sentence must be text when present.
    if result["natural_sentence"] is not None:

        result["natural_sentence"] = str(
            result["natural_sentence"]
        )

    return result

# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize_result(
    sign_sequence,
    result
):
    """
    Normalize whitespace/capitalization and apply deterministic
    project-specific cases.
    """

    sequence = " ".join(
        sign_sequence.strip().upper().split()
    )

    # Known project case takes priority.
    if sequence in SPECIAL_CASES:

        return SPECIAL_CASES[
            sequence
        ].copy()

    return validate_result(
        result
    )

# ------------------------------------------------------------
# MAIN SEMANTIC FUNCTION
# ------------------------------------------------------------

def understand_sign_sequence(
    sign_sequence
):
    """
    Convert an ISL gloss sequence into semantic JSON.

    This function is also imported by the live BiGRU pipeline.
    """

    if not isinstance(
        sign_sequence,
        str
    ):

        raise TypeError(
            "sign_sequence must be a string."
        )

    sign_sequence = " ".join(
        sign_sequence.strip().split()
    )

    if not sign_sequence:

        return {
            "intent": "UNKNOWN",
            "subject": None,
            "action": None,
            "object": None,
            "natural_sentence": ""
        }

    # --------------------------------------------------------
    # Fast path for known project sequences.
    # This avoids unnecessary Qwen generation.
    # --------------------------------------------------------

    normalized_sequence = sign_sequence.upper()

    if normalized_sequence in SPECIAL_CASES:

        return SPECIAL_CASES[
            normalized_sequence
        ].copy()

    # --------------------------------------------------------
    # Qwen semantic interpretation
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content":
                "Interpret this recognized ISL gloss sequence:\n"
                + sign_sequence
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    if torch.cuda.is_available():

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    result = extract_json(
        response
    )

    return normalize_result(
        sign_sequence,
        result
    )

# ------------------------------------------------------------
# TEST FUNCTION
# ------------------------------------------------------------

def print_test(sequence):

    print()
    print("=" * 70)
    print("SIGN SEQUENCE:", sequence)
    print("=" * 70)

    try:

        result = understand_sign_sequence(
            sequence
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )

    except Exception as error:

        print(
            "ERROR:",
            error
        )

# ------------------------------------------------------------
# TEST SET
# ------------------------------------------------------------

if __name__ == "__main__":

    test_sequences = [

        "I WANT WATER",

        "YOU NEED HELP",

        "WE NEED WATER",

        "FAMILY NEED FOOD",

        "I NEED AMBULANCE",

        "HELLO",

        "THANK_YOU",

        "I WANT FOOD",

        "I NEED MEDICINE",

        "I PAIN",

        "I PAIN HELP",

        "HELP",

        "EMERGENCY",

        # This one checks the Qwen fallback path.
        "YOU WANT WATER"
    ]

    for sequence in test_sequences:

        print_test(
            sequence
        )

    print()
    print("=" * 70)
    print(
        "ISL SEMANTIC UNDERSTANDING TEST COMPLETED"
    )
    print("=" * 70)
