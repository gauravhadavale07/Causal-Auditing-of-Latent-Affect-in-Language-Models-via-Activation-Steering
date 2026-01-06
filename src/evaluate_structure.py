import re
import numpy as np

NEGATIONS = {"not", "no", "never", "n't"}
FIRST_PERSON = {"i", "me", "my", "mine", "i'm", "i’ve", "i’d"}

def structural_scores(text):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]

    if len(tokens) == 0:
        return None

    neg_rate = sum(t in NEGATIONS for t in tokens) / len(tokens)
    fp_rate = sum(t in FIRST_PERSON for t in tokens) / len(tokens)
    sent_lens = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    avg_sent_len = np.mean(sent_lens) if sent_lens else 0
    repetition = 1 - len(set(tokens)) / len(tokens)

    return {
        "negation_rate": neg_rate,
        "first_person_rate": fp_rate,
        "avg_sentence_len": avg_sent_len,
        "repetition_rate": repetition,
        "sentence_count": len(sentences),
    }
