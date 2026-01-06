import spacy
import json
from transformers import pipeline

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load skills database
with open("skills.json", "r") as f:
    SKILL_DB = json.load(f)

# Load BERT zero-shot classifier
bert_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def extract_skills(text):
    extracted = set()
    doc = nlp(text)

    # spaCy noun-chunk matching
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower()
        for category, skill_list in SKILL_DB.items():
            for skill in skill_list:
                if skill in phrase:
                    extracted.add(skill)

    # BERT classification scoring
    all_skills = SKILL_DB["technical"] + SKILL_DB["soft"]
    bert_output = bert_classifier(text, candidate_labels=all_skills)

    for label, score in zip(bert_output["labels"], bert_output["scores"]):
        if score > 0.5:   # confidence threshold
            extracted.add(label)

    return list(extracted)
