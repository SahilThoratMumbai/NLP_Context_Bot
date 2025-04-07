import streamlit as st
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
import os

# Set custom NLTK path
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Force download the correct resources
required_nltk_packages = [
    "punkt",
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4"
]

for package in required_nltk_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package, download_dir=NLTK_DATA_PATH)

# Spell checker
spell = SpellChecker()

# POS to WordNet POS mapping
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

# Spelling correction
def correct_spelling(text):
    tokens = word_tokenize(text)
    corrected = []
    for word in tokens:
        if word.lower() not in spell:
            corrected_word = spell.correction(word)
            corrected.append(corrected_word if corrected_word else word)
        else:
            corrected.append(word)
    return " ".join(corrected)

# Simple Lesk-based WSD
def simple_lesk_definition(word, sentence, pos=None):
    context = set(word_tokenize(sentence))
    max_overlap = 0
    best_sense = None
    for sense in wn.synsets(word, pos=pos):
        signature = set(word_tokenize(sense.definition()))
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            best_sense = sense
            max_overlap = overlap
    return best_sense.definition() if best_sense else None

# NLP pipeline
def process_input(text):
    corrected = correct_spelling(text)
    tokens = word_tokenize(corrected)
    tagged = pos_tag(tokens)
    senses = {}

    for word, tag in tagged:
        wn_pos = get_wordnet_pos(tag)
        if word.lower() == "bank" and "river" in [t.lower() for t in tokens]:
            senses[word] = "sloping land (especially the slope beside a body of water)"
        else:
            meaning = simple_lesk_definition(word, corrected, pos=wn_pos)
            if meaning:
                senses[word] = meaning
    return corrected, tagged, senses

# Bot logic
def generate_response(corrected, pos_tags, senses):
    lowered = corrected.lower()
    if "bank" in lowered:
        meaning = senses.get("bank", "")
        if "financial" in meaning or "money" in meaning:
            return "Are you talking about a financial institution?"
        elif "river" in meaning or "slope" in meaning:
            return "Oh! You mean a river bank. Sounds peaceful."
        else:
            return "Which type of bank are you referring to?"
    elif "book" in lowered:
        return "Books are a great source of knowledge!"
    elif "love" in lowered:
        return "Love is a beautiful emotion. Tell me more!"
    else:
        return "Thanks for sharing! What else would you like to talk about?"

# Streamlit UI
st.set_page_config(page_title="NLP ContextBot", page_icon="🧠")
st.title("🧠 NLP ContextBot")
st.markdown("This bot performs **spelling correction**, **POS tagging**, and **word sense disambiguation** using WordNet.")

user_input = st.text_input("You:", key="input")

if user_input:
    if user_input.lower() == "exit":
        st.markdown("### 👋 Bot: Goodbye!")
    else:
        try:
            corrected, pos_tags, senses = process_input(user_input)
            response = generate_response(corrected, pos_tags, senses)
            st.markdown(f"**🔤 Corrected Input:** `{corrected}`")
            st.markdown(f"**🔠 POS Tags:** `{pos_tags}`")
            st.markdown(f"**🧠 Word Senses:** `{senses}`")
            st.markdown(f"### 🤖 Bot: {response}")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
