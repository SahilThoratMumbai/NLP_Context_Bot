import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from pywsd.lesk import cosine_lesk
from spellchecker import SpellChecker
import os

# ========== Setup ==========
# Ensure local nltk_data path exists
NLTK_DATA_PATH = "./nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download necessary NLTK data if not found
resources = [
    ("tokenizers/punkt", "punkt"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("corpora/stopwords", "stopwords"),
]

for path, name in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, download_dir=NLTK_DATA_PATH)

# Initialize spell checker
spell = SpellChecker()


# ========== Helper Functions ==========

# Convert POS tag to WordNet format
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
        return wn.NOUN  # default to noun


# Spell correction
def correct_spelling(text):
    tokens = word_tokenize(text)
    corrected_tokens = []
    for word in tokens:
        if word.lower() not in spell:
            corrected = spell.correction(word)
            corrected_tokens.append(corrected if corrected else word)
        else:
            corrected_tokens.append(word)
    return ' '.join(corrected_tokens)


# NLP pipeline
def process_input(user_input):
    corrected = correct_spelling(user_input)
    tokens = word_tokenize(corrected)
    pos_tags = pos_tag(tokens)
    disambiguated = {}

    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)

        # Custom override for river + bank
        if word.lower() == "bank" and "river" in [t.lower() for t in tokens]:
            disambiguated[word] = "sloping land (especially the slope beside a body of water)"
        else:
            context = ' '.join(tokens)
            sense = cosine_lesk(context, word, pos=wn_pos)
            if sense:
                disambiguated[word] = sense.definition()
    return corrected, pos_tags, disambiguated


# Response logic
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


# ========== Streamlit UI ==========

st.set_page_config(page_title="NLP ContextBot", page_icon="🧠")
st.title("🧠 NLP ContextBot")
st.markdown("This chatbot performs **spelling correction**, **POS tagging**, and **word sense disambiguation** using the Lesk algorithm.")

user_input = st.text_input("You:", key="input")

if user_input:
    if user_input.lower() == 'exit':
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
            st.error(f"⚠️ Something went wrong: {str(e)}")
