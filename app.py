import streamlit as st
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from spellchecker import SpellChecker
import os

# ✅ Setup nltk_data path and download resources if missing
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

required_resources = [
    "punkt",
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4"
]

for res in required_resources:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res, download_dir=nltk_data_path)

# ✅ Spell checker setup
spell = SpellChecker()

# ✅ Map POS tags to WordNet
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

# ✅ Spell correction
def correct_spelling(text):
    tokens = word_tokenize(text)
    corrected = [spell.correction(w) if w.lower() not in spell else w for w in tokens]
    return " ".join(corrected)

# ✅ Word Sense Disambiguation using simplified Lesk
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

# ✅ NLP pipeline
def process_input(text):
    corrected = correct_spelling(text)
    tokens = word_tokenize(corrected)
    tagged = pos_tag(tokens)
    senses = {}

    for word, tag in tagged:
        wn_pos = get_wordnet_pos(tag)
        senses[word] = simple_lesk_definition(word, corrected, pos=wn_pos)
    return corrected, tagged, senses

# ✅ Bot response generation
def generate_response(corrected, pos_tags, senses):
    lowered = corrected.lower()
    if "bank" in lowered:
        meaning = senses.get("bank", "")
        if meaning and "financial" in meaning:
            return "Are you talking about a financial institution?"
        elif meaning and "river" in meaning:
            return "Ah, the river bank then!"
        else:
            return "Could you clarify which bank you mean?"
    return "Thanks for your message! Anything else on your mind?"

# ✅ Streamlit UI
st.set_page_config(page_title="NLP ContextBot", page_icon="🧠")
st.title("🧠 NLP ContextBot")
st.markdown("This bot performs **spelling correction**, **POS tagging**, and **word sense disambiguation** using WordNet.")

user_input = st.text_input("You:", key="input")

if user_input:
    try:
        corrected, pos_tags, senses = process_input(user_input)
        response = generate_response(corrected, pos_tags, senses)

        st.markdown(f"**✅ Corrected Input:** `{corrected}`")
        st.markdown(f"**🔠 POS Tags:** `{pos_tags}`")
        st.markdown(f"**🧠 Word Senses:** `{senses}`")
        st.markdown(f"### 🤖 Bot: {response}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
