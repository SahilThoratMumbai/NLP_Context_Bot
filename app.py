import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from spellchecker import SpellChecker

# ========== Download NLTK Resources ==========
nltk_packages = [
    "punkt",
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4",
    "stopwords"
]

for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

# ========== NLP Helper Functions ==========
spell = SpellChecker()

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

def correct_spelling(text):
    tokens = word_tokenize(text)
    corrected = []
    for word in tokens:
        if word.lower() not in spell:
            suggestion = spell.correction(word)
            corrected.append(suggestion if suggestion else word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

def simple_lesk_definition(word, context_sentence, pos=None):
    max_overlap = 0
    best_sense = None
    context = set(word_tokenize(context_sentence))
    for sense in wn.synsets(word, pos=pos):
        signature = set(word_tokenize(sense.definition()))
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    return best_sense.definition() if best_sense else None

def process_input(user_input):
    corrected = correct_spelling(user_input)
    tokens = word_tokenize(corrected)
    tags = pos_tag(tokens)
    senses = {}

    for word, tag in tags:
        wn_pos = get_wordnet_pos(tag)
        if word.lower() == "bank" and "river" in [t.lower() for t in tokens]:
            senses[word] = "sloping land (especially the slope beside a body of water)"
        else:
            definition = simple_lesk_definition(word, corrected, pos=wn_pos)
            if definition:
                senses[word] = definition
    return corrected, tags, senses

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
st.markdown("This chatbot performs **spelling correction**, **POS tagging**, and **word sense disambiguation** using WordNet.")

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
