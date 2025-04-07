import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from pywsd.lesk import cosine_lesk
from spellchecker import SpellChecker
import os
import time

# ✅ Must be the first Streamlit command
st.set_page_config(
    page_title="🧠 NLP ContextBot Pro", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✨ Custom CSS for beautiful UI
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .pos-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        font-size: 0.85rem;
        color: white;
    }
    .noun { background-color: #4cc9f0; }
    .verb { background-color: #f72585; }
    .adj { background-color: #7209b7; }
    .adv { background-color: #3a0ca3; }
    .other { background-color: #4361ee; }
    .bot-response {
        background-color: #e9f7fe;
        border-left: 4px solid #4cc9f0;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ✨ Set up NLTK data path
nltk_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_path, exist_ok=True)
nltk.data.path.append(nltk_path)

# ✅ Download required NLTK resources
nltk_packages = [
    "punkt", "averaged_perceptron_tagger", 
    "wordnet", "omw-1.4", "stopwords"
]
for package in nltk_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package, download_dir=nltk_path)

# 🧠 Initialize spell checker
spell = SpellChecker()

# 🔤 POS converter (same as original)
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

# 🧹 Spelling correction (same as original)
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

# 🧠 NLP processing (same as original)
def process_input(user_input):
    corrected = correct_spelling(user_input)
    tokens = word_tokenize(corrected)
    pos_tags = pos_tag(tokens)
    disambiguated = {}

    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        if word.lower() == "bank" and "river" in [t.lower() for t in tokens]:
            disambiguated[word] = "sloping land (especially the slope beside a body of water)"
        else:
            context = ' '.join(tokens)
            sense = cosine_lesk(context, word, pos=wn_pos)
            if sense:
                disambiguated[word] = sense.definition()
    return corrected, pos_tags, disambiguated

# 💬 Bot response (same logic with enhanced UI)
def generate_response(corrected, pos_tags, senses):
    lowered = corrected.lower()
    if "bank" in lowered:
        meaning = senses.get("bank", "")
        if "financial" in meaning or "money" in meaning:
            return "🏦 Are you talking about a financial institution?"
        elif "river" in meaning or "slope" in meaning:
            return "🌊 Oh! You mean a river bank. Sounds peaceful."
        else:
            return "🤔 Which type of bank are you referring to?"
    elif "book" in lowered:
        return "📚 Books are a great source of knowledge!"
    elif "love" in lowered:
        return "❤️ Love is a beautiful emotion. Tell me more!"
    else:
        return "🙂 Thanks for sharing! What else would you like to talk about?"

# ✨ Beautiful UI Components
def display_pos_tags(tags):
    """Visualize POS tags with color coding"""
    pos_html = "<div>"
    for word, tag in tags:
        pos_class = ""
        if tag.startswith('NN'): pos_class = "noun"
        elif tag.startswith('VB'): pos_class = "verb"
        elif tag.startswith('JJ'): pos_class = "adj"
        elif tag.startswith('RB'): pos_class = "adv"
        else: pos_class = "other"
        
        pos_html += f"""
        <span class="pos-tag {pos_class}">
            {word} <small>({tag})</small>
        </span>
        """
    pos_html += "</div>"
    st.markdown(pos_html, unsafe_allow_html=True)

def display_word_senses(senses):
    """Visualize word senses with expandable sections"""
    if not senses:
        st.info("No specific word senses detected")
        return
    
    for word, definition in senses.items():
        with st.expander(f"🔍 {word.capitalize()}", expanded=True):
            st.markdown(f"**Definition:** {definition}")
            st.progress(70)  # Visual confidence indicator

# ✨ Main App UI
st.markdown("<h1 class='fade-in'>🧠 NLP ContextBot Pro</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='fade-in'>
This chatbot performs <b>spelling correction</b>, <b>POS tagging</b>, and <b>word sense disambiguation</b> 
using the Lesk algorithm with enhanced visualization.
</div>
""", unsafe_allow_html=True)

# Sidebar with examples
with st.sidebar:
    st.markdown("### 💡 Try these examples:")
    examples = [
        "I went to the bank to deposit money",
        "The river bank was beautiful",
        "I love reading books",
        "Tell me about your favorite book"
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.input = example

# User input
user_input = st.text_input(
    "💬 You:", 
    key="input",
    placeholder="Type your message here..."
)

if user_input:
    if user_input.lower() == 'exit':
        st.success("👋 Goodbye! Refresh the page to start over.")
    else:
        with st.spinner("🔍 Analyzing your input..."):
            start_time = time.time()
            corrected, pos_tags, senses = process_input(user_input)
            response = generate_response(corrected, pos_tags, senses)
            processing_time = time.time() - start_time
        
        # Display results in a beautiful layout
        st.markdown("---")
        
        # Corrected text
        with st.container():
            st.markdown("### 🔤 Corrected Input")
            st.markdown(f"<div class='result-card'>{corrected}</div>", unsafe_allow_html=True)
        
        # POS Tags
        with st.container():
            st.markdown("### 🔠 Part-of-Speech Tags")
            display_pos_tags(pos_tags)
        
        # Word Senses
        with st.container():
            st.markdown("### 🧠 Word Sense Disambiguation")
            display_word_senses(senses)
        
        # Bot Response
        st.markdown("### 🤖 Bot Response")
        st.markdown(f"<div class='bot-response'>{response}</div>", unsafe_allow_html=True)
        
        # Processing time (hidden by default)
        st.caption(f"Processed in {processing_time:.2f} seconds")
        
        # Success effect
        st.balloons()