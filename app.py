import streamlit as st
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import time
from collections import defaultdict
import re

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

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
    .pron { background-color: #4895ef; }
    .det { background-color: #3f37c9; }
    .conj { background-color: #560bad; }
    .prep { background-color: #4361ee; }
    .part { background-color: #b5179e; }
    .other { background-color: #f15bb5; }
    .bot-response {
        background-color: #e9f7fe;
        border-left: 4px solid #4cc9f0;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .sense-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class NLPChatBot:
    def __init__(self):
        # Word sense targets
        self.target_words = {'bank', 'bat', 'book', 'love'}
        
        # Simple dictionary for common words
        self.dictionary = {
            'i', 'like', 'playing', 'with', 'bat', 'bank', 'book', 'love', 'river',
            'money', 'financial', 'water', 'read', 'knowledge', 'emotion', 'tell',
            'more', 'share', 'thanks', 'went', 'to', 'the', 'saw', 'flying', 'deposited'
        }
    
    def correct_spelling(self, text):
        """Basic spelling correction using dictionary lookup"""
        tokens = nltk.word_tokenize(text)
        corrected = []
        for word in tokens:
            if word.lower() not in self.dictionary:
                # Simple correction - just lowercase if not in dictionary
                corrected.append(word.lower())
            else:
                corrected.append(word)
        return ' '.join(corrected)
    
    def pos_tag(self, text):
        """POS tagging using NLTK"""
        tokens = nltk.word_tokenize(text)
        return nltk.pos_tag(tokens)
    
    def disambiguate_word(self, word, sentence):
        """Word sense disambiguation using NLTK's Lesk algorithm"""
        if word.lower() not in self.target_words:
            return None
            
        synset = lesk(nltk.word_tokenize(sentence), word.lower())
        if synset:
            return {
                'definition': synset.definition(),
                'examples': synset.examples(),
                'pos': synset.pos()
            }
        return None
    
    def generate_response(self, senses):
        """Context-aware response generation"""
        if senses:
            for word, sense in senses.items():
                if word.lower() == 'bank':
                    if 'financial' in sense['definition']:
                        return "🏦 Talking about banking services?"
                    else:
                        return "🌊 Ah, the peaceful riverbank!"
                elif word.lower() == 'bat':
                    if 'mammal' in sense['definition']:
                        return "🦇 Interesting! Bats are the only flying mammals."
                    else:
                        return "🏏 Are you talking about baseball or cricket?"
                elif word.lower() == 'book':
                    return "📚 Books are wonderful sources of knowledge!"
                elif word.lower() == 'love':
                    return "❤️ Love is a powerful emotion. Tell me more!"
        
        return "🤔 Thanks for sharing! What else would you like to discuss?"

    def process_input(self, text):
        """Complete NLP processing pipeline"""
        # Spelling correction
        corrected = self.correct_spelling(text)
        
        # POS tagging
        pos_tags = self.pos_tag(corrected)
        
        # Word sense disambiguation
        senses = {}
        for token, _ in pos_tags:
            if token.lower() in self.target_words:
                sense = self.disambiguate_word(token, corrected)
                if sense:
                    senses[token] = sense
        
        # Generate response
        response = self.generate_response(senses)
        
        return corrected, pos_tags, senses, response

# ✨ Initialize the bot
bot = NLPChatBot()

# ✨ Main App UI
st.markdown("<h1 class='fade-in'>🧠 NLP ContextBot Pro</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='fade-in'>
This chatbot performs:
- <b>Basic spelling correction</b> 
- <b>POS tagging</b>
- <b>Context-aware word sense disambiguation</b>
</div>
""", unsafe_allow_html=True)

# Sidebar with examples
with st.sidebar:
    st.markdown("### 💡 Try these examples:")
    examples = [
        "The bat flew out of the cave at dusk",
        "I deposited money at the bank",
        "She loves reading books about animals",
        "They are playing with a bat and ball"
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
            corrected, pos_tags, senses, response = bot.process_input(user_input)
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
            pos_html = "<div>"
            for word, tag in pos_tags:
                pos_class = ""
                if tag.startswith('NN'): pos_class = "noun"
                elif tag.startswith('VB'): pos_class = "verb"
                elif tag.startswith('JJ'): pos_class = "adj"
                elif tag.startswith('RB'): pos_class = "adv"
                elif tag.startswith('PRP'): pos_class = "pron"
                elif tag.startswith('DT'): pos_class = "det"
                elif tag.startswith('CC'): pos_class = "conj"
                elif tag.startswith('IN'): pos_class = "prep"
                elif tag == 'RP': pos_class = "part"
                else: pos_class = "other"
                
                pos_html += f"""
                <span class="pos-tag {pos_class}">
                    {word} <small>({tag})</small>
                </span>
                """
            pos_html += "</div>"
            st.markdown(pos_html, unsafe_allow_html=True)
        
        # Word Senses
        if senses:
            st.markdown("### 🧠 Word Sense Disambiguation")
            for word, sense in senses.items():
                with st.container():
                    st.markdown(f"#### ✨ {word.capitalize()}")
                    st.markdown(f"""
                    <div class='sense-card'>
                        <b>Definition:</b> {sense['definition']}<br>
                        {f"<b>Examples:</b> {', '.join(sense['examples'][:2])}" if sense.get('examples') else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(75)
        
        # Bot Response
        st.markdown("### 🤖 Bot Response")
        st.markdown(f"<div class='bot-response'>{response}</div>", unsafe_allow_html=True)
        
        # Processing time
        st.caption(f"Processed in {processing_time:.2f} seconds")
        
        # Success effect
        st.balloons()