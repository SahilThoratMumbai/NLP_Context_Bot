import streamlit as st
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import time
import re

# Download all required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# ✅ Streamlit configuration
st.set_page_config(
    page_title="🧠 NLP ContextBot Pro", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✨ Custom CSS
st.markdown("""
<style>
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
    .bot-response {
        background-color: #e9f7fe;
        border-left: 4px solid #4cc9f0;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class NLPChatBot:
    def __init__(self):
        self.target_words = {'bank', 'bat', 'book', 'love'}
        self.dictionary = {
            'i', 'like', 'playing', 'with', 'bat', 'bank', 'book', 'love', 'river',
            'money', 'financial', 'water', 'read', 'knowledge', 'emotion', 'tell',
            'more', 'share', 'thanks', 'went', 'to', 'the', 'saw', 'flying', 'deposited'
        }

    def tokenize(self, text):
        """Simplified tokenizer that doesn't rely on punkt"""
        return re.findall(r"\w+(?:'\w+)?|\S", text)

    def correct_spelling(self, text):
        """Basic spelling correction"""
        tokens = self.tokenize(text)
        corrected = []
        for word in tokens:
            if word.lower() not in self.dictionary:
                corrected.append(word.lower())
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def pos_tag(self, text):
        """POS tagging with fallback to simple regex if NLTK fails"""
        try:
            tokens = self.tokenize(text)
            return nltk.pos_tag(tokens)
        except:
            # Fallback simple tagger
            tags = []
            for token in self.tokenize(text):
                if token.lower() in {'i', 'you', 'he', 'she', 'it', 'we', 'they'}:
                    tags.append((token, 'PRP'))
                elif token.lower() in {'the', 'a', 'an'}:
                    tags.append((token, 'DT'))
                elif token.endswith('ing'):
                    tags.append((token, 'VBG'))
                elif token.endswith('ed'):
                    tags.append((token, 'VBD'))
                elif token[0].isupper():
                    tags.append((token, 'NNP'))
                else:
                    tags.append((token, 'NN'))
            return tags

    def disambiguate_word(self, word, sentence):
        """Word sense disambiguation with fallback"""
        if word.lower() not in self.target_words:
            return None
            
        try:
            synset = lesk(self.tokenize(sentence), word.lower())
            if synset:
                return {
                    'definition': synset.definition(),
                    'examples': synset.examples()
                }
        except:
            # Fallback simple senses
            if word.lower() == 'bank':
                return {'definition': '🏦 Financial institution'}
            elif word.lower() == 'bat':
                return {'definition': '🦇 Flying mammal'}
        return None

    def generate_response(self, senses):
        """Simplified response generation"""
        if senses:
            if 'bank' in senses:
                return "🏦 Talking about banking services?"
            if 'bat' in senses:
                return "🦇 Interesting! Bats are nocturnal."
        return "🤔 Thanks for sharing! What else would you like to discuss?"

    def process_input(self, text):
        """Robust processing pipeline"""
        corrected = self.correct_spelling(text)
        pos_tags = self.pos_tag(corrected)
        senses = {}
        
        for token, _ in pos_tags:
            if token.lower() in self.target_words:
                sense = self.disambiguate_word(token, corrected)
                if sense:
                    senses[token] = sense
        
        response = self.generate_response(senses)
        return corrected, pos_tags, senses, response

# Initialize the bot
bot = NLPChatBot()

# Main App
st.title("🧠 NLP ContextBot Pro")
st.markdown("Basic NLP processing with word sense disambiguation")

# User input
user_input = st.text_input("💬 You:", placeholder="Type your message here...")

if user_input:
    with st.spinner("Processing..."):
        try:
            corrected, pos_tags, senses, response = bot.process_input(user_input)
            
            st.markdown("### 🔤 Corrected Input")
            st.code(corrected)
            
            st.markdown("### 🔠 POS Tags")
            pos_html = "<div>"
            for word, tag in pos_tags:
                pos_class = "noun" if tag.startswith('NN') else "verb" if tag.startswith('VB') else "other"
                pos_html += f"<span class='pos-tag {pos_class}'>{word} ({tag})</span> "
            st.markdown(pos_html + "</div>", unsafe_allow_html=True)
            
            if senses:
                st.markdown("### 🧠 Word Senses")
                for word, sense in senses.items():
                    st.markdown(f"**{word}**: {sense['definition']}")
            
            st.markdown("### 🤖 Response")
            st.markdown(f"<div class='bot-response'>{response}</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try a different input")