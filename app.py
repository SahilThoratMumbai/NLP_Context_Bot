import streamlit as st
import re
from collections import defaultdict

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
    .prep { background-color: #4361ee; }
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
        # Context words for bank disambiguation
        self.river_context = {'river', 'water', 'stream', 'sit', 'sitting', 'near', 'by', 'side'}
        self.bank_context = {'money', 'account', 'deposit', 'withdraw', 'loan', 'financial'}
        
        # POS tagging rules
        self.pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her'}
        self.prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'of', 'near'}
        self.be_verbs = {'am', 'is', 'are', 'was', 'were'}
    
    def tokenize(self, text):
        """Simple regex tokenizer that doesn't require NLTK"""
        return re.findall(r"\w+(?:'\w+)?|\S", text)
    
    def pos_tag(self, tokens):
        """Rule-based POS tagger"""
        pos_tags = []
        for i, token in enumerate(tokens):
            lower_token = token.lower()
            
            # Determine POS tag
            if lower_token in self.pronouns:
                pos_tags.append((token, 'PRP'))
            elif lower_token in self.be_verbs:
                pos_tags.append((token, 'VBP'))
            elif lower_token in self.prepositions:
                pos_tags.append((token, 'IN'))
            elif token.endswith('ing'):
                pos_tags.append((token, 'VBG'))
            elif token.lower() == 'bank':
                pos_tags.append((token, 'NN'))
            else:
                # Default to noun
                pos_tags.append((token, 'NN'))
        return pos_tags
    
    def disambiguate_bank(self, tokens):
        """Determine if bank is river or financial"""
        context_words = set(token.lower() for token in tokens)
        
        river_matches = len(context_words & self.river_context)
        bank_matches = len(context_words & self.bank_context)
        
        if river_matches > bank_matches:
            return "🌊 Sloping land beside a body of water (river bank)"
        else:
            return "🏦 Financial institution (money bank)"
    
    def generate_response(self, bank_sense):
        """Generate appropriate response"""
        if "river" in bank_sense:
            return "🌊 Yes, riverbanks are beautiful places to relax!"
        else:
            return "🏦 Are you discussing financial matters?"

# Initialize the bot
bot = NLPChatBot()

# Streamlit UI
st.title("🧠 NLP ContextBot Pro")
st.markdown("This version uses a self-contained tokenizer and POS tagger with no external data requirements")

user_input = st.text_input("💬 You:", "I am sitting near river bank")

if user_input:
    # Process input
    tokens = bot.tokenize(user_input)
    pos_tags = bot.pos_tag(tokens)
    
    # Word sense disambiguation
    senses = {}
    if 'bank' in [token.lower() for token in tokens]:
        senses['bank'] = bot.disambiguate_bank(tokens)
    
    # Generate response
    response = bot.generate_response(senses.get('bank', ''))
    
    # Display results
    st.markdown("### 🔠 POS Tags")
    st.write(" ".join([f"{word} ({tag})" for word, tag in pos_tags]))
    
    if senses:
        st.markdown("### 🧠 Word Senses")
        for word, sense in senses.items():
            st.markdown(f"**{word}**: {sense}")
    
    st.markdown("### 🤖 Response")
    st.markdown(f"<div class='bot-response'>{response}</div>", unsafe_allow_html=True)