import streamlit as st
import re
import time
from collections import defaultdict
import pandas as pd

# ========== Self-Contained NLP Implementation ==========
class NLPChatBot:
    def __init__(self):
        # Custom word senses with emojis
        self.word_senses = {
            "bank": {
                "financial": "💰 Financial institution that handles money",
                "river": "🌊 Sloping land beside a body of water"
            },
            "book": {
                "reading": "📖 A written or printed work",
                "reserve": "🔄 To arrange for something in advance"
            },
            "love": {
                "emotion": "❤️ Strong feeling of affection",
                "score": "🎾 Zero in tennis"
            }
        }
        
        # POS tag mapping with colors
        self.pos_colors = {
            'NOUN': '#4CC9F0', 
            'VERB': '#F72585',
            'ADJ': '#7209B7',
            'ADV': '#3A0CA3',
            'OTHER': '#4361EE'
        }
        
        # Common words dictionary
        self.dictionary = {
            'hello', 'hi', 'bank', 'book', 'love', 'river',
            'money', 'financial', 'water', 'read', 'knowledge',
            'emotion', 'tell', 'more', 'share', 'thanks'
        }

    def tokenize(self, text):
        return re.findall(r"\w+(?:'\w+)?|\S", text)

    def pos_tag(self, tokens):
        tags = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token.endswith('ing'):
                tags.append((token, 'VBG'))
            elif lower_token.endswith('ed'):
                tags.append((token, 'VBD'))
            elif lower_token.endswith('ly'):
                tags.append((token, 'RB'))
            elif lower_token.endswith('s'):
                tags.append((token, 'NNS'))
            elif lower_token[0].isupper() and len(token) > 1:
                tags.append((token, 'NNP'))
            elif lower_token in {'is', 'am', 'are', 'was', 'were'}:
                tags.append((token, 'VB'))
            else:
                tags.append((token, 'NN'))
        return tags

    def correct_spelling(self, text):
        tokens = self.tokenize(text)
        corrected = []
        for word in tokens:
            if word.lower() not in self.dictionary:
                corrected.append(word.lower())
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def disambiguate_word(self, word, context):
        word = word.lower()
        context_words = set(w.lower() for w in self.tokenize(context))
        
        if word in self.word_senses:
            senses = self.word_senses[word]
            for sense, definition in senses.items():
                if sense in context_words:
                    return definition
            return next(iter(senses.values()))
        return None

    def process_input(self, text):
        corrected = self.correct_spelling(text)
        tokens = self.tokenize(corrected)
        tags = self.pos_tag(tokens)
        senses = {}
        
        for token, tag in tags:
            meaning = self.disambiguate_word(token, corrected)
            if meaning:
                senses[token] = meaning
        
        return corrected, tags, senses

    def generate_response(self, corrected, pos_tags, senses):
        lowered = corrected.lower()
        
        if "bank" in senses:
            if "financial" in senses["bank"]:
                return "Are you talking about a financial institution?"
            else:
                return "Oh! You mean a river bank. Sounds peaceful."
        
        if "book" in lowered:
            return "Books are a great source of knowledge!"
        
        if "love" in lowered:
            return "Love is a beautiful emotion. Tell me more!"
        
        return "Thanks for sharing! What else would you like to talk about?"

# ========== Beautiful UI Components ==========
def display_pos_tags(tags):
    pos_df = pd.DataFrame(tags, columns=["Word", "POS"])
    
    # Map to simplified POS tags
    pos_map = {
        'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN',
        'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
        'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
        'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV'
    }
    
    pos_df['Simple_POS'] = pos_df['POS'].map(lambda x: pos_map.get(x, 'OTHER'))
    
    # Display as colored tags
    tags_html = ""
    for _, row in pos_df.iterrows():
        color = bot.pos_colors[row['Simple_POS']]
        tags_html += f"""
        <span style='display: inline-block; margin: 0.2em; padding: 0.4em 0.6em; 
        border-radius: 0.5em; background-color: {color}; color: white; font-weight: bold;'>
        {row['Word']} <small>({row['POS']})</small>
        </span>
        """
    st.markdown(tags_html, unsafe_allow_html=True)

def display_word_senses(senses):
    if not senses:
        st.warning("No specific word senses detected")
        return
    
    for word, sense in senses.items():
        with st.expander(f"✨ {word.capitalize()}", expanded=True):
            st.markdown(f"**Meaning:** {sense}")
            st.progress(70)  # Visual indicator of confidence

def show_emoji(response):
    if "bank" in response.lower() and "financial" in response.lower():
        st.markdown("<h1 style='text-align: center;'>💰</h1>", unsafe_allow_html=True)
    elif "bank" in response.lower():
        st.markdown("<h1 style='text-align: center;'>🌊</h1>", unsafe_allow_html=True)
    elif "book" in response.lower():
        st.markdown("<h1 style='text-align: center;'>📚</h1>", unsafe_allow_html=True)
    elif "love" in response.lower():
        st.markdown("<h1 style='text-align: center;'>❤️</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center;'>🧠</h1>", unsafe_allow_html=True)

# ========== Main App ==========
bot = NLPChatBot()

# Configure page
st.set_page_config(
    page_title="NLP Magic Bot", 
    page_icon="✨",
    layout="wide"
)

# Custom header
st.markdown("""
<style>
.header {
    font-size: 2.5em;
    color: #6a3093;
    padding: 10px;
    text-align: center;
    border-bottom: 2px solid #6a3093;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">✨ NLP Magic Bot</div>', unsafe_allow_html=True)
st.markdown("Discover the magic of natural language processing")

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This interactive bot demonstrates:
    - 📝 Text processing
    - 🏷️ POS tagging
    - 🔍 Word sense disambiguation
    """)
    st.markdown("---")
    st.markdown("**Try phrases like:**")
    st.markdown("- 'I love reading books'")
    st.markdown("- 'The river bank is beautiful'")
    st.markdown("- 'I need to visit the bank'")

# User input
user_input = st.text_input("Type your message here...", key="input")

if user_input:
    if user_input.lower() == 'exit':
        st.success("Goodbye! Refresh the page to start over.")
    else:
        with st.spinner("Analyzing your text..."):
            time.sleep(0.5)  # Simulate processing
            
            # Process input
            corrected, pos_tags, senses = bot.process_input(user_input)
            
            # Display results in columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Results")
                st.success(f"**Corrected:** {corrected}")
                
                response = bot.generate_response(corrected, pos_tags, senses)
                st.markdown(f"**Response:** {response}")
                show_emoji(response)
            
            with col2:
                st.subheader("Analysis")
                tab1, tab2 = st.tabs(["POS Tags", "Word Senses"])
                
                with tab1:
                    display_pos_tags(pos_tags)
                
                with tab2:
                    display_word_senses(senses)
            
            # Success effect
            st.balloons()