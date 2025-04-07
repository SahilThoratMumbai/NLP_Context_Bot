import streamlit as st
import re
import time
from collections import defaultdict
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain
from streamlit_lottie import st_lottie
import json
import requests

# ========== Setup ==========
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
brain_animation = load_lottie("https://assets1.lottiefiles.com/packages/lf20_vybwn7df.json")
book_animation = load_lottie("https://assets1.lottiefiles.com/packages/lf20_sk5h1kfn.json")
love_animation = load_lottie("https://assets1.lottiefiles.com/packages/lf20_rycdstfb.json")
bank_animation = load_lottie("https://assets1.lottiefiles.com/packages/lf20_0fhjyd5r.json")

# ========== Self-Contained NLP ==========
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

def show_animation(response):
    if "bank" in response.lower() and "financial" in response.lower():
        st_lottie(bank_animation, height=200, key="bank")
    elif "bank" in response.lower():
        st_lottie(bank_animation, height=200, key="river")
    elif "book" in response.lower():
        st_lottie(book_animation, height=200, key="book")
    elif "love" in response.lower():
        st_lottie(love_animation, height=200, key="love")
    else:
        st_lottie(brain_animation, height=200, key="default")

# ========== Main App ==========
bot = NLPChatBot()

# Configure page
st.set_page_config(
    page_title="NLP Magic Bot", 
    page_icon="✨",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("✨ NLP Magic Bot")
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

# Main content
colored_header(
    label="✨ NLP Magic Bot",
    description="Discover the magic of natural language processing",
    color_name="violet-70"
)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    if user_input.lower() == 'exit':
        rain(
            emoji="👋",
            font_size=40,
            falling_speed=5,
            animation_length=1,
        )
        st.success("Goodbye! Refresh the page to start over.")
    else:
        with st.spinner("🔮 Analyzing your text..."):
            time.sleep(1)  # Simulate processing
            
            # Process input
            corrected, pos_tags, senses = bot.process_input(user_input)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📝 Results", "🏷️ POS Tags", "🔍 Word Senses"])
            
            with tab1:
                st.subheader("Corrected Text")
                st.success(corrected)
                
                st.subheader("Bot Response")
                response = bot.generate_response(corrected, pos_tags, senses)
                st.markdown(f"### {response}")
                show_animation(response)
            
            with tab2:
                st.subheader("Part-of-Speech Tags")
                st.caption("Color-coded by word type")
                display_pos_tags(pos_tags)
                
                # POS statistics
                pos_counts = defaultdict(int)
                for _, tag in pos_tags:
                    simple_tag = 'NOUN' if tag.startswith('NN') else \
                               'VERB' if tag.startswith('VB') else \
                               'ADJ' if tag.startswith('JJ') else \
                               'ADV' if tag.startswith('RB') else 'OTHER'
                    pos_counts[simple_tag] += 1
                
                st.subheader("POS Distribution")
                st.bar_chart(pos_counts)
            
            with tab3:
                st.subheader("Word Sense Disambiguation")
                display_word_senses(senses)
                
                if senses:
                    st.balloons()