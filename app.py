import streamlit as st
import nltk
import os
import re
import time
from collections import defaultdict
from spellchecker import SpellChecker

# ✅ Must be the first Streamlit command
st.set_page_config(
    page_title="🧠 NLP ContextBot Pro", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✨ Custom CSS for animations and styling
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .pos-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        font-size: 0.85rem;
    }
    .noun { background-color: #4cc9f0; color: white; }
    .verb { background-color: #f72585; color: white; }
    .adj { background-color: #7209b7; color: white; }
    .adv { background-color: #3a0ca3; color: white; }
    .other { background-color: #4361ee; color: white; }
    .highlight {
        background-color: #fff3cd;
        padding: 0.2rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ✨ Set up NLTK with robust error handling
def setup_nltk():
    try:
        nltk_path = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_path, exist_ok=True)
        nltk.data.path.append(nltk_path)
        
        required_packages = [
            "punkt", "averaged_perceptron_tagger", 
            "wordnet", "omw-1.4", "stopwords"
        ]
        
        for package in required_packages:
            try:
                nltk.data.find(package)
            except LookupError:
                nltk.download(package, download_dir=nltk_path)
        
        return True
    except Exception as e:
        st.error(f"⚠️ NLTK setup error: {str(e)}")
        return False

if not setup_nltk():
    st.warning("Some NLP features may be limited")

# 🧠 Enhanced NLP Components
class NLPChatBot:
    def __init__(self):
        self.spell = SpellChecker()
        self.special_cases = {
            "bank": {
                "financial": ["money", "account", "loan", "deposit"],
                "river": ["water", "fish", "stream", "shore"]
            },
            "bat": {
                "sports": ["baseball", "cricket", "hit", "game"],
                "animal": ["fly", "wing", "mammal", "cave"]
            },
            "book": {
                "reading": ["read", "novel", "author", "page"],
                "reserve": ["reservation", "ticket", "appointment"]
            }
        }
        
        # Add common words to spell checker
        self.spell.word_frequency.load_words([
            'hello', 'bank', 'book', 'bat', 'love', 'river',
            'financial', 'money', 'reading', 'sports', 'animal'
        ])

    def tokenize(self, text):
        """Robust tokenizer with fallback"""
        try:
            return nltk.word_tokenize(text)
        except:
            # Fallback to simple tokenizer
            return re.findall(r"\w+(?:'\w+)?|\S", text)

    def pos_tag(self, tokens):
        """POS tagging with fallback"""
        try:
            return nltk.pos_tag(tokens)
        except:
            # Fallback to simple POS tagging
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
                else:
                    tags.append((token, 'NN'))
            return tags

    def correct_spelling(self, text):
        """Enhanced spelling correction"""
        tokens = self.tokenize(text)
        corrected = []
        for word in tokens:
            if word.lower() not in self.spell:
                suggestion = self.spell.correction(word)
                corrected.append(suggestion if suggestion else word)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def get_wordnet_pos(self, treebank_tag):
        """POS to WordNet mapping"""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        return None

    def disambiguate_word(self, word, context):
        """Enhanced word sense disambiguation"""
        word_lower = word.lower()
        context_words = set(w.lower() for w in self.tokenize(context))
        
        # Check special cases first
        if word_lower in self.special_cases:
            for sense, triggers in self.special_cases[word_lower].items():
                if any(trigger in context_words for trigger in triggers):
                    return self.get_sense_definition(word_lower, sense)
        
        # Try WordNet lookup
        try:
            tags = self.pos_tag([word])
            wn_pos = self.get_wordnet_pos(tags[0][1])
            synsets = wn.synsets(word, pos=wn_pos)
            if synsets:
                return synsets[0].definition()
        except:
            pass
        
        return None

    def get_sense_definition(self, word, sense_type):
        """Get predefined definitions for special cases"""
        definitions = {
            "bank": {
                "financial": "🏦 Financial institution that handles money",
                "river": "🌊 Sloping land beside a body of water"
            },
            "bat": {
                "sports": "🏏 Club used in baseball or cricket",
                "animal": "🦇 Flying mammal with wings"
            },
            "book": {
                "reading": "📖 Written or printed work",
                "reserve": "📅 To arrange something in advance"
            }
        }
        return definitions.get(word, {}).get(sense_type, None)

    def process_input(self, text):
        """Robust processing pipeline"""
        try:
            corrected = self.correct_spelling(text)
            tokens = self.tokenize(corrected)
            tags = self.pos_tag(tokens)
            senses = {}
            
            for word, tag in tags:
                definition = self.disambiguate_word(word, corrected)
                if definition:
                    senses[word] = definition
            
            return corrected, tags, senses
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return text, [], {}

    def generate_response(self, corrected, senses):
        """Context-aware response generation with emojis"""
        lowered = corrected.lower()
        
        if "bat" in senses:
            if "sports" in senses["bat"]:
                return "🏏 Are you talking about baseball or cricket?"
            return "🦇 Interesting! Bats are the only flying mammals."
        
        if "bank" in senses:
            if "financial" in senses["bank"]:
                return "🏦 Talking about banking services?"
            return "🌊 Ah, the peaceful riverbank!"
        
        if "book" in lowered:
            return "📚 Books are wonderful sources of knowledge!"
        
        if "love" in lowered:
            return "❤️ Love is a powerful emotion. Tell me more!"
        
        interesting_words = [w for w in senses if w.lower() not in ['i', 'you', 'the']]
        if interesting_words:
            return f"✨ Interesting! Tell me more about {interesting_words[0]}."
        return "🤔 Thanks for sharing! What else would you like to discuss?"

# ✨ Initialize the bot
bot = NLPChatBot()

# ✨ App UI
st.markdown("<h1 class='fade-in'>🧠 NLP ContextBot Pro</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='fade-in'>
This enhanced bot performs <span class='highlight'>spelling correction</span>, 
<span class='highlight'>POS tagging</span>, and <span class='highlight'>word sense disambiguation</span> 
with robust error handling and beautiful visuals.
</div>
""", unsafe_allow_html=True)

# Sidebar with examples
with st.sidebar:
    st.markdown("### 💡 Try these examples:")
    examples = [
        "I went to the bank to deposit money",
        "The bat flew out of the cave",
        "I love reading books",
        "Can you book a table for dinner?"
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.input = example

# User input
user_input = st.text_input(
    "💬 Type your message:", 
    key="input",
    placeholder="Try: 'I like playing with bat' or 'I need to visit the bank'..."
)

if user_input:
    if user_input.lower() == "exit":
        st.success("👋 Goodbye! Refresh the page to start over.")
    else:
        with st.spinner("🔍 Analyzing your message..."):
            time.sleep(0.5)  # Simulate processing time
            corrected, pos_tags, senses = bot.process_input(user_input)
            response = bot.generate_response(corrected, senses)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📝 Results", "🏷️ POS Tags", "🔍 Word Senses"])
        
        with tab1:
            st.markdown("### 📝 Processed Text")
            st.markdown(f"**Original:** `{user_input}`")
            st.markdown(f"**Corrected:** `{corrected}`")
            
            st.markdown("### 🤖 Bot Response")
            st.markdown(f'<div class="card">{response}</div>', unsafe_allow_html=True)
            
            # Show appropriate emoji based on response
            if "🏏" in response:
                st.markdown("<h2 style='text-align: center;'>🏏</h2>", unsafe_allow_html=True)
            elif "🦇" in response:
                st.markdown("<h2 style='text-align: center;'>🦇</h2>", unsafe_allow_html=True)
            elif "🏦" in response:
                st.markdown("<h2 style='text-align: center;'>🏦</h2>", unsafe_allow_html=True)
            elif "🌊" in response:
                st.markdown("<h2 style='text-align: center;'>🌊</h2>", unsafe_allow_html=True)
            elif "📚" in response:
                st.markdown("<h2 style='text-align: center;'>📚</h2>", unsafe_allow_html=True)
            elif "❤️" in response:
                st.markdown("<h2 style='text-align: center;'>❤️</h2>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 🏷️ Part-of-Speech Tags")
            if pos_tags:
                # Create color-coded POS tags
                pos_html = "<div>"
                for word, tag in pos_tags:
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
                
                # POS statistics
                pos_counts = defaultdict(int)
                for _, tag in pos_tags:
                    simple_tag = 'NOUN' if tag.startswith('NN') else \
                               'VERB' if tag.startswith('VB') else \
                               'ADJ' if tag.startswith('JJ') else \
                               'ADV' if tag.startswith('RB') else 'OTHER'
                    pos_counts[simple_tag] += 1
                
                st.markdown("#### 📊 POS Distribution")
                st.bar_chart(pos_counts)
            else:
                st.warning("No POS tags available")
        
        with tab3:
            st.markdown("### 🔍 Word Senses")
            if senses:
                for word, definition in senses.items():
                    with st.expander(f"✨ {word.capitalize()}", expanded=True):
                        st.markdown(f"**Definition:** {definition}")
                        st.progress(75)  # Confidence indicator
            else:
                st.info("No specific word senses detected")
        
        # Success effect
        st.balloons()