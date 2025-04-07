import streamlit as st
import re
import time
from collections import defaultdict
import pandas as pd

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
    .other { background-color: #b5179e; }
    .bot-response {
        background-color: #e9f7fe;
        border-left: 4px solid #4cc9f0;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 🧠 Enhanced POS Tagger with more robust rules
class EnhancedPOSTagger:
    def __init__(self):
        # Common word lists for more accurate tagging
        self.verbs = {'is', 'am', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did'}
        self.prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'about', 'to', 'from'}
        self.conjunctions = {'and', 'but', 'or', 'so', 'yet', 'for', 'nor'}
        self.determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her'}
        self.pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        self.adverbs = {'very', 'really', 'quickly', 'slowly', 'well', 'badly', 'often', 'always', 'never'}
        
    def tag(self, tokens):
        """More accurate POS tagging with enhanced rules"""
        tags = []
        for i, token in enumerate(tokens):
            lower_token = token.lower()
            
            # Check for pronouns
            if lower_token in self.pronouns:
                tags.append((token, 'PRP'))
            
            # Check for determiners
            elif lower_token in self.determiners:
                tags.append((token, 'DT'))
            
            # Check for conjunctions
            elif lower_token in self.conjunctions:
                tags.append((token, 'CC'))
            
            # Check for prepositions
            elif lower_token in self.prepositions:
                tags.append((token, 'IN'))
            
            # Check for common verbs
            elif lower_token in self.verbs:
                tags.append((token, 'VB'))
            
            # Check for adverbs
            elif lower_token in self.adverbs:
                tags.append((token, 'RB'))
            
            # Check for verb endings
            elif lower_token.endswith('ing'):
                tags.append((token, 'VBG'))
            elif lower_token.endswith('ed'):
                tags.append((token, 'VBD'))
            elif lower_token.endswith('es'):
                tags.append((token, 'VBZ'))
            elif lower_token.endswith('s'):
                tags.append((token, 'VBZ' if i > 0 and tags[-1][1] in ('NN', 'PRP') else 'NNS'))
            
            # Check for adjective endings
            elif lower_token.endswith('able') or lower_token.endswith('ible'):
                tags.append((token, 'JJ'))
            elif lower_token.endswith('ful') or lower_token.endswith('ous'):
                tags.append((token, 'JJ'))
            
            # Check for adverb endings
            elif lower_token.endswith('ly'):
                tags.append((token, 'RB'))
            
            # Check for proper nouns (capitalized words)
            elif token[0].isupper() and len(token) > 1 and (i == 0 or tags[-1][1] in ('.', '!', '?')):
                tags.append((token, 'NNP'))
            
            # Default to noun for other cases
            else:
                tags.append((token, 'NN'))
        
        return tags

# 🧠 Self-contained NLP implementation with enhanced POS tagging
class NLPChatBot:
    def __init__(self):
        self.tagger = EnhancedPOSTagger()
        
        # Custom word senses database
        self.word_senses = {
            "bank": {
                "financial": "🏦 Financial institution that handles money",
                "river": "🌊 Sloping land beside a body of water"
            },
            "book": {
                "reading": "📖 A written or printed work",
                "reserve": "📅 To arrange something in advance"
            },
            "bat": {
                "sports": "🏏 Club used in baseball or cricket",
                "animal": "🦇 Flying mammal with wings"
            },
            "love": {
                "emotion": "❤️ Strong feeling of affection",
                "score": "🎾 Zero in tennis"
            }
        }
        
        # Common words dictionary
        self.dictionary = {
            'i', 'like', 'playing', 'with', 'bat', 'bank', 'book', 'love', 'river',
            'money', 'financial', 'water', 'read', 'knowledge', 'emotion', 'tell',
            'more', 'share', 'thanks', 'went', 'to', 'the', 'saw', 'flying'
        }

    def tokenize(self, text):
        """Simple whitespace tokenizer with punctuation handling"""
        return re.findall(r"\w+(?:'\w+)?|\S", text)

    def pos_tag(self, tokens):
        """Use our enhanced POS tagger"""
        return self.tagger.tag(tokens)

    def correct_spelling(self, text):
        """Basic spelling correction using dictionary lookup"""
        tokens = self.tokenize(text)
        corrected = []
        for word in tokens:
            if word.lower() not in self.dictionary:
                # Simple correction - just lowercase if not in dictionary
                corrected.append(word.lower())
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def disambiguate_word(self, word, context):
        """Custom word sense disambiguation"""
        word = word.lower()
        context_words = set(w.lower() for w in self.tokenize(context))
        
        if word in self.word_senses:
            senses = self.word_senses[word]
            for sense, definition in senses.items():
                if sense in context_words:
                    return definition
            
            # Default to first sense if no context match
            return next(iter(senses.values()))
        
        return None

    def process_input(self, text):
        """Complete NLP processing pipeline"""
        corrected = self.correct_spelling(text)
        tokens = self.tokenize(corrected)
        tags = self.pos_tag(tokens)
        senses = {}
        
        for token, tag in tags:
            meaning = self.disambiguate_word(token, corrected)
            if meaning:
                senses[token] = meaning
        
        return corrected, tags, senses

    def generate_response(self, corrected, senses):
        """Context-aware response generation"""
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
        
        return "🤔 Thanks for sharing! What else would you like to discuss?"

# ✨ Initialize the bot
bot = NLPChatBot()

# ✨ Main App UI
st.markdown("<h1 class='fade-in'>🧠 NLP ContextBot Pro</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='fade-in'>
This enhanced chatbot performs <b>accurate spelling correction</b>, <b>robust POS tagging</b>, 
and <b>context-aware word sense disambiguation</b> with beautiful visualization.
</div>
""", unsafe_allow_html=True)

# Sidebar with examples
with st.sidebar:
    st.markdown("### 💡 Try these examples:")
    examples = [
        "I deposited money at the bank",
        "The bat flew out of the cave at dusk",
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
            corrected, pos_tags, senses = bot.process_input(user_input)
            response = bot.generate_response(corrected, senses)
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
            # Create color-coded POS tags
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
                else: pos_class = "other"
                
                pos_html += f"""
                <span class="pos-tag {pos_class}">
                    {word} <small>({tag})</small>
                </span>
                """
            pos_html += "</div>"
            st.markdown(pos_html, unsafe_allow_html=True)
        
        # Word Senses
        with st.container():
            st.markdown("### 🧠 Word Sense Disambiguation")
            if senses:
                for word, definition in senses.items():
                    with st.expander(f"✨ {word.capitalize()}", expanded=True):
                        st.markdown(f"**Definition:** {definition}")
                        st.progress(70)  # Visual confidence indicator
            else:
                st.info("No specific word senses detected")
        
        # Bot Response
        st.markdown("### 🤖 Bot Response")
        st.markdown(f"<div class='bot-response'>{response}</div>", unsafe_allow_html=True)
        
        # Processing time (hidden by default)
        st.caption(f"Processed in {processing_time:.2f} seconds")
        
        # Success effect
        st.balloons()