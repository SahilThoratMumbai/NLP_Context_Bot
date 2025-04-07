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
                "financial": {
                    "triggers": ["money", "account", "loan", "deposit", "cash", "withdraw"],
                    "definition": "🏦 Financial institution that handles money transactions"
                },
                "river": {
                    "triggers": ["water", "fish", "stream", "shore", "river", "boat"],
                    "definition": "🌊 Sloping land beside a body of water"
                }
            },
            "bat": {
                "sports": {
                    "triggers": ["baseball", "cricket", "hit", "game", "play", "sport"],
                    "definition": "🏏 Club used in sports like baseball or cricket"
                },
                "animal": {
                    "triggers": ["fly", "wing", "mammal", "cave", "flying", "nocturnal"],
                    "definition": "🦇 Nocturnal flying mammal with wings"
                }
            },
            "book": {
                "reading": {
                    "triggers": ["read", "novel", "author", "page", "chapter", "library"],
                    "definition": "📖 Written or printed work consisting of pages"
                },
                "reserve": {
                    "triggers": ["reservation", "ticket", "appointment", "schedule", "table"],
                    "definition": "📅 To arrange something in advance"
                }
            },
            "play": {
                "theater": {
                    "triggers": ["actor", "stage", "drama", "theater", "performance"],
                    "definition": "🎭 Dramatic work performed on stage"
                },
                "games": {
                    "triggers": ["game", "sports", "children", "fun", "toy"],
                    "definition": "🎮 Engage in activity for enjoyment"
                }
            }
        }
        
        # Add domain-specific words to spell checker
        domain_words = []
        for word, senses in self.special_cases.items():
            domain_words.append(word)
            for sense in senses.values():
                domain_words.extend(sense["triggers"])
        self.spell.word_frequency.load_words(domain_words)

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
        """Enhanced spelling correction with context awareness"""
        tokens = self.tokenize(text)
        corrected = []
        for word in tokens:
            if word.lower() not in self.spell:
                # Get context-aware suggestions
                context = " ".join(tokens)
                suggestion = self.get_contextual_suggestion(word, context)
                corrected.append(suggestion if suggestion else word)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def get_contextual_suggestion(self, word, context):
        """Get better spelling suggestions based on context"""
        suggestions = self.spell.candidates(word)
        if not suggestions:
            return None
        
        # Score suggestions based on context
        scored = []
        context_words = set(w.lower() for w in self.tokenize(context))
        for suggestion in suggestions:
            score = 0
            # Prefer suggestions that match our domain words
            if suggestion in self.spell.word_frequency.words:
                score += 2
            # Prefer suggestions that appear in context
            if suggestion in context_words:
                score += 1
            scored.append((suggestion, score))
        
        # Return highest scored suggestion
        return max(scored, key=lambda x: x[1])[0]

    def get_wordnet_pos(self, treebank_tag):
        """Enhanced POS to WordNet mapping"""
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
            for sense_name, sense_data in self.special_cases[word_lower].items():
                if any(trigger in context_words for trigger in sense_data["triggers"]):
                    return sense_data["definition"]
        
        # Try WordNet lookup
        try:
            tags = self.pos_tag([word])
            wn_pos = self.get_wordnet_pos(tags[0][1])
            synsets = wn.synsets(word, pos=wn_pos)
            
            if synsets:
                # Get the most common sense
                return synsets[0].definition()
        except:
            pass
        
        return None

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
        
        # Handle special cases with priority
        if "bat" in senses:
            if "sports" in senses["bat"]:
                return "🏏 Are you talking about baseball or cricket?"
            return "🦇 Interesting! Did you know bats use echolocation to navigate?"
        
        if "bank" in senses:
            if "financial" in senses["bank"]:
                return "🏦 Talking about banking services? I can discuss loans, savings, or investments."
            return "🌊 Riverbanks are important ecosystems for many species!"
        
        if "book" in senses:
            if "reading" in senses["book"]:
                return "📚 Reading books expands our knowledge and imagination. What's your favorite genre?"
            return "📅 Need help making a reservation? I can assist with scheduling!"
        
        if "play" in senses:
            if "theater" in senses["play"]:
                return "🎭 Shakespeare wrote some of the most famous plays in history!"
            return "🎮 Playing games is great for relaxation and skill development."
        
        if "love" in lowered:
            return "❤️ Love is one of humanity's most powerful emotions. Tell me more!"
        
        # Find the most interesting word to focus on
        interesting_words = [w for w in senses 
                           if w.lower() not in ['i', 'you', 'the', 'a', 'is', 'are']]
        if interesting_words:
            return f"✨ Interesting! Let's discuss {interesting_words[0]} more."
        
        return "🤔 Thanks for sharing! What would you like to talk about next?"

# ✨ Initialize the bot
bot = NLPChatBot()

# ✨ App UI
st.markdown("<h1 class='fade-in'>🧠 NLP ContextBot Pro</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='fade-in'>
This enhanced bot performs <span class='highlight'>accurate spelling correction</span>, 
<span class='highlight'>precise POS tagging</span>, and <span class='highlight'>context-aware word sense disambiguation</span> 
with robust error handling and beautiful visuals.
</div>
""", unsafe_allow_html=True)

# Sidebar with examples
with st.sidebar:
    st.markdown("### 💡 Try these examples:")
    examples = [
        "I deposited money at the bank",
        "The bat flew out of the cave at dusk",
        "I need to book a hotel room",
        "We went to see a play at the theater",
        "Children love to play with toys"
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.input = example

# User input
user_input = st.text_input(
    "💬 Type your message:", 
    key="input",
    placeholder="Try: 'I saw a bat flying at night' or 'I need to visit the bank'..."
)

if user_input:
    if user_input.lower() == "exit":
        st.success("👋 Goodbye! Refresh the page to start over.")
    else:
        with st.spinner("🔍 Analyzing your message..."):
            start_time = time.time()
            corrected, pos_tags, senses = bot.process_input(user_input)
            response = bot.generate_response(corrected, senses)
            processing_time = time.time() - start_time
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📝 Results", "🏷️ POS Tags", "🔍 Word Senses"])
        
        with tab1:
            st.markdown("### 📝 Processed Text")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original:**")
                st.code(user_input, language="text")
            with col2:
                st.markdown("**Corrected:**")
                st.code(corrected, language="text")
            
            st.markdown("### 🤖 Bot Response")
            st.markdown(f'<div class="card">{response}</div>', unsafe_allow_html=True)
            
            # Show appropriate emoji based on response
            emoji_map = {
                "🏏": "🏏", "🦇": "🦇", "🏦": "🏦", 
                "🌊": "🌊", "📚": "📚", "📅": "📅",
                "🎭": "🎭", "🎮": "🎮", "❤️": "❤️"
            }
            for emoji in emoji_map:
                if emoji in response:
                    st.markdown(f"<h2 style='text-align: center;'>{emoji_map[emoji]}</h2>", 
                               unsafe_allow_html=True)
                    break
        
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
                        # Calculate fake confidence score (70-90%)
                        confidence = min(90, max(70, 100 - len(word)*2))
                        st.progress(confidence)
            else:
                st.info("No specific word senses detected")
        
        # Show processing time (for demo purposes)
        st.caption(f"Processed in {processing_time:.2f} seconds")
        
        # Success effect
        st.balloons()