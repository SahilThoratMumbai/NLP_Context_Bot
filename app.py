import streamlit as st
import spacy
from textblob import TextBlob
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import time
from collections import defaultdict
from spacy import displacy
import en_core_web_sm

# Load Spacy model
nlp = en_core_web_sm.load()

# ✅ Must be the first Streamlit command
st.set_page_config(
    page_title="🧠 NLP ContextBot Pro+", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✨ Enhanced Custom CSS
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
    .dep-tree {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedNLPChatBot:
    def __init__(self):
        # Initialize conversation memory
        self.conversation_history = []
        
        # Word sense targets
        self.target_words = {'bank', 'bat', 'book', 'love', 'play', 'run', 'fly', 'light'}
    
    def correct_spelling(self, text):
        """Improved spelling correction using TextBlob"""
        blob = TextBlob(text)
        return str(blob.correct())
    
    def pos_tag(self, text):
        """Advanced POS tagging using Spacy"""
        doc = nlp(text)
        return [(token.text, token.tag_) for token in doc]
    
    def tokenize(self, text):
        """Tokenization using Spacy"""
        doc = nlp(text)
        return [token.text for token in doc]
    
    def disambiguate_word(self, word, sentence):
        """Enhanced WSD using NLTK's Lesk algorithm"""
        if word.lower() not in self.target_words:
            return None
            
        synset = lesk(sentence.split(), word.lower())
        if synset:
            return {
                'definition': synset.definition(),
                'examples': synset.examples(),
                'pos': synset.pos()
            }
        return None
    
    def analyze_sentiment(self, text):
        """Sentiment analysis using TextBlob"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def extract_entities(self, text):
        """Named Entity Recognition using Spacy"""
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def generate_response(self, text, senses, entities):
        """Context-aware response generation"""
        response = ""
        
        # Check for specific entities first
        if entities:
            for entity, label in entities:
                if label == 'ORG' and any(w in text.lower() for w in ['bank', 'financial']):
                    return "🏦 I see you mentioned a financial organization. Are you asking about banking services?"
                elif label == 'GPE':
                    return f"🌍 You mentioned {entity}. Are you asking about something location-specific?"
        
        # Check word senses
        if senses:
            for word, sense in senses.items():
                if word.lower() == 'bank':
                    if 'financial' in sense['definition']:
                        return "🏦 Talking about banking services? I can help with financial questions."
                    else:
                        return "🌊 Ah, riverbanks are such peaceful places! Are you discussing nature?"
                elif word.lower() == 'bat':
                    if 'mammal' in sense['definition']:
                        return "🦇 Fascinating creatures! Did you know bats use echolocation?"
                    else:
                        return "⚾ Baseball or cricket? I love sports discussions!"
                elif word.lower() == 'love':
                    return "❤️ Love is such a profound emotion. Would you like to share more?"
        
        # Default responses based on sentiment
        sentiment = self.analyze_sentiment(text)
        if sentiment['polarity'] > 0.5:
            return "😊 You seem positive! What else would you like to discuss?"
        elif sentiment['polarity'] < -0.5:
            return "😔 I sense some negative sentiment. Would you like to talk about it?"
        
        return "🤔 Interesting! Could you tell me more about that?"

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
        
        # Entity recognition
        entities = self.extract_entities(corrected)
        
        # Generate response
        response = self.generate_response(corrected, senses, entities)
        
        # Update conversation history
        self.conversation_history.append({
            'input': text,
            'corrected': corrected,
            'pos_tags': pos_tags,
            'senses': senses,
            'entities': entities,
            'response': response
        })
        
        return corrected, pos_tags, senses, entities, response

# ✨ Initialize the bot
bot = EnhancedNLPChatBot()

# ✨ Main App UI
st.markdown("<h1 class='fade-in'>🧠 NLP ContextBot Pro+</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='fade-in'>
This <b>advanced</b> chatbot performs:
- <b>Accurate spelling correction</b> 
- <b>State-of-the-art POS tagging</b>
- <b>Context-aware word sense disambiguation</b>
- <b>Named entity recognition</b>
- <b>Sentiment analysis</b>
</div>
""", unsafe_allow_html=True)

# Sidebar with examples
with st.sidebar:
    st.markdown("### 💡 Try these examples:")
    examples = [
        "The bat flew out of the cave at dusk",
        "I deposited money at Bank of America",
        "She loves reading books about Paris",
        "They are playing with a bat and ball in London",
        "I'm feeling really happy today!"
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
        with st.spinner("🔍 Analyzing your input with advanced NLP..."):
            start_time = time.time()
            corrected, pos_tags, senses, entities, response = bot.process_input(user_input)
            processing_time = time.time() - start_time
        
        # Display results in a beautiful layout
        st.markdown("---")
        
        # Corrected text
        with st.container():
            st.markdown("### 🔤 Corrected Input")
            st.markdown(f"<div class='result-card'>{corrected}</div>", unsafe_allow_html=True)
        
        # Linguistic Analysis
        with st.expander("🔍 Advanced Linguistic Analysis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏷️ Part-of-Speech Tags")
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
                
                # Sentiment Analysis
                sentiment = TextBlob(user_input).sentiment
                st.markdown(f"""
                #### 😊 Sentiment Analysis
                <div class='sense-card'>
                    <b>Polarity:</b> {sentiment.polarity:.2f} ({"Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral"})<br>
                    <b>Subjectivity:</b> {sentiment.subjectivity:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🏛️ Named Entities")
                if entities:
                    entity_html = "<div style='display: flex; flex-wrap: wrap; gap: 0.5rem;'>"
                    for entity, label in entities:
                        entity_html += f"""
                        <span style='background-color: #f0f0f0; padding: 0.25rem 0.5rem; 
                            border-radius: 4px; font-size: 0.85rem;'>
                            {entity} <small>({label})</small>
                        </span>
                        """
                    entity_html += "</div>"
                    st.markdown(entity_html, unsafe_allow_html=True)
                else:
                    st.info("No named entities detected")
                
                # Dependency Parse Visualization
                doc = nlp(corrected)
                html = displacy.render(doc, style="dep", options={'compact': True, 'distance': 100})
                st.markdown("#### 🎋 Dependency Parse")
                st.markdown(f"<div class='dep-tree'>{html}</div>", unsafe_allow_html=True)
        
        # Word Sense Disambiguation
        if senses:
            st.markdown("### 🧠 Contextual Understanding")
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
        st.markdown("### 🤖 Intelligent Response")
        st.markdown(f"<div class='bot-response'>{response}</div>", unsafe_allow_html=True)
        
        # Processing time
        st.caption(f"Processed in {processing_time:.2f} seconds with advanced NLP models")
        
        # Success effect
        st.balloons()