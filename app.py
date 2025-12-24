import streamlit as st
import joblib
import re
import nltk
import random
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# CONFIG & THEME SETUP 
st.set_page_config(
    page_title="SafeSpace AI",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS for the "SafeSpace" Aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F8FF;
        color: #333333;
    }
    h1 {
        color: #005b96 !important;
        font-family: 'Helvetica', sans-serif;
    }
    p, .stMarkdown, div[data-testid="stChatMessageContent"] {
        color: #333333 !important;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# SIDEBAR RESOURCES 
# --- 2. SIDEBAR RESOURCES ---
with st.sidebar:
    st.header("🌿 SafeSpace Resources")
    
    # CSS to style the sidebar separately
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #E0F7FA; /* Light Teal */
    }
    [data-testid="stSidebar"] * {
        color: #005b96 !important; /* Dark Blue Text */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    **You are not alone.** If you are in crisis, please reach out to professional help:
    
    * **Emergency:** 911 (US) / 999 (UK)
    * **Suicide & Crisis Lifeline:** 988
    * **Crisis Text Line:** Text HOME to 741741
    
    *This AI is a supportive companion, not a licensed therapist.*
    """)
    st.image("https://placehold.co/200x200/E0F7FA/005b96?text=Wellness", width=150)

# LOGIC & LOADING 
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')

@st.cache_resource
def load_brain():
    model = joblib.load('final_best_model.pkl')
    vec = joblib.load('tfidf_vectorizer.pkl')
    return model, vec

try:
    model, vectorizer = load_brain()
except FileNotFoundError:
    st.error("❌ Brain files not found. Please run the Training Notebook first!")
    st.stop()

# SAFETY GUARDRAILS
# Triggers for immediate crisis support
crisis_keywords = {
    'suicide', 'kill', 'die', 'death', 'end it', 'hurt myself', 
    'beat', 'hit', 'punch', 'slap', 'abuse', 'rape', 'assault', 'threat', 'weapon',
    'domestic', 'violence', 'safe', 'danger'
}

def check_safety(text):
    """Returns True if the input contains crisis keywords"""
    text = text.lower()
    for word in crisis_keywords:
        # Check for whole words (prevents false positives like 'beatles')
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            return True
    return False

# RESPONSE BANK
response_bank = {
    0: [ # Sadness
        "I'm truly sorry you're feeling this way. I'm here to listen. 💙",
        "It sounds like things are heavy right now. Sending you strength.",
        "It's okay not to be okay. Take all the time you need.",
        "I hear you, and your feelings are valid. Do you want to talk more about it?"
    ],
    1: [ # Joy
        "That is wonderful news! I'm so happy for you! 🌟",
        "It sounds like a great moment! Hold onto this feeling! 😄",
        "Yay! I love hearing positive updates. Tell me more!",
        "That's fantastic! You deserve this happiness."
    ],
    2: [ # Love
        "That is so heartwarming. Love is a beautiful thing. 🥰",
        "It sounds like you have a lot of love in your heart. ❤️",
        "That's really sweet. Thanks for sharing that with me.",
        "Connection is so important. I'm glad you're feeling this."
    ],
    3: [ # Anger
        "I can hear the frustration in your words. It's okay to be mad. 😤",
        "That sounds incredibly difficult. You have a right to feel this way.",
        "Take a deep breath. I'm here to listen if you want to let it out.",
        "I'm listening. Sometimes venting is the best medicine."
    ],
    4: [ # Fear
        "It is completely normal to feel scared. You are safe here. 🛡️",
        "Take a slow, deep breath with me. One step at a time.",
        "You are braver than you feel right now. We can get through this.",
        "That does sound daunting, but you don't have to face it alone."
    ],
    5: [ # Surprise
        "Wow! That is quite a twist! 😲",
        "I wasn't expecting that either! Life is full of surprises.",
        "That's a moment to remember! How do you feel about it?",
        "Really? That is certainly unexpected!"
    ]
}

# CLEANING FUNCTION 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negation_words = {'no', 'not', 'nor', 'neither', 'never', "don't", "aren't", "couldn't", "didn't", "won't"}
stop_words = stop_words - negation_words 

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_input(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words if w not in stop_words]
    return ' '.join(cleaned)

# THE UI 
st.title("🌿 SafeSpace AI")
st.markdown("### A judgment-free zone to share your feelings.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Area
if prompt := st.chat_input("How are you feeling right now?"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. SAFETY CHECK FIRST (Override AI)
    if check_safety(prompt):
        # Force a Crisis Response
        full_response = "🚨 **It sounds like you are in a dangerous or difficult situation.** \n\nI am an AI, not a human, and I want to make sure you are safe. Please consider reaching out to a professional who can help immediately:\n\n* **Domestic Violence Hotline:** 800-799-SAFE (7233)\n* **Crisis Text Line:** Text HOME to 741741\n* **Emergency:** 911"
        emotion_label = "Crisis Alert 🚨"
    
    else:
        # 3. Normal AI Prediction
        clean_prompt = clean_input(prompt)
        vec = vectorizer.transform([clean_prompt])
        pred_id = model.predict(vec)[0]
        
        # Get Response
        conversational_reply = random.choice(response_bank.get(pred_id, ["I'm listening..."]))
        
        # Labels
        labels = {0: 'Sadness 😢', 1: 'Joy 😃', 2: 'Love 🥰', 3: 'Anger 😡', 4: 'Fear 😨', 5: 'Surprise 😲'}
        emotion_label = labels.get(pred_id, "Unknown")
        
        full_response = f"{conversational_reply} \n\n*(Detected: {emotion_label})*"
    
    # 4. Final Output
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.markdown(full_response)