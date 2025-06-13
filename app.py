import streamlit as st
import pickle
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from scipy.sparse import hstack
import xgboost as xgb
import os

# Configure NLTK data path first
nltk_data_dir = './nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add the path before downloading
nltk.data.path.append(nltk_data_dir)

# Download NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)  
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)  # Add this for newer NLTK versions
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

# Load models with error handling
@st.cache_resource
def load_models():
    models = {}
    try:
        # Load XGBoost model with compatibility fix
        with open('xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
            # Fix for XGBoost version compatibility
            if not hasattr(xgb_model, 'feature_types'):
                xgb_model.feature_types = None
            models['xgb_model'] = xgb_model
        
        with open('tfidf_vectorizer_ml.pkl', 'rb') as f:
            models['tfidf'] = pickle.load(f)
        
        with open('tfidf1_vectorizer_ml.pkl', 'rb') as f:
            models['tfidf1'] = pickle.load(f)
        
        with open('label_encoder_ml.pkl', 'rb') as f:
            models['label_encoder'] = pickle.load(f)
        
        with open('naive_bayes_model.pkl', 'rb') as f:
            models['nb_model'] = pickle.load(f)
        
        with open('logistic_regression_model.pkl', 'rb') as f:
            models['logreg'] = pickle.load(f)
        
        with open('svm_model.pkl', 'rb') as f:
            models['svm_linear'] = pickle.load(f)
            
        return models
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return len(sentences)

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[\U00010000-\U0010FFFF]", "", text)
    allowed_chars = set(string.ascii_letters + "áéíóúãõàâêôç ")
    text = ''.join(c for c in text if c in allowed_chars)
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess(text):
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
        tokens = [stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return text  # Return original text if preprocessing fails

# Streamlit UI
st.title("Sentiment Analysis with Machine Learning Models")
st.markdown("Input a Tweet, and predict the sentiment using the selected model.")

# Load models
models = load_models()

if models is not None:
    # Example texts for testing - 4 examples for each mood
    example_texts = {
        "Positive": [
            "I'm having the most amazing day! The weather is beautiful and I just got great news about my job promotion. Life is wonderful! 😊",
            "Just finished an incredible workout and feeling absolutely fantastic! My energy levels are through the roof today! 💪✨",
            "Today marks my 25th birthday and I'm having the most magical celebration surrounded by everyone I love! My family surprised me with a party that brought tears of happiness to my eyes. All my closest friends traveled from different cities just to be here with me. The decorations are absolutely beautiful, the cake is delicious, and the love in this room is overwhelming in the best possible way. I've received so many heartfelt messages, thoughtful gifts, and warm hugs that my heart feels like it might burst from joy. Looking back on this past year, I've grown so much as a person and achieved things I never thought possible. I'm entering this new year of life with so much excitement and gratitude. Best birthday ever and I'll remember this day forever! 🎂🎈❤️🥳",
            "Finally got accepted to my dream university! All the hard work has paid off. I'm so excited for this new chapter! 🎓🌟"
        ],
        "Negative": [
            "The darkness is consuming me again and I can't find a way out. Every morning feels like climbing an impossible mountain just to get out of bed. I'm so tired of fighting this battle in my head every single day.",
            "I don't see the point anymore. Everything feels meaningless and I'm just a burden to everyone around me. Maybe they'd be better off without me here. The pain never stops and I'm exhausted from pretending I'm okay.",
            "My borderline personality disorder is destroying every relationship I have. One minute I love someone desperately, the next I hate them with equal intensity. I can't control these emotions and it's tearing me apart. People always leave in the end anyway.",
            "Living with severe depression feels like drowning in an ocean of darkness with no hope of reaching the surface. I've been struggling for months now, and despite therapy and medication, some days the weight of existence feels unbearable. Simple tasks that others take for granted - showering, eating, even breathing - require tremendous effort that I barely have left in me. I used to be someone who could handle challenges, but now I feel like I'm losing a battle against my own mind every single day. The worst part is the isolation and emptiness that follows me everywhere, making me question if there's any purpose to continuing this fight. My family tries to understand, but I can see the worry and helplessness in their eyes when they look at me, which only adds to my guilt about being such a burden. I'm tired of putting on a mask and pretending everything is fine when inside I'm completely falling apart. 😞💔🌧️"
        ],
        "Neutral": [
            "Just finished my morning coffee and reading the news. Time to start working on my project.",
            "Attended our monthly department meeting this afternoon where we discussed the third quarter performance metrics and outlined our strategy for the next fiscal period. The presentation covered various aspects of our current projects, budget allocations, and timeline adjustments that need to be implemented. We reviewed the client feedback from recent deliverables and identified areas where we can improve our processes. The team leads provided updates on their respective areas of responsibility, and we scheduled follow-up meetings for more detailed discussions on specific initiatives. Overall, it was a productive session that covered all the necessary business items on our agenda. I took notes on the action items assigned to my department and will need to coordinate with other team members to ensure we meet our deadlines. Standard business operations continuing as expected. 📊💼📋",
            "Went grocery shopping and picked up some vegetables. Planning to cook dinner later tonight.",
            "Completed my daily commute to the office this morning, taking the usual route through downtown traffic. Left the house at 7:30 AM and arrived at the office parking garage by 8:15 AM, which is pretty standard timing for a regular workday. Traffic was moving at a reasonable pace with no major delays or accidents reported on the radio. I listened to a podcast about productivity tips during the drive, which was moderately interesting and helped pass the time. The weather was clear and visibility was good, making for safe driving conditions. Once I arrived at the office, I grabbed my laptop bag and headed up to the third floor where my desk is located. Started the workday by checking emails and reviewing my schedule for the day. Pretty routine morning overall, nothing out of the ordinary to report. 🚗🏢⏰"
        ],
        "Mixed": [
            "Started therapy for my anxiety disorder today and I have complicated feelings about it. Part of me is hopeful that this might finally help me manage these overwhelming panic attacks that have been controlling my life. But I'm also terrified of opening up about my darkest thoughts.",
            "My psychiatrist adjusted my antidepressant medication this week. I'm grateful to have access to treatment and I know it takes time to work, but the side effects are making me feel worse in some ways. I'm caught between hope for improvement and frustration with the process.",
            "Finally told my family about my eating disorder and self-harm struggles. Their support means everything to me and I feel relieved to not carry this secret alone anymore. But I'm also scared about disappointing them and worried they'll watch my every move now. Recovery feels possible and impossible at the same time.",
            "I've been managing my bipolar disorder for two years now, and today perfectly captures the complexity of living with this condition. This morning I woke up feeling incredibly energetic and optimistic - I cleaned my entire apartment, started three new creative projects, and felt like I could conquer the world. The hypomanic episodes can feel amazing in the moment, like I'm finally the person I'm supposed to be, productive and full of ideas and ambition. But by this afternoon, I could feel the familiar crash beginning as my mood started to shift downward. Now I'm sitting here feeling exhausted and empty, looking at all the unfinished projects I started with such enthusiasm just hours ago. It's frustrating because I know this pattern so well, yet I still get caught up in the highs and devastated by the lows. My medication helps stabilize me most of the time, and my therapist has given me tools to recognize these mood swings, but some days I still feel like I'm on an emotional roller coaster that I can't get off. I'm grateful for the treatment and support I have, but I also mourn the person I might have been without this constant battle in my brain. 🎢💊😔✨"
        ]
    }
    
    # Create columns for example buttons
    st.markdown("### Quick Examples:")
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize session state for input text
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    # Import random for selecting examples
    import random
    
    with col1:
        if st.button("😊 Positive"):
            st.session_state.input_text = random.choice(example_texts["Positive"])
            st.rerun()
    
    with col2:
        if st.button("😞 Negative"):
            st.session_state.input_text = random.choice(example_texts["Negative"])
            st.rerun()
    
    with col3:
        if st.button("😐 Neutral"):
            st.session_state.input_text = random.choice(example_texts["Neutral"])
            st.rerun()
    
    with col4:
        if st.button("🤔 Mixed"):
            st.session_state.input_text = random.choice(example_texts["Mixed"])
            st.rerun()
    
    # Text input area with session state
    input_text = st.text_area(
        "Enter Tweet for Prediction:",
        value=st.session_state.input_text,
        height=100,
        help="Click one of the example buttons above to fill with sample text, or type your own."
    )
    
    # Clear button
    if st.button("🗑️ Clear Text"):
        st.session_state.input_text = ""
        st.rerun()

    model_choice = st.selectbox(
        "Select Model for Prediction:",
        ["XGBoost", "Naive Bayes", "Logistic Regression", "Linear SVM"]
    )

    if st.button("Check Tweet Sentiment"):
        if input_text:
            try:
                cleaned_text = clean_text(input_text)
                preprocessed_text = preprocess(cleaned_text)
                
                tfidf_vectorized = models['tfidf'].transform([preprocessed_text])
                tfidf_vectorized1 = models['tfidf1'].transform([preprocessed_text])

                num_features = [[len(input_text), count_sentences(input_text)]]
                num_features = hstack([tfidf_vectorized, num_features]) 

                if model_choice == "XGBoost":
                    # Convert sparse matrix to dense for XGBoost compatibility
                    pred = models['xgb_model'].predict(num_features.toarray())
                    model_name = "XGBoost"
                elif model_choice == "Naive Bayes":
                    pred = models['nb_model'].predict(num_features)
                    model_name = "Naive Bayes"
                elif model_choice == "Logistic Regression":
                    pred = models['logreg'].predict(num_features)
                    model_name = "Logistic Regression"
                else:  
                    pred = models['svm_linear'].predict(tfidf_vectorized1.toarray())
                    model_name = "Linear SVM"

                st.subheader(f"{model_name} Prediction: {models['label_encoder'].inverse_transform(pred)}")

                st.subheader("Preprocessed Text:")
                st.write(preprocessed_text)
                
                # TF-IDF Analysis
                tfidf_values = tfidf_vectorized.toarray()[0]
                tfidf_features = models['tfidf'].get_feature_names_out()

                word_tfidf = {word: tfidf_values[i] for i, word in enumerate(tfidf_features) if tfidf_values[i] > 0}
                sorted_word_tfidf = dict(sorted(word_tfidf.items(), key=lambda item: item[1], reverse=True))
                
                top_words = list(sorted_word_tfidf.keys())[:10]
                top_values = list(sorted_word_tfidf.values())[:10]

                if top_words:  # Only create visualization if there are words to display
                    st.subheader("Top 10 TF-IDF Words")
                    
                    # Create DataFrame for Streamlit chart
                    chart_data = pd.DataFrame({
                        'Words': top_words,
                        'TF-IDF Score': top_values
                    })
                    
                    # Display as horizontal bar chart using Streamlit
                    st.bar_chart(
                        chart_data.set_index('Words'),
                        height=400,
                        use_container_width=True
                    )
                    
                    # Alternative: Display as a table with color coding
                    st.subheader("TF-IDF Scores Table")
                    
                    # Create a styled dataframe
                    styled_df = chart_data.copy()
                    styled_df['TF-IDF Score'] = styled_df['TF-IDF Score'].round(4)
                    styled_df = styled_df.reset_index(drop=True)
                    styled_df.index = styled_df.index + 1  # Start index from 1
                    
                    # Display with metrics for top 3 words
                    col1, col2, col3 = st.columns(3)
                    if len(top_words) >= 3:
                        with col1:
                            st.metric(
                                label=f"🥇 Top Word: {top_words[0]}", 
                                value=f"{top_values[0]:.4f}"
                            )
                        with col2:
                            st.metric(
                                label=f"🥈 Second: {top_words[1]}", 
                                value=f"{top_values[1]:.4f}"
                            )
                        with col3:
                            st.metric(
                                label=f"🥉 Third: {top_words[2]}", 
                                value=f"{top_values[2]:.4f}"
                            )
                    
                    # Display full table
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=False
                    )
                    
                else:
                    st.warning("No significant words found for TF-IDF visualization.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Please enter some text for prediction.")
else:
    st.error("Failed to load models. Please check if all model files are present in the current directory.")