import streamlit as st
import pandas as pd
from PIL import Image
import requests
import json
import datetime
import random
from typing import Dict, List, Tuple
import pytz
from groq import Groq
from astrapy import DataAPIClient
import hashlib

# Constants
GROQ_API_KEY = "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"
client_groq = Groq(api_key=GROQ_API_KEY)

# Astra DB initialization
ASTRA_DB_TOKEN = "AstraCS:oSJoCzqxlAegFLyAcmYikgbL:050926825d078529b97eb3d67fbacc38afc895ce6a1bacf1e869f5a4672addd7"
ASTRA_DB_API_ENDPOINT = "https://5cbd5a6b-f228-4672-b5bf-38cddab69aa6-us-east-2.apps.astra.datastax.com"

client_astra = DataAPIClient(ASTRA_DB_TOKEN)
db_astra = client_astra.get_database_by_api_endpoint(ASTRA_DB_API_ENDPOINT)
print(f"Connected to Astra DB: {db_astra.list_collection_names()}")

# Ensure the 'soul' collection exists
try:
    db_astra.get_collection('soul')
except Exception:
    db_astra.create_collection('soul')
    print("Created 'soul' collection.")


# Helper functions for chunking and storing data
def chunk_content(content: str, chunk_size: int = 7500) -> List[str]:
    """Split content into chunks that fit within Astra DB's size limits"""
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]


def retrieve_response(collection, response_type: str, **query_params) -> str:
    """Retrieve and reconstruct chunked response with improved error handling"""
    try:
        # First, try to find any matching document
        first_chunk = collection.find_one({
            'type': response_type,
            **query_params
        })

        if not first_chunk:
            return None

        # If there's only one chunk, return it directly
        if 'chunk_id' not in first_chunk:
            return first_chunk['content']

        # Otherwise, retrieve all chunks and reconstruct
        chunks = list(collection.find({
            'type': response_type,
            'chunk_id': first_chunk['chunk_id']
        }).sort([('chunk_index', 1)]))  # Materialize the cursor into a list

        # Verify we have all chunks
        if not chunks:
            return None
            
        expected_chunks = first_chunk['total_chunks']
        if len(chunks) != expected_chunks:
            return None

        # Reconstruct content
        sorted_chunks = sorted(chunks, key=lambda x: x['chunk_index'])
        return ''.join(chunk['content'] for chunk in sorted_chunks)

    except Exception as e:
        print(f"Error retrieving response: {str(e)}")
        return None

def store_response(collection, response_type: str, content: str, **metadata) -> str:
    """Store response in chunks with improved error handling"""
    try:
        chunks = chunk_content(content)
        chunk_id = f"{response_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Store each chunk with retries
        for i, chunk in enumerate(chunks):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    document = {
                        'type': response_type,
                        'chunk_id': chunk_id,
                        'chunk_index': i,
                        'content': chunk,
                        'total_chunks': len(chunks),
                        'timestamp': datetime.datetime.now().isoformat(),
                        **metadata
                    }
                    collection.insert_one(document)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise e
                    time.sleep(1)  # Wait before retry

        return chunk_id

    except Exception as e:
        print(f"Error storing response: {str(e)}")
        return None


# Constants
ZODIAC_DATES = [
    ((3, 21), (4, 19), "Aries"),
    ((4, 20), (5, 20), "Taurus"),
    ((5, 21), (6, 20), "Gemini"),
    ((6, 21), (7, 22), "Cancer"),
    ((7, 23), (8, 22), "Leo"),
    ((8, 23), (9, 22), "Virgo"),
    ((9, 23), (10, 22), "Libra"),
    ((10, 23), (11, 21), "Scorpio"),
    ((11, 22), (12, 21), "Sagittarius"),
    ((12, 22), (1, 19), "Capricorn"),
    ((1, 20), (2, 18), "Aquarius"),
    ((2, 19), (3, 20), "Pisces")
]

ZODIAC_CHARACTERISTICS = {
    "Aries": {
        "element": "Fire",
        "ruling_planet": "Mars",
        "qualities": ["Leadership", "Courage", "Energy"],
        "compatible_signs": ["Leo", "Sagittarius", "Gemini"],
        "lucky_numbers": [1, 8, 17],
        "lucky_colors": ["Red", "Orange"],
        "lucky_days": ["Tuesday", "Saturday"]
    }
    # Add characteristics for other signs here
}

LANGUAGES = {
    "Hindi": "hi",
    "Marathi": "mr",
    "English": "en"
}


@st.cache_data(ttl=3600)
def get_groq_response(prompt: str, language_code: str) -> str:
    """Get response from Groq API with improved handling and language preference"""
    try:
        completion = client_groq.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert spiritual guide providing responses in {language_code}. Ensure your responses are thorough and well-structured in this language."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            stream=False
        )

        response = completion.choices[0].message.content.strip()

        if response.endswith(('...', '…')) or len(response) < 100:
            st.warning("The response may be incomplete. Please try regenerating.")

        return response
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}. Please try again later."
        st.error(error_msg)
        return error_msg


def user_details_sidebar() -> tuple:
    """Create and handle the sidebar for user details input"""
    st.sidebar.header("Personal Details")

    name = st.sidebar.text_input("Name")
    dob = st.sidebar.date_input(
        "Date of Birth",
        min_value=datetime.date(1900, 1, 1),
        max_value=datetime.date.today()
    )
    time_of_birth = st.sidebar.time_input("Time of Birth")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    state = st.sidebar.text_input("State")
    city = st.sidebar.text_input("City")
    language = st.sidebar.selectbox("Preferred Language", list(LANGUAGES.keys()))

    return name, dob, time_of_birth, gender, state, city, LANGUAGES[language]


def get_zodiac_sign(dob: datetime.date) -> str:
    """Determine zodiac sign based on date of birth"""
    day = dob.day
    month = dob.month

    for (start_month, start_day), (end_month, end_day), sign in ZODIAC_DATES:
        if ((month == start_month and day >= start_day) or
                (month == end_month and day <= end_day)):
            return sign
    if (month == 12 and day >= 22) or (month == 1 and day <= 19):
        return "Capricorn"
    return "Unknown"


def get_daily_horoscope(sign: str, language_code: str) -> str:
    """Generate or fetch detailed daily horoscope"""
    collection = db_astra.get_collection('soul')
    today = datetime.date.today().isoformat()

    existing = retrieve_response(
        collection,
        'daily_horoscope',
        sign=sign,
        language=language_code,
        date=today
    )

    if existing:
        return existing

    prompt = f"""As an expert astrologer, provide a comprehensive daily horoscope for {sign} in {language_code}.
    Include these detailed sections:

    General Outlook:
    - Overall energy and influences for the day
    - Key opportunities and challenges
    - Planetary influences affecting your sign

    Love & Relationships:
    - Romantic prospects and partner dynamics
    - Family and friendship insights
    - Social interactions and opportunities

    Career & Money:
    - Professional opportunities and challenges
    - Financial trends and advice
    - Workplace dynamics and project outcomes

    Health & Wellness:
    - Physical and mental well-being
    - Energy levels and stress management
    - Wellness recommendations"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'daily_horoscope',
        response,
        sign=sign,
        language=language_code,
        date=today
    )

    return response


def get_gemstone_recommendation(sign: str, language_code: str) -> str:
    """Get comprehensive gemstone recommendations with language preference"""
    collection = db_astra.get_collection('soul')

    existing = retrieve_response(
        collection,
        'gemstone_recommendation',
        sign=sign,
        language=language_code
    )

    if existing:
        return existing

    prompt = f"""As an expert gemologist and astrologer, provide detailed gemstone recommendations for {sign} in {language_code}.
    Include these comprehensive sections:

    Primary Gemstones:
    - Detailed description of main recommended stones
    - Specific metaphysical properties
    - Astrological significance
    - Quality indicators to look for

    Wearing Guidelines:
    - Best metals for setting
    - Specific fingers for different stones
    - Most auspicious times to start wearing
    - Required rituals or preparations"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'gemstone_recommendation',
        response,
        sign=sign,
        language=language_code
    )

    return response


def generate_kundali(dob: datetime.date, time_of_birth: datetime.time, language_code: str) -> str:
    """Generate comprehensive kundali analysis with language preference"""
    collection = db_astra.get_collection('soul')

    existing = retrieve_response(
        collection,
        'kundali_analysis',
        dob=dob.isoformat(),
        time_of_birth=time_of_birth.isoformat(),
        language=language_code
    )

    if existing:
        return existing

    prompt = f"""As a Vedic astrology expert, provide a detailed kundali analysis for:
    Date: {dob}
    Time: {time_of_birth}
    in {language_code}.

    Please include these comprehensive sections:

    Planetary Positions:
    - Detailed analysis of all major planets
    - House positions and their significance
    - Key conjunctions and aspects
    - Dasha periods and their effects"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'kundali_analysis',
        response,
        dob=dob.isoformat(),
        time_of_birth=time_of_birth.isoformat(),
        language=language_code
    )

    return response


def analyze_palm_image(image: Image, language_code: str) -> str:
    """Generate detailed palm reading analysis with language preference"""
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    collection = db_astra.get_collection('soul')

    existing = retrieve_response(
        collection,
        'palm_reading',
        image_hash=image_hash,
        language=language_code
    )

    if existing:
        return existing

    prompt = f"""As an experienced palmist, provide a comprehensive palm reading in {language_code}:

    Major Lines Analysis:
    - Life Line: Length, quality, branches, and islands
    - Heart Line: Shape, depth, and special markings
    - Head Line: Path, breaks, and connections
    - Fate Line: Presence, strength, and variations"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'palm_reading',
        response,
        image_hash=image_hash,
        language=language_code
    )

    return response


def get_meditation_guidance(sign: str, language_code: str) -> str:
    """Provide meditation guidance based on zodiac sign with language preference"""
    collection = db_astra.get_collection('soul')

    existing = retrieve_response(
        collection,
        'meditation_guidance',
        sign=sign,
        language=language_code
    )

    if existing:
        return existing

    prompt = f"""Provide meditation guidance tailored for {sign} in {language_code}. Include:
    - Type of meditation (e.g., mindfulness, visualization)
    - Focus points or mantras
    - Duration and frequency suggestions
    - Tips for achieving deeper states of meditation
    - Benefits specific to the zodiac sign's characteristics"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'meditation_guidance',
        response,
        sign=sign,
        language=language_code
    )

    return response


def get_workout_recommendations(sign: str, language_code: str) -> str:
    """Generate workout recommendations based on zodiac sign with language preference"""
    collection = db_astra.get_collection('soul')

    existing = retrieve_response(
        collection,
        'workout_recommendation',
        sign=sign,
        language=language_code
    )

    if existing:
        return existing

    prompt = f"""Provide workout recommendations for {sign} in {language_code}. Include:
    - Types of exercises (e.g., yoga, strength training, cardio)
    - Specific exercises or routines
    - How often to exercise
    - Benefits linked to the zodiac sign's traits
    - Considerations for physical and mental balance"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'workout_recommendation',
        response,
        sign=sign,
        language=language_code
    )

    return response


def predict_future_triggers(sign: str, language_code: str) -> str:
    """Predict future astrological triggers and how to prepare for them"""
    collection = db_astra.get_collection('soul')

    existing = retrieve_response(
        collection,
        'future_triggers',
        sign=sign,
        language=language_code
    )

    if existing:
        return existing

    prompt = f"""For {sign}, predict potential astrological triggers in the near future in {language_code}. Include:
    - Types of triggers (e.g., planetary transits, retrogrades)
    - Expected impacts on life areas (career, relationships, health)
    - Strategies or preventive measures to manage these triggers
    - Timing of these events if possible"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'future_triggers',
        response,
        sign=sign,
        language=language_code
    )

    return response


def get_pooja_recommendation(sign: str, language_code: str) -> str:
    """Provide Pooja recommendations based on zodiac sign"""
    collection = db_astra.get_collection('soul')

    existing = retrieve_response(
        collection,
        'pooja_recommendation',
        sign=sign,
        language=language_code
    )
    if existing:
        return existing

    prompt = f"""For {sign}, recommend specific Poojas or rituals in {language_code}. Include:
        - Types of Poojas or rituals recommended for spiritual growth
        - When to perform these Poojas (e.g., days of the week, specific times)
        - Mantras, offerings, or deities to focus on
        - The purpose or expected benefits of each Pooja
        - Any precautions or preparations needed"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'pooja_recommendation',
        response,
        sign=sign,
        language=language_code
    )

    return response


def spiritual_chatbot(query: str, sign: str, language_code: str) -> str:
    """Handle spiritual queries through a chatbot with zodiac context"""
    collection = db_astra.get_collection('soul')

    existing = retrieve_response(
        collection,
        'spiritual_chatbot',
        query=query,
        sign=sign,
        language=language_code
    )

    if existing:
        return existing

    if "marriage" in query.lower() or "ex" in query.lower():
        prompt = f"""As a spiritual guide and astrologer, provide an analysis for someone with the zodiac sign {sign} regarding '{query}' in {language_code}. Include:
            - Astrological insights based on current and upcoming planetary transits
            - Predict a specific date or time frame for the event if possible
            - Explain the astrological influences that might lead to this event
            - Offer practical spiritual steps or considerations for preparation or action"""
    else:
        prompt = f"""As a spiritual guide, respond to this query for someone with the zodiac sign {sign} in {language_code}: '{query}'. Use your knowledge of astrology to:
            - Provide insightful answers or advice related to the zodiac traits
            - Include any relevant astrological prediction or influence if applicable
            - Offer practical spiritual steps or considerations tailored to this sign"""

    response = get_groq_response(prompt, language_code)

    store_response(
        collection,
        'spiritual_chatbot',
        response,
        query=query,
        sign=sign,
        language=language_code
    )

    return response


def display_zodiac_info(sign: str):
    """Display comprehensive zodiac sign information"""
    if sign in ZODIAC_CHARACTERISTICS:
        info = ZODIAC_CHARACTERISTICS[sign]
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Element:** {info['element']}")
            st.write(f"**Ruling Planet:** {info['ruling_planet']}")
            st.write(f"**Lucky Numbers:** {', '.join(map(str, info['lucky_numbers']))}")
            st.write(f"**Lucky Colors:** {', '.join(info['lucky_colors'])}")

        with col2:
            st.write("**Key Qualities:**")
            for quality in info['qualities']:
                st.write(f"- {quality}")
            st.write("**Compatible Signs:**")
            for compatible in info['compatible_signs']:
                st.write(f"- {compatible}")


def main():
    st.set_page_config(
        page_title="Karmakode - AI Spiritual Guide",
        page_icon="🔮",
        layout="wide"
    )

    st.title("🔮 Karmakode - Your AI Spiritual Guide")

    # Get user details
    name, dob, time_of_birth, gender, state, city, language_code = user_details_sidebar()

    if name and dob and time_of_birth:
        sign = get_zodiac_sign(dob)
        st.success(f"Welcome {name}! Your zodiac sign is {sign} ♈")

        tabs = st.tabs([
            "🌟 Horoscope", "💎 Gemstones", "🎯 Kundali", "✋ Palm Reading",
            "🧘 Meditation", "🏋️ Workout", "🌠 Future Triggers", "🕉️ Pooja", "🗣️ Chatbot"
        ])

        with tabs[0]:
            st.header("Your Daily Horoscope")
            if st.button("Generate Horoscope", key="horoscope_btn"):
                with st.spinner("Generating your personalized horoscope..."):
                    horoscope = get_daily_horoscope(sign, language_code)
                    st.markdown(horoscope)
                    display_zodiac_info(sign)

        with tabs[1]:
            st.header("Gemstone Recommendations")
            if st.button("Get Recommendations", key="gemstone_btn"):
                with st.spinner("Analyzing your gemstone compatibility..."):
                    recommendations = get_gemstone_recommendation(sign, language_code)
                    st.markdown(recommendations)

        with tabs[2]:
            st.header("Kundali Analysis")
            if st.button("Generate Kundali", key="kundali_btn"):
                with st.spinner("Generating your kundali analysis..."):
                    analysis = generate_kundali(dob, time_of_birth, language_code)
                    st.markdown(analysis)

        
        with tabs[3]:
            st.header("Palm Reading")
            uploaded_file = st.file_uploader("Upload palm image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
               image = Image.open(uploaded_file)
               cols = st.columns(2)

               with cols[0]:
                # Updated deprecated parameter
                  st.image(image, caption="Your Palm Image", use_container_width=True)

               with cols[1]:
                   if st.button("Analyze Palm", key="palm_btn"):
                        with st.spinner("Analyzing your palm..."):
                            reading = analyze_palm_image(image, language_code)
                            st.markdown(reading)


        with tabs[4]:
            st.header("Meditation Guidance")
            if st.button("Get Meditation Guidance", key="meditation_btn"):
                with st.spinner("Fetching your meditation guidance..."):
                    guidance = get_meditation_guidance(sign, language_code)
                    st.markdown(guidance)

        with tabs[5]:
            st.header("Workout Recommendations")
            if st.button("Get Workout Recommendations", key="workout_btn"):
                with st.spinner("Generating workout recommendations..."):
                    workout = get_workout_recommendations(sign, language_code)
                    st.markdown(workout)

        with tabs[6]:
            st.header("Predictive Triggers")
            if st.button("View Future Triggers", key="triggers_btn"):
                with st.spinner("Looking into the astral forecast..."):
                    triggers = predict_future_triggers(sign, language_code)
                    st.markdown(triggers)

        with tabs[7]:
            st.header("Pooja Recommendations")
            if st.button("Get Pooja Recommendations", key="pooja_btn"):
                with st.spinner("Consulting the sacred texts..."):
                    pooja = get_pooja_recommendation(sign, language_code)
                    st.markdown(pooja)

        with tabs[8]:
            st.header("Spiritual Chatbot")
            query = st.text_input("Ask a spiritual question")
            if st.button("Submit Query", key="chatbot_btn"):
                with st.spinner("Consulting with the spirits..."):
                    response = spiritual_chatbot(query, sign, language_code)
                    st.markdown(response)

    else:
        st.info("👋 Please fill in your details in the sidebar to begin your spiritual journey.")

    st.markdown("---")
    st.markdown(
        f"""
            💫  Karmakode - Your AI Spiritual Guide

            Powered by advanced AI for personalized spiritual insights in {language_code}. Remember that while technology can provide guidance, 
            your intuition and personal wisdom are your best guides on your spiritual journey.
            """
    )


if __name__ == "__main__":
    main()
