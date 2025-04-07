import streamlit as st
from PIL import Image
import os
import sys
import time

# Add module paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import UI components
from ui.home_page import render_home
from ui.crop_page import render_crop_page
from ui.disease_page import render_disease_page
from ui.weather_page import render_weather_page
from ui.market_page import render_market_page
from utils.session_state import get_session_state

def set_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="AgriSync - AI Farm Assistant",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load and apply custom CSS
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def sidebar_navigation():
    """Create the sidebar navigation menu"""
    with st.sidebar:
        st.image("assets/images/agrisync-logo.png", width=200)
        st.markdown("## AgriSync")
        st.markdown("##### Voice-First AI Farm Assistant")
        
        st.markdown("---")
        
        # Language selector
        languages = ["English", "हिंदी (Hindi)", "தமிழ் (Tamil)", "తెలుగు (Telugu)", 
                    "ಕನ್ನಡ (Kannada)", "മലയാളം (Malayalam)", "ਪੰਜਾਬੀ (Punjabi)"]
        selected_language = st.selectbox("Select Language", languages)
        
        st.markdown("---")
        
        # Navigation links
        pages = {
            "Home": "home",
            "Crop Recommendations": "crops",
            "Disease Detection": "disease",
            "Weather Forecast": "weather",
            "Market Prices": "market"
        }
        
        selected_page = None
        for page_name, page_id in pages.items():
            if st.button(page_name, key=f"nav-{page_id}"):
                selected_page = page_id
        
        if selected_page:
            st.session_state.current_page = selected_page
        
        st.markdown("---")
        
        # Voice assistant toggle
        voice_enabled = st.toggle("Enable Voice Assistant", value=True)
        if voice_enabled:
            st.session_state.voice_enabled = True
            st.info("Voice assistant is active! Click the microphone icon to speak.")
        else:
            st.session_state.voice_enabled = False
            
        st.markdown("---")
        
        st.markdown("### About")
        st.markdown("""
        AgriSync is an AI-powered platform designed to empower small and marginal 
        farmers in India through accessible agricultural intelligence.
        """)
        
        st.markdown("[GitHub Repository](https://github.com/yourusername/agrisync)")

def main():
    """Main application entry point"""
    # Set up the page configuration
    set_page_config()
    
    # Initialize session state if needed
    session_state = get_session_state()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True
        
    # Create the sidebar navigation
    sidebar_navigation()
    
    # Render the appropriate page content based on the navigation
    current_page = st.session_state.current_page
    
    if current_page == "home":
        render_home()
    elif current_page == "crops":
        render_crop_page()
    elif current_page == "disease":
        render_disease_page()
    elif current_page == "weather":
        render_weather_page()
    elif current_page == "market":
        render_market_page()

if __name__ == "__main__":
    main()