import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

if gemini_key and gemini_key != "YOUR ACTUAL_GEMINI_API_KEY":
    
    print("Gemini API key is set correctly.")
    print(f"Key starts with: {gemini_key[:4]}...")
else:
    print("Gemini API key is not set or is still the placeholder value.")
    print("Please set the GEMINI_API_KEY in your .env file.")