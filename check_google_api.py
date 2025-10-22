import os
from dotenv import load_dotenv


def check_google_api_key():
    """
    Check if Google API key is properly configured for Gemini models.

    Validates that:
    - GOOGLE_API_KEY environment variable is set
    - The key is not a placeholder value

    Returns:
        bool: True if API key is valid, False otherwise
    """
    # Load environment variables from .env file
    load_dotenv()

    google_key = os.getenv("GOOGLE_API_KEY")

    if google_key and google_key != "your-google-api-key-here":
        print("Google API key is set correctly.")
        print(f"Key starts with: {google_key[:4]}...")
        return True
    else:
        print("Google API key is not set or is still the placeholder value.")
        print("Please set the GOOGLE_API_KEY in your .env file.")
        return False


if __name__ == "__main__":
    # Run as standalone script
    check_google_api_key()
