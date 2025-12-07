import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_available_models():
    """Check what Gemini models are available."""
    try:
        import google.generativeai as genai

        # Configure with your API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(" No GOOGLE_API_KEY found in environment")
            return

        genai.configure(api_key=api_key)

        print("Checking available Gemini models...")
        print("=" * 50)

        # List all models
        models = genai.list_models()

        # Filter for models that support generateContent
        available_models = []
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                available_models.append(model)

        if available_models:
            print(f"Found {len(available_models)} models that support text generation:")
            print()
            for model in available_models:
                print(f"   Model: {model.name}")
                print(f"   Display Name: {model.display_name}")
                print(f"   Description: {model.description}")
                print(f"   Methods: {', '.join(model.supported_generation_methods)}")
                print()
        else:
            print("No models found that support text generation")

        # Also show all models for reference
        print("All available models:")
        for model in models:
            print(f"   - {model.name} ({model.display_name})")

    except ImportError:
        print("google-generativeai package not installed")
        print("   Run: pip install google-generativeai")
    except Exception as e:
        print(f"Error checking models: {e}")


if __name__ == "__main__":
    check_available_models()
