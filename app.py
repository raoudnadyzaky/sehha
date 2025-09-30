import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import Vertex AI SDK
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- App Initialization ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# --- Available Clinics (Should match the frontend) ---
CLINICS_LIST = [
    "الباطنة-والجهاز-الهضمي-والكبد", "مسالك", "باطنة-عامة", "غدد-صماء-وسكر",
    "القلب-والإيكو", "السونار-والدوبلكس", "جراحة-التجميل", "عظام", "جلدية-وليزر"
]
CLINICS_STRING = ", ".join([f'"{c}"' for c in CLINICS_LIST])

# --- Vertex AI Initialization ---
# The model is initialized once when the application starts for best performance
model = None
try:
    # Automatically get project and location from Cloud Run environment
    project_id = os.environ.get("GCP_PROJECT")
    location = "us-central1" # Explicitly set for accuracy

    if not project_id:
        raise ValueError("GCP_PROJECT environment variable not set. This is required for Vertex AI initialization.")

    vertexai.init(project=project_id, location=location)

    # Use a stable and available model
    model = GenerativeModel("gemini-1.5-flash-001")

    print(f"Vertex AI initialized successfully in project '{project_id}' at location '{location}'.")
    print(f"Model '{model._model_name}' loaded and ready.")

except Exception as e:
    # If model initialization fails, log a critical error
    # The /api/recommend endpoint will be disabled
    print(f"CRITICAL ERROR: Failed to initialize Vertex AI. The /api/recommend endpoint will not work. Error: {e}")
    model = None # Ensure model is None if initialization fails

# --- Endpoints ---

@app.route('/')
def serve_index():
    """Serves the main frontend file."""
    return send_from_directory('static', 'index.html')

@app.route("/api/recommend", methods=["POST"])
def recommend_clinic():
    """
    Receives patient symptoms and uses the AI model to suggest the most suitable clinic.
    """
    # Check if the model was loaded successfully at startup
    if model is None:
        return jsonify({"error": "AI model is not available due to a server initialization error."}), 503

    # Validate that the request contains JSON data
    if not request.is_json:
        return jsonify({"error": "Invalid request: Content-Type must be application/json."}), 400

    try:
        data = request.get_json()
        symptoms = data.get('symptoms')

        # Validate that symptoms are provided and are a non-empty string
        if not symptoms or not isinstance(symptoms, str) or len(symptoms.strip()) < 3:
            return jsonify({"error": "Symptoms must be provided as a non-empty string."}), 400

        # Model configuration to ensure a JSON response
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7,
            max_output_tokens=1024,
        )

        # A clear and specific prompt for the model
        prompt = f"""
        You are an expert medical assistant in a hospital. Your task is to analyze a patient's complaint and recommend a maximum of two clinics from the available list.
        The list of available clinic IDs is: [{CLINICS_STRING}]

        Patient's complaint: "{symptoms}"

        Your task:
        1. Identify the most likely primary clinic based on the symptoms.
        2. Explain to the patient in simple, direct Arabic **why** you recommended this clinic.
        3. If there is another strong possibility, identify a secondary clinic and explain its reasoning.
        4. If the complaint is very vague (e.g., "I'm tired" or "exhausted"), recommend "باطنة-عامة" (General Medicine) and explain that a general check-up is the best starting point.
        5. Your response **MUST** be in JSON format only, with no text or markdown before or after it. It must follow this exact structure:
        {{
          "recommendations": [
            {{ "id": "recommended_clinic_id", "reason": "Clear explanation of the choice." }}
          ]
        }}
        """

        # Send the request to Vertex AI
        response = model.generate_content(prompt, generation_config=generation_config)

        # Check if the response contains any text
        if not response.text:
            print("ERROR: Gemini API returned an empty response.")
            return jsonify({"error": "The AI model returned an empty response."}), 500

        # Try to parse the response as JSON
        try:
            json_response = json.loads(response.text)
            # Validate that the response contains the required key
            if "recommendations" not in json_response or not isinstance(json_response["recommendations"], list):
                raise ValueError("JSON response is missing 'recommendations' list.")
            return jsonify(json_response)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: Failed to parse JSON response from model. Error: {e}. Response text: {response.text}")
            return jsonify({"error": "The AI model returned a malformed response."}), 500

    except Exception as e:
        # Handle any other unexpected errors
        print(f"ERROR in /api/recommend: {str(e)}")
        # Do not expose internal error details to the user for security
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

# --- Run the App ---
if __name__ == "__main__":
    # This part is for local development; Cloud Run uses Gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
