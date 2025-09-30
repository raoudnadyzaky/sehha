import os
import json
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- تهيئة التطبيق ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# --- متغيرات عالمية لتتبع حالة النموذج ---
# سنستخدم قفل لضمان أن عملية التحقق من النموذج آمنة
model_lock = threading.Lock()
model = None
model_initialization_error = None

# --- قائمة العيادات المتاحة ---
CLINICS_LIST = [
    "الباطنة-والجهاز-الهضمي-والكبد", "مسالك", "باطنة-عامة", "غدد-صماء-وسكر", 
    "القلب-والإيكو", "السونار-والدوبلكس", "جراحة-التجميل", "عظام", "جلدية-وليزر"
]
CLINICS_STRING = ", ".join([f'"{c}"' for c in CLINICS_LIST])

def initialize_vertex_ai():
    """
    دالة مخصصة لتهيئة النموذج. سيتم تشغيلها في الخلفية
    لتجنب إبطاء بدء تشغيل التطبيق.
    """
    global model, model_initialization_error
    try:
        project_id = os.environ.get("GCP_PROJECT")
        location = "us-central1"
        
        if not project_id:
            raise ValueError("GCP_PROJECT environment variable not set.")

        print("Starting Vertex AI initialization...")
        vertexai.init(project=project_id, location=location)
        
        loaded_model = GenerativeModel("gemini-1.5-flash-001")
        
        # استخدام القفل لتحديث المتغير العام بأمان
        with model_lock:
            model = loaded_model
        
        print(f"SUCCESS: Vertex AI model '{model._model_name}' loaded and ready.")

    except Exception as e:
        model_initialization_error = e
        print(f"CRITICAL ERROR: Failed to initialize Vertex AI model. Error: {e}")

# --- بدء عملية تهيئة النموذج في thread منفصل ---
# هذا يسمح للتطبيق بالبدء بسرعة بينما يتم تحميل النموذج في الخلفية
initialization_thread = threading.Thread(target=initialize_vertex_ai)
initialization_thread.start()


# --- نقاط الوصول (Endpoints) ---

@app.route('/')
def serve_index():
    """يقدم ملف الواجهة الأمامية الرئيسي"""
    return send_from_directory('static', 'index.html')

@app.route('/ready')
def readiness_check():
    """
    نقطة فحص الجاهزية (Startup Probe).
    يستخدمها Cloud Run ليعرف متى يكون التطبيق جاهزاً لاستقبال الطلبات.
    """
    with model_lock:
        if model is not None:
            # إذا تم تحميل النموذج بنجاح، نرسل إشارة نجاح
            return "OK", 200
        elif model_initialization_error is not None:
            # إذا فشل التحميل، نرسل خطأ دائم
            return f"Initialization failed: {model_initialization_error}", 500
        else:
            # إذا كان التحميل لا يزال جارياً، نطلب من Cloud Run الانتظار والمحاولة مرة أخرى
            return "Model is not ready yet.", 503


@app.route("/api/recommend", methods=["POST"])
def recommend_clinic():
    """يستقبل شكوى المريض ويقترح العيادة الأنسب."""
    with model_lock:
        # التحقق من جاهزية النموذج قبل معالجة الطلب
        if model is None:
            error_message = f"AI model is not available. Error: {model_initialization_error}" if model_initialization_error else "AI model is still initializing."
            return jsonify({"error": error_message}), 503 # 503 Service Unavailable

    # ... باقي الكود يبقى كما هو ...
    if not request.is_json:
        return jsonify({"error": "Invalid request: Content-Type must be application/json."}), 400
    try:
        data = request.get_json()
        symptoms = data.get('symptoms')
        if not symptoms or not isinstance(symptoms, str) or len(symptoms.strip()) < 3:
            return jsonify({"error": "Symptoms must be provided as a non-empty string."}), 400
        
        generation_config = GenerationConfig(response_mime_type="application/json", temperature=0.7)
        prompt = f"""
        أنت مساعد طبي خبير... (باقي الـ prompt كما هو)
        قائمة معرفات (IDs) العيادات المتاحة هي: [{CLINICS_STRING}]
        شكوى المريض: "{symptoms}"
        ...
        """
        response = model.generate_content(prompt, generation_config=generation_config)
        return jsonify(json.loads(response.text))
    except Exception as e:
        print(f"ERROR in /api/recommend: {str(e)}")
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
