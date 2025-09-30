import os
import json
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import Vertex AI SDK
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- تهيئة التطبيق ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# --- متغيرات عالمية لتتبع حالة النموذج ---
# يتم استخدام قفل لضمان الوصول الآمن للنموذج بين الـ threads
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
    تقوم بتهيئة نموذج Vertex AI في thread منفصل في الخلفية
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

        # تحديث المتغير العام للنموذج بأمان باستخدام القفل
        with model_lock:
            model = loaded_model

        print(f"SUCCESS: Vertex AI model '{model._model_name}' loaded and ready.")

    except Exception as e:
        model_initialization_error = e
        print(f"CRITICAL ERROR: Failed to initialize Vertex AI model. Error: {e}")

# --- بدء عملية تهيئة النموذج في thread منفصل ---
initialization_thread = threading.Thread(target=initialize_vertex_ai)
initialization_thread.start()


# --- نقاط الوصول (Endpoints) ---

@app.route('/')
def serve_index():
    """تقوم بتقديم ملف الواجهة الأمامية الرئيسي."""
    return send_from_directory('static', 'index.html')

@app.route('/ready')
def readiness_check():
    """
    نقطة فحص الجاهزية (Readiness Probe).
    يستخدمها Cloud Run لمعرفة متى يكون التطبيق جاهزاً لاستقبال الطلبات.
    """
    with model_lock:
        if model is not None:
            # إذا تم تحميل النموذج بنجاح، يتم إرسال إشارة نجاح
            return "OK", 200
        elif model_initialization_error is not None:
            # إذا فشل التحميل، يتم إرسال خطأ دائم
            return f"Initialization failed: {model_initialization_error}", 500
        else:
            # إذا كان التحميل لا يزال جارياً، نطلب من Cloud Run الانتظار والمحاولة مرة أخرى
            return "Model is not ready yet.", 503

@app.route("/api/recommend", methods=["POST"])
def recommend_clinic():
    """تستقبل شكوى المريض وتقترح العيادة الأنسب."""
    with model_lock:
        # التحقق من جاهزية النموذج قبل معالجة الطلب
        if model is None:
            error_message = f"AI model is not available. Error: {model_initialization_error}" if model_initialization_error else "AI model is still initializing."
            return jsonify({"error": error_message}), 503 # 503 Service Unavailable

    # التحقق من أن الطلب يحتوي على بيانات JSON
    if not request.is_json:
        return jsonify({"error": "Invalid request: Content-Type must be application/json."}), 400

    try:
        data = request.get_json()
        symptoms = data.get('symptoms')

        # التحقق من وجود نص الشكوى وأنه ليس فارغاً
        if not symptoms or not isinstance(symptoms, str) or len(symptoms.strip()) < 3:
            return jsonify({"error": "Symptoms must be provided as a non-empty string."}), 400

        # إعدادات النموذج لضمان الحصول على رد بصيغة JSON
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7,
            max_output_tokens=1024,
        )

        # النص الكامل والمفصل للنموذج (تم استعادته)
        prompt = f"""
        أنت مساعد طبي خبير في مستشفى. مهمتك هي تحليل شكوى المريض واقتراح أفضل عيادتين بحد أقصى من قائمة العيادات المتاحة.
        قائمة معرفات (IDs) العيادات المتاحة هي: [{CLINICS_STRING}]

        شكوى المريض: "{symptoms}"

        المطلوب منك:
        1.  حدد العيادة الأساسية الأكثر احتمالاً بناءً على الأعراض.
        2.  اشرح للمريض بلغة عربية بسيطة ومباشرة **لماذا** قمت بترشيح هذه العيادة.
        3.  إذا كان هناك احتمال آخر قوي، حدد عيادة ثانوية واشرح سببها.
        4.  إذا كانت الشكوى غامضة جداً (مثل "أنا متعب" أو "مرهق")، قم بترشيح "باطنة-عامة" واشرح أن الفحص العام هو أفضل نقطة بداية.
        5.  ردك **يجب** أن يكون بصيغة JSON فقط، بدون أي نصوص أو علامات قبله أو بعده. يجب أن يكون على هذا الشكل بالضبط:
        {{
          "recommendations": [
            {{ "id": "ID_العيادة_المقترحة", "reason": "شرح سبب الاختيار بوضوح." }}
          ]
        }}
        """

        # إرسال الطلب إلى Vertex AI
        response = model.generate_content(prompt, generation_config=generation_config)

        # التحقق من وجود محتوى في الرد
        if not response.text:
            print("ERROR: Gemini API returned an empty response.")
            return jsonify({"error": "The AI model returned an empty response."}), 500

        # محاولة تحليل الرد كـ JSON
        try:
            json_response = json.loads(response.text)
            if "recommendations" not in json_response or not isinstance(json_response["recommendations"], list):
                raise ValueError("JSON response is missing 'recommendations' list.")
            return jsonify(json_response)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: Failed to parse JSON response from model. Error: {e}. Response text: {response.text}")
            return jsonify({"error": "The AI model returned a malformed response."}), 500

    except Exception as e:
        print(f"ERROR in /api/recommend: {str(e)}")
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
