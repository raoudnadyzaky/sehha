import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import Vertex AI SDK
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- تهيئة التطبيق ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# --- قائمة العيادات المتاحة (يجب أن تطابق الواجهة الأمامية) ---
CLINICS_LIST = [
    "الباطنة-والجهاز-الهضمي-والكبد", "مسالك", "باطنة-عامة", "غدد-صماء-وسكر", 
    "القلب-والإيكو", "السونار-والدوبلكس", "جراحة-التجميل", "عظام", "جلدية-وليزر"
]
CLINICS_STRING = ", ".join([f'"{c}"' for c in CLINICS_LIST])

# --- تهيئة Vertex AI ---
# يتم تهيئة النموذج مرة واحدة عند بدء تشغيل التطبيق لضمان أفضل أداء
try:
    # الحصول على إعدادات المشروع والموقع من بيئة Cloud Run تلقائياً
    project_id = os.environ.get("GCP_PROJECT")
    location = "us-central1" # تحديد الموقع بشكل صريح لضمان الدقة
    
    if not project_id:
        raise ValueError("GCP_PROJECT environment variable not set. This is required for Vertex AI initialization.")

    vertexai.init(project=project_id, location=location)
    
    # استخدام أحدث نموذج متوفر ومستقر
    model = GenerativeModel("gemini-1.5-flash-001") 
    
    print(f"Vertex AI initialized successfully in project '{project_id}' at location '{location}'.")
    print(f"Model '{model._model_name}' loaded and ready.")

except Exception as e:
    # في حالة فشل تهيئة النموذج، يتم تسجيل خطأ فادح ومنع التطبيق من معالجة الطلبات
    print(f"CRITICAL ERROR: Failed to initialize Vertex AI. The /api/recommend endpoint will not work. Error: {e}")
    model = None

# --- نقاط الوصول (Endpoints) ---

@app.route('/')
def serve_index():
    """يقوم بتقديم ملف الواجهة الأمامية الرئيسي"""
    return send_from_directory('static', 'index.html')

@app.route("/api/recommend", methods=["POST"])
def recommend_clinic():
    """
    يستقبل شكوى المريض ويستخدم نموذج الذكاء الاصطناعي لاقتراح العيادة الأنسب.
    """
    # التحقق من أن النموذج تم تحميله بنجاح عند بدء التشغيل
    if model is None:
        return jsonify({"error": "AI model is not available due to a server initialization error."}), 503

    # التحقق من أن الطلب يحتوي على بيانات JSON
    if not request.is_json:
        return jsonify({"error": "Invalid request: Content-Type must be application/json."}), 400

    try:
        data = request.get_json()
        symptoms = data.get('symptoms')
        
        # التحقق من وجود نص الشكوى
        if not symptoms or not isinstance(symptoms, str) or len(symptoms.strip()) < 3:
            return jsonify({"error": "Symptoms must be provided as a non-empty string."}), 400
        
        # إعدادات النموذج لضمان الحصول على رد بصيغة JSON
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7, # درجة من الإبداعية للحصول على تفسيرات جيدة
            max_output_tokens=1024,
        )

        # بناء Prompt واضح ومحدد للنموذج
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
            # التحقق من أن الرد يحتوي على المفتاح المطلوب
            if "recommendations" not in json_response or not isinstance(json_response["recommendations"], list):
                 raise ValueError("JSON response is missing 'recommendations' list.")
            return jsonify(json_response)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: Failed to parse JSON response from model. Error: {e}. Response text: {response.text}")
            return jsonify({"error": "The AI model returned a malformed response."}), 500
        
    except Exception as e:
        # معالجة أي أخطاء غير متوقعة أثناء التنفيذ
        print(f"ERROR in /api/recommend: {str(e)}")
        # لا تعرض تفاصيل الخطأ للمستخدم النهائي لدواعي الأمان
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

# --- تشغيل التطبيق ---
if __name__ == "__main__":
    # هذا الجزء مخصص للتشغيل المحلي، Cloud Run يستخدم Gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
