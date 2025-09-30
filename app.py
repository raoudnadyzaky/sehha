import os
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import base64

# تهيئة تطبيق فلاسك لخدمة الملفات الثابتة من مجلد "static"
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# قائمة العيادات المحدثة بناءً على الجدول الجديد
CLINICS_LIST = """
"الباطنة-والجهاز-الهضمي-والكبد", "مسالك", "باطنة-عامة", "غدد-صماء-وسكر", "القلب-والإيكو",
"السونار-والدوبلكس", "جراحة-التجميل", "عظام", "جلدية-وليزر"
"""

# المسار الخاص بصفحة الموقع الرئيسية
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

# المسار الخاص بتوصية العيادات بناءً على الأعراض
@app.route("/api/recommend", methods=["POST"])
def recommend_clinic():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms')
        if not symptoms:
            return jsonify({"error": "Missing symptoms"}), 400
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("CRITICAL ERROR: GEMINI_API_KEY is not set in environment variables.")
            return jsonify({"error": "Server configuration error."}), 500

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        أنت مساعد طبي خبير ومحترف في مستشفى كبير. مهمتك هي تحليل شكوى المريض بدقة واقتراح أفضل عيادتين بحد أقصى من قائمة العيادات المتاحة.
        قائمة معرفات (IDs) العيادات المتاحة هي: [{CLINICS_LIST}]
        شكوى المريض: "{symptoms}"
        ردك **يجب** أن يكون بصيغة JSON فقط، بدون أي نصوص أو علامات قبله أو بعده، ويحتوي على قائمة اسمها "recommendations" بداخلها عناصر تحتوي على "id" و "reason".
        """
        
        response = model.generate_content(prompt)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        json_response = json.loads(cleaned_text)
        return jsonify(json_response)
        
    except Exception as e:
        print(f"ERROR in /api/recommend: {str(e)}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
