# =============================================================================
#  Smart Student Guide - Web Edition (تطبيق الويب المتكامل)
#  خادم Flask + محرك استرجاع المعلومات (TF-IDF) + محرك الاستنتاج الأكاديمي
# =============================================================================

import os
import sys
import csv
import json
import logging
import threading
import webbrowser
from collections import defaultdict
# pyrefly: ignore[missing-import]
from flask import Flask, render_template, request, jsonify

# استيراد خوارزميات استرجاع المعلومات
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # دعم العربية في مخرجات JSON

# =============================================================================
#  مسارات الملفات
# =============================================================================
COURSES_FILE   = "courses.csv"
PREREQ_FILE    = "prerequisites.json"
DOCS_DIR       = "docs"
LOG_FILE       = "queries.log"

ATTENDANCE_DOC = os.path.join(DOCS_DIR, "attendance_policy.txt")
GPA_DOC        = os.path.join(DOCS_DIR, "gpa_system.txt")
REG_DOC        = os.path.join(DOCS_DIR, "registration_rules.txt")

# =============================================================================
#  إعداد نظام السجل (UTF-8 Logging)
# =============================================================================
logger = logging.getLogger("SmartGuideWeb")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_fh)

def log_query(action: str, details: str = ""):
    """تسجيل العمليات في queries.log."""
    logger.info(f"[{action}] {details}")


# =============================================================================
#  تهيئة البيانات التلقائية
# =============================================================================
def initialize_data():
    """إنشاء الملفات ببيانات واقعية إذا لم تكن موجودة."""
    # 1. courses.csv
    if not os.path.exists(COURSES_FILE):
        courses = [
            ["course_id", "course_name", "instructor", "room", "credit_hours"],
            ["CS101",   "مقدمة في البرمجة",                       "د. أحمد محمد",    "معمل الحاسب 1",          3],
            ["CS102",   "هياكل البيانات والخوارزميات",             "د. محمود حسن",    "مدرج 2",                 3],
            ["MATH101", "تفاضل وتكامل",                           "د. إبراهيم علي",  "مدرج 1",                 3],
            ["MATH102", "جبر خطي وإحصاء",                         "د. سارة خالد",    "قاعة 104",               3],
            ["AI201",   "أساسيات الذكاء الاصطناعي",               "د. علي عبدالله",  "قاعة 201",               3],
            ["AI202",   "تعلم الآلة (Machine Learning)",           "د. كريم وليد",    "معمل الذكاء الاصطناعي", 3],
            ["IS201",   "قواعد البيانات",                          "د. منى رضا",      "معمل 3",                 3],
            ["ROB201",  "مقدمة في الروبوتات والأنظمة المدمجة",    "د. طارق أحمد",    "معمل الروبوت",           3],
        ]
        with open(COURSES_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(courses)

    # 2. prerequisites.json
    if not os.path.exists(PREREQ_FILE):
        prereqs = {
            "CS101":   [],
            "MATH101": [],
            "CS102":   ["CS101"],
            "MATH102": ["MATH101"],
            "AI201":   ["CS101", "MATH101"],
            "AI202":   ["AI201", "MATH102"],
            "IS201":   ["CS101"],
            "ROB201":  ["CS101", "MATH101"],
        }
        with open(PREREQ_FILE, "w", encoding="utf-8") as f:
            json.dump(prereqs, f, ensure_ascii=False, indent=4)

    # 3. docs/
    os.makedirs(DOCS_DIR, exist_ok=True)

    if not os.path.exists(ATTENDANCE_DOC):
        content = """لائحة الحضور والغياب - كلية الذكاء الاصطناعي والحاسبات

المادة الأولى: الحضور الإلزامي
يُعدّ حضور المحاضرات والأعمال العملية والميدانية إلزامياً لجميع طلاب الكلية.
يجب على الطالب حضور ما لا يقل عن 75% من إجمالي المحاضرات المقررة لكل مادة في الفصل الدراسي.

المادة الثانية: الحد الأقصى للغياب
الحد الأقصى المسموح به للغياب هو 25% من إجمالي ساعات المادة.
يُحتسب الغياب بعذر وبدون عذر ضمن نسبة الغياب الإجمالية حتى يتم البت فيه من لجنة شؤون الطلاب.
في حالة تجاوز نسبة الغياب 10% وحتى 20%: يُرسل للطالب إنذار أكاديمي أول عبر البوابة الإلكترونية.

المادة الثالثة: الحرمان من دخول الامتحان النهائي (الرسوب بالغياب)
إذا تجاوز الطالب نسبة الغياب المقررة (25%) في أي مادة دون عذر قهري معتمد، يُحرَم رسمياً من دخول الامتحان النهائي لتلك المادة.
تُرصد للطالب درجة صفر في خانة الامتحان النهائي مع تسجيل تقدير (DN - حرمان).
يحق للطالب تقديم التماس تظلم رسمي إلى وكيل الكلية لشؤون التعليم والطلاب خلال أسبوع واحد من إعلان كشوف الحرمان.

المادة الرابعة: حالات الغياب بعذر مقبول
تُقبل الأعذار في الحالات الاستثنائية التالية فقط:
1. التنويم في المستشفى أو إجراء عمليات جراحية عاجلة مع تقديم تقرير طبي معتمد من الإدارة الطبية بالجامعة.
2. حالات الوفاة للأقارب من الدرجة الأولى (الأب، الأم، الأخ، الأخت) مع إرفاق شهادة الوفاة.
3. تمثيل الكلية أو الجامعة في البطولات والمؤتمرات والمسابقات الرسمية بخطاب تكليف مسبق من رعاية الشباب.
4. الظروف القاهرة غير المتوقعة التي يقرر مجلس الكلية قبولها.
"""
        with open(ATTENDANCE_DOC, "w", encoding="utf-8") as f:
            f.write(content)

    if not os.path.exists(GPA_DOC):
        content = """نظام التقديرات وحساب المعدل التراكمي (CGPA) - نظام الساعات المعتمدة

أولاً: جدول توزيع التقديرات والنقاط
الدرجة المئوية    التقدير اللاتيني    النقاط (Grade Points)    التقدير المكافئ
------------------------------------------------------------------------
90% فأكثر         A+                 4.0                      ممتاز مرتفع
85% إلى أقل من 90% A                  3.7                      ممتاز
80% إلى أقل من 85% B+                 3.3                      جيد جداً مرتفع
75% إلى أقل من 80% B                  3.0                      جيد جداً
70% إلى أقل من 75% C+                 2.7                      جيد مرتفع
65% إلى أقل من 70% C                  2.3                      جيد
60% إلى أقل من 65% D+                 2.0                      مقبول مرتفع
55% إلى أقل من 60% D                  1.7                      مقبول (الحد الأدنى للنجاح)
أقل من 55%        F                  0.0                      راسب

ثانياً: شروط النجاح الأكاديمي
الحد الأدنى للنجاح في أي مقرر دراسي هو الحصول على تقدير (D) بنسبة 55% من إجمالي الدرجة النهائية.
المقرر الحاصل فيه الطالب على تقدير (F) لا تُحتسب ساعاته المعتمدة ضمن الساعات المنجزة، ويجب على الطالب إعادة دراسته والتسجيل فيه مرة أخرى.

ثالثاً: معادلة حساب المعدل التراكمي (GPA)
المعدل = مجموع (نقاط كل مقرر × عدد ساعاته المعتمدة) ÷ مجموع الساعات المعتمدة الإجمالي
المعدل التراكمي العام المطلوب للتخرج هو 2.0 على الأقل من 4.0.

رابعاً: التحذير الأكاديمي ولوائح الفصل
1. يوضع الطالب تحت الملاحظة الأكاديمية (Academic Probation) إذا انخفض معدله التراكمي عن 2.0 في أي فصل دراسي.
2. الطالب المنذر أكاديمياً يُمنع من تسجيل أكثر من 12 ساعة معتمدة في الفصل التالي للتركيز على رفع معدله.
3. إذا استمر الطالب تحت الملاحظة لثلاثة فصول دراسية متتالية يُعرض موقفه على مجلس الكلية للنظر في فصله أكاديمياً أو منحه فرصة أخيرة.

خامساً: مرتبة الشرف والتفوق
تُمنح مرتبة الشرف للطالب الذي يتخرج بمعدل تراكمي 3.5 فأكثر بشرط ألا يكون قد رسب في أي مقرر دراسي طوال سنوات دراسته بالكلية.
"""
        with open(GPA_DOC, "w", encoding="utf-8") as f:
            f.write(content)

    if not os.path.exists(REG_DOC):
        content = """قواعد التسجيل والإرشاد الأكاديمي - كلية الحاسبات والذكاء الاصطناعي

أولاً: العبء الدراسي (الحد الأدنى والأقصى للساعات المعتمدة)
1. الحد الأدنى للتسجيل في الفصل الدراسي الاعتيادي: 12 ساعة معتمدة.
2. الحد الأقصى للعبء الدراسي للطالب العادي (معدل 2.0 إلى 3.49): 18 ساعة معتمدة.
3. استثناء التفوق (معدل 3.5 فأكثر): يُسمح للطالب بتسجيل حتى 21 ساعة معتمدة بعد موافقة المرشد الأكاديمي والعميد.
4. الطلاب المنذرون أكاديمياً (معدل أقل من 2.0): الحد الأقصى المسموح به هو 12 ساعة معتمدة فقط.
5. الفصل الصيفي (Summer Semester): الحد الأقصى للتسجيل هو 6 ساعات معتمدة (أو 7 ساعات إذا كانت تشمل مقرر تدريب أو تخرج).

ثانياً: المتطلبات السابقة للمقررات (Prerequisites)
لا يجوز للطالب بأي حال من الأحوال التسجيل في مقرر دراسي ما لم يكن قد اجتاز المتطلب السابق (Prerequisite) بنجاح (تقدير D على الأقل).
يُلغى تسجيل أي مقرر يتم بشكل خاطئ دون استيفاء متطلبه السابق فور مراجعة الجداول وقبل بدء الأسبوع الثالث.

ثالثاً: مهام المرشد الأكاديمي
1. يُخصص لكل طالب مرشد أكاديمي من أعضاء هيئة التدريس لمساعدته في بناء خطته الدراسية واختيار المقررات المناسبة.
2. لا يُعتمد الجدول الدراسي للطالب إلا بعد الحصول على الموافقة الإلكترونية من المرشد الأكاديمي.
3. يتولى المرشد متابعة الطلاب المتعثرين وتقديم الدعم الإرشادي المناسب لتحسين معدلاتهم.

رابعاً: فترات الحذف والإضافة والانسحاب (Add & Drop)
1. فترة الحذف والإضافة: خلال الأسبوعين الأول والثاني من بدء الفصل الدراسي، ولا يُثبت المقرر في السجل عند حذفه خلال هذه الفترة.
2. الانسحاب بعذر (Withdrawal - W): يجوز للطالب الانسحاب من مقرر حتى نهاية الأسبوع الثامن، وتُسجل له علامة (W) ولا تدخل في حساب المعدل.
3. الانسحاب الإجباري/الرسوب (WF): الانسحاب بعد الأسبوع الثامن يُعد رسوباً بالانسحاب (WF) ويُحسب كتقدير F بنقاط صفرية في المعدل.
"""
        with open(REG_DOC, "w", encoding="utf-8") as f:
            f.write(content)


# =============================================================================
#  إدارة الملفات (CSV / JSON)
# =============================================================================
def load_courses() -> list[dict]:
    courses = []
    if not os.path.exists(COURSES_FILE):
        return courses
    with open(COURSES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            courses.append(dict(row))
    return courses


def save_courses(courses: list[dict]):
    if not courses:
        return
    fieldnames = ["course_id", "course_name", "instructor", "room", "credit_hours"]
    with open(COURSES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(courses)


def load_prerequisites() -> dict:
    if not os.path.exists(PREREQ_FILE):
        return {}
    with open(PREREQ_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_prerequisites(prereqs: dict):
    with open(PREREQ_FILE, "w", encoding="utf-8") as f:
        json.dump(prereqs, f, ensure_ascii=False, indent=4)


# =============================================================================
#  محرك استرجاع المعلومات (IR Engine - TF-IDF & Cosine Similarity)
# =============================================================================
class WebIREngine:
    def __init__(self):
        self.doc_names = []
        self.doc_texts = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.inverted_index = {}
        self.build()

    def build(self):
        self.doc_names.clear()
        self.doc_texts.clear()
        self.inverted_index.clear()

        doc_texts_map = {}
        if os.path.exists(DOCS_DIR):
            for fname in os.listdir(DOCS_DIR):
                if fname.endswith(".txt"):
                    fpath = os.path.join(DOCS_DIR, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                    if text.strip():
                        self.doc_names.append(fname)
                        self.doc_texts.append(text)
                        doc_texts_map[fname] = text

        if not self.doc_texts:
            return

        # بناء الفهرس المعكوس
        idx = defaultdict(set)
        for doc_name, text in doc_texts_map.items():
            words = text.replace("\n", " ").replace(".", " ").replace("،", " ").split()
            for word in words:
                clean = word.strip().lower()
                if len(clean) > 2:
                    idx[clean].add(doc_name)
        self.inverted_index = {w: sorted(docs) for w, docs in idx.items()}

        # بناء نموذج TF-IDF
        self.vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def search(self, query: str, top_n: int = 3):
        if not self.vectorizer or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_indices = similarities.argsort()[::-1]

        results = []
        for idx in ranked_indices[:top_n]:
            score = float(similarities[idx])
            if score > 0.01:
                # استخراج أفضل فقرة مطابقة
                paragraphs = [p.strip() for p in self.doc_texts[idx].split("\n\n") if p.strip()]
                best_snippet = paragraphs[0] if paragraphs else self.doc_texts[idx][:300]
                results.append({
                    "file": self.doc_names[idx],
                    "score": round(score * 100, 1),
                    "snippet": best_snippet[:400]
                })
        return results


# تهيئة البيانات ومحرك البحث
initialize_data()
ir_engine = WebIREngine()


# =============================================================================
#  محرك الاستنتاج الأكاديمي (Inference Engine)
# =============================================================================
def get_available_courses(passed_courses: list[str]):
    prereqs = load_prerequisites()
    all_courses = load_courses()
    passed_set = set(c.upper().strip() for c in passed_courses if c.strip())

    available = []
    locked = []

    for course in all_courses:
        cid = course["course_id"].upper().strip()
        if cid in passed_set:
            continue

        reqs = prereqs.get(cid, [])
        missing_reqs = [r for r in reqs if r.upper() not in passed_set]

        if not missing_reqs:
            available.append(course)
        else:
            item = dict(course)
            item["missing_prereqs"] = missing_reqs
            locked.append(item)

    return available, locked


# =============================================================================
#  المسارات والواجهات (Routes & API Endpoints)
# =============================================================================

@app.route("/")
def index():
    """الصفحة الرئيسية للموقع."""
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """إحصائيات النظام العامة."""
    courses = load_courses()
    return jsonify({
        "courses_count": len(courses),
        "docs_count": len(ir_engine.doc_names),
        "terms_count": len(ir_engine.inverted_index),
        "status": "Online"
    })


@app.route("/api/courses", methods=["GET"])
def api_courses():
    """الحصول على جميع المقررات وشجرة المتطلبات."""
    courses = load_courses()
    prereqs = load_prerequisites()
    for c in courses:
        cid = c.get("course_id", "")
        c["prereqs"] = prereqs.get(cid, [])
    return jsonify({"courses": courses})


@app.route("/api/search", methods=["POST"])
def api_search():
    """البحث الذكي باستخدام TF-IDF و Cosine Similarity."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"results": []})

    log_query("WEB_SEARCH", f"استعلام المستخدم: {query}")
    results = ir_engine.search(query, top_n=3)
    return jsonify({"results": results, "query": query})


@app.route("/api/advisor", methods=["POST"])
def api_advisor():
    """محرك الاستنتاج الأكاديمي."""
    data = request.get_json() or {}
    passed = data.get("passed", [])
    available, locked = get_available_courses(passed)
    
    log_query("WEB_ADVISOR", f"مجتاز: {passed} | متاح: {[c['course_id'] for c in available]}")
    return jsonify({
        "available": available,
        "locked": locked,
        "passed_count": len(passed)
    })


@app.route("/api/doc/<filename>", methods=["GET"])
def api_get_doc(filename):
    """قراءة نص لائحة كاملة."""
    # تأمين المسار ضد Directory Traversal
    safe_name = os.path.basename(filename)
    fpath = os.path.join(DOCS_DIR, safe_name)
    if not os.path.exists(fpath):
        return jsonify({"error": "المستند غير موجود"}), 404

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    log_query("WEB_READ_DOC", f"قراءة المستند: {safe_name}")
    return jsonify({"filename": safe_name, "content": content})


@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    """التحقق من كلمة مرور المسؤول."""
    data = request.get_json() or {}
    pwd = data.get("password", "")
    if pwd == "admin123":
        log_query("ADMIN_LOGIN", "تسجيل دخول ناجح للوحة الإدارة")
        return jsonify({"success": True})
    log_query("ADMIN_LOGIN_FAIL", "محاولة دخول خاطئة")
    return jsonify({"success": False, "message": "كلمة المرور غير صحيحة"}), 401


@app.route("/api/admin/course/add", methods=["POST"])
def api_admin_add_course():
    """إضافة مقرر جديد."""
    data = request.get_json() or {}
    cid = data.get("course_id", "").strip().upper()
    name = data.get("course_name", "").strip()
    inst = data.get("instructor", "").strip()
    room = data.get("room", "").strip()
    hours = data.get("credit_hours", "3").strip()
    prereqs_list = data.get("prerequisites", [])

    if not cid or not name:
        return jsonify({"error": "رمز واسم المادة مطلوبان"}), 400

    courses = load_courses()
    if any(c["course_id"].upper() == cid for c in courses):
        return jsonify({"error": "المقرر موجود بالفعل"}), 400

    courses.append({
        "course_id": cid,
        "course_name": name,
        "instructor": inst,
        "room": room,
        "credit_hours": hours
    })
    save_courses(courses)

    prereqs = load_prerequisites()
    prereqs[cid] = [p.strip().upper() for p in prereqs_list if p.strip()]
    save_prerequisites(prereqs)

    log_query("ADMIN_ADD_COURSE", f"إضافة المادة: {cid} - {name}")
    return jsonify({"success": True, "message": f"تمت إضافة المقرر {cid} بنجاح"})


@app.route("/api/admin/course/delete", methods=["POST"])
def api_admin_delete_course():
    """حذف مقرر."""
    data = request.get_json() or {}
    cid = data.get("course_id", "").strip().upper()

    courses = load_courses()
    new_courses = [c for c in courses if c["course_id"].upper() != cid]
    save_courses(new_courses)

    prereqs = load_prerequisites()
    if cid in prereqs:
        del prereqs[cid]
    for k, v in prereqs.items():
        if cid in v:
            v.remove(cid)
    save_prerequisites(prereqs)

    log_query("ADMIN_DELETE_COURSE", f"حذف المادة: {cid}")
    return jsonify({"success": True, "message": f"تم حذف المقرر {cid}"})


@app.route("/api/admin/doc/add", methods=["POST"])
def api_admin_add_doc():
    """إضافة مستند جديد وإعادة بناء الفهرس."""
    data = request.get_json() or {}
    fname = data.get("filename", "").strip()
    content = data.get("content", "").strip()

    if not fname or not content:
        return jsonify({"error": "اسم الملف والمحتوى مطلوبان"}), 400

    if not fname.endswith(".txt"):
        fname += ".txt"

    safe_name = os.path.basename(fname)
    fpath = os.path.join(DOCS_DIR, safe_name)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

    ir_engine.build()
    log_query("ADMIN_ADD_DOC", f"إضافة المستند: {safe_name}")
    return jsonify({"success": True, "message": f"تم حفظ المستند {safe_name} وإعادة تدريب الفهرس"})


@app.route("/api/admin/logs", methods=["GET"])
def api_admin_logs():
    """عرض سجل العمليات."""
    if not os.path.exists(LOG_FILE):
        return jsonify({"logs": []})
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return jsonify({"logs": lines[-40:]})  # آخر 40 عملية


@app.route("/api/admin/reset", methods=["POST"])
def api_admin_reset():
    """استعادة البيانات الافتراضية."""
    for file in [COURSES_FILE, PREREQ_FILE, ATTENDANCE_DOC, GPA_DOC, REG_DOC]:
        if os.path.exists(file):
            os.remove(file)
    initialize_data()
    ir_engine.build()
    log_query("ADMIN_RESET_DATA", "استعادة البيانات الافتراضية")
    return jsonify({"success": True, "message": "تمت استعادة البيانات الافتراضية بنجاح"})


# =============================================================================
#  تشغيل الخادم مع فتح المتصفح التلقائي
# =============================================================================
def open_browser():
    """فتح المتصفح تلقائياً عند تشغيل الخادم."""
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("=" * 65)
    print("  🎓 Smart Student Guide — Web Edition")
    print("  🌐 جاري تشغيل خادم الويب على: http://127.0.0.1:5000")
    print("  🖥️ سيتم فتح المتصفح تلقائياً الآن...")
    print("=" * 65)

    # تشغيل فتح المتصفح في خيط مستقل بعد ثانية واحدة
    threading.Timer(1.2, open_browser).start()

    # تشغيل تطبيق Flask
    app.run(host="127.0.0.1", port=5000, debug=False)
