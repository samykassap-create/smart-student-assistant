# =============================================================================
#  Smart Student Guide - دليل الطالب الذكي (Premium Edition)
#  مشروع جامعي متكامل: استرجاع المعلومات + تمثيل المعرفة + إدارة قواعد البيانات
#  واجهة تفاعلية احترافية (Premium CLI) مخصصة لكليات الحاسبات والذكاء الاصطناعي
# =============================================================================

import os
import sys
import csv
import json
import logging
import datetime
from collections import defaultdict

# استيراد مكتبات الذكاء الاصطناعي واسترجاع المعلومات
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# استيراد مكتبات التصميم المتقدم
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

# دعم تشبيك الحروف العربية للطرفيات (VS Code Terminal)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False

# =============================================================================
#  إعدادات النظام العامة
# =============================================================================
FIX_ARABIC_TERMINAL = True  # تفعيل معالجة الحروف العربية داخل تيرمنال VS Code
if os.name == "nt":
    os.system("")  # تفعيل دعم VT100 ورموز ANSI في طرفية ويندوز
console = Console()

COURSES_FILE   = "courses.csv"
PREREQ_FILE    = "prerequisites.json"
DOCS_DIR       = "docs"
LOG_FILE       = "queries.log"

ATTENDANCE_DOC = os.path.join(DOCS_DIR, "attendance_policy.txt")
GPA_DOC        = os.path.join(DOCS_DIR, "gpa_system.txt")
REG_DOC        = os.path.join(DOCS_DIR, "registration_rules.txt")

# =============================================================================
#  دالة معالجة النصوص العربية الذكية
# =============================================================================
def ar(text) -> str:
    """
    معالجة النصوص العربية تلقائياً لتظهر متصلة ومنسقة بالاتجاه السليم
    داخل تيرمنال VS Code وأي طرفية أخرى.
    """
    if text is None:
        return ""
    text_str = str(text)
    if not FIX_ARABIC_TERMINAL or not HAS_BIDI:
        return text_str
    try:
        if any('\u0600' <= char <= '\u06FF' for char in text_str):
            return get_display(arabic_reshaper.reshape(text_str))
        return text_str
    except Exception:
        return text_str


def clear_screen():
    """مسح الشاشة لمنح المستخدم تجربة تصفح نظيفة ومنظمة."""
    os.system('cls' if os.name == 'nt' else 'clear')


def pause(message: str = "اضغط Enter للمتابعة..."):
    """توقف مؤقت لقراءة المخرجات قبل العودة للقائمة."""
    try:
        input(f"\n  [Enter] {ar(message)}")
    except (EOFError, KeyboardInterrupt):
        pass


def get_arabic_input(prompt: str = "") -> str:
    """
    استقبال مدخلات المستخدم من لوحة المفاتيح مع تشبيك الحروف العربية
    لحظياً وبشكل حي على الشاشة داخل تيرمنال VS Code.
    """
    formatted_prompt = prompt.strip()
    if not (formatted_prompt.endswith(":") or formatted_prompt.endswith(">")):
        formatted_prompt += ":"

    # إذا كان الإدخال غير تفاعلي (Piped/Testing)، نستخدم دالة الإدخال العادية
    if not sys.stdin.isatty():
        try:
            return input(f"\n  {formatted_prompt} ").strip()
        except EOFError:
            return ""

    # على نظام ويندوز وباستخدام msvcrt للالتقاط الحي للضغطات
    if os.name == "nt":
        try:
            import msvcrt
            sys.stdout.write(f"\n  {formatted_prompt} ")
            sys.stdout.flush()
            buf = []
            while True:
                ch = msvcrt.getwch()
                if ch in ('\r', '\n'):
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    break
                elif ch == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
                elif ch in ('\x00', '\xe0'):  # مفاتيح الأسهم والوظائف الخاصة
                    msvcrt.getwch()
                    continue
                elif ch in ('\x08', '\x7f'):  # زر المسح Backspace
                    if buf:
                        buf.pop()
                else:
                    buf.append(ch)

                # تشبيك وإعادة رسم السطر لحظياً
                cur = "".join(buf)
                display_text = ar(cur) if cur else ""
                sys.stdout.write(f"\r\033[K  {formatted_prompt} {display_text}")
                sys.stdout.flush()

            return "".join(buf).strip()
        except Exception:
            pass

    return input(f"\n  {formatted_prompt} ").strip()


def gui_arabic_input(title: str = "البحث الذكي", prompt: str = "أدخل عبارة البحث باللغة العربية:") -> str:
    """
    فتح نافذة إدخال ويندوز صغيرة (Native Windows Dialog)
    تتيح الكتابة واللصق بالعربية بكل سلاسة ومحاذاة طبيعية من اليمين لليسار.
    """
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        res = simpledialog.askstring(title, prompt, parent=root)
        root.destroy()
        return res.strip() if res else ""
    except Exception:
        return ""



# =============================================================================
#  إعداد نظام تسجيل العمليات (UTF-8 Logging)
# =============================================================================
logger = logging.getLogger("SmartGuide")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_fh)

def log_query(action: str, details: str = ""):
    """تسجيل كل استفسار أو عملية بحث في ملف السجل queries.log بترميز UTF-8."""
    logger.info(f"[{action}] {details}")


# =============================================================================
#  دالة التهيئة التلقائية - Realistic Auto-Generated Data
# =============================================================================
def initialize_data():
    """
    تفحص وجود الملفات المطلوبة، وإذا لم تكن موجودة تقوم بإنشائها فوراً
    ببيانات دقيقة وواقعية تحاكي كليات الذكاء الاصطناعي والحاسبات المصرية.
    """
    created_any = False

    # 1. إنشاء ملف المقررات courses.csv
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
        created_any = True

    # 2. إنشاء شجرة المتطلبات السابقة prerequisites.json
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
        created_any = True

    # 3. إنشاء مجلد الوثائق واللوائح docs/
    os.makedirs(DOCS_DIR, exist_ok=True)

    # وثيقة لائحة الحضور والغياب
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
        created_any = True

    # وثيقة نظام التقديرات وحساب المعدل التراكمي
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
المعدل التراكمي التراكمي العام المطلوب للتخرج هو 2.0 على الأقل من 4.0.

رابعاً: التحذير الأكاديمي ولوائح الفصل
1. يوضع الطالب تحت الملاحظة الأكاديمية (Academic Probation) إذا انخفض معدله التراكمي عن 2.0 في أي فصل دراسي.
2. الطالب المنذر أكاديمياً يُمنع من تسجيل أكثر من 12 ساعة معتمدة في الفصل التالي للتركيز على رفع معدله.
3. إذا استمر الطالب تحت الملاحظة لثلاثة فصول دراسية متتالية يُعرض موقفه على مجلس الكلية للنظر في فصله أكاديمياً أو منحه فرصة أخيرة.

خامساً: مرتبة الشرف والتفوق
تُمنح مرتبة الشرف للطالب الذي يتخرج بمعدل تراكمي 3.5 فأكثر بشرط ألا يكون قد رسب في أي مقرر دراسي طوال سنوات دراسته بالكلية.
"""
        with open(GPA_DOC, "w", encoding="utf-8") as f:
            f.write(content)
        created_any = True

    # وثيقة قواعد التسجيل والإرشاد الأكاديمي
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
        created_any = True

    return created_any


# =============================================================================
#  وحدة التعامل مع الملفات (CSV / JSON / Docs)
# =============================================================================
def load_courses() -> list[dict]:
    """تحميل بيانات المقررات من ملف CSV."""
    courses = []
    if not os.path.exists(COURSES_FILE):
        return courses
    with open(COURSES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            courses.append(dict(row))
    return courses


def save_courses(courses: list[dict]):
    """حفظ قائمة المقررات إلى ملف CSV."""
    if not courses:
        return
    fieldnames = ["course_id", "course_name", "instructor", "room", "credit_hours"]
    with open(COURSES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(courses)


def load_prerequisites() -> dict:
    """تحميل شجرة المتطلبات السابقة من ملف JSON."""
    if not os.path.exists(PREREQ_FILE):
        return {}
    with open(PREREQ_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_prerequisites(prereqs: dict):
    """حفظ شجرة المتطلبات إلى ملف JSON."""
    with open(PREREQ_FILE, "w", encoding="utf-8") as f:
        json.dump(prereqs, f, ensure_ascii=False, indent=4)


def load_doc_file(filepath: str) -> str:
    """قراءة محتوى ملف نصي بترميز UTF-8."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# =============================================================================
#  وحدة استرجاع المعلومات (Information Retrieval - IR)
# =============================================================================
def build_inverted_index(doc_texts: dict[str, str]) -> dict[str, list[str]]:
    """بناء Inverted Index من الوثائق النصية."""
    index = defaultdict(set)
    for doc_name, text in doc_texts.items():
        words = text.replace("\n", " ").replace(".", " ").replace("،", " ").split()
        for word in words:
            clean = word.strip().lower()
            if len(clean) > 2:
                index[clean].add(doc_name)
    return {word: sorted(docs) for word, docs in index.items()}


class IREngine:
    """محرك استرجاع المعلومات المعتمد على TF-IDF و Cosine Similarity."""
    def __init__(self):
        self.doc_names: list[str] = []
        self.doc_texts: list[str] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.inverted_index: dict = {}
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
                    text = load_doc_file(fpath)
                    if text.strip():
                        self.doc_names.append(fname)
                        self.doc_texts.append(text)
                        doc_texts_map[fname] = text

        if not self.doc_texts:
            return

        self.inverted_index = build_inverted_index(doc_texts_map)
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def search(self, query: str, top_n: int = 3) -> list[dict]:
        """البحث في الوثائق وحساب تشابه جيب التمام."""
        if self.vectorizer is None or self.tfidf_matrix is None or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_indices = similarities.argsort()[::-1]

        results = []
        for idx in ranked_indices[:top_n]:
            score = float(similarities[idx])
            if score > 0.01:
                # العثور على أفضل فقرة مطابقة كمعاينة
                paragraphs = [p.strip() for p in self.doc_texts[idx].split("\n\n") if p.strip()]
                best_snippet = paragraphs[0] if paragraphs else self.doc_texts[idx][:250]
                results.append({
                    "file": self.doc_names[idx],
                    "score": round(score * 100, 1),
                    "snippet": best_snippet[:350],
                })
        return results


# =============================================================================
#  محرك الاستنتاج الأكاديمي (Knowledge & Inference Engine)
# =============================================================================
def get_available_courses(passed_courses: list[str]) -> tuple[list[dict], list[dict]]:
    """
    محرك استنتاج يحدد المقررات المتاحة والمقررات المقفلة للطالب.
    المدخلات: قائمة رموز المواد المجتازة (مثل ['CS101', 'MATH101'])
    المخرجات: (المقررات المتاحة للتسجيل, المقررات المقفلة مع المتطلبات الناقصة)
    """
    prereqs = load_prerequisites()
    all_courses = load_courses()
    passed_set = set(c.upper().strip() for c in passed_courses if c.strip())

    available = []
    locked = []

    for course in all_courses:
        cid = course["course_id"].upper().strip()

        # تخطي المادة إذا كان الطالب قد اجتازها بالفعل
        if cid in passed_set:
            continue

        reqs = prereqs.get(cid, [])
        missing_reqs = [r for r in reqs if r.upper() not in passed_set]

        if not missing_reqs:
            available.append(course)
        else:
            locked_info = dict(course)
            locked_info["missing_prereqs"] = missing_reqs
            locked.append(locked_info)

    return available, locked


# =============================================================================
#  مكونات الواجهة البصرية الفاخرة (UI Components)
# =============================================================================
def render_header(subtitle: str = ""):
    """عرض الشعار والترويسة الرئيسية بتصميم بطاقة عصرية."""
    clear_screen()
    
    title_text = f"[bold bright_yellow]🎓 SMART STUDENT GUIDE[/bold bright_yellow]  |  [bold bright_cyan]{ar('دليل الطالب الذكي')}[/bold bright_cyan]"
    desc_text = f"[dim white]{ar('نظام الإرشاد الأكاديمي الذكي واسترجاع المعلومات')}[/dim white]"
    
    if subtitle:
        desc_text += f"\n[bold bright_green]◀ {ar(subtitle)} ▶[/bold bright_green]"

    panel = Panel(
        f"{title_text}\n{desc_text}",
        box=box.ROUNDED,
        border_style="bright_blue",
        padding=(0, 2),
    )
    console.print(panel)


def render_dashboard_bar(ir_engine: IREngine):
    """عرض شريط الحالة والإحصائيات السريعة للنظام."""
    courses_count = len(load_courses())
    docs_count = len(ir_engine.doc_names)
    terms_count = len(ir_engine.inverted_index)

    status_str = (
        f"[bright_cyan]📚 {ar('المقررات')}: {courses_count}[/bright_cyan]   "
        f"[bright_magenta]📄 {ar('اللوائح')}: {docs_count}[/bright_magenta]   "
        f"[bright_yellow]🧠 {ar('الكلمات المفهرسة')}: {terms_count}[/bright_yellow]   "
        f"[bright_green]⚡ {ar('الحالة')}: {ar('نشط ومتصل')}[/bright_green]"
    )
    console.print(Panel(status_str, box=box.HORIZONTALS, border_style="dim blue", padding=(0, 1)))


def render_courses_table(courses: list[dict], title: str = "دليل المقررات الدراسية"):
    """عرض جدول مقررات دراسية بتنسيق ملون وعالي الجودة."""
    if not courses:
        console.print(f"[yellow]{ar('لا توجد مقررات لعرضها.')}[/yellow]")
        return

    prereqs = load_prerequisites()
    table = Table(
        title=f"[bold bright_cyan]{ar(title)}[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold bright_yellow",
        expand=True,
    )

    table.add_column(ar("الرمز"), style="bold cyan", justify="center", width=10)
    table.add_column(ar("اسم المادة"), style="bold white", width=30)
    table.add_column(ar("المحاضر"), style="green", width=18)
    table.add_column(ar("القاعة / المعمل"), style="yellow", width=22)
    table.add_column(ar("الساعات"), justify="center", style="magenta", width=8)
    table.add_column(ar("المتطلبات السابقة"), style="dim white", width=20)

    for c in courses:
        cid = c.get("course_id", "")
        req_list = prereqs.get(cid, [])
        req_str = ", ".join(req_list) if req_list else ar("بدون متطلب")
        
        table.add_row(
            cid,
            ar(c.get("course_name", "")),
            ar(c.get("instructor", "")),
            ar(c.get("room", "")),
            str(c.get("credit_hours", "")),
            ar(req_str),
        )

    console.print(table)


# =============================================================================
#  صفحات وميزات بوابة الطالب (Student Portal)
# =============================================================================
def student_course_catalog():
    """استعراض دليل المقررات الكامل للكلية."""
    render_header("دليل المقررات الدراسية")
    courses = load_courses()
    render_courses_table(courses, "جميع المقررات المعتمدة في الخطة الدراسية")
    log_query("STUDENT_VIEW_COURSES", f"استعراض {len(courses)} مقرراً")
    pause()


def student_advisor_check():
    """محرك الاستنتاج الأكاديمي: فحص المقررات المتاحة للتسجيل."""
    render_header("محرك الاستنتاج الذكي - فحص المواد المتاحة للتسجيل")
    
    courses = load_courses()
    all_cids = [c["course_id"] for c in courses]

    info_panel = Panel(
        f"[bold bright_white]{ar('تعليمات الإرشاد الأكاديمي:')}[/bold bright_white]\n"
        f"• {ar('أدخل رموز المواد التي اجتزتها بنجاح مفصولة بفاصلة (مثال: CS101, MATH101)')}\n"
        f"• {ar('إذا كنت طالباً مستجداً ولم تجتز أي مادة بعد، اضغط Enter مباشرة')}\n"
        f"• [dim cyan]{ar('المقررات المسجلة في النظام')}: {', '.join(all_cids)}[/dim cyan]",
        box=box.ROUNDED,
        border_style="cyan",
    )
    console.print(info_panel)

    raw_input = get_arabic_input("[?] المواد المجتازة")
    passed = [p.strip().upper() for p in raw_input.split(",") if p.strip()]

    available, locked = get_available_courses(passed)
    log_query("STUDENT_CHECK_COURSES", f"مجتاز: {passed} | متاح: {[c['course_id'] for c in available]}")

    # عرض ملخص حالة الطالب
    completed_hours = sum(int(c.get("credit_hours", 3)) for c in courses if c["course_id"].upper() in passed)
    summary_text = (
        f"[bold bright_green]✔ {ar('المواد المجتازة')}: {len(passed)} {ar('مادة')} ({completed_hours} {ar('ساعة معتمدة')})[/bold bright_green]\n"
        f"[bold bright_cyan]⚡ {ar('المقررات المتاحة للتسجيل فوراً')}: {len(available)} {ar('مقرر')}[/bold bright_cyan]\n"
        f"[bold yellow]🔒 {ar('المقررات المغلقة حالياً')}: {len(locked)} {ar('مقرر')}[/bold yellow]"
    )
    console.print(Panel(summary_text, box=box.ROUNDED, border_style="green", title=f"📊 {ar('ملخص السجل الأكاديمي')}"))

    # جدول المواد المتاحة
    if available:
        render_courses_table(available, "المقررات التي يحق لك التسجيل فيها الآن")
    else:
        console.print(f"[yellow]{ar('لا توجد مقررات جديدة متاحة للتسجيل حالياً.')}[/yellow]")

    # جدول المواد المغلقة مع سبب القفل
    if locked:
        locked_table = Table(
            title=f"[bold bright_yellow]{ar('المقررات المغلقة (تتطلب مواد سابقة لم تجتزها بعد)')}[/bold bright_yellow]",
            box=box.ROUNDED,
            border_style="yellow",
            header_style="bold red",
            expand=True,
        )
        locked_table.add_column(ar("الرمز"), style="bold white", justify="center", width=10)
        locked_table.add_column(ar("اسم المادة"), style="white", width=30)
        locked_table.add_column(ar("المتطلبات المتبقية الناقصة"), style="bold red", width=35)
        locked_table.add_column(ar("الساعات"), justify="center", style="magenta", width=8)

        for c in locked:
            missing_str = ", ".join(c.get("missing_prereqs", []))
            locked_table.add_row(
                c["course_id"],
                ar(c.get("course_name", "")),
                ar(f"ينقصك اجتياز: {missing_str}"),
                str(c.get("credit_hours", "3")),
            )
        console.print(locked_table)

    pause()


def student_ir_search(ir_engine: IREngine):
    """
    البحث الدلالي الذكي المباشر في لوائح وقوانين الكلية.
    يستقبل استفسار الطالب بحرية باللغة العربية، ويقوم بتشبيك الحروف لحظياً
    أثناء الكتابة، ثم يطبق خوارزمية TF-IDF ومقياس Cosine Similarity لترتيب النتائج.
    """
    while True:
        render_header("محرك البحث الذكي في لوائح وقوانين الكلية (IR Search Engine)")
        
        info_box = Panel(
            f"[bold bright_yellow]⚡ {ar('نظام استرجاع المعلومات (Information Retrieval):')}[/bold bright_yellow]\n"
            f"• {ar('اكتب استفسارك بحرية باللغة العربية (مثال: عقوبة الغياب، حساب المعدل، الساعات المسموحة، حرمان الامتحان)')}\n"
            f"• {ar('ستتشابك الحروف العربية وتنتظم تلقائياً أمامك على الشاشة أثناء الكتابة')}\n"
            f"• [dim white]{ar('اضغط Enter مباشرة دون كتابة للعودة لقائمة الطالب')}[/dim white]",
            box=box.ROUNDED,
            border_style="bright_blue",
        )
        console.print(info_box)

        # استقبال الاستعلام مباشرة بالكتابة الحرة مع تشبيك الحروف اللحظي
        query = get_arabic_input(f"[🔍] {ar('أدخل استفسارك')} > ")

        # إذا ضغط Enter بدون كتابة، يعود للقائمة
        if not query:
            break

        # تسجيل وتنفيذ خوارزمية البحث بـ TF-IDF و Cosine Similarity
        log_query("STUDENT_SEARCH", f"استعلام حر: {query}")
        results = ir_engine.search(query, top_n=3)

        if not results:
            console.print(f"\n[bold red]✖ {ar('لم يتم العثور على أي تطابق مع استفسارك.')}[/bold red]")
            console.print(f"[dim white]{ar('جرب صياغة أخرى مثل: غياب، معدل، تسجيل، ساعات، مرشد...')}[/dim white]")
        else:
            console.print(f"\n[bold bright_green]🎯 {ar('نتائج البحث المطابقة للاستعلام:')} \"{ar(query)}\"[/bold bright_green]\n")
            
            for i, res in enumerate(results, 1):
                score_bar = "█" * int(res["score"] / 10) + "░" * (10 - int(res["score"] / 10))
                score_badge = f"[bold green]{res['score']}% {score_bar}[/bold green]"
                
                panel_content = (
                    f"[bold bright_yellow]📄 {ar('الملف المصدر')}:[/bold bright_yellow] [bold white]{res['file']}[/bold white]   "
                    f"[bold bright_yellow]📊 {ar('درجة التشابه')}:[/bold bright_yellow] {score_badge}\n"
                    f"────────────────────────────────────────────────────────────\n"
                    f"[white]{ar(res['snippet'])}...[/white]"
                )
                console.print(Panel(panel_content, box=box.ROUNDED, border_style="cyan", title=f"[bold cyan]#{i}[/bold cyan]"))

        # خيار فوري للبحث مجدداً أو العودة
        again = input(f"\n  [?] {ar('اضغط Enter للبحث عن شيء آخر (أو اكتب 0 للعودة)')}: ").strip()
        if again == "0":
            break




def student_view_full_docs():
    """استعراض اللوائح الجامعية المعتمدة بالكامل داخل واجهة القراءة."""
    while True:
        render_header("استعراض اللوائح والأنظمة الجامعية الرسمية")
        
        menu_text = (
            f"[bold cyan][1][/bold cyan] 📜 {ar('لائحة الحضور والغياب والحرمان')}\n"
            f"[bold cyan][2][/bold cyan] 📊 {ar('نظام التقديرات وحساب المعدل التراكمي (GPA)')}\n"
            f"[bold cyan][3][/bold cyan] 📋 {ar('قواعد التسجيل والإرشاد الأكاديمي والعبء الدراسي')}\n"
            f"[bold cyan][0][/bold cyan] ↩️ {ar('العودة لقائمة الطالب')}"
        )
        console.print(Panel(menu_text, box=box.ROUNDED, border_style="bright_blue", title=f"📖 {ar('اختر اللائحة المطلوب قراءتها')}"))

        try:
            choice = input(f"\n  [?] {ar('اختيارك')}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        doc_map = {
            "1": ("لائحة الحضور والغياب", ATTENDANCE_DOC),
            "2": ("نظام التقديرات والمعدل التراكمي", GPA_DOC),
            "3": ("قواعد التسجيل والإرشاد الأكاديمي", REG_DOC),
        }

        if choice in doc_map:
            doc_title, doc_path = doc_map[choice]
            content = load_doc_file(doc_path)
            render_header(doc_title)
            console.print(Panel(ar(content), box=box.ROUNDED, border_style="bright_green", title=f"📜 {ar(doc_title)}"))
            log_query("STUDENT_READ_DOC", f"قراءة: {doc_title}")
            pause()
        elif choice == "0":
            break
        else:
            console.print(f"[red]{ar('اختيار غير صحيح.')}[/red]")
            pause()


def student_gpa_calculator():
    """حاسبة المعدل الفصلي والتراكمي حسب نظام الساعات المعتمدة في الجامعات المصرية."""
    render_header("حاسبة المعدل التراكمي (GPA Calculator)")
    
    calc_intro = Panel(
        f"[bold bright_yellow]{ar('مقياس النقاط المعتمد في كلية الذكاء الاصطناعي:')}[/bold bright_yellow]\n"
        f"A+: 4.0 | A: 3.7 | B+: 3.3 | B: 3.0 | C+: 2.7 | C: 2.3 | D+: 2.0 | D: 1.7 | F: 0.0",
        box=box.ROUNDED,
        border_style="yellow",
    )
    console.print(calc_intro)

    grade_points = {
        "A+": 4.0, "A": 3.7, "B+": 3.3, "B": 3.0,
        "C+": 2.7, "C": 2.3, "D+": 2.0, "D": 1.7, "F": 0.0
    }

    try:
        count_str = input(f"\n  [?] {ar('كم عدد المواد التي تود حساب معدلها؟')}: ").strip()
        if not count_str.isdigit() or int(count_str) <= 0:
            console.print(f"[red]{ar('يرجى إدخال رقم صحيح.')}[/red]")
            pause()
            return
        
        count = int(count_str)
        total_points = 0.0
        total_hours = 0
        details = []

        for i in range(1, count + 1):
            console.print(f"\n[bold cyan]◀ {ar('المادة رقم')} {i} ▶[/bold cyan]")
            code = input(f"   {ar('رمز أو اسم المادة (مثال: CS101)')}: ").strip() or f"Course_{i}"
            
            while True:
                grade = input(f"   {ar('التقدير (A+, A, B+, B, C+, C, D+, D, F)')}: ").strip().upper()
                if grade in grade_points:
                    break
                console.print(f"   [red]{ar('تقدير غير صحيح، اختر من التقديرات المعتمدة.')}[/red]")

            while True:
                hrs_str = input(f"   {ar('عدد الساعات المعتمدة (الافتراضي 3)')}: ").strip() or "3"
                if hrs_str.isdigit() and int(hrs_str) > 0:
                    hrs = int(hrs_str)
                    break
                console.print(f"   [red]{ar('يرجى إدخال عدد ساعات صحيح.')}[/red]")

            pts = grade_points[grade] * hrs
            total_points += pts
            total_hours += hrs
            details.append((code, grade, hrs, pts))

        # حساب وعرض النتيجة
        final_gpa = total_points / total_hours if total_hours > 0 else 0.0

        if final_gpa >= 3.5:
            standing = "ممتاز مرتفع (مرشح لمرتبة الشرف)"
            standing_color = "bright_green"
        elif final_gpa >= 3.0:
            standing = "جيد جداً"
            standing_color = "green"
        elif final_gpa >= 2.5:
            standing = "جيد مرتفع"
            standing_color = "cyan"
        elif final_gpa >= 2.0:
            standing = "مقبول (وضع أكاديمي سليم)"
            standing_color = "yellow"
        else:
            standing = "إنذار أكاديمي (تحت الملاحظة)"
            standing_color = "bold red"

        result_panel = Panel(
            f"[bold white]{ar('إجمالي الساعات المعتمدة')}:[/bold white] [bold cyan]{total_hours}[/bold cyan] {ar('ساعة')}\n"
            f"[bold white]{ar('مجموع النقاط المكتسبة')}:[/bold white] [bold yellow]{round(total_points, 2)}[/bold yellow]\n"
            f"[bold white]{ar('المعدل المحسوب (GPA)')}:[/bold white] [bold {standing_color}]{final_gpa:.2f} / 4.00[/bold {standing_color}]\n"
            f"[bold white]{ar('الحالة الأكاديمية')}:[/bold white] [{standing_color}]{ar(standing)}[/{standing_color}]",
            box=box.DOUBLE,
            border_style=standing_color,
            title=f"🎯 {ar('نتيجة حساب المعدل')}",
        )
        console.print(result_panel)
        log_query("STUDENT_GPA_CALC", f"حساب المعدل: {final_gpa:.2f} لساعات {total_hours}")

    except Exception as e:
        console.print(f"[red]{ar('حدث خطأ أثناء الحساب')}: {e}[/red]")

    pause()


# =============================================================================
#  صفحات وميزات بوابة الإدارة (Admin Portal)
# =============================================================================
def admin_add_course():
    """إضافة مقرر جديد إلى ملف courses.csv وملف prerequisites.json."""
    render_header("إضافة مقرر دراسي جديد")
    courses = load_courses()
    prereqs = load_prerequisites()

    cid = input(f"\n  [?] {ar('رمز المقرر (مثال: CS205)')}: ").strip().upper()
    if not cid:
        console.print(f"[yellow]{ar('تم إلغاء الإضافة.')}[/yellow]")
        pause()
        return

    if any(c["course_id"].upper() == cid for c in courses):
        console.print(f"\n[bold red]✖ {ar('المقرر موجود بالفعل في قاعدة البيانات!')}[/bold red]")
        pause()
        return

    name = input(f"  [?] {ar('اسم المادة بالعربية')}: ").strip()
    inst = input(f"  [?] {ar('اسم عضو هيئة التدريس (المحاضر)')}: ").strip()
    room = input(f"  [?] {ar('القاعة أو المعمل')}: ").strip()
    hrs  = input(f"  [?] {ar('عدد الساعات المعتمدة (الافتراضي 3)')}: ").strip() or "3"

    prereq_input = input(f"  [?] {ar('المتطلبات السابقة مفصولة بفاصلة (اتركه فارغاً إذا لم توجد)')}: ").strip()
    prereq_list = [p.strip().upper() for p in prereq_input.split(",") if p.strip()]

    courses.append({
        "course_id": cid,
        "course_name": name,
        "instructor": inst,
        "room": room,
        "credit_hours": hrs,
    })
    save_courses(courses)

    prereqs[cid] = prereq_list
    save_prerequisites(prereqs)

    log_query("ADMIN_ADD_COURSE", f"إضافة المادة {cid} - {name} متطلبات: {prereq_list}")
    console.print(f"\n[bold bright_green]✔ {ar(f'تمت إضافة المقرر {cid} وتحديث شجرة المتطلبات بنجاح.')}[/bold bright_green]")
    pause()


def admin_edit_course():
    """تعديل بيانات مقرر دراسي قائم."""
    render_header("تعديل بيانات مقرر دراسي")
    courses = load_courses()
    render_courses_table(courses, "المقررات الحالية المسجلة")

    cid = input(f"\n  [?] {ar('أدخل رمز المادة المراد تعديلها')}: ").strip().upper()
    course = next((c for c in courses if c["course_id"].upper() == cid), None)

    if not course:
        console.print(f"\n[red]✖ {ar(f'لم يتم العثور على المقرر {cid}')}[/red]")
        pause()
        return

    console.print(Panel(
        f"[bold cyan]{ar('القيم الحالية')} (اضغط Enter للإبقاء على نفس القيمة دون تغيير):[/bold cyan]\n"
        f"• {ar('الاسم')}: {course.get('course_name')}\n"
        f"• {ar('المحاضر')}: {course.get('instructor')}\n"
        f"• {ar('القاعة')}: {course.get('room')}\n"
        f"• {ar('الساعات')}: {course.get('credit_hours')}",
        box=box.ROUNDED,
        border_style="yellow",
    ))

    new_name = input(f"  [?] {ar('اسم المادة الجديد')}: ").strip()
    new_inst = input(f"  [?] {ar('اسم المحاضر الجديد')}: ").strip()
    new_room = input(f"  [?] {ar('القاعة الجديدة')}: ").strip()
    new_hrs  = input(f"  [?] {ar('الساعات المعتمدة الجديدة')}: ").strip()

    if new_name: course["course_name"] = new_name
    if new_inst: course["instructor"] = new_inst
    if new_room: course["room"] = new_room
    if new_hrs:  course["credit_hours"] = new_hrs

    save_courses(courses)
    log_query("ADMIN_EDIT_COURSE", f"تعديل المادة: {cid}")
    console.print(f"\n[bold green]✔ {ar(f'تم حفظ التعديلات على المقرر {cid} بنجاح.')}[/bold green]")
    pause()


def admin_delete_course():
    """حذف مقرر دراسي من ملف CSV وشجرة المتطلبات."""
    render_header("حذف مقرر دراسي")
    courses = load_courses()
    render_courses_table(courses, "المقررات المسجلة")

    cid = input(f"\n  [?] {ar('أدخل رمز المادة المراد حذفها نهائياً')}: ").strip().upper()
    course = next((c for c in courses if c["course_id"].upper() == cid), None)

    if not course:
        console.print(f"\n[red]✖ {ar(f'المقرر {cid} غير موجود.')}[/red]")
        pause()
        return

    confirm = input(f"  [⚠️] {ar(f'هل أنت متأكد من حذف المقرر {cid} نهائياً؟ (y/n)')}: ").strip().lower()
    if confirm == "y":
        courses = [c for c in courses if c["course_id"].upper() != cid]
        save_courses(courses)

        prereqs = load_prerequisites()
        if cid in prereqs:
            del prereqs[cid]
        # إزالة هذا المقرر من متطلبات المواد الأخرى
        for k, v in prereqs.items():
            if cid in v:
                v.remove(cid)
        save_prerequisites(prereqs)

        log_query("ADMIN_DELETE_COURSE", f"حذف المادة: {cid}")
        console.print(f"\n[bold green]✔ {ar(f'تم حذف المقرر {cid} بنجاح.')}[/bold green]")
    else:
        console.print(f"[yellow]{ar('تم إلغاء عملية الحذف.')}[/yellow]")
    
    pause()


def admin_add_document(ir_engine: IREngine):
    """إضافة وثيقة/لائحة جديدة وإعادة بناء فهرس البحث تلقائياً."""
    render_header("إضافة لائحة أو مستند جامعي جديد")

    fname = input(f"\n  [?] {ar('اسم الملف الإنجليزي (مثال: examination_rules.txt)')}: ").strip()
    if not fname:
        console.print(f"[yellow]{ar('تم الإلغاء.')}[/yellow]")
        pause()
        return

    if not fname.endswith(".txt"):
        fname += ".txt"

    console.print(f"\n[bold cyan]{ar('أدخل نص اللائحة/المستند (اكتب كلمة END في سطر مستقل عند الانتهاء):')}[/bold cyan]")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break

    content = "\n".join(lines).strip()
    if not content:
        console.print(f"[yellow]{ar('لم يتم إدخال نص، تم الإلغاء.')}[/yellow]")
        pause()
        return

    fpath = os.path.join(DOCS_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

    # إعادة بناء فهرس TF-IDF
    ir_engine.build()
    log_query("ADMIN_ADD_DOC", f"إضافة المستند: {fname}")
    console.print(f"\n[bold green]✔ {ar(f'تم حفظ المستند {fname} وإعادة بناء فهرس البحث النصي بنجاح.')}[/bold green]")
    pause()


def admin_view_logs():
    """استعراض سجل العمليات والبحث queries.log بتنسيق جدول ملون."""
    render_header("سجل الاستفسارات والعمليات (queries.log)")

    if not os.path.exists(LOG_FILE):
        console.print(f"[yellow]{ar('ملف السجل فارغ حالياً.')}[/yellow]")
        pause()
        return

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        console.print(f"[yellow]{ar('لا توجد سجلات لعرضها.')}[/yellow]")
        pause()
        return

    recent_lines = lines[-25:]  # عرض آخر 25 عملية
    table = Table(
        title=f"[bold bright_cyan]{ar(f'آخر {len(recent_lines)} عملية مسجلة في النظام')}[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold yellow",
        expand=True,
    )
    table.add_column(ar("التاريخ والوقت"), style="cyan", width=20)
    table.add_column(ar("المستوى"), justify="center", style="green", width=8)
    table.add_column(ar("نوع العملية والتفاصيل"), style="white")

    for line in recent_lines:
        parts = line.split(" | ")
        if len(parts) >= 3:
            time_str = parts[0]
            lvl = parts[1]
            msg = " | ".join(parts[2:])
            table.add_row(time_str, lvl, ar(msg))
        else:
            table.add_row("-", "LOG", ar(line))

    console.print(table)
    pause()


def admin_reset_data(ir_engine: IREngine):
    """إعادة تهيئة البيانات الافتراضية للكلية."""
    render_header("إعادة ضبط البيانات الافتراضية")
    console.print(f"[bold red]⚠️  {ar('تحذير: سيتم حذف التعديلات الحالية واستعادة البيانات النموذجية الأولية للكلية!')}[/bold red]")
    
    confirm = input(f"\n  [?] {ar('هل أنت متأكد؟ اكتب YES للتأكيد')}: ").strip()
    if confirm == "YES":
        for file in [COURSES_FILE, PREREQ_FILE, ATTENDANCE_DOC, GPA_DOC, REG_DOC]:
            if os.path.exists(file):
                os.remove(file)
        initialize_data()
        ir_engine.build()
        log_query("ADMIN_RESET_DATA", "استعادة البيانات الافتراضية للمشروع")
        console.print(f"\n[bold green]✔ {ar('تمت استعادة البيانات الافتراضية وإعادة بناء الفهارس بنجاح.')}[/bold green]")
    else:
        console.print(f"[yellow]{ar('تم إلغاء عملية إعادة الضبط.')}[/yellow]")
    
    pause()


# =============================================================================
#  القوائم التفاعلية الرئيسية (Navigation Menus)
# =============================================================================
def student_menu(ir_engine: IREngine):
    """قائمة بوابة الطالب التفاعلية."""
    while True:
        render_header("بوابة الطالب - Student Portal")
        render_dashboard_bar(ir_engine)

        menu_card = (
            f"[bold bright_cyan][1][/bold bright_cyan]  📋  {ar('دليل وتفاصيل المقررات الدراسية')}\n"
            f"[bold bright_cyan][2][/bold bright_cyan]  🎯  {ar('محرك الاستنتاج الذكي: فحص المقررات المتاحة للتسجيل')}\n"
            f"[bold bright_cyan][3][/bold bright_cyan]  🔍  {ar('البحث الذكي في لوائح ونصوص الكلية (TF-IDF & Cosine)')}\n"
            f"[bold bright_cyan][4][/bold bright_cyan]  📖  {ar('قراءة واستعراض اللوائح الجامعية المعتمدة بالكامل')}\n"
            f"[bold bright_cyan][5][/bold bright_cyan]  🧮  {ar('حاسبة المعدل التراكمي والفصلي التفاعلية (GPA Calculator)')}\n"
            f"────────────────────────────────────────────────────────────\n"
            f"[bold bright_red][0][/bold bright_red]  ↩️  {ar('العودة إلى القائمة الرئيسية')}"
        )
        console.print(Panel(menu_card, box=box.ROUNDED, border_style="cyan", title=f"🎓 {ar('الخدمات الطلابية المتاحة')}"))

        try:
            choice = input(f"\n  [?] {ar('اختر رقم الخدمة')}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            student_course_catalog()
        elif choice == "2":
            student_advisor_check()
        elif choice == "3":
            student_ir_search(ir_engine)
        elif choice == "4":
            student_view_full_docs()
        elif choice == "5":
            student_gpa_calculator()
        elif choice == "0":
            break
        else:
            console.print(f"[red]{ar('اختيار غير صحيح، الرجاء اختيار رقم من القائمة.')}[/red]")
            pause()


def admin_menu(ir_engine: IREngine):
    """قائمة لوحة تحكم المسؤول التفاعلية."""
    while True:
        render_header("لوحة تحكم إدارة النظام - Admin Portal")
        render_dashboard_bar(ir_engine)

        menu_card = (
            f"[bold bright_yellow][1][/bold bright_yellow]  📋  {ar('استعراض المقررات الحالية والشجرة الأكاديمية')}\n"
            f"[bold bright_yellow][2][/bold bright_yellow]  ➕  {ar('إضافة مقرر دراسي جديد وقاعته ومتطلباته')}\n"
            f"[bold bright_yellow][3][/bold bright_yellow]  ✏️   {ar('تعديل بيانات مقرر دراسي قائم')}\n"
            f"[bold bright_yellow][4][/bold bright_yellow]  🗑️   {ar('حذف مقرر دراسي من الخطة')}\n"
            f"[bold bright_yellow][5][/bold bright_yellow]  📄  {ar('إضافة لائحة / وثيقة جديدة (وإعادة بناء الفهرس تلقائياً)')}\n"
            f"[bold bright_yellow][6][/bold bright_yellow]  📜  {ar('استعراض سجل استفسارات وعمليات الطلاب (Live Log)')}\n"
            f"[bold bright_yellow][7][/bold bright_yellow]  🔄  {ar('استعادة البيانات النموذجية الافتراضية للكلية')}\n"
            f"────────────────────────────────────────────────────────────\n"
            f"[bold bright_red][0][/bold bright_red]  ↩️  {ar('تسجيل الخروج والعودة للقائمة الرئيسية')}"
        )
        console.print(Panel(menu_card, box=box.ROUNDED, border_style="bright_yellow", title=f"🛡️ {ar('صلاحيات المسؤول الأكاديمي')}"))

        choice = input(f"\n  [?] {ar('اختر رقم الإجراء')}: ").strip()

        if choice == "1":
            render_header("المقررات المسجلة")
            render_courses_table(load_courses(), "المقررات الحالية في الخطة")
            pause()
        elif choice == "2":
            admin_add_course()
        elif choice == "3":
            admin_edit_course()
        elif choice == "4":
            admin_delete_course()
        elif choice == "5":
            admin_add_document(ir_engine)
        elif choice == "6":
            admin_view_logs()
        elif choice == "7":
            admin_reset_data(ir_engine)
        elif choice == "0":
            break
        else:
            console.print(f"[red]{ar('اختيار غير صحيح، الرجاء اختيار رقم من القائمة.')}[/red]")
            pause()


def show_about():
    """عرض معلومات عن المشروع والمعايير البرمجية المنفذة."""
    render_header("حول مشروع دليل الطالب الذكي")
    about_text = (
        f"[bold bright_yellow]Smart Student Guide (دليل الطالب الذكي)[/bold bright_yellow]\n"
        f"[dim white]مشروع جامعي متكامل لكلية الحاسبات والذكاء الاصطناعي[/dim white]\n\n"
        f"[bold bright_cyan]المحاور والتقنيات البرمجية المطبقة:[/bold bright_cyan]\n"
        f"1. [green]استرجاع المعلومات (Information Retrieval):[/green] بناء Inverted Index وتطبيق خوارزمية TF-IDF ومقياس Cosine Similarity.\n"
        f"2. [green]تمثيل المعرفة ومحرك الاستنتاج (Inference Engine):[/green] شجرة المتطلبات السابقة (Prerequisites DAG) وتحديد أهليات التسجيل.\n"
        f"3. [green]إدارة البيانات والملفات (File Handling):[/green] قراءة وتعديل ملفات CSV و JSON و TXT وسجل الأحداث queries.log.\n"
        f"4. [green]واجهة مستخدم احترافية (Premium CLI):[/green] دعم الخط العربي، بطاقات تفاعلية، وجداول ملونة باستخدام مكتبة Rich."
    )
    console.print(Panel(ar(about_text), box=box.ROUNDED, border_style="bright_blue", title=f"ℹ️ {ar('بطاقة المشروع')}"))
    pause()


# =============================================================================
#  نقطة الدخول الرئيسية للتطبيق
# =============================================================================
def main():
    """الدالة الرئيسية لتشغيل التطبيق."""
    # 1. تهيئة البيانات التلقائية
    initialize_data()

    # 2. بناء محرك استرجاع المعلومات والفهرس
    ir_engine = IREngine()

    # 3. تسجيل بدء الجلسة
    log_query("SESSION_START", "تشغيل التطبيق وتهيئة الواجهة بنجاح")

    while True:
        render_header()
        render_dashboard_bar(ir_engine)

        welcome_card = (
            f"[bold bright_cyan][1][/bold bright_cyan]  🎓  [bold white]{ar('دخول كطالب (Student Portal)')}[/bold white]\n"
            f"       [dim white]{ar('استعراض المقررات، فحص المواد المتاحة، البحث في اللوائح، وحساب المعدل')}[/dim white]\n\n"
            f"[bold bright_cyan][2][/bold bright_cyan]  🛡️   [bold white]{ar('دخول كمسؤول (Admin Dashboard)')}[/bold white]\n"
            f"       [dim white]{ar('إدارة المقررات، إضافة لوائح، ومراقبة سجل العمليات (كلمة السر: admin123)')}[/dim white]\n\n"
            f"[bold bright_cyan][3][/bold bright_cyan]  ℹ️   [bold white]{ar('حول المشروع والتقنيات المستخدمة (About)')}[/bold white]\n\n"
            f"────────────────────────────────────────────────────────────\n"
            f"[bold bright_red][0][/bold bright_red]  🚪  [bold red]{ar('إنهاء البرنامج والخروج')}[/bold red]"
        )
        console.print(Panel(welcome_card, box=box.ROUNDED, border_style="bright_blue", title=f"🌟 {ar('مرحباً بك - اختر نوع الدخول')}"))

        try:
            choice = input(f"\n  [?] {ar('اختر رقماً')}: ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "0"

        if choice == "1":
            student_menu(ir_engine)
        elif choice == "2":
            pwd = input(f"\n  [🔒] {ar('أدخل كلمة مرور المسؤول')}: ").strip()
            if pwd == "admin123":
                log_query("ADMIN_LOGIN_SUCCESS", "تسجيل دخول ناجح للمسؤول")
                admin_menu(ir_engine)
            else:
                console.print(f"\n[bold red]✖ {ar('كلمة المرور غير صحيحة!')}[/bold red]")
                log_query("ADMIN_LOGIN_FAIL", "محاولة دخول خاطئة")
                pause()
        elif choice == "3":
            show_about()
        elif choice == "0":
            log_query("SESSION_END", "خروج من التطبيق")
            clear_screen()
            farewell = Panel(
                f"[bold bright_green]🎓 {ar('شكراً لاستخدامك دليل الطالب الذكي.')}[/bold bright_green]\n"
                f"[dim white]{ar('نتمنى لك فصلاً دراسياً حافلاً بالنجاح والتفوق الأكاديمي!')}[/dim white]",
                box=box.ROUNDED,
                border_style="green",
            )
            console.print(farewell)
            break
        else:
            console.print(f"[red]{ar('اختيار غير صحيح، الرجاء اختيار رقم من القائمة.')}[/red]")
            pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nتم إنهاء البرنامج.")
        sys.exit(0)
