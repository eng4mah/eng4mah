
# --------------------------------------------------------------
# RegTech Assistance – Streamlit UI (مُحدَّث)
# --------------------------------------------------------------

import os
import re
from pathlib import Path
from typing import List, Dict, Generator, Optional, Tuple

import streamlit as st

# --------------------------- Cerebras SDK ---------------------------
try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:                                   # pragma: no cover
    st.error("⚠️ يرجى تثبيت Cerebras SDK: `pip install cerebras-cloud-sdk`")
    st.stop()

# --------------------------- Optional tokeniser --------------------
# tiktoken هو نفس المحلل الذي تستعمله نماذج OpenAI/Cerebras.
# إذا لم يتوفر، نعود إلى حساب تقريبي (عدد الكلمات).
try:
    import tiktoken
except Exception:
    tiktoken = None

def count_tokens(text: str) -> int:
    """عدد الرموز بنفس طريقة Cerebras (مع fallback بسيط)."""
    if not text:
        return 0
    if tiktoken:
        try:
            # نموذج gpt‑oss‑120b يستخدم نفس الترميز العام (cl100k_base)
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # تقريب بسيط – كل كلمة ≈ 1.33 رمز (مقاربة للنماذج الكبيرة)
    return int(len(text.split()) * 1.33)


# --------------------------- PDF / DOCX / Image utilities ----------
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import pdfminer.high_level as pdfminer
except Exception:
    pdfminer = None

try:
    import fitz                     # PyMuPDF
except Exception:
    fitz = None

try:
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
except Exception:
    convert_from_path = None
    Image = None
    pytesseract = None

try:
    import docx
except Exception:
    docx = None

# --------------------------- Paths & config -------------------------
def get_repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


REPO_ROOT = get_repo_root()
FOLDER_PATH = os.path.join(REPO_ROOT, "legal_docs")
SUMMARY_FOLDER_PATH = os.path.join(REPO_ROOT, "summary")
RULES_FILE_PATH = os.path.join(REPO_ROOT, "RULES.txt")

API_KEY = os.getenv("CEREBRAS_API_KEY") or st.secrets.get("CEREBRAS_API_KEY", "")
if not API_KEY:                                           # pragma: no cover
    st.error("❗ مفتاح Cerebras غير مُعرَّف.")
    st.stop()
client = Cerebras(api_key=API_KEY)

# --------------------------- Feature flags -------------------------
USE_SUMMARY_FOR_SELECTION = True
USE_TWO_STEP_SELECTION = True
MAX_DOCS_FOR_SELECTION = 1
SELECTION_RETRY_ATTEMPTS = 3

# --------------------------- Helpers -------------------------------
def safe_run(fn):
    """Decorator لالتقاط جميع الاستثناءات وعرضها في واجهة Streamlit."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                         # pragma: no cover
            st.error(f"❌ خطأ غير متوقع: {exc}")
            st.exception(exc)
    return wrapper


# ---------- PDF cascade ----------
def _extract_pdf_with_pypdf(path: str) -> str:
    if not pypdf:
        return ""
    try:
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _extract_pdf_with_pdfminer(path: str) -> str:
    if not pdfminer:
        return ""
    try:
        return pdfminer.extract_text(path) or ""
    except Exception:
        return ""


def _extract_pdf_with_fitx(path: str) -> str:
    if not fitz:
        return ""
    try:
        doc = fitz.open(path)
        return "".join(page.get_text() for page in doc)
    except Exception:
        return ""


def _ocr_pdf(path: str) -> str:
    if not (convert_from_path and pytesseract and Image):
        return ""
    try:
        images = convert_from_path(path, dpi=300, fmt="png")
        txt = ""
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            txt += pytesseract.image_to_string(img) + "\n"
        return txt
    except Exception:
        return ""


def _read_docx(path: str) -> str:
    if not docx:
        return ""
    try:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""


def _read_image(path: str) -> str:
    if not (pytesseract and Image):
        return ""
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        return ""


def read_file_content(file_path: str) -> Tuple[str, str]:
    """
    إرجاع (اسم‑الملف, النص المستخرج) لأي صيغة مدعومة.
    PDFs → cascade، DOCX → python‑docx، صور → OCR، غير ذلك → نص عادي.
    """
    file_name = os.path.basename(file_path)
    text = ""

    if file_name.lower().endswith(".pdf"):
        text = _extract_pdf_with_pypdf(file_path)
        if not text.strip():
            text = _extract_pdf_with_pdfminer(file_path)
        if not text.strip():
            text = _extract_pdf_with_fitx(file_path)
        if not text.strip():
            text = _ocr_pdf(file_path)

    elif file_name.lower().endswith(".docx"):
        text = _read_docx(file_path)

    elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        text = _read_image(file_path)

    else:   # نص عادي
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            pass

    if not text.strip():
        st.warning(f"⚠️ لا يوجد محتوى مقروء في `{file_name}`.")
    return file_name, text


def get_document_names_from_folder() -> List[str]:
    if not os.path.isdir(FOLDER_PATH):
        st.warning(f"⚠️ المجلد غير موجود: {FOLDER_PATH}")
        return []
    return [
        f for f in os.listdir(FOLDER_PATH)
        if f.lower().endswith((".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"))
    ]


def load_rules_file() -> str:
    if os.path.exists(RULES_FILE_PATH):
        _, txt = read_file_content(RULES_FILE_PATH)
        return txt
    return "لا توجد تعليمات خاصة. اتبع التعليمات العامة."


def load_all_summaries() -> str:
    if not os.path.isdir(SUMMARY_FOLDER_PATH):
        return "لا توجد ملخصات متاحة."
    out = ""
    for fname in sorted(os.listdir(SUMMARY_FOLDER_PATH)):
        if fname.lower().endswith(".txt"):
            _, txt = read_file_content(os.path.join(SUMMARY_FOLDER_PATH, fname))
            out += f"---\n{txt}\n---\n\n"
    return out or "لم يُعثر على ملخصات."


def get_document_selection_with_summaries(
    question: str,
    doc_names: List[str],
    rules: str,
    summaries: str,
    max_docs: int,
) -> Optional[List[int]]:
    doc_list = "\n".join([f"#{i}: {n}" for i, n in enumerate(doc_names)])

    user_prompt = f"""
مهمتك هي اختيار الأنسب من المستندات للإجابة على سؤال المستخدم، بحد أقصى {max_docs} ملف.
استخدم القواعد والملخصات.

سؤال المستخدم:
{question}

القواعد الإلزامية:
{rules}

ملخصات الملفات:
{summaries}

قائمة الملفات المتاحة:
{doc_list}

أرسل أرقام الملفات المختارة بصيغة #N أو #NONE إذا لا شيء مناسب.
""".strip()

    for attempt in range(SELECTION_RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": user_prompt}],
                model="gpt-oss-120b",
            )
            content = resp.choices[0].message.content or ""
            nums = re.findall(r"#(\d+)", content)

            if nums:
                indices = [int(x) for x in nums if 0 <= int(x) < len(doc_names)]
                if indices:
                    return indices[:max_docs]

            if "#NONE" in content.upper():
                return []
        except Exception as e:                     # pragma: no cover
            st.error(f"❌ محاولة اختيار الوثائق فشلت ({attempt+1}/{SELECTION_RETRY_ATTEMPTS}): {e}")
            continue
    return None


def get_cerebras_response_stream(prompt: str, chat_history: List[Dict[str, str]]) -> Generator[str, None, None]:
    messages = chat_history + [{"role": "user", "content": prompt}]
    try:
        stream = client.chat.completions.create(
            messages=messages,
            model="gpt-oss-120b",
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:                         # pragma: no cover
        err = f"❌ خطأ في الاتصال بالـ API: {e}"
        st.error(err)
        yield err


def build_prompt(task: str, rules: str, context: str, question: Optional[str] = None) -> str:
    q_sec = f"\nسؤال المستخدم:\n---\n{question}\n---\n" if question else ""
    return f"""أنت مساعد قانوني متخصص "RegTech Assistance". اتبع التعليمات بدقة:

قواعد ملزمة:
{rules}

المهمة:
{task}

الوثائق:
{context if context else "لم يتم اختيار أو العثور على وثائق."}
{q_sec}
"""


# ------------------------------------------------------------------
# 7️⃣  واجهة Streamlit
# ------------------------------------------------------------------
@safe_run
def main():
    st.set_page_config(page_title="RegTech Assistance", page_icon="⚖️", layout="centered")

    # ----------------- CSS عام -----------------
    st.markdown(
        """
        <style>
        body { direction: rtl; text-align: right; }
        input, textarea { text-align: right !important; }
        .stChatMessage, .stChatMessage > div { opacity: 1 !important; }

        /* زر المسح الثابت – أسفل‑يسار */
        .clear-btn {
            position: fixed;
            bottom: 12px;
            left: 12px;
            z-index: 9999;
            background: none;
            border: none;
            font-size: 1.8rem;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- زر المسح الثابت -----------------
    # يجرى إعادة تحميل الصفحة => مسح كل الـ session state
    st.markdown(
        """
        <button class="clear-btn" title="مسح المحادثة"
                onclick="window.location.reload();">🧹</button>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- Session State -----------------
    for key, default in {
        "messages": [],
        "doc_names": [],
        "rules_content": "",
        "docs_loaded_first_time": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ----------------- أول تحميل للملفات/المجلد -----------------
    if not st.session_state.docs_loaded_first_time:
        with st.spinner("جاري تهيئة النظام…"):
            os.makedirs(FOLDER_PATH, exist_ok=True)
            os.makedirs(SUMMARY_FOLDER_PATH, exist_ok=True)

            if not os.path.exists(RULES_FILE_PATH):
                with open(RULES_FILE_PATH, "w", encoding="utf-8") as f:
                    f.write("لا توجد تعليمات خاصة. اتبع التعليمات العامة.")

            st.session_state.doc_names = get_document_names_from_folder()
            st.session_state.rules_content = load_rules_file()
            st.session_state.docs_loaded_first_time = True
            st.toast(f"✅ عُثر على {len(st.session_state.doc_names)} وثيقة.", icon="✅")

    # ----------------- عرض السجل -----------------
    if not st.session_state.messages:
        st.markdown(
            "<div style='color:#a0a0a0; text-align:right'>مرحباً! كيف يمكنني مساعدتك في وثائقك اليوم؟ ⚖️</div>",
            unsafe_allow_html=True,
        )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_review_notice"):
                st.info(msg["content"])
            else:
                st.markdown(msg["content"])

    # ----------------- إدخال المستخدم -----------------
    if prompt := st.chat_input("اسأل سؤالاً عن مستنداتك …"):
        # حفظ سؤال المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ------------ المساعد ------------
        with st.chat_message("assistant"):
            final_context = ""
            context_docs = []

            # ----- اختيار الوثائق إذا كان مفعلاً -----
            if USE_TWO_STEP_SELECTION and st.session_state.doc_names:
                with st.spinner("تحليل السؤال واختيار الوثائق…"):
                    all_summaries = load_all_summaries()
                    selected_idxs = get_document_selection_with_summaries(
                        prompt,
                        st.session_state.doc_names,
                        st.session_state.rules_content,
                        all_summaries,
                        MAX_DOCS_FOR_SELECTION,
                    )

                if selected_idxs is None:
                    err = "❌ فشل اختيار الوثيقة بعد عدة محاولات. رجاءً أعد المحاولة."
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    st.rerun()
                elif not selected_idxs:
                    st.info("⚙️ لم تُحدَّد وثائق صالحة – سيُستند الرد إلى القواعد فقط.")
                else:
                    chosen = [
                        st.session_state.doc_names[i]
                        for i in selected_idxs
                        if 0 <= i < len(st.session_state.doc_names)
                    ]
                    notice = "سأراجِع الوثائق التالية:\n\n" + "\n".join(f"- `{n}`" for n in chosen)
                    st.info(notice)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": notice, "is_review_notice": True}
                    )
                    with st.spinner("تحميل محتوى الوثائق…"):
                        for n in chosen:
                            _, txt = read_file_content(os.path.join(FOLDER_PATH, n))
                            if txt:
                                context_docs.append({"source": n, "content": txt})
                    final_context = "\n\n---\n\n".join(
                        f"المصدر: {d['source']}\n\nالمحتوى: {d['content']}"
                        for d in context_docs
                    )
            else:
                st.info("⚙️ اختيار الوثائق معطل – سيتم الاعتماد على القواعد فقط.")

            # ----- توليد الرد -----
            with st.spinner("توليد الرد…"):
                task = "أجب على سؤال المستخدم بالاعتماد على الوثائق فقط."
                full_prompt = build_prompt(task, st.session_state.rules_content, final_context, prompt)

                # تحضير التاريخ (بدون رسائل “notice”)
                history_for_api = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]   # باستثناء سؤال المستخدم الأخير
                    if not m.get("is_review_notice")
                ]

                placeholder = st.empty()
                answer = ""

                for chunk in get_cerebras_response_stream(full_prompt, history_for_api):
                    answer += chunk
                    placeholder.markdown(answer + "▌")

                # عدد الرموز (حسب محلل Cerebras أو تقريبي)
                token_cnt = count_tokens(answer)
                answer_with_tokens = f"{answer} ({token_cnt}t)"
                placeholder.markdown(answer_with_tokens)

                # حفظ الرد في السجل
                if answer:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer_with_tokens}
                    )

            # ----- تمرير الصفحة إلى الأسفل تلقائيًا -----
            st.markdown(
                """
                <script>
                const el = window.parent.document.querySelector('.stApp');
                if (el) { el.scrollTo({top: el.scrollHeight, behavior: 'smooth'}); }
                </script>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()