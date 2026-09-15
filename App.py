import io
import os
import re
import html
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Translate Preserv",
    page_icon="🌐",
    layout="wide",
)


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(0, 180, 170, 0.10),
                transparent 35%
            ),
            #071014;
        color: #e8f5f3;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .app-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #8da6a3;
        margin-bottom: 25px;
    }

    .card {
        background: rgba(15, 29, 33, 0.86);
        border: 1px solid rgba(70, 190, 180, 0.16);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .status {
        border-radius: 12px;
        padding: 12px 15px;
        background: rgba(20, 100, 100, 0.16);
        border: 1px solid rgba(60, 200, 190, 0.18);
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    button[kind="primary"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SECRETS
# ============================================================

def secret(name, default=None):
    try:
        value = st.secrets.get(name)
        if value is not None:
            return value
    except Exception:
        pass

    return os.getenv(name, default)


GEMINI_API_KEY = secret("GEMINI_API_KEY")
GEMINI_MODEL = secret("GEMINI_MODEL", "gemini-3.8-flash")

OPENAI_API_KEY = secret("OPENAI_API_KEY")
OPENAI_MODEL = secret("OPENAI_MODEL", "gpt-5.6-luna")


# ============================================================
# PROVIDERS
# ============================================================

PROVIDERS = {}

if GEMINI_API_KEY:
    PROVIDERS["Gemini"] = {
        "key": GEMINI_API_KEY,
        "model": GEMINI_MODEL,
    }

if OPENAI_API_KEY:
    PROVIDERS["OpenAI"] = {
        "key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
    }


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = [
    "English",
    "Persian",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Italian",
    "Portuguese",
    "Turkish",
    "Russian",
    "Chinese",
    "Japanese",
    "Korean",
    "Hindi",
    "Urdu",
    "Dutch",
    "Polish",
    "Swedish",
    "Norwegian",
    "Danish",
]


RTL_LANGUAGES = {
    "Persian",
    "Arabic",
    "Urdu",
    "Hebrew",
}


def is_rtl(language):
    return language in RTL_LANGUAGES


# ============================================================
# TRANSLATION PROMPT
# ============================================================

TRANSLATION_SYSTEM = """
You are a professional document translation engine.

Your job is to translate the supplied document segment faithfully.

Rules:

1. Translate completely.
2. Never summarize.
3. Never explain.
4. Never add commentary.
5. Never remove information.
6. Preserve names, numbers, dates and technical terms.
7. Preserve paragraph boundaries.
8. Preserve list structure.
9. Preserve placeholders.
10. Preserve URLs, emails and file names.
11. Do not invent missing content.
12. Preserve the author's tone and style.
13. Keep terminology consistent.
14. Do not translate proper names unless appropriate.
15. Return ONLY the translation.
16. Do not surround the translation with quotation marks.
17. Do not add markdown fences.
18. Keep the same logical order.
"""


# ============================================================
# NORMALIZATION
# ============================================================

def clean_translation(text):
    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"^```(?:text|html)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


# ============================================================
# GEMINI
# ============================================================

def translate_gemini(text, target_language):
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is not configured.")

    if genai is None:
        raise RuntimeError("google-genai package is unavailable.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
Translate the following document segment into {target_language}.

Do not summarize it.
Do not explain it.
Return only the translated content.

DOCUMENT SEGMENT:
{text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=TRANSLATION_SYSTEM,
            temperature=0.2,
        ),
    )

    return clean_translation(response.text)


# ============================================================
# OPENAI
# ============================================================

def translate_openai(text, target_language):
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key is not configured.")

    if OpenAI is None:
        raise RuntimeError("openai package is unavailable.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
Translate the following document segment into {target_language}.

Do not summarize.
Do not explain.
Return only the translated content.

DOCUMENT SEGMENT:
{text}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=TRANSLATION_SYSTEM,
        input=prompt,
        temperature=0.2,
    )

    return clean_translation(response.output_text)


# ============================================================
# PROVIDER ROUTER
# ============================================================

def translate_once(text, target_language, provider_name):
    if not text.strip():
        return ""

    if provider_name == "Gemini":
        return translate_gemini(text, target_language)

    if provider_name == "OpenAI":
        return translate_openai(text, target_language)

    raise RuntimeError(
        f"Unsupported translation provider: {provider_name}"
    )


# ============================================================
# RETRY
# ============================================================

def translate_with_retry(
    text,
    target_language,
    provider_name,
    retries=3,
):
    last_error = None

    for attempt in range(retries):
        try:
            return translate_once(
                text,
                target_language,
                provider_name,
            )

        except Exception as exc:
            last_error = exc

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Translation failed after {retries} attempts: {last_error}"
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text, max_chars=6000):
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= max_chars:
            candidate = (
                current + "\n\n" + paragraph
                if current
                else paragraph
            )

            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)

                current = paragraph

        else:
            # Long paragraph:
            # split without destroying the whole document.
            words = paragraph.split()

            buffer = ""

            for word in words:
                candidate = (
                    buffer + " " + word
                    if buffer
                    else word
                )

                if len(candidate) <= max_chars:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(
                            current + "\n\n" + buffer
                            if current
                            else buffer
                        )

                    current = ""
                    buffer = word

            if buffer:
                current = (
                    current + "\n\n" + buffer
                    if current
                    else buffer
                )

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# PARALLEL TRANSLATION
# ============================================================

def translate_chunks_parallel(
    chunks,
    target_language,
    provider_name,
    workers=4,
    progress_callback=None,
):
    if not chunks:
        return []

    workers = max(1, min(int(workers), 8))

    results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=workers) as executor:

        future_map = {
            executor.submit(
                translate_with_retry,
                chunk,
                target_language,
                provider_name,
            ): index
            for index, chunk in enumerate(chunks)
        }

        completed = 0

        for future in as_completed(future_map):
            index = future_map[future]

            results[index] = future.result()

            completed += 1

            if progress_callback:
                progress_callback(
                    completed,
                    len(chunks),
                )

    return results


# ============================================================
# PDF ANALYSIS
# ============================================================

def analyze_pdf(data):
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is not installed."
        )

    document = fitz.open(
        stream=data,
        filetype="pdf",
    )

    total_pages = len(document)

    native_pages = 0
    scanned_pages = 0
    total_text_chars = 0

    for page in document:
        text = page.get_text("text").strip()

        if text:
            native_pages += 1
            total_text_chars += len(text)
        else:
            scanned_pages += 1

    document.close()

    return {
        "pages": total_pages,
        "native_pages": native_pages,
        "scanned_pages": scanned_pages,
        "text_chars": total_text_chars,
    }


# ============================================================
# PDF STRUCTURED BLOCK EXTRACTION
# ============================================================

def extract_pdf_blocks(page):
    """
    Extract real text blocks with geometry and style.

    We deliberately do NOT merge the whole page into one rectangle.
    """

    data = page.get_text(
        "dict",
        sort=True,
    )

    blocks = []

    block_id = 0

    for raw_block in data.get("blocks", []):

        # type 0 = text block
        if raw_block.get("type") != 0:
            continue

        lines = raw_block.get("lines", [])

        span_texts = []

        sizes = []
        fonts = []
        colors = []

        for line in lines:

            for span in line.get("spans", []):

                text = span.get("text", "")

                if text:
                    span_texts.append(text)

                if span.get("size"):
                    sizes.append(float(span["size"]))

                if span.get("font"):
                    fonts.append(span["font"])

                if span.get("color") is not None:
                    colors.append(span["color"])

        text = "".join(span_texts).strip()

        if not text:
            continue

        bbox = raw_block.get("bbox")

        if not bbox:
            continue

        blocks.append(
            {
                "id": block_id,
                "text": text,
                "bbox": tuple(bbox),
                "font_size": (
                    sum(sizes) / len(sizes)
                    if sizes
                    else 10.0
                ),
                "font": (
                    fonts[0]
                    if fonts
                    else "helv"
                ),
                "color": (
                    colors[0]
                    if colors
                    else 0
                ),
            }
        )

        block_id += 1

    return blocks


# ============================================================
# PDF TRANSLATION MEMORY
# ============================================================

class TranslationMemory:
    def __init__(self):
        self.cache = {}

    def get(self, source, language, provider):
        key = (
            provider,
            language,
            source.strip(),
        )

        return self.cache.get(key)

    def set(
        self,
        source,
        language,
        provider,
        translated,
    ):
        key = (
            provider,
            language,
            source.strip(),
        )

        self.cache[key] = translated


# ============================================================
# FONT / COLOR HELPERS
# ============================================================

def pdf_color(color_int):
    try:
        r = (color_int >> 16) & 255
        g = (color_int >> 8) & 255
        b = color_int & 255

        return (
            r / 255,
            g / 255,
            b / 255,
        )

    except Exception:
        return (0, 0, 0)


def html_escape_text(text):
    return html.escape(
        text
    ).replace(
        "\n",
        "<br>",
    )


# ============================================================
# INSERT TRANSLATED BLOCK
# ============================================================

def insert_translated_block(
    page,
    block,
    translated,
    target_language,
):
    rect = fitz.Rect(block["bbox"])

    # Small safety margin.
    rect.x0 += 0.5
    rect.y0 += 0.5
    rect.x1 -= 0.5
    rect.y1 -= 0.5

    if rect.width <= 2 or rect.height <= 2:
        return False

    # Estimate a suitable font size.
    original_size = max(
        6,
        min(
            block.get("font_size", 10),
            30,
        ),
    )

    rtl = is_rtl(target_language)

    alignment = "right" if rtl else "left"

    color = pdf_color(
        block.get("color", 0)
    )

    r = int(color[0] * 255)
    g = int(color[1] * 255)
    b = int(color[2] * 255)

    safe = html_escape_text(
        translated
    )

    css = f"""
    * {{
        font-family: sans-serif;
        font-size: {original_size}pt;
        color: rgb({r},{g},{b});
        line-height: 1.15;
        text-align: {alignment};
    }}
    """

    html_content = (
        f'<div dir="{"rtl" if rtl else "ltr"}">'
        f"{safe}"
        f"</div>"
    )

    # Try original size first.
    for scale in [
        1.0,
        0.92,
        0.84,
        0.76,
        0.68,
    ]:

        fontsize = max(
            5.5,
            original_size * scale,
        )

        current_css = f"""
        * {{
            font-family: sans-serif;
            font-size: {fontsize}pt;
            color: rgb({r},{g},{b});
            line-height: 1.15;
            text-align: {alignment};
        }}
        """

        try:
            spare_height = rect.height

            result = page.insert_htmlbox(
                rect,
                html_content,
                css=current_css,
                overlay=True,
            )

            # PyMuPDF returns a value that can indicate
            # whether the content fit. We accept the
            # first successful rendering.
            if result is not None:
                return True

        except Exception:
            pass

    # Last fallback.
    try:
        page.insert_textbox(
            rect,
            translated,
            fontsize=max(
                5.5,
                original_size * 0.68,
            ),
            fontname="helv",
            color=color,
            align=(
                fitz.TEXT_ALIGN_RIGHT
                if rtl
                else fitz.TEXT_ALIGN_LEFT
            ),
            overlay=True,
        )

        return True

    except Exception:
        return False


# ============================================================
# PDF PAGE TRANSLATION
# ============================================================

def translate_pdf(
    data,
    target_language,
    provider_name,
    workers=4,
):
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is not installed."
        )

    document = fitz.open(
        stream=data,
        filetype="pdf",
    )

    total_pages = len(document)

    progress_bar = st.progress(
        0,
        text="Preparing PDF...",
    )

    status = st.empty()

    memory = TranslationMemory()

    translated_blocks_total = 0
    untouched_pages = 0

    for page_index in range(total_pages):

        page = document[page_index]

        status.info(
            f"Processing page {page_index + 1} "
            f"of {total_pages}"
        )

        blocks = extract_pdf_blocks(page)

        # ----------------------------------------------------
        # Native PDF page
        # ----------------------------------------------------

        if blocks:

            source_texts = [
                block["text"]
                for block in blocks
            ]

            translated = [None] * len(blocks)

            missing_indices = []

            for i, source in enumerate(source_texts):

                cached = memory.get(
                    source,
                    target_language,
                    provider_name,
                )

                if cached:
                    translated[i] = cached
                else:
                    missing_indices.append(i)

            if missing_indices:

                missing_texts = [
                    source_texts[i]
                    for i in missing_indices
                ]

                translated_missing = []

                for source in missing_texts:

                    chunks = split_text(
                        source,
                        max_chars=6000,
                    )

                    parts = translate_chunks_parallel(
                        chunks,
                        target_language,
                        provider_name,
                        workers=workers,
                    )

                    translated_missing.append(
                        "\n\n".join(parts)
                    )

                for local_index, block_index in enumerate(
                    missing_indices
                ):
                    result = translated_missing[
                        local_index
                    ]

                    translated[
                        block_index
                    ] = result

                    memory.set(
                        source_texts[block_index],
                        target_language,
                        provider_name,
                        result,
                    )

            # ------------------------------------------------
            # Replace block-by-block
            # ------------------------------------------------

            successful_blocks = 0

            # Redact only original text areas.
            for block in blocks:

                rect = fitz.Rect(
                    block["bbox"]
                )

                # Slightly expand to avoid remnants.
                rect.x0 -= 0.7
                rect.y0 -= 0.7
                rect.x1 += 0.7
                rect.y1 += 0.7

                page.add_redact_annot(
                    rect,
                    fill=(1, 1, 1),
                )

            page.apply_redactions()

            for i, block in enumerate(blocks):

                result = translated[i]

                if not result:
                    continue

                ok = insert_translated_block(
                    page,
                    block,
                    result,
                    target_language,
                )

                if ok:
                    successful_blocks += 1

            translated_blocks_total += successful_blocks

        # ----------------------------------------------------
        # Scanned/image-only page
        # ----------------------------------------------------

        else:
            untouched_pages += 1

        progress = (
            (page_index + 1)
            / max(total_pages, 1)
        )

        progress_bar.progress(
            progress,
            text=(
                f"Processed "
                f"{page_index + 1}/{total_pages} pages"
            ),
        )

    status.success(
        "PDF translation completed."
    )

    output = document.tobytes(
        garbage=4,
        deflate=True,
    )

    document.close()

    return output, {
        "pages": total_pages,
        "translated_blocks": translated_blocks_total,
        "scanned_pages": untouched_pages,
    }


# ============================================================
# TXT
# ============================================================

def translate_text_file(
    data,
    target_language,
    provider_name,
    workers=4,
):
    text = data.decode(
        "utf-8",
        errors="replace",
    )

    chunks = split_text(
        text,
        max_chars=6000,
    )

    translated = translate_chunks_parallel(
        chunks,
        target_language,
        provider_name,
        workers=workers,
    )

    return "\n\n".join(
        translated
    ).encode("utf-8")


# ============================================================
# DOCX
# ============================================================

def translate_docx(
    data,
    target_language,
    provider_name,
    workers=4,
):
    if Document is None:
        raise RuntimeError(
            "python-docx is not installed."
        )

    document = Document(
        io.BytesIO(data)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(
                paragraph
            )

    source_texts = [
        p.text
        for p in paragraphs
    ]

    translated = [None] * len(
        paragraphs
    )

    if source_texts:

        translated = translate_chunks_parallel(
            source_texts,
            target_language,
            provider_name,
            workers=workers,
        )

    for paragraph, translation in zip(
        paragraphs,
        translated,
    ):

        paragraph.clear()

        run = paragraph.add_run(
            translation
        )

        if is_rtl(target_language):
            paragraph.alignment = 2

    output = io.BytesIO()

    document.save(output)

    return output.getvalue()


# ============================================================
# MAIN FILE ROUTER
# ============================================================

def translate_file(
    file_bytes,
    filename,
    target_language,
    provider_name,
    workers=4,
):
    extension = (
        filename.lower()
        .split(".")[-1]
    )

    if extension == "pdf":

        return translate_pdf(
            file_bytes,
            target_language,
            provider_name,
            workers,
        )

    if extension == "txt":

        result = translate_text_file(
            file_bytes,
            target_language,
            provider_name,
            workers,
        )

        return result, {}

    if extension == "docx":

        result = translate_docx(
            file_bytes,
            target_language,
            provider_name,
            workers,
        )

        return result, {}

    raise ValueError(
        "Supported formats currently: PDF, DOCX, TXT."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">Translate Preserv</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Translate documents while preserving their structure and layout.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# PROVIDER CHECK
# ============================================================

if not PROVIDERS:

    st.error(
        "No translation API is configured. "
        "Add your provider key in Streamlit Secrets."
    )

    st.stop()


# ============================================================
# SETTINGS
# ============================================================

with st.container():

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        target_language = st.selectbox(
            "Language",
            LANGUAGES,
            index=1,
        )

    with col2:

        provider_name = st.selectbox(
            "Translation engine",
            list(PROVIDERS.keys()),
        )

    with col3:

        workers = st.slider(
            "Parallel workers",
            min_value=1,
            max_value=8,
            value=4,
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# MODE
# ============================================================

mode = st.radio(
    "Mode",
    ["Text", "File"],
    horizontal=True,
)


# ============================================================
# TEXT MODE
# ============================================================

if mode == "Text":

    text = st.text_area(
        "Text",
        height=260,
        placeholder="Write or paste text...",
    )

    if st.button(
        "Translate",
        type="primary",
        use_container_width=True,
    ):

        if not text.strip():

            st.warning(
                "Enter some text first."
            )

        else:

            with st.spinner(
                "Translating..."
            ):

                chunks = split_text(
                    text,
                    max_chars=6000,
                )

                translated = translate_chunks_parallel(
                    chunks,
                    target_language,
                    provider_name,
                    workers=workers,
                )

                result = "\n\n".join(
                    translated
                )

            st.success(
                "Translation completed."
            )

            st.text_area(
                "Result",
                result,
                height=320,
            )


# ============================================================
# FILE MODE
# ============================================================

else:

    uploaded = st.file_uploader(
        "Upload",
        type=[
            "pdf",
            "docx",
            "txt",
        ],
        label_visibility="collapsed",
    )

    if uploaded:

        file_bytes = uploaded.getvalue()

        st.markdown(
            f"""
            <div class="status">
            <b>{html.escape(uploaded.name)}</b>
            &nbsp;·&nbsp;
            {len(file_bytes) / 1024 / 1024:.2f} MB
            </div>
            """,
            unsafe_allow_html=True,
        )

        # PDF analysis
        if uploaded.name.lower().endswith(".pdf"):

            try:

                analysis = analyze_pdf(
                    file_bytes
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Pages",
                    analysis["pages"],
                )

                c2.metric(
                    "Native",
                    analysis["native_pages"],
                )

                c3.metric(
                    "Scanned",
                    analysis["scanned_pages"],
                )

                c4.metric(
                    "Text",
                    f"{analysis['text_chars']:,}",
                )

                if analysis["scanned_pages"] > 0:

                    st.warning(
                        "This PDF contains scanned/image-only "
                        "pages. Those pages require OCR before "
                        "they can be translated."
                    )

            except Exception as exc:

                st.warning(
                    f"PDF analysis failed: {exc}"
                )

        if st.button(
            "Translate document",
            type="primary",
            use_container_width=True,
        ):

            try:

                result, stats = translate_file(
                    file_bytes,
                    uploaded.name,
                    target_language,
                    provider_name,
                    workers=workers,
                )

                base_name = os.path.splitext(
                    uploaded.name
                )[0]

                extension = (
                    uploaded.name.lower()
                    .split(".")[-1]
                )

                if extension == "pdf":

                    output_name = (
                        f"{base_name}_"
                        f"{target_language.lower()}."
                        f"pdf"
                    )

                    st.success(
                        "PDF translation completed."
                    )

                    if stats:

                        st.info(
                            f"Pages: {stats.get('pages', 0)} · "
                            f"Translated blocks: "
                            f"{stats.get('translated_blocks', 0)} · "
                            f"Scanned pages requiring OCR: "
                            f"{stats.get('scanned_pages', 0)}"
                        )

                elif extension == "docx":

                    output_name = (
                        f"{base_name}_"
                        f"{target_language.lower()}."
                        f"docx"
                    )

                    st.success(
                        "DOCX translation completed."
                    )

                else:

                    output_name = (
                        f"{base_name}_"
                        f"{target_language.lower()}.txt"
                    )

                    st.success(
                        "Text translation completed."
                    )

                st.download_button(
                    "Download translated file",
                    data=result,
                    file_name=output_name,
                    mime=(
                        "application/pdf"
                        if extension == "pdf"
                        else (
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                            if extension == "docx"
                            else "text/plain"
                        )
                    ),
                    use_container_width=True,
                )

            except Exception as exc:

                st.error(
                    f"Translation failed: {exc}"
                )

                st.exception(exc)
