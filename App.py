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

    return os.getenv(
        name,
        default,
    )


GEMINI_API_KEY = secret(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = secret(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

OPENAI_API_KEY = secret(
    "OPENAI_API_KEY"
)

OPENAI_MODEL = secret(
    "OPENAI_MODEL",
    "gpt-5.6-luna",
)


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
# OCR LANGUAGES
# ============================================================

# These correspond to the language packages
# installed in packages.txt.
OCR_LANGUAGES = (
    "fas+eng+ara+deu+fra+spa+ita+tur+rus+por+"
    "nld+pol+swe+dan+nor+fin+ell+heb+hin+urd+pus+"
    "chi_sim+chi_tra+jpn+kor"
)


# ============================================================
# TRANSLATION SYSTEM PROMPT
# ============================================================

TRANSLATION_SYSTEM = """
You are a professional document translation engine.

Translate the supplied document segment faithfully and completely.

Rules:

1. Never summarize.
2. Never explain.
3. Never add commentary.
4. Never remove information.
5. Preserve names, numbers, dates and technical terms.
6. Preserve paragraph structure.
7. Preserve list structure.
8. Preserve placeholders.
9. Preserve URLs, emails and file names.
10. Do not invent missing content.
11. Preserve the author's tone and style.
12. Keep terminology consistent.
13. Do not translate proper names unless appropriate.
14. Return ONLY the translation.
15. Do not add markdown fences.
16. Do not add quotation marks around the whole translation.
17. Preserve the logical order of the source.
"""


# ============================================================
# CLEAN TRANSLATION
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

def translate_gemini(
    text,
    target_language,
):

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini API key is not configured."
        )

    if genai is None:
        raise RuntimeError(
            "google-genai package is unavailable."
        )

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    prompt = f"""
Translate the following document segment into
{target_language}.

Translate everything.

Do not summarize.
Do not explain.
Return only the translation.

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

    return clean_translation(
        response.text
    )


# ============================================================
# OPENAI
# ============================================================

def translate_openai(
    text,
    target_language,
):

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OpenAI API key is not configured."
        )

    if OpenAI is None:
        raise RuntimeError(
            "openai package is unavailable."
        )

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    prompt = f"""
Translate the following document segment into
{target_language}.

Translate everything.

Do not summarize.
Do not explain.
Return only the translation.

DOCUMENT SEGMENT:

{text}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=TRANSLATION_SYSTEM,
        input=prompt,
        temperature=0.2,
    )

    return clean_translation(
        response.output_text
    )


# ============================================================
# PROVIDER ROUTER
# ============================================================

def translate_once(
    text,
    target_language,
    provider_name,
):

    if not text.strip():
        return ""

    if provider_name == "Gemini":
        return translate_gemini(
            text,
            target_language,
        )

    if provider_name == "OpenAI":
        return translate_openai(
            text,
            target_language,
        )

    raise RuntimeError(
        f"Unsupported provider: {provider_name}"
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
                time.sleep(
                    2 ** attempt
                )

    raise RuntimeError(
        f"Translation failed after "
        f"{retries} attempts: {last_error}"
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    max_chars=6000,
):

    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

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
                    chunks.append(
                        current
                    )

                current = paragraph

        else:

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
                            buffer
                        )

                    buffer = word

            if buffer:

                if current:

                    chunks.append(
                        current
                    )

                current = buffer

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
):

    if not chunks:
        return []

    workers = max(
        1,
        min(
            int(workers),
            8,
        ),
    )

    results = [None] * len(
        chunks
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        future_map = {
            executor.submit(
                translate_with_retry,
                chunk,
                target_language,
                provider_name,
            ): index
            for index, chunk
            in enumerate(chunks)
        }

        for future in as_completed(
            future_map
        ):

            index = future_map[
                future
            ]

            results[index] = (
                future.result()
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

    total_pages = len(
        document
    )

    native_pages = 0
    scanned_pages = 0
    total_text_chars = 0

    for page in document:

        text = page.get_text(
            "text",
            sort=True,
        ).strip()

        if text:

            native_pages += 1

            total_text_chars += len(
                text
            )

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
# BUILD BLOCKS
# ============================================================

def build_blocks_from_dict(
    data
):

    blocks = []

    block_id = 0

    for raw_block in data.get(
        "blocks",
        [],
    ):

        # Text blocks only.
        if raw_block.get(
            "type"
        ) != 0:
            continue

        lines = raw_block.get(
            "lines",
            [],
        )

        span_texts = []
        sizes = []
        fonts = []
        colors = []

        for line in lines:

            for span in line.get(
                "spans",
                [],
            ):

                text = span.get(
                    "text",
                    "",
                )

                if text:
                    span_texts.append(
                        text
                    )

                if span.get("size"):
                    sizes.append(
                        float(
                            span["size"]
                        )
                    )

                if span.get("font"):
                    fonts.append(
                        span["font"]
                    )

                if span.get(
                    "color"
                ) is not None:

                    colors.append(
                        span["color"]
                    )

        text = "".join(
            span_texts
        ).strip()

        if not text:
            continue

        bbox = raw_block.get(
            "bbox"
        )

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
# PDF BLOCK EXTRACTION + OCR
# ============================================================

def extract_pdf_blocks(
    page,
    ocr_languages=OCR_LANGUAGES,
):

    # --------------------------------------------------------
    # First: native PDF text
    # --------------------------------------------------------

    native_text = page.get_text(
        "text",
        sort=True,
    ).strip()

    if native_text:

        data = page.get_text(
            "dict",
            sort=True,
        )

        return (
            build_blocks_from_dict(
                data
            ),
            False,
        )

    # --------------------------------------------------------
    # Second: OCR
    # --------------------------------------------------------

    try:

        text_page = page.get_textpage_ocr(
            language=ocr_languages,
            dpi=300,
            full=True,
        )

        ocr_data = page.get_text(
            "dict",
            textpage=text_page,
            sort=True,
        )

        blocks = build_blocks_from_dict(
            ocr_data
        )

        return (
            blocks,
            True,
        )

    except Exception as exc:

        raise RuntimeError(
            "OCR failed. "
            f"Languages: {ocr_languages}. "
            f"Error: {exc}"
        )


# ============================================================
# TRANSLATION MEMORY
# ============================================================

class TranslationMemory:

    def __init__(self):

        self.cache = {}

    def get(
        self,
        source,
        language,
        provider,
    ):

        key = (
            provider,
            language,
            source.strip(),
        )

        return self.cache.get(
            key
        )

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
# COLOR
# ============================================================

def pdf_color(
    color_int
):

    try:

        r = (
            color_int >> 16
        ) & 255

        g = (
            color_int >> 8
        ) & 255

        b = (
            color_int
        ) & 255

        return (
            r / 255,
            g / 255,
            b / 255,
        )

    except Exception:

        return (
            0,
            0,
            0,
        )


# ============================================================
# HTML ESCAPE
# ============================================================

def html_escape_text(
    text
):

    return (
        html.escape(
            text
        )
        .replace(
            "\n",
            "<br>",
        )
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

    rect = fitz.Rect(
        block["bbox"]
    )

    rect.x0 += 0.5
    rect.y0 += 0.5
    rect.x1 -= 0.5
    rect.y1 -= 0.5

    if (
        rect.width <= 2
        or rect.height <= 2
    ):
        return False

    original_size = max(
        6,
        min(
            block.get(
                "font_size",
                10,
            ),
            30,
        ),
    )

    rtl = is_rtl(
        target_language
    )

    alignment = (
        "right"
        if rtl
        else "left"
    )

    color = pdf_color(
        block.get(
            "color",
            0,
        )
    )

    r = int(
        color[0] * 255
    )

    g = int(
        color[1] * 255
    )

    b = int(
        color[2] * 255
    )

    safe = html_escape_text(
        translated
    )

    html_content = (
        f'<div dir="'
        f'{"rtl" if rtl else "ltr"}'
        f'">'
        f"{safe}"
        f"</div>"
    )

    # Try several font sizes.
    for scale in [
        1.00,
        0.92,
        0.84,
        0.76,
        0.68,
        0.60,
    ]:

        fontsize = max(
            5.5,
            original_size * scale,
        )

        css = f"""
        * {{
            font-family: sans-serif;
            font-size: {fontsize}pt;
            color: rgb({r},{g},{b});
            line-height: 1.15;
            text-align: {alignment};
        }}
        """

        try:

            page.insert_htmlbox(
                rect,
                html_content,
                css=css,
                overlay=True,
                scale_low=0,
            )

            return True

        except Exception:

            continue

    # Fallback.
    try:

        page.insert_textbox(
            rect,
            translated,
            fontsize=max(
                5.5,
                original_size * 0.60,
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
# COVER ORIGINAL TEXT
# ============================================================

def cover_block(
    page,
    block,
    scanned=False,
):

    rect = fitz.Rect(
        block["bbox"]
    )

    rect.x0 -= 0.8
    rect.y0 -= 0.8
    rect.x1 += 0.8
    rect.y1 += 0.8

    if scanned:

        # For scanned pages, the text is part
        # of the image. We therefore place a white
        # rectangle over the OCR text area.
        shape = page.new_shape()

        shape.draw_rect(
            rect
        )

        shape.finish(
            color=None,
            fill=(1, 1, 1),
            fill_opacity=1,
        )

        shape.commit()

    else:

        page.add_redact_annot(
            rect,
            fill=(1, 1, 1),
        )


# ============================================================
# TRANSLATE PDF
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

    total_pages = len(
        document
    )

    progress_bar = st.progress(
        0,
        text="Preparing PDF...",
    )

    status = st.empty()

    memory = TranslationMemory()

    translated_blocks_total = 0
    scanned_pages = 0

    for page_index in range(
        total_pages
    ):

        page = document[
            page_index
        ]

        status.info(
            f"Processing page "
            f"{page_index + 1} "
            f"of {total_pages}"
        )

        # ----------------------------------------------------
        # Native extraction OR OCR
        # ----------------------------------------------------

        blocks, used_ocr = (
            extract_pdf_blocks(
                page
            )
        )

        if used_ocr:

            scanned_pages += 1

            status.info(
                f"Page "
                f"{page_index + 1} "
                f"of {total_pages} "
                f"— OCR processing..."
            )

        # ----------------------------------------------------
        # Nothing recognized
        # ----------------------------------------------------

        if not blocks:

            progress = (
                (page_index + 1)
                / max(
                    total_pages,
                    1,
                )
            )

            progress_bar.progress(
                progress,
                text=(
                    f"Processed "
                    f"{page_index + 1}/"
                    f"{total_pages}"
                ),
            )

            continue

        # ----------------------------------------------------
        # Translate each block
        # ----------------------------------------------------

        translated = [
            None
        ] * len(blocks)

        missing_indices = []

        for i, block in enumerate(
            blocks
        ):

            source = block[
                "text"
            ]

            cached = memory.get(
                source,
                target_language,
                provider_name,
            )

            if cached:

                translated[i] = cached

            else:

                missing_indices.append(
                    i
                )

        for i in missing_indices:

            source = blocks[i][
                "text"
            ]

            chunks = split_text(
                source,
                max_chars=6000,
            )

            parts = (
                translate_chunks_parallel(
                    chunks,
                    target_language,
                    provider_name,
                    workers=workers,
                )
            )

            result = "\n\n".join(
                parts
            )

            translated[i] = result

            memory.set(
                source,
                target_language,
                provider_name,
                result,
            )

        # ----------------------------------------------------
        # Cover original text
        # ----------------------------------------------------

        for block in blocks:

            cover_block(
                page,
                block,
                scanned=used_ocr,
            )

        # Native text redaction annotations
        # must be applied after all annotations
        # have been created.
        if not used_ocr:

            try:

                page.apply_redactions()

            except Exception:

                pass

        # ----------------------------------------------------
        # Insert translations
        # ----------------------------------------------------

        successful = 0

        for i, block in enumerate(
            blocks
        ):

            result = translated[i]

            if not result:
                continue

            ok = (
                insert_translated_block(
                    page,
                    block,
                    result,
                    target_language,
                )
            )

            if ok:
                successful += 1

        translated_blocks_total += (
            successful
        )

        progress = (
            (page_index + 1)
            / max(
                total_pages,
                1,
            )
        )

        progress_bar.progress(
            progress,
            text=(
                f"Processed "
                f"{page_index + 1}/"
                f"{total_pages} pages"
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
        "translated_blocks":
            translated_blocks_total,
        "scanned_pages":
            scanned_pages,
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

    translated = (
        translate_chunks_parallel(
            chunks,
            target_language,
            provider_name,
            workers=workers,
        )
    )

    return "\n\n".join(
        translated
    ).encode(
        "utf-8"
    )


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

    for paragraph in (
        document.paragraphs
    ):

        text = paragraph.text.strip()

        if text:

            paragraphs.append(
                paragraph
            )

    source_texts = [
        p.text
        for p in paragraphs
    ]

    translated = (
        translate_chunks_parallel(
            source_texts,
            target_language,
            provider_name,
            workers=workers,
        )
        if source_texts
        else []
    )

    for paragraph, translation in zip(
        paragraphs,
        translated,
    ):

        paragraph.clear()

        run = paragraph.add_run(
            translation
        )

        if is_rtl(
            target_language
        ):

            paragraph.alignment = 2

    output = io.BytesIO()

    document.save(
        output
    )

    return output.getvalue()


# ============================================================
# FILE ROUTER
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

        result = (
            translate_text_file(
                file_bytes,
                target_language,
                provider_name,
                workers,
            )
        )

        return result, {}

    if extension == "docx":

        result = (
            translate_docx(
                file_bytes,
                target_language,
                provider_name,
                workers,
            )
        )

        return result, {}

    raise ValueError(
        "Supported formats: PDF, DOCX, TXT."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">'
    'Translate Preserv'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Translate documents while preserving '
    'their structure and layout.'
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

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        target_language = (
            st.selectbox(
                "Language",
                LANGUAGES,
                index=1,
            )
        )

    with col2:

        provider_name = (
            st.selectbox(
                "Translation engine",
                list(
                    PROVIDERS.keys()
                ),
            )
        )

    with col3:

        workers = st.slider(
            "Parallel workers",
            min_value=1,
            max_value=8,
            value=2,
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
    [
        "Text",
        "File",
    ],
    horizontal=True,
)


# ============================================================
# TEXT MODE
# ============================================================

if mode == "Text":

    text = st.text_area(
        "Text",
        height=260,
        placeholder=(
            "Write or paste text..."
        ),
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

                translated = (
                    translate_chunks_parallel(
                        chunks,
                        target_language,
                        provider_name,
                        workers=workers,
                    )
                )

                result = (
                    "\n\n".join(
                        translated
                    )
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

        file_bytes = (
            uploaded.getvalue()
        )

        st.markdown(
            f"""
            <div class="status">
            <b>
            {html.escape(uploaded.name)}
            </b>
            &nbsp;·&nbsp;
            {len(file_bytes) / 1024 / 1024:.2f} MB
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # PDF ANALYSIS
        # ----------------------------------------------------

        if uploaded.name.lower().endswith(
            ".pdf"
        ):

            try:

                analysis = analyze_pdf(
                    file_bytes
                )

                c1, c2, c3, c4 = (
                    st.columns(4)
                )

                c1.metric(
                    "Pages",
                    analysis[
                        "pages"
                    ],
                )

                c2.metric(
                    "Native",
                    analysis[
                        "native_pages"
                    ],
                )

                c3.metric(
                    "Scanned",
                    analysis[
                        "scanned_pages"
                    ],
                )

                c4.metric(
                    "Text",
                    f"{analysis['text_chars']:,}",
                )

                if analysis[
                    "scanned_pages"
                ] > 0:

                    st.info(
                        "Scanned pages detected. "
                        "OCR will be used automatically."
                    )

            except Exception as exc:

                st.warning(
                    f"PDF analysis failed: {exc}"
                )

        # ----------------------------------------------------
        # TRANSLATE
        # ----------------------------------------------------

        if st.button(
            "Translate document",
            type="primary",
            use_container_width=True,
        ):

            try:

                result, stats = (
                    translate_file(
                        file_bytes,
                        uploaded.name,
                        target_language,
                        provider_name,
                        workers=workers,
                    )
                )

                base_name = (
                    os.path.splitext(
                        uploaded.name
                    )[0]
                )

                extension = (
                    uploaded.name.lower()
                    .split(".")[-1]
                )

                if extension == "pdf":

                    output_name = (
                        f"{base_name}_"
                        f"{target_language.lower()}"
                        f".pdf"
                    )

                    st.success(
                        "PDF translation completed."
                    )

                    st.info(
                        f"Pages: "
                        f"{stats.get('pages', 0)} · "
                        f"Translated blocks: "
                        f"{stats.get('translated_blocks', 0)} · "
                        f"OCR pages: "
                        f"{stats.get('scanned_pages', 0)}"
                    )

                elif extension == "docx":

                    output_name = (
                        f"{base_name}_"
                        f"{target_language.lower()}"
                        f".docx"
                    )

                    st.success(
                        "DOCX translation completed."
                    )

                else:

                    output_name = (
                        f"{base_name}_"
                        f"{target_language.lower()}"
                        f".txt"
                    )

                    st.success(
                        "Text translation completed."
                    )

                if extension == "pdf":

                    mime = (
                        "application/pdf"
                    )

                elif extension == "docx":

                    mime = (
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    )

                else:

                    mime = "text/plain"

                st.download_button(
                    "Download translated file",
                    data=result,
                    file_name=output_name,
                    mime=mime,
                    use_container_width=True,
                )

            except Exception as exc:

                st.error(
                    f"Translation failed: {exc}"
                )

                st.exception(exc)
