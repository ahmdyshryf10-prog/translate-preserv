import os
import io
import time
import csv
import html
from pathlib import Path

import streamlit as st
from openai import OpenAI

try:
    import fitz
except ImportError:
    fitz = None

try:
    from docx import Document
except ImportError:
    Document = None


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="Translate Preserv",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 2rem;
    }

    .upload-box {
        padding: 1rem;
        border-radius: 16px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    .result-box {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌐 Translate Preserv</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Translate text, documents and long books while preserving structure.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# OPENAI
# =========================================================

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None


# =========================================================
# LANGUAGES
# =========================================================

LANGUAGES = [
    "Persian",
    "English",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Italian",
    "Portuguese",
    "Dutch",
    "Russian",
    "Ukrainian",
    "Polish",
    "Czech",
    "Slovak",
    "Romanian",
    "Hungarian",
    "Greek",
    "Turkish",
    "Hebrew",
    "Urdu",
    "Hindi",
    "Bengali",
    "Indonesian",
    "Malay",
    "Vietnamese",
    "Thai",
    "Chinese Simplified",
    "Chinese Traditional",
    "Japanese",
    "Korean",
    "Swedish",
    "Norwegian",
    "Danish",
    "Finnish",
    "Persian (Afghanistan)",
    "Pashto",
    "Kurdish",
    "Azerbaijani",
    "Armenian",
    "Georgian",
]


# =========================================================
# LANGUAGE DIRECTION
# =========================================================

RTL_LANGUAGES = {
    "Persian",
    "Persian (Afghanistan)",
    "Arabic",
    "Hebrew",
    "Urdu",
    "Pashto",
    "Kurdish",
}


def is_rtl(language):
    return language in RTL_LANGUAGES


# =========================================================
# MODEL
# =========================================================

MODEL = os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-5-mini")


# =========================================================
# TRANSLATION ENGINE
# =========================================================

SYSTEM_PROMPT = """
You are a professional high-quality document translation engine.

Translate accurately and naturally.

Rules:

1. Translate the meaning faithfully.
2. Do not summarize.
3. Do not explain the translation.
4. Do not add commentary.
5. Do not remove information.
6. Preserve names, numbers, dates and technical terminology.
7. Preserve paragraph structure.
8. Preserve line and section markers when supplied.
9. Never invent missing text.
10. Keep the same order as the input.
11. For books, preserve the author's tone and style.
12. Do not translate proper names unless the target language normally requires it.
13. Preserve URLs, email addresses and file names.
14. Preserve placeholders such as {{name}}, [IMAGE], [TABLE], etc.
"""


def translate_text(text, target_language):
    """
    Translate ordinary text.
    """

    if not text or not text.strip():
        return ""

    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Target language: {target_language}\n\n"
                    f"TEXT:\n{text}"
                ),
            },
        ],
    )

    return response.output_text.strip()


# =========================================================
# BATCH TRANSLATION
# =========================================================

def translate_batch(items, target_language):
    """
    Translate multiple numbered blocks in one API request.

    This is much faster than calling the API once for every
    sentence/span.
    """

    if not items:
        return []

    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    joined = []

    for i, text in enumerate(items):
        joined.append(
            f"<<<BLOCK_{i}>>>\n{text}\n<<<END_BLOCK_{i}>>>"
        )

    prompt = f"""
Target language: {target_language}

Translate every block below.

IMPORTANT:
- Return every block.
- Keep the exact block markers.
- Do not merge blocks.
- Do not remove blocks.
- Do not add explanations.

{chr(10).join(joined)}
"""

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    output = response.output_text.strip()

    results = []

    for i in range(len(items)):
        start_marker = f"<<<BLOCK_{i}>>>"
        end_marker = f"<<<END_BLOCK_{i}>>>"

        if start_marker not in output:
            results.append("")
            continue

        part = output.split(start_marker, 1)[1]

        if end_marker in part:
            part = part.split(end_marker, 1)[0]

        results.append(part.strip())

    return results


# =========================================================
# TEXT CHUNKING
# =========================================================

def split_text(text, max_chars=7000):
    """
    Split long text into reasonably sized chunks.
    """

    text = text.replace("\r\n", "\n")

    paragraphs = text.split("\n")

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) + 1 <= max_chars:
            current += paragraph + "\n"

        else:

            if current.strip():
                chunks.append(current.strip())

            current = paragraph + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


# =========================================================
# TXT / MD
# =========================================================

def extract_text_file(data):
    return data.decode("utf-8", errors="replace")


# =========================================================
# DOCX
# =========================================================

def extract_docx(data):

    if Document is None:
        raise RuntimeError(
            "python-docx is not installed."
        )

    document = Document(
        io.BytesIO(data)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(
                paragraph.text
            )

    return "\n\n".join(paragraphs)


# =========================================================
# CSV
# =========================================================

def extract_csv(data):

    text = data.decode(
        "utf-8",
        errors="replace"
    )

    reader = csv.reader(
        io.StringIO(text)
    )

    rows = []

    for row in reader:
        rows.append(
            " | ".join(row)
        )

    return "\n".join(rows)


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_pages(data):

    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is not installed."
        )

    document = fitz.open(
        stream=data,
        filetype="pdf",
    )

    pages = []

    for page in document:

        text = page.get_text(
            "text"
        )

        pages.append(text)

    document.close()

    return pages


# =========================================================
# PDF TRANSLATION
# =========================================================

def translate_pdf(data, target_language):

    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is not installed."
        )

    document = fitz.open(
        stream=data,
        filetype="pdf",
    )

    total_pages = len(document)

    progress = st.progress(
        0,
        text="Preparing PDF..."
    )

    status = st.empty()

    for page_index in range(total_pages):

        page = document[page_index]

        text = page.get_text(
            "text"
        ).strip()

        if not text:
            progress.progress(
                (page_index + 1) / total_pages,
                text=f"Page {page_index + 1}/{total_pages}",
            )
            continue

        chunks = split_text(
            text,
            max_chars=6500
        )

        translated_chunks = []

        # Batch several chunks together
        batch_size = 4

        for start in range(
            0,
            len(chunks),
            batch_size
        ):

            batch = chunks[
                start:start + batch_size
            ]

            translated = translate_batch(
                batch,
                target_language
            )

            translated_chunks.extend(
                translated
            )

        translated_text = "\n\n".join(
            translated_chunks
        )

        # Cover original text with white area.
        # Images remain because we only cover text
        # areas detected by the PDF text layer.
        blocks = page.get_text(
            "blocks"
        )

        text_rects = []

        for block in blocks:

            if len(block) >= 5:

                block_text = str(
                    block[4]
                ).strip()

                if block_text:
                    rect = fitz.Rect(
                        block[:4]
                    )
                    text_rects.append(
                        rect
                    )

        # Redact all text blocks first.
        for rect in text_rects:
            page.add_redact_annot(
                rect,
                fill=(1, 1, 1)
            )

        if text_rects:
            page.apply_redactions()

        # Reinsert translated text.
        #
        # We use the entire page text region rather than
        # translating every tiny span.
        if text_rects:

            union_rect = text_rects[0]

            for rect in text_rects[1:]:
                union_rect |= rect

            margin = 3

            union_rect.x0 += margin
            union_rect.y0 += margin
            union_rect.x1 -= margin
            union_rect.y1 -= margin

            direction = (
                "rtl"
                if is_rtl(target_language)
                else "ltr"
            )

            safe_text = html.escape(
                translated_text
            ).replace(
                "\n",
                "<br>"
            )

            html_content = f"""
            <div dir="{direction}"
                 style="
                    font-family: sans-serif;
                    font-size: 10pt;
                    line-height: 1.35;
                    text-align: {'right' if direction == 'rtl' else 'left'};
                 ">
                {safe_text}
            </div>
            """

            try:

                page.insert_htmlbox(
                    union_rect,
                    html_content,
                    css="",
                )

            except Exception:

                # Fallback for environments where
                # HTML insertion is unavailable.
                page.insert_textbox(
                    union_rect,
                    translated_text,
                    fontsize=9,
                    fontname="helv",
                    color=(0, 0, 0),
                )

        progress.progress(
            (page_index + 1) / total_pages,
            text=f"Translating page {page_index + 1}/{total_pages}",
        )

        status.write(
            f"Page {page_index + 1} of {total_pages}"
        )

    output = document.tobytes(
        garbage=4,
        deflate=True,
    )

    document.close()

    return output


# =========================================================
# GENERAL DOCUMENT TRANSLATION
# =========================================================

def translate_long_text(
    text,
    target_language
):

    chunks = split_text(
        text,
        max_chars=7000
    )

    progress = st.progress(
        0,
        text="Translating..."
    )

    translated = []

    batch_size = 4

    for start in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            start:start + batch_size
        ]

        result = translate_batch(
            batch,
            target_language
        )

        translated.extend(
            result
        )

        completed = min(
            start + batch_size,
            len(chunks)
        )

        progress.progress(
            completed / len(chunks),
            text=f"{completed}/{len(chunks)} sections",
        )

    return "\n\n".join(
        translated
    )


# =========================================================
# MAIN UI
# =========================================================

target_language = st.selectbox(
    "Translate to",
    LANGUAGES,
    index=0,
)


st.markdown("###")


input_mode = st.radio(
    "Input",
    ["Text", "File"],
    horizontal=True,
    label_visibility="collapsed",
)


# =========================================================
# TEXT MODE
# =========================================================

if input_mode == "Text":

    text = st.text_area(
        "Write your text",
        height=250,
        placeholder="Write or paste your text here...",
        label_visibility="collapsed",
    )

    if st.button(
        "Translate",
        type="primary",
        use_container_width=True,
    ):

        if not api_key:
            st.error(
                "OPENAI_API_KEY is not configured."
            )

        elif not text.strip():

            st.warning(
                "Please enter some text."
            )

        else:

            try:

                with st.spinner(
                    "Translating..."
                ):

                    result = translate_long_text(
                        text,
                        target_language,
                    )

                st.success(
                    "Translation completed."
                )

                st.text_area(
                    "Translation",
                    result,
                    height=400,
                )

                st.download_button(
                    "Download TXT",
                    result.encode("utf-8"),
                    file_name="translation.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    f"Translation failed: {e}"
                )


# =========================================================
# FILE MODE
# =========================================================

else:

    uploaded = st.file_uploader(
        "Upload",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "csv",
            "html",
            "htm",
        ],
        label_visibility="collapsed",
    )

    if uploaded:

        file_name = uploaded.name
        extension = Path(
            file_name
        ).suffix.lower()

        st.info(
            f"📄 {file_name}"
        )

        if st.button(
            "Translate file",
            type="primary",
            use_container_width=True,
        ):

            if not api_key:

                st.error(
                    "OPENAI_API_KEY is not configured."
                )

            else:

                try:

                    data = uploaded.read()

                    # -------------------------------------
                    # PDF
                    # -------------------------------------

                    if extension == ".pdf":

                        with st.spinner(
                            "Processing PDF..."
                        ):

                            result = translate_pdf(
                                data,
                                target_language,
                            )

                        st.success(
                            "PDF translation completed."
                        )

                        st.download_button(
                            "Download translated PDF",
                            result,
                            file_name=(
                                Path(file_name).stem
                                + "_translated.pdf"
                            ),
                            mime="application/pdf",
                            use_container_width=True,
                        )

                    # -------------------------------------
                    # DOCX
                    # -------------------------------------

                    elif extension == ".docx":

                        text = extract_docx(
                            data
                        )

                        result = translate_long_text(
                            text,
                            target_language,
                        )

                        st.success(
                            "Translation completed."
                        )

                        st.text_area(
                            "Translation",
                            result,
                            height=450,
                        )

                        st.download_button(
                            "Download translation",
                            result.encode(
                                "utf-8"
                            ),
                            file_name=(
                                Path(file_name).stem
                                + "_translated.txt"
                            ),
                            mime="text/plain",
                            use_container_width=True,
                        )

                    # -------------------------------------
                    # TXT / MD
                    # -------------------------------------

                    elif extension in {
                        ".txt",
                        ".md",
                    }:

                        text = extract_text_file(
                            data
                        )

                        result = translate_long_text(
                            text,
                            target_language,
                        )

                        st.success(
                            "Translation completed."
                        )

                        st.text_area(
                            "Translation",
                            result,
                            height=450,
                        )

                        st.download_button(
                            "Download translation",
                            result.encode(
                                "utf-8"
                            ),
                            file_name=(
                                Path(file_name).stem
                                + "_translated.txt"
                            ),
                            mime="text/plain",
                            use_container_width=True,
                        )

                    # -------------------------------------
                    # CSV
                    # -------------------------------------

                    elif extension == ".csv":

                        text = extract_csv(
                            data
                        )

                        result = translate_long_text(
                            text,
                            target_language,
                        )

                        st.success(
                            "Translation completed."
                        )

                        st.text_area(
                            "Translation",
                            result,
                            height=450,
                        )

                        st.download_button(
                            "Download translation",
                            result.encode(
                                "utf-8"
                            ),
                            file_name=(
                                Path(file_name).stem
                                + "_translated.txt"
                            ),
                            mime="text/plain",
                            use_container_width=True,
                        )

                    # -------------------------------------
                    # HTML
                    # -------------------------------------

                    elif extension in {
                        ".html",
                        ".htm",
                    }:

                        text = data.decode(
                            "utf-8",
                            errors="replace"
                        )

                        result = translate_long_text(
                            text,
                            target_language,
                        )

                        st.success(
                            "Translation completed."
                        )

                        st.text_area(
                            "Translation",
                            result,
                            height=450,
                        )

                        st.download_button(
                            "Download translation",
                            result.encode(
                                "utf-8"
                            ),
                            file_name=(
                                Path(file_name).stem
                                + "_translated.txt"
                            ),
                            mime="text/plain",
                            use_container_width=True,
                        )

                except Exception as e:

                    st.error(
                        f"Translation failed: {e}"
                    )


# =========================================================
# API WARNING
# =========================================================

if not api_key:

    st.caption(
        "API key is not configured. Set OPENAI_API_KEY in the environment."
      )
