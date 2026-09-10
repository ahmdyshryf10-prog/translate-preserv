import os
import fitz
import streamlit as st
from openai import OpenAI


st.set_page_config(
    page_title="Translate Preserv",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Translate Preserv")
st.write(
    "AI document translator that keeps the original PDF layout, "
    "images, tables and formatting as much as possible."
)


# -----------------------------
# OpenAI
# -----------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.warning(
        "OPENAI_API_KEY is not configured yet. "
        "The application will need an API key before translation."
    )

client = OpenAI(api_key=api_key) if api_key else None


# -----------------------------
# Translation
# -----------------------------

def translate_text(text, target_language):
    if not text.strip():
        return text

    if client is None:
        return text

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a professional document translator. "
                    "Translate the user's text accurately. "
                    "Do not explain anything. "
                    "Do not summarize. "
                    "Keep numbers, names, technical terms and "
                    "structure whenever possible."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Translate the following text into "
                    f"{target_language}:\n\n{text}"
                )
            }
        ]
    )

    return response.output_text.strip()


# -----------------------------
# PDF translation
# -----------------------------

def translate_pdf(pdf_bytes, target_language):

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    progress = st.progress(0)
    total_pages = len(document)

    for page_number, page in enumerate(document):

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:

            if "lines" not in block:
                continue

            for line in block["lines"]:

                for span in line["spans"]:

                    original_text = span["text"].strip()

                    if not original_text:
                        continue

                    translated_text = translate_text(
                        original_text,
                        target_language
                    )

                    if not translated_text:
                        continue

                    rect = fitz.Rect(span["bbox"])

                    # Remove original text
                    page.add_redact_annot(rect)

                    # Add translated text
                    page.apply_redactions()

                    page.insert_textbox(
                        rect,
                        translated_text,
                        fontsize=max(5, span["size"]),
                        fontname="helv",
                        color=(0, 0, 0),
                        align=0
                    )

        progress.progress((page_number + 1) / total_pages)

    output = document.tobytes()
    document.close()

    return output


# -----------------------------
# User interface
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

target_language = st.selectbox(
    "Target language",
    [
        "Persian",
        "English",
        "Arabic",
        "German",
        "French",
        "Spanish",
        "Turkish"
    ]
)


if uploaded_file:

    st.success(
        f"File selected: {uploaded_file.name}"
    )

    if st.button("🚀 Translate PDF"):

        if not api_key:
            st.error(
                "OPENAI_API_KEY is missing. "
                "Configure it before translating."
            )

        else:

            with st.spinner("Translating document..."):

                try:

                    result = translate_pdf(
                        uploaded_file.read(),
                        target_language
                    )

                    st.success("Translation completed!")

                    st.download_button(
                        label="⬇️ Download translated PDF",
                        data=result,
                        file_name="translated.pdf",
                        mime="application/pdf"
                    )

                except Exception as e:

                    st.error(
                        f"Translation failed: {str(e)}"
                    )
