import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

import pytesseract
from pdf2image import convert_from_path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

os.environ.setdefault(
    "USER_AGENT",
    "IIUI-Policy-Assistant/1.0"
)

st.set_page_config(
    page_title="IIUI Policy Assistant",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# ENVIRONMENT / API KEY
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning(
        "GROQ_API_KEY is not configured. "
        "Please add it to Streamlit Secrets."
    )


# ============================================================
# CONSTANTS
# ============================================================

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Current Groq-supported model.
GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# ACCESS MATRIX
# ============================================================

ACCESS_MATRIX = {
    "Faculty Member": [
        "Faculty policies",
        "Academic policies",
        "Examination policies",
        "University policies",
        "General University",
    ],
    "Student": [
        "Student policies",
        "Academic policies",
        "Examination policies",
        "University policies",
        "General University",
    ],
    "Outsider / Prospective Student": [
        "Admission policies",
        "General University",
        "University policies",
    ],
}


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_signature" not in st.session_state:
    st.session_state.index_signature = None


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():
    """
    Load the Hugging Face embedding model once and reuse it.

    This does not require an API key.
    The model runs locally on the Streamlit server.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# GROQ LLM
# ============================================================

@st.cache_resource
def get_llm():
    """
    Create the Groq LLM using Groq's OpenAI-compatible API.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return ChatOpenAI(
        model=GROQ_MODEL,
        temperature=0,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


# ============================================================
# OCR FUNCTIONS
# ============================================================

def ocr_pdf(file_path):
    """
    Extract text from a scanned PDF using OCR.
    """

    documents = []

    try:
        images = convert_from_path(
            file_path,
            dpi=300
        )

        for page_number, image in enumerate(images, start=1):

            text = pytesseract.image_to_string(
                image,
                lang="eng"
            )

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "page": page_number,
                            "type": "OCR PDF",
                        },
                    )
                )

    except Exception as e:
        st.error(
            f"OCR failed for PDF {file_path}: {str(e)}"
        )

    return documents


def ocr_image(file_path):
    """
    Extract text from an image using OCR.
    """

    try:

        image = Image.open(file_path)

        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        if not text.strip():
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "page": 1,
                    "type": "OCR Image",
                },
            )
        ]

    except Exception as e:

        st.error(
            f"OCR failed for image {file_path}: {str(e)}"
        )

        return []


# ============================================================
# SCANNED PDF DETECTION
# ============================================================

def is_scanned_pdf(file_path):
    """
    Detect whether a PDF appears to be scanned.

    If extracted text is extremely small, OCR is used.
    """

    try:

        loader = PyPDFLoader(str(file_path))

        documents = loader.load()

        total_text = sum(
            len(document.page_content.strip())
            for document in documents
        )

        return total_text < 100

    except Exception:
        return True


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_document(file_path):
    """
    Load a document based on its file extension.
    """

    extension = Path(file_path).suffix.lower()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == ".pdf":

        if is_scanned_pdf(file_path):

            return ocr_pdf(file_path)

        try:

            loader = PyPDFLoader(
                str(file_path)
            )

            documents = loader.load()

            for document in documents:

                document.metadata.setdefault(
                    "source",
                    str(file_path)
                )

                document.metadata.setdefault(
                    "type",
                    "PDF"
                )

            return documents

        except Exception as e:

            st.warning(
                f"Normal PDF extraction failed. "
                f"Trying OCR: {str(e)}"
            )

            return ocr_pdf(file_path)

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    if extension == ".docx":

        try:

            loader = Docx2txtLoader(
                str(file_path)
            )

            documents = loader.load()

            for document in documents:

                document.metadata.setdefault(
                    "source",
                    str(file_path)
                )

                document.metadata.setdefault(
                    "type",
                    "DOCX"
                )

            return documents

        except Exception as e:

            st.error(
                f"Failed to load DOCX {file_path}: {str(e)}"
            )

            return []

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    if extension == ".txt":

        try:

            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

            documents = loader.load()

            for document in documents:

                document.metadata.setdefault(
                    "source",
                    str(file_path)
                )

                document.metadata.setdefault(
                    "type",
                    "TXT"
                )

            return documents

        except UnicodeDecodeError:

            try:

                loader = TextLoader(
                    str(file_path),
                    encoding="latin-1"
                )

                return loader.load()

            except Exception as e:

                st.error(
                    f"Failed to load TXT {file_path}: {str(e)}"
                )

                return []

        except Exception as e:

            st.error(
                f"Failed to load TXT {file_path}: {str(e)}"
            )

            return []

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if extension in [".png", ".jpg", ".jpeg"]:

        return ocr_image(file_path)

    return []


# ============================================================
# VECTOR STORE
# ============================================================

def build_vectorstore(file_paths, url=None):
    """
    Build FAISS vector store from uploaded documents
    and optionally a web URL.
    """

    documents = []

    # --------------------------------------------------------
    # Load local documents
    # --------------------------------------------------------

    for file_path in file_paths:

        loaded_documents = load_document(
            file_path
        )

        documents.extend(
            loaded_documents
        )

    # --------------------------------------------------------
    # Load URL
    # --------------------------------------------------------

    if url and url.strip():

        try:

            st.info(
                f"Loading URL: {url}"
            )

            loader = WebBaseLoader(
                url.strip()
            )

            url_documents = loader.load()

            for document in url_documents:

                document.metadata.setdefault(
                    "source",
                    url.strip()
                )

                document.metadata.setdefault(
                    "type",
                    "Web Page"
                )

            documents.extend(
                url_documents
            )

        except Exception as e:

            st.error(
                f"Failed to load URL: {str(e)}"
            )

    # --------------------------------------------------------
    # Validate documents
    # --------------------------------------------------------

    if not documents:

        raise ValueError(
            "No readable documents were found."
        )

    # --------------------------------------------------------
    # Default metadata
    # --------------------------------------------------------

    for document in documents:

        document.metadata.setdefault(
            "category",
            "General University"
        )

        document.metadata.setdefault(
            "role_access",
            "Public"
        )

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(
        documents
    )

    if not chunks:

        raise ValueError(
            "No text chunks were generated from the documents."
        )

    # --------------------------------------------------------
    # Hugging Face Embeddings
    # --------------------------------------------------------

    embeddings = get_embeddings()

    # --------------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ============================================================
# SOURCE FORMATTER
# ============================================================

def format_sources(documents):
    """
    Format retrieved documents as source references.
    """

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page"
        )

        if page is not None:

            try:
                page_number = int(page) + 1
            except Exception:
                page_number = page

            sources.append(
                f"{source} - Page {page_number}"
            )

        else:

            sources.append(
                str(source)
            )

    # Remove duplicates while preserving order

    return list(
        dict.fromkeys(sources)
    )


# ============================================================
# POLICY ASSISTANT
# ============================================================

def ask_policy_assistant(
    question,
    vectorstore,
    role
):
    """
    Retrieve relevant policy documents and
    ask Groq to answer using only those documents.
    """

    if vectorstore is None:

        return (
            "Please upload and index policy documents first.",
            []
        )

    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    retrieved_documents = vectorstore.similarity_search(
        question,
        k=4
    )

    if not retrieved_documents:

        return (
            "I could not find relevant information "
            "in the available IIUI policy documents.",
            []
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page"
        )

        category = document.metadata.get(
            "category",
            "General University"
        )

        context_parts.append(
            f"""
SOURCE {index}
Category: {category}
Source: {source}
Page: {page if page is not None else "N/A"}

CONTENT:
{document.page_content}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt_template = """
You are the IIUI Policy Assistant.

Your task is to answer questions using ONLY the
provided IIUI policy context.

User role:
{role}

Important rules:

1. Answer only from the provided policy context.
2. Do not invent or assume university rules.
3. If the answer is not available in the context,
   clearly say that the available policy documents
   do not contain enough information.
4. Give a concise but useful answer.
5. If relevant, mention the source document.
6. Do not claim something is official unless the
   provided context supports it.
7. If the question is unrelated to IIUI policies,
   politely explain that you can only answer
   IIUI policy-related questions.
8. Prefer clear bullet points when explaining
   multiple rules or requirements.

POLICY CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "role",
            "context",
            "question",
        ],
    )

    formatted_prompt = prompt.format(
        role=role,
        context=context,
        question=question,
    )

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    try:

        llm = get_llm()

        response = llm.invoke(
            formatted_prompt
        )

        answer = response.content

    except Exception as e:

        answer = (
            f"Unable to get a response from Groq API.\n\n"
            f"Error: {str(e)}"
        )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = format_sources(
        retrieved_documents
    )

    return answer, sources


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

def save_uploaded_file(uploaded_file):
    """
    Save Streamlit uploaded file to local storage.
    """

    file_path = (
        UPLOAD_DIR /
        uploaded_file.name
    )

    with open(
        file_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return str(file_path)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎓 IIUI Policy Assistant"
)

st.sidebar.markdown(
    "Upload IIUI policy documents and ask questions."
)

st.sidebar.divider()

role = st.sidebar.selectbox(
    "Select your role",
    [
        "Faculty Member",
        "Student",
        "Outsider / Prospective Student",
    ],
)

st.sidebar.divider()

st.sidebar.subheader(
    "Policy Documents"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT or Images",
    type=[
        "pdf",
        "docx",
        "txt",
        "png",
        "jpg",
        "jpeg",
    ],
    accept_multiple_files=True,
)

url = st.sidebar.text_input(
    "Policy URL (optional)",
    placeholder="https://example.com/policy"
)

index_button = st.sidebar.button(
    "📚 Build / Rebuild Index",
    use_container_width=True,
)


# ============================================================
# INDEX DOCUMENTS
# ============================================================

if index_button:

    if not uploaded_files and not url.strip():

        st.sidebar.warning(
            "Please upload at least one document "
            "or provide a URL."
        )

    else:

        try:

            with st.spinner(
                "Processing documents and building index..."
            ):

                file_paths = []

                for uploaded_file in uploaded_files:

                    file_path = save_uploaded_file(
                        uploaded_file
                    )

                    file_paths.append(
                        file_path
                    )

                vectorstore = build_vectorstore(
                    file_paths=file_paths,
                    url=url,
                )

                st.session_state.vectorstore = (
                    vectorstore
                )

                st.session_state.messages = []

                st.session_state.index_signature = (
                    tuple(file_paths),
                    url.strip(),
                )

            st.sidebar.success(
                "Policy index created successfully."
            )

        except Exception as e:

            st.sidebar.error(
                f"Document indexing failed: {str(e)}"
            )


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "🎓 IIUI Policy Assistant"
)

st.caption(
    "Ask questions about IIUI policies, rules and procedures."
)

# ------------------------------------------------------------
# Role information
# ------------------------------------------------------------

allowed_categories = ACCESS_MATRIX.get(
    role,
    []
)

with st.expander(
    f"Current Role: {role}"
):

    st.write(
        "The assistant is configured for this role."
    )

    st.write(
        "Relevant policy categories:"
    )

    for category in allowed_categories:

        st.write(
            f"• {category}"
        )


# ============================================================
# CHECK INDEX
# ============================================================

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload your policy documents from the sidebar "
        "and click **Build / Rebuild Index**."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"• {source}"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask an IIUI policy question..."
)


if question:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        if st.session_state.vectorstore is None:

            answer = (
                "Please upload and index the IIUI "
                "policy documents first."
            )

            sources = []

        else:

            with st.spinner(
                "Searching IIUI policies..."
            ):

                answer, sources = (
                    ask_policy_assistant(
                        question=question,
                        vectorstore=(
                            st.session_state.vectorstore
                        ),
                        role=role,
                    )
                )

        st.markdown(
            answer
        )

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    st.write(
                        f"• {source}"
                    )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "IIUI Policy Assistant • Powered by Groq + "
    "Local Hugging Face Embeddings + FAISS"
)
