import os

import streamlit as st
from dotenv import load_dotenv

# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# ============================================================
# OCR IMPORTS
# ============================================================

import pytesseract
from pdf2image import convert_from_path
from PIL import Image


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()

# Identify requests made by WebBaseLoader
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
# OPENAI API KEY CHECK
# ============================================================

if not os.getenv("OPENAI_API_KEY"):
    st.warning(
        "OPENAI_API_KEY is not configured. "
        "Please add it to your .env file before using "
        "the assistant."
    )


# ============================================================
# 2. SIDEBAR — ROLE SELECTION
# ============================================================

st.sidebar.title("🎓 IIUI Policy Assistant")

st.sidebar.markdown(
    "Ask questions about IIUI policies, academic regulations, "
    "admissions, fees, etc."
)

user_role = st.sidebar.radio(
    "You are:",
    (
        "Faculty Member",
        "Student",
        "Outsider / Prospective Student",
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Knowledge Base")


uploaded_files = st.sidebar.file_uploader(
    "Upload policy documents (PDF, DOCX, TXT, PNG, JPG)",
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


url_input = st.sidebar.text_input(
    "Or add an IIUI URL",
    placeholder="https://www.iiu.edu.pk",
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_signature" not in st.session_state:
    st.session_state.index_signature = None


# ============================================================
# 4. ACCESS CONTROL MATRIX
# ============================================================

ACCESS_MATRIX = {
    "Faculty Member": [
        "Admissions",
        "Fees",
        "Academic",
        "Examination",
        "Student Affairs",
        "Faculty / HR",
        "Scholarships",
        "Internship",
        "Hostel",
        "Transport",
        "Hospital",
        "Departments",
        "General University",
        "Public",
    ],

    "Student": [
        "Admissions",
        "Fees",
        "Academic",
        "Examination",
        "Student Affairs",
        "Scholarships",
        "Internship",
        "Hostel",
        "Transport",
        "Hospital",
        "Departments",
        "General University",
        "Public",
    ],

    "Outsider / Prospective Student": [
        "Admissions",
        "Fees",
        "Public",
        "General University",
        "Scholarships",
        "Hostel",
        "Departments",
    ],
}


# ============================================================
# 5. OCR FUNCTIONS
# ============================================================

def ocr_pdf(file_path):
    """
    Extract text from a scanned PDF using OCR.
    """

    text = ""

    try:
        images = convert_from_path(
            file_path,
            dpi=300,
        )

        for page_number, img in enumerate(
            images,
            start=1,
        ):

            page_text = pytesseract.image_to_string(
                img,
                lang="eng",
            )

            text += (
                f"\n--- Page {page_number} ---\n"
                f"{page_text}"
            )

    except Exception as e:

        st.warning(
            f"OCR failed for {file_path}: {e}"
        )

    return text


def ocr_image(file_path):
    """
    Extract text from an image using OCR.
    """

    try:

        img = Image.open(file_path)

        return pytesseract.image_to_string(
            img,
            lang="eng",
        )

    except Exception as e:

        st.warning(
            f"OCR failed for {file_path}: {e}"
        )

        return ""


def is_scanned_pdf(file_path):
    """
    Return True if the PDF contains almost no
    extractable text.
    """

    try:

        docs = PyPDFLoader(
            file_path
        ).load()

        total_text = "".join(
            document.page_content.strip()
            for document in docs
        )

        return len(total_text) < 100

    except Exception:

        return True


# ============================================================
# 6. DOCUMENT INGESTION
# ============================================================

@st.cache_resource(show_spinner=False)
def build_vectorstore(file_paths, url):
    """
    Load documents, OCR scanned documents,
    split documents into chunks, and create
    a FAISS vector store.
    """

    documents = []
    logs = []

    # --------------------------------------------------------
    # Uploaded documents
    # --------------------------------------------------------

    for path in file_paths:

        extension = (
            path.lower()
            .split(".")[-1]
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if extension == "pdf":

            if is_scanned_pdf(path):

                logs.append(
                    "🔍 Scanned PDF detected: "
                    f"{os.path.basename(path)} — "
                    "running OCR..."
                )

                ocr_text = ocr_pdf(
                    path
                )

                if ocr_text.strip():

                    documents.append(
                        Document(
                            page_content=ocr_text,
                            metadata={
                                "source": path,
                                "type": "scanned_pdf",
                            },
                        )
                    )

            else:

                pdf_documents = (
                    PyPDFLoader(path).load()
                )

                documents.extend(
                    pdf_documents
                )

        # ----------------------------------------------------
        # DOCX
        # ----------------------------------------------------

        elif extension == "docx":

            docx_documents = (
                Docx2txtLoader(path).load()
            )

            documents.extend(
                docx_documents
            )

        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------

        elif extension == "txt":

            txt_documents = (
                TextLoader(
                    path,
                    encoding="utf-8",
                ).load()
            )

            documents.extend(
                txt_documents
            )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        elif extension in [
            "png",
            "jpg",
            "jpeg",
        ]:

            logs.append(
                "🖼️ Image detected: "
                f"{os.path.basename(path)} — "
                "running OCR..."
            )

            ocr_text = ocr_image(
                path
            )

            if ocr_text.strip():

                documents.append(
                    Document(
                        page_content=ocr_text,
                        metadata={
                            "source": path,
                            "type": "image",
                        },
                    )
                )

    # ========================================================
    # URL
    # ========================================================

    if url:

        try:

            url_documents = (
                WebBaseLoader(url).load()
            )

            documents.extend(
                url_documents
            )

        except Exception as e:

            logs.append(
                f"Could not load URL: {e}"
            )

    # ========================================================
    # No documents
    # ========================================================

    if not documents:

        return None, logs

    # ========================================================
    # Split documents
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(
        documents
    )

    # ========================================================
    # Add metadata
    # ========================================================

    for chunk in chunks:

        chunk.metadata.setdefault(
            "category",
            "General University",
        )

        chunk.metadata.setdefault(
            "role_access",
            "Public",
        )

    # ========================================================
    # Create embeddings
    # ========================================================

    embeddings = OpenAIEmbeddings()

    # ========================================================
    # Create FAISS
    # ========================================================

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    return vectorstore, logs


# ============================================================
# 7. UPLOAD SIGNATURE
# ============================================================

def _upload_signature(
    uploaded_files,
    url,
):
    """
    Create a stable signature of current inputs.
    """

    names = tuple(
        sorted(
            file.name
            for file in (
                uploaded_files or []
            )
        )
    )

    return (
        names,
        url or "",
    )


current_sig = _upload_signature(
    uploaded_files,
    url_input,
)


# ============================================================
# 8. PROCESS UPLOADED FILES
# ============================================================

if (
    uploaded_files or url_input
) and (
    current_sig
    != st.session_state.index_signature
):

    os.makedirs(
        "data/documents",
        exist_ok=True,
    )

    saved_paths = []

    # --------------------------------------------------------
    # Save uploaded files
    # --------------------------------------------------------

    for uploaded_file in (
        uploaded_files or []
    ):

        path = os.path.join(
            "data/documents",
            uploaded_file.name,
        )

        with open(
            path,
            "wb",
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        saved_paths.append(
            path
        )

    # --------------------------------------------------------
    # Build vector store
    # --------------------------------------------------------

    with st.spinner(
        "Indexing documents. "
        "This may take time for scanned files..."
    ):

        try:

            vectorstore, logs = (
                build_vectorstore(
                    tuple(saved_paths),
                    url_input,
                )
            )

            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.index_signature = (
                current_sig
            )

            # ------------------------------------------------
            # Logs
            # ------------------------------------------------

            for msg in logs:

                st.sidebar.info(
                    msg
                )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            if vectorstore is not None:

                document_count = len(
                    saved_paths
                )

                url_text = (
                    " + URL"
                    if url_input
                    else ""
                )

                st.sidebar.success(
                    f"✅ Indexed "
                    f"{document_count} "
                    f"document(s)"
                    f"{url_text}"
                )

        except Exception as e:

            st.error(
                "Document indexing failed: "
                f"{e}"
            )


# ============================================================
# 9. RAG PROMPT
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """
You are the IIUI Policy Assistant.

You answer questions using ONLY the IIUI
policy information provided in the context.

User role:
{user_role}

Authorized categories:
{authorized_categories}

IMPORTANT RULES:

1. Do NOT invent IIUI policies.

2. Do NOT use general knowledge to create
   or assume an IIUI policy.

3. If the answer cannot be found in the
   provided context, clearly say:

   "The information was not found in the
   available IIUI policy documents."

4. Prefer the latest and active policy
   information when multiple sources exist.

5. If the context contains a source document
   or page number, mention it.

6. If the policy contains conditions,
   dates, exceptions, eligibility requirements,
   fees, or procedures, mention them accurately.

7. Answer clearly and concisely.

8. Do not claim that a policy exists unless
   it is supported by the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# ============================================================
# 10. CREATE PROMPT
# ============================================================

def get_prompt(user_role):
    """
    Create the RAG prompt for the selected role.
    """

    authorized_categories = ", ".join(
        ACCESS_MATRIX.get(
            user_role,
            ["Public"],
        )
    )

    prompt_text = (
        SYSTEM_PROMPT_TEMPLATE
        .replace(
            "{user_role}",
            user_role,
        )
        .replace(
            "{authorized_categories}",
            authorized_categories,
        )
    )

    return PromptTemplate(
        template=prompt_text,
        input_variables=[
            "context",
            "question",
        ],
    )


# ============================================================
# 11. RAG ANSWER
# ============================================================

def ask_policy_assistant(
    vectorstore,
    user_role,
    user_query,
):
    """
    Retrieve relevant documents from FAISS,
    send the retrieved context to the LLM,
    and return the answer and sources.

    This intentionally does NOT use RetrievalQA.
    """

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
        }
    )

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    source_documents = retriever.invoke(
        user_query
    )

    # --------------------------------------------------------
    # No relevant documents
    # --------------------------------------------------------

    if not source_documents:

        return (
            "The information was not found in the "
            "available IIUI policy documents.",
            [],
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for index, document in enumerate(
        source_documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page",
            "N/A",
        )

        context_parts.append(
            f"""
--- SOURCE {index} ---
Document: {os.path.basename(str(source))}
Page: {page}

{document.page_content}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = get_prompt(
        user_role
    )

    formatted_prompt = prompt.format(
        context=context,
        question=user_query,
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    response = llm.invoke(
        formatted_prompt
    )

    answer = response.content

    return (
        answer,
        source_documents,
    )


# ============================================================
# 12. MAIN CHAT UI
# ============================================================

st.title(
    "🎓 IIUI Policy Assistant"
)

st.caption(
    f"You are logged in as: **{user_role}**"
)

st.markdown(
    "Ask questions about IIUI policies, "
    "academic regulations, admissions, fees "
    "and university procedures."
)


# ============================================================
# 13. DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 14. CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask your question..."
)


if user_query:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_query
        )

    # --------------------------------------------------------
    # Check vector store
    # --------------------------------------------------------

    if (
        st.session_state.vectorstore
        is None
    ):

        warning = (
            "⚠️ Please upload at least one "
            "policy document or add a URL "
            "in the sidebar to begin."
        )

        with st.chat_message(
            "assistant"
        ):

            st.warning(
                warning
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": warning,
            }
        )

    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching IIUI policies..."
            ):

                try:

                    # ------------------------------------------------
                    # Ask assistant
                    # ------------------------------------------------

                    answer, sources = (
                        ask_policy_assistant(
                            st.session_state.vectorstore,
                            user_role,
                            user_query,
                        )
                    )

                    # ------------------------------------------------
                    # Display answer
                    # ------------------------------------------------

                    st.markdown(
                        answer
                    )

                    # ------------------------------------------------
                    # Display sources
                    # ------------------------------------------------

                    if sources:

                        st.markdown(
                            "---"
                        )

                        st.markdown(
                            "**📖 Sources:**"
                        )

                        seen = set()

                        for document in sources:

                            title = document.metadata.get(
                                "source",
                                "Unknown",
                            )

                            page = document.metadata.get(
                                "page",
                                "N/A",
                            )

                            key = (
                                f"{title}-{page}"
                            )

                            if key in seen:
                                continue

                            seen.add(
                                key
                            )

                            st.markdown(
                                f"- "
                                f"`{os.path.basename(str(title))}` "
                                f"— Page {page}"
                            )

                    # ------------------------------------------------
                    # Source text for chat history
                    # ------------------------------------------------

                    if sources:

                        parts = []

                        for document in sources:

                            source = os.path.basename(
                                str(
                                    document.metadata.get(
                                        "source",
                                        "?",
                                    )
                                )
                            )

                            page = document.metadata.get(
                                "page",
                                "?",
                            )

                            source_item = (
                                f"{source} "
                                f"(p.{page})"
                            )

                            if (
                                source_item
                                not in parts
                            ):

                                parts.append(
                                    source_item
                                )

                        source_text = (
                            "\n\n**Sources:** "
                            + ", ".join(parts)
                        )

                    else:

                        source_text = ""

                    # ------------------------------------------------
                    # Save assistant response
                    # ------------------------------------------------

                    full_response = (
                        answer
                        + source_text
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                        }
                    )

                except Exception as e:

                    error_message = (
                        f"Error: {e}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )


# ============================================================
# 15. FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "MVP v1.0 — Grounded answers with citations. "
    "Built with Streamlit + LangChain + FAISS + OCR."
)
