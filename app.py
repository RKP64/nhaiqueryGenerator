import re
import streamlit as st
import pypdfium2 as pdfium
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os 

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS, AzureSearch
from langchain_core.messages import HumanMessage

# ─── AZURE CONFIG: Read from st.secrets for Streamlit Cloud deployment ────────
try:
    AZURE_API_TYPE = st.secrets["AZURE_API_TYPE"]
    AZURE_API_BASE = st.secrets["AZURE_API_BASE"]
    AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
    AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]

    CHAT_DEPLOYMENT = st.secrets["CHAT_DEPLOYMENT"]
    EMBEDDING_DEPLOYMENT = st.secrets["EMBEDDING_DEPLOYMENT"]

    AZURE_SEARCH_SERVICE_ENDPOINT = st.secrets["AZURE_SEARCH_SERVICE_ENDPOINT"]
    AZURE_SEARCH_ADMIN_KEY = st.secrets["AZURE_SEARCH_ADMIN_KEY"]
    AZURE_SEARCH_INDEX_NAME = st.secrets["AZURE_SEARCH_INDEX_NAME"]
except KeyError as e:
    st.error(f"Configuration Error: Missing Streamlit secret: '{e}'. Please ensure all Azure credentials are set in your Streamlit Cloud app's secrets.")
    st.stop()


# ─── INIT EMBEDDINGS & LLM ─────────────────────────────────────────────────────
# Corrected: Removed extra 'Open' in AzureOpenAIEmbeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_API_BASE,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    # For text-embedding-3-large, default dimensions are 3072.
    # If your Azure Search index has a specific vector dimension configured (e.g., 1536),
    # you can uncomment and set it here to truncate the embeddings:
    # dimensions=1536 # Example
)

llm = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_API_BASE,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    temperature=0.0,
    max_tokens=500,
    verbose=False,
)

# ─── STREAMLIT SETUP ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KPMG RFP Query Assistant", page_icon="KPMG_logo.png", layout="wide"
)

# Header with KPMG logo
col1, col2 = st.columns([1, 8], gap="small")
with col1:
    try:
        # For Streamlit Cloud, ensure 'KPMG_logo.png' is in the same directory
        # as app.py or provide a full path if it's elsewhere in the repo.
        logo = Image.open("KPMG_logo.png")
        st.image(logo, width=80) # Adjusted width for better fit, original was 400
    except FileNotFoundError:
        st.image("https://via.placeholder.com/80", width=80, caption="Logo") # Fallback
with col2:
    st.title("📄 KPMG RFP Query Assistant")

# Initialize session-state defaults for persistent data across Streamlit reruns
for key in ("full_text", "chunks", "vector_store", "index_built", "pdf_bytes"):
    st.session_state.setdefault(key, None)
st.session_state.setdefault("main_app_mode", "Runtime Analysis") # Default to Runtime Analysis
st.session_state.setdefault("last_app_mode", None) # Track last mode for reset logic

# ─── SIDEBAR: MAIN APP SELECTION ─────────────────────────────────────────────
st.sidebar.title("App Selection")
current_main_app_mode = st.sidebar.radio(
    "Choose Application Mode:",
    ("Historical Analysis", "Runtime Analysis"),
    index=0 if st.session_state.main_app_mode == "Historical Analysis" else 1,
    key="main_app_mode_selector"
)

# Check if app mode has changed, and reset relevant session states if so
if st.session_state.last_app_mode != current_main_app_mode:
    st.session_state.full_text = None
    st.session_state.chunks = None
    st.session_state.vector_store = None
    st.session_state.index_built = False
    st.session_state.pdf_bytes = None
    st.session_state.last_app_mode = current_main_app_mode
    st.rerun() # Rerun to apply state changes immediately

st.session_state.main_app_mode = current_main_app_mode


# ─── CONDITIONAL SIDEBAR CONTENT BASED ON APP MODE ───────────────────────────
excel_file = st.sidebar.file_uploader(
    "Upload Bidder Queries (CSV/XLSX)", type=["csv", "xlsx"]
)

# Variables for controlling UI visibility later
rfp_size_mode = None
show_excerpts = False
viz_option = "None"
current_vector_store_type = None


if st.session_state.main_app_mode == "Historical Analysis":
    # ─── HISTORICAL ANALYSIS MODE (Azure AI Search) ──────────────────────────
    st.sidebar.subheader("Historical Analysis Settings")
    st.sidebar.info(
        "This mode queries a pre-existing Azure AI Search index. "
        "It assumes the index has been populated from a backend process."
    )
    current_vector_store_type = "Cloud Database (Azure AI Search)"

    # Initialize Azure Search store for Historical Analysis
    if st.session_state.vector_store is None or not isinstance(st.session_state.vector_store, AzureSearch):
        try:
            st.session_state.vector_store = AzureSearch(
                azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
                azure_search_key=AZURE_SEARCH_ADMIN_KEY, # Admin key needed for index operations, or query key for just search
                index_name=AZURE_SEARCH_INDEX_NAME,
                embedding_function=embeddings.embed_query,
                vector_key="contentVector" # Corrected vector field name based on image
            )
            st.session_state.index_built = True # Assume index exists and is ready
            st.sidebar.success(f"Connected to cloud database '{AZURE_SEARCH_INDEX_NAME}'.")
        except Exception as e:
            st.sidebar.error(f"Failed to connect to Azure AI Search: {e}. Check Streamlit secrets and index availability.")
            st.session_state.index_built = False
    else:
        st.session_state.index_built = True # If already connected from a previous run

    show_excerpts = st.sidebar.checkbox("🔍 Show Azure Search Excerpts", value=False, key="historical_show_excerpts")
    viz_option = st.sidebar.selectbox(
        "Visualization",
        ["None", "Bar chart of response lengths", "Pie chart of answer coverage"],
        key="historical_viz_option"
    )

elif st.session_state.main_app_mode == "Runtime Analysis":
    # ─── RUNTIME ANALYSIS MODE (FAISS) ───────────────────────────────────────
    st.sidebar.subheader("Runtime Analysis Settings")
    pdf_file_uploader = st.sidebar.file_uploader("Upload Tender PDF", type="pdf", key="runtime_pdf_uploader")

    # Handle PDF upload and state management for Runtime Analysis
    if pdf_file_uploader is not None:
        st.session_state.pdf_bytes = pdf_file_uploader.read()
    else:
        st.session_state.pdf_bytes = None
        st.session_state.full_text = None
        st.session_state.chunks = None
        st.session_state.vector_store = None
        st.session_state.index_built = False

    if st.session_state.pdf_bytes:
        try:
            pdf_document = pdfium.PdfDocument(st.session_state.pdf_bytes)
            num_pages = len(pdf_document)
            pdf_document.close()
            rfp_size_mode = "Large Size RFP" if num_pages >= 25 else "Small Size RFP"
            st.sidebar.markdown(f"📄 **Detected RFP Size:** `{rfp_size_mode}` ({num_pages} pages)")
        except pdfium.PdfiumError as e:
            st.error(f"Error loading PDF: {e}. Please ensure the PDF is not corrupted or malformed.")
            st.session_state.pdf_bytes = None
            rfp_size_mode = None
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred while processing the PDF: {e}")
            st.session_state.pdf_bytes = None
            rfp_size_mode = None
            st.stop()
    else:
        rfp_size_mode = None

    current_vector_store_type = "Local Database (FAISS)"

    if st.session_state.pdf_bytes and rfp_size_mode == "Large Size RFP":
        st.sidebar.markdown(f"Using **{current_vector_store_type}** for indexing.")
        cs = st.sidebar.slider("Chunk size", 200, 2000, 600, help="Chars per chunk", key="faiss_chunk_size")
        co = st.sidebar.slider(
            "Chunk overlap", 50, 1000, 150, help="Overlap between chunks", key="faiss_chunk_overlap"
        )

        if st.sidebar.button(f"Build {current_vector_store_type} Index", key="build_faiss_index_button"):
            with st.spinner(f"Processing PDF and building {current_vector_store_type} index..."):
                CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)")
                pages = []
                pdf_document = pdfium.PdfDocument(st.session_state.pdf_bytes)
                for i in range(len(pdf_document)):
                    page = pdf_document.get_page(i)
                    text_page = page.get_textpage()
                    text = text_page.get_text_range() or ""
                    text_page.close()
                    page.close()
                    
                    clause = None
                    for line in text.splitlines():
                        m = CLAUSE_RE.match(line.strip())
                        if m:
                            clause = m.group(1)
                            break
                    pages.append(
                        Document(page_content=text, metadata={"page": i + 1, "clause": clause})
                    )
                pdf_document.close()

                splitter = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=co)
                chunks = splitter.split_documents(pages)
                st.session_state.chunks = chunks
                st.session_state.full_text = "".join(d.page_content for d in pages)

                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.index_built = True
                st.sidebar.success(f"Built FAISS index over {len(chunks)} chunks.")

        show_excerpts = st.sidebar.checkbox("🔍 Show FAISS Excerpts", value=False, key="faiss_show_excerpts")
        viz_option = st.sidebar.selectbox(
            "Visualization",
            ["None", "Bar chart of response lengths", "Pie chart of answer coverage"],
            key="faiss_viz_option"
        )
    else:
        st.info("Upload a PDF to begin Runtime Analysis. Large PDFs will use FAISS for indexing.")
        show_excerpts = False
        viz_option = "None"


# ─── MAIN PANEL: INITIAL CHECKS ───────────────────────────────────────────────
if st.session_state.main_app_mode == "Historical Analysis":
    if not st.session_state.index_built:
        st.info("Attempting to connect to cloud database (Azure AI Search). Please ensure credentials are set correctly and the index is available.")
        st.stop()
    st.success("Ready for Historical Analysis queries using Azure AI Search.")

elif st.session_state.main_app_mode == "Runtime Analysis":
    if not st.session_state.pdf_bytes:
        st.info("Please upload a tender PDF to begin Runtime Analysis.")
        st.stop()
    
    if rfp_size_mode == "Small Size RFP" and st.session_state.full_text is None:
        try:
            reader = pdfium.PdfDocument(st.session_state.pdf_bytes)
            full_text_list = []
            for i in range(len(reader)):
                page = reader.get_page(i)
                text_page = page.get_textpage()
                full_text_list.append(text_page.get_text_range() or "")
                text_page.close()
                page.close()
            reader.close()
            st.session_state.full_text = "".join(full_text_list)
            st.success(f"Loaded full PDF text ({len(st.session_state.full_text):,} chars).")
        except pdfium.PdfiumError as e:
            st.error(f"Error extracting text from PDF: {e}. The PDF may be unreadable.")
            st.session_state.full_text = None
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during text extraction: {e}")
            st.session_state.full_text = None
            st.stop()

    if rfp_size_mode == "Large Size RFP" and not st.session_state.index_built:
        st.info(f"In sidebar: choose chunk settings and click **Build {current_vector_store_type} Index**.")
        st.stop()


# ─── ANSWER HELPERS ────────────────────────────────────────────────────────────
def _format_references(docs):
    """Helper function to extract and format references from retrieved documents."""
    refs = []
    if docs:
        top_meta = docs[0].metadata
        clause = top_meta.get("clause")
        page = top_meta.get("page")
        if clause:
            refs.append(f"Clause {clause}")
        if page:
            refs.append(f"pg {page}")
    return f" ({', '.join(refs)})" if refs else ""


def _display_excerpts(docs):
    """Helper function to display retrieved document excerpts."""
    if st.session_state.main_app_mode == "Historical Analysis":
        current_show_excerpts = st.session_state.get('historical_show_excerpts', False)
    else:
        current_show_excerpts = st.session_state.get('faiss_show_excerpts', False)

    if current_show_excerpts:
        st.subheader("Relevant Excerpts from RFP:")
        if not docs:
            st.info("No relevant excerpts found for this query.")
            return

        for i, d in enumerate(docs, start=1):
            meta = d.metadata
            clause_info = f"Clause {meta.get('clause')}, " if meta.get('clause') else ""
            page_info = f"pg {meta.get('page')}" if meta.get('page') else ""
            st.markdown(
                f"**Excerpt {i} ({clause_info}{page_info}):** \n"
                f"```\n{d.page_content[:300]}...\n```"
            )
            st.divider()


def answer_direct(q: str) -> str:
    """Answers a query using the full PDF text (for Small Size RFP in Runtime Analysis)."""
    prompt = (
        "You are KPMG Tender Authority. Using *only* the provided RFP document, "
        "answer in ≤20 words, formal tone. If not specified, reply ‘Not specified in the RFP.’\n\n"
        f"RFP Document:\n{st.session_state.full_text}\n\n"
        f"Query: {q}\n"
        f"Answer:"
    )
    return llm([HumanMessage(content=prompt)]).content.strip()


def answer_faiss(q: str) -> str:
    """Answers a query using retrieved chunks from FAISS vector store (for Runtime Analysis)."""
    docs = st.session_state.vector_store.similarity_search(q, k=3)
    _display_excerpts(docs)

    excerpt_text = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "You are KPMG Tender Authority. Using *only* the provided RFP excerpts, "
        "answer in ≤20 words, formal tone. If the information is not present in the excerpts, reply ‘Not specified in the RFP.’\n\n"
        f"RFP Excerpts:\n{excerpt_text}\n\n"
        f"Query: {q}\n"
        f"Answer:"
    )
    ans = llm([HumanMessage(content=prompt)]).content.strip()
    return f"{ans}{_format_references(docs)}"


def answer_azure_search(q: str) -> str:
    """Answers a query using retrieved chunks from Azure AI Search vector store with semantic ranking (for Historical Analysis)."""
    docs = []
    try:
        # Corrected: Using the specific semantic configuration name and vector_key is set in init
        docs = st.session_state.vector_store.similarity_search(
            q,
            k=3,
            search_type="semantic_hybrid",
            semantic_configuration_name="azureml-default" # Set to the confirmed semantic config name
        )
    except Exception as e:
        st.warning(f"Azure AI Search retrieval failed with Semantic Hybrid. Falling back to Hybrid search. Error: {e}")
        try:
            # Fallback to Hybrid search, vector_key is already set during initialization
            docs = st.session_state.vector_store.similarity_search(q, k=3, search_type="hybrid")
        except Exception as e_hybrid:
            st.warning(f"Hybrid search also failed. Falling back to basic similarity search. Error: {e_hybrid}")
            # Fallback to basic similarity, vector_key is already set
            docs = st.session_state.vector_store.similarity_search(q, k=3, search_type="similarity")
    
    _display_excerpts(docs)

    excerpt_text = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "You are KPMG Tender Authority. Using *only* the provided RFP excerpts, "
        "answer in ≤20 words, formal tone. If the information is not present in the excerpts, reply ‘Not specified in the RFP.’\n\n"
        f"RFP Excerpts:\n{excerpt_text}\n\n"
        f"Query: {q}\n"
        f"Answer:"
    )
    ans = llm([HumanMessage(content=prompt)]).content.strip()
    return f"{ans}{_format_references(docs)}"


# ─── QUERY UPLOAD & RESPONSES ─────────────────────────────────────────────────
if excel_file:
    df = (
        pd.read_csv(excel_file)
        if excel_file.name.lower().endswith(".csv")
        else pd.read_excel(excel_file)
    )
    if "Query" not in df.columns:
        st.error("Your file must contain a column named **Query**.")
        st.stop()

    st.subheader("Loaded Bidder Queries")
    st.dataframe(df, use_container_width=True)

    # Store current show_excerpts state in session for _display_excerpts helper
    # This must be done after show_excerpts is set in the conditional blocks above
    if st.session_state.main_app_mode == "Historical Analysis":
        st.session_state.show_excerpts_current_mode = st.session_state.get('historical_show_excerpts', False)
    else: # Runtime Analysis
        st.session_state.show_excerpts_current_mode = st.session_state.get('faiss_show_excerpts', False)


    if st.button("Generate Responses"):
        answer_func = None
        if st.session_state.main_app_mode == "Historical Analysis":
            if not st.session_state.index_built:
                st.error("Cloud database is not connected. Please check sidebar for connection status.")
                st.stop()
            answer_func = answer_azure_search
        elif st.session_state.main_app_mode == "Runtime Analysis":
            if rfp_size_mode == "Small Size RFP":
                answer_func = answer_direct
            elif rfp_size_mode == "Large Size RFP":
                if not st.session_state.index_built:
                    st.error("FAISS index is not built. Please build it in the sidebar.")
                    st.stop()
                answer_func = answer_faiss
            else:
                st.error("Cannot determine appropriate answering method for Runtime Analysis. Please ensure a PDF is uploaded and index is built if it's a Large RFP.")
                st.stop()
        
        if answer_func:
            with st.spinner("Answering queries... This may take a moment for large files or cloud services."):
                df["Response"] = (
                    df["Query"]
                    .fillna("")
                    .apply(answer_func)
                )

            st.subheader("Generated Responses")
            st.dataframe(df, use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Responses CSV", csv_out, "responses.csv", "text/csv")

            st.subheader("Response Analytics")
            # Get the correct viz_option based on the active mode
            # viz_option is set locally in the conditional blocks above (faiss_viz_option or historical_viz_option)
            current_viz_option = viz_option 

            if current_viz_option == "Bar chart of response lengths":
                lengths = df["Response"].str.len().fillna(0)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(df.index.astype(str), lengths, color='skyblue')
                ax.set_xlabel("Query Index")
                ax.set_ylabel("Response Length (characters)")
                ax.set_title("Length of Generated Responses per Query")
                st.pyplot(fig)

            elif current_viz_option == "Pie chart of answer coverage":
                covered_counts = (
                    df["Response"]
                    .str.strip()
                    .astype(bool)
                    .value_counts()
                    .reindex([True, False], fill_value=0)
                )
                
                labels = ["Answered", "Blank/Unanswered"]
                sizes = [covered_counts[True], covered_counts[False]]
                
                if sum(sizes) == 0:
                    st.info("No responses to visualize yet or all responses are empty.")
                else:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=['#66b3ff','#99ff99'])
                    ax.axis('equal')
                    ax.set_title("Proportion of Answered vs. Unanswered Queries")
                    st.pyplot(fig)
        else:
            st.error("An internal error occurred: No answering function determined.")
else:
    st.info("Please upload your CSV/XLSX of queries to generate responses.")
