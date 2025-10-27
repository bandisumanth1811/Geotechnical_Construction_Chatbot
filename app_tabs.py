# app.py — four-tab layout (Chat, Photos, Knowledge Base, Settings)

import os, json, shutil
from datetime import datetime
from typing import List
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import streamlit.components.v1 as components

# --- RAG deps ---
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------------- Config ----------------
load_dotenv()
st.set_page_config(page_title="AI in Geotechnical Construction", page_icon="🏛️", layout="wide")

VECTORSTORE_DIR = "vectorstore"
METADATA_PATH   = os.path.join(VECTORSTORE_DIR, "metadata.json")
PHOTOS_DIR      = "photos"
DEEP_FOUNDATION_DIR = os.path.join(os.path.dirname(__file__), "Deep_foundation")
SHALLOW_FOUNDATION_DIR = os.path.join(os.path.dirname(__file__), "Shallow_foundation")

# Create directories if they don't exist
os.makedirs(DEEP_FOUNDATION_DIR, exist_ok=True)
os.makedirs(SHALLOW_FOUNDATION_DIR, exist_ok=True)

MODEL_NAME      = "gpt-4o-mini"
TEMPERATURE     = 0.0

PRIMARY_BLUE = "#0F4C81"
SUNY_GOLD    = "#FFC72C"
APP_BG       = "#F4F6FA"
PANEL_BG     = "#ffffff"

# ---------------- Utilities ----------------
def _load_faiss(path: str, embeddings: OpenAIEmbeddings) -> FAISS:
    try:
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except TypeError:
        return FAISS.load_local(path, embeddings)

def list_pdfs_in_cwd() -> List[str]:
    return [f for f in sorted(os.listdir()) if os.path.isfile(f) and f.lower().endswith(".pdf")]

def list_images() -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    imgs = [os.path.join(PHOTOS_DIR, f) for f in sorted(os.listdir(PHOTOS_DIR)) if f.lower().endswith(exts)]
    return imgs

def list_images_from_dir(directory: str) -> List[str]:
    """List all images from a specific directory."""
    exts = (".png", ".jpg", ".jpeg", ".webp")
    if not os.path.exists(directory):
        return []
    imgs = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.lower().endswith(exts)]
    return imgs

def fix_image_orientation(img: Image.Image) -> Image.Image:
    """Fix image orientation based on EXIF data to prevent unwanted rotation."""
    try:
        # Get EXIF data
        exif = img._getexif()
        if exif is not None:
            # EXIF orientation tag
            orientation_key = 274  # EXIF orientation tag key
            if orientation_key in exif:
                orientation = exif[orientation_key]
                
                # Apply rotation based on orientation
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # No EXIF data or orientation info
        pass
    
    return img

def load_docs_from_files(pdf_files):
    docs = []
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": os.path.basename(pdf), "page": i + 1}))
        except Exception as e:
            st.warning(f"Could not read {pdf}: {e}")
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def build_or_load_vectorstore(api_key: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    if os.path.isdir(VECTORSTORE_DIR):
        return _load_faiss(VECTORSTORE_DIR, embeddings)

    pdfs = list_pdfs_in_cwd()
    if not pdfs:
        return None

    docs = load_docs_from_files(pdfs)
    if not docs:
        return None

    chunks = split_docs(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vs.save_local(VECTORSTORE_DIR)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({"built_at": datetime.now().isoformat(timespec="seconds"), "pdf_files": pdfs}, f, indent=2)
    return vs

def make_chain(vectorstore, api_key: str):
    llm = ChatOpenAI(temperature=TEMPERATURE, model=MODEL_NAME, api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
    )

def get_api_key() -> str:
    # precedence: session override -> env var
    key = st.session_state.get("OPENAI_API_KEY_OVERRIDE", "") or os.getenv("OPENAI_API_KEY", "")
    return key.strip()

def ensure_conversation():
    """Initialize RAG chain when possible."""
    if st.session_state.get("conversation"):
        return
    api_key = get_api_key()
    if not api_key:
        st.session_state["rag_status"] = "missing_key"
        return
    vs = build_or_load_vectorstore(api_key)
    if vs is None:
        st.session_state["rag_status"] = "no_pdfs"
        return
    st.session_state["conversation"] = make_chain(vs, api_key)
    st.session_state["rag_status"] = "ready"

def tidy_response(text: str) -> str:
    """Cleans and formats model output for better Markdown readability."""
    if not text:
        return ""
    text = text.strip()
    # Ensure paragraphs have a blank line
    text = text.replace("\n-", "\n\n-")
    text = text.replace("** ", "**")
    text = text.replace(" **", "**")
    return text

# ---------------- Session ----------------
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation" not in st.session_state: st.session_state.conversation = None
if "rag_status" not in st.session_state: st.session_state.rag_status = "init"
if "input_key" not in st.session_state: st.session_state.input_key = 0

# ---------------- Styles / Header ----------------
st.markdown(
    f"""
    <style>
    .stApp {{ background:{APP_BG}; }}
    .app-card {{ background:{PANEL_BG}; border-radius:18px; box-shadow:0 8px 24px rgba(16,24,40,.08);
                 padding:0; overflow:hidden; border:1px solid rgba(16,24,40,.06); }}
    .headerband {{ background:{PRIMARY_BLUE}; color:white; border-radius:12px; padding:18px 22px; margin-bottom:14px; }}
    .title {{ font-size:26px; font-weight:800; letter-spacing:-.02em; }}
    .subtitle {{ opacity:.95; font-size:13px; margin-top:4px; }}
    .bubble {{ border-radius:12px; padding:14px 16px; background:#F2F6FB; border:1px solid #E7ECF3; color:#111827; margin:10px 0 0 0; }}
    .bubble.alt {{ background:#F7F9FB; }}
    .user-pill {{ display:inline-block; padding:4px 10px; border-radius:999px; background:#EEF2F7; border:1px solid #E3E8EF; color:#495567; font-size:12px; font-weight:600; margin-right:8px; }}
    .gallery-img {{ cursor: pointer; border-radius: 8px; transition: transform 0.2s; }}
    .gallery-img:hover {{ transform: scale(1.05); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    .section-divider {{ border-top: 2px solid {PRIMARY_BLUE}; margin: 20px 0; }}
    
    /* ChatGPT-like chat interface */
    .chat-messages-container {{
        height: 500px;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 20px;
        margin-bottom: 20px;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        background: #FAFBFC;
    }}
    .chat-messages-container::-webkit-scrollbar {{
        width: 8px;
    }}
    .chat-messages-container::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 4px;
    }}
    .chat-messages-container::-webkit-scrollbar-thumb {{
        background: #888;
        border-radius: 4px;
    }}
    .chat-messages-container::-webkit-scrollbar-thumb:hover {{
        background: #555;
    }}
    .fixed-input-container {{
        position: relative;
        background: white;
        padding: 16px 20px;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }}
    .chat-message {{
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
    }}
    .chat-message.user {{
        align-items: flex-end;
    }}
    .chat-message.assistant {{
        align-items: flex-start;
    }}
    .message-content {{
        max-width: 80%;
        padding: 16px 20px;
        border-radius: 18px;
        line-height: 1.6;
        word-wrap: break-word;
    }}
    .message-content.user {{
        background: {PRIMARY_BLUE};
        color: white;
        border-bottom-right-radius: 4px;
    }}
    .message-content.assistant {{
        background: #F7F9FC;
        color: #1F2937;
        border: 1px solid #E5E7EB;
        border-bottom-left-radius: 4px;
    }}
    .message-label {{
        font-size: 12px;
        font-weight: 600;
        color: #6B7280;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .chat-input-area {{
        padding: 20px 24px;
        background: white;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-shrink: 0;
    }}
    .welcome-message {{
        text-align: center;
        padding: 60px 20px;
        color: #6B7280;
    }}
    .welcome-message h3 {{
        color: {PRIMARY_BLUE};
        margin-bottom: 12px;
        font-size: 24px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="headerband">
      <div class="title">AI in Geotechnical Construction</div>
      <div class="subtitle">SUNY Polytechnic Institute • Intelligent Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Tabs ----------------
tab_chat, tab_photos, tab_kb, tab_references, tab_settings = st.tabs(
    ["💬 Chat", "🖼️ Photos", "🧠 Knowledge Base", "🔗 References", "⚙️ Settings"]
)

# --- Chat Tab ---
with tab_chat:
    # RAG auto-init
    ensure_conversation()
    
    # Scrollable chat messages container
    st.markdown('<div class="app-card" style="padding:20px;">', unsafe_allow_html=True)
    
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("👋 Hello! I can help answer questions about geotechnical construction. Ask me anything!")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    st.markdown('</div>', unsafe_allow_html=True)
    
    ## messages_html = ""
    # if not st.session_state.messages:
    #     # Welcome message
    #     messages_html = """
    #         <div class="welcome-message">
    #             <h3>👋 Welcome to AI Geotechnical Assistant</h3>
    #             <p>Ask me anything about soil mechanics, foundations, construction methods, and more.</p>
    #             <p style="font-size: 14px; margin-top: 12px;">💡 Try asking: "What are the types of shallow foundations?"</p>
    #         </div>
    #     """
    # else:
    #     # Build message history HTML with inline styles
    #     for msg in st.session_state.messages:
    #         if msg["role"] == "user":
    #             messages_html += f"""
    #                 <div class="chat-message" style="font-size: 16px; line-height: 1.6; margin-bottom: 8px;">
    #                     <span style="font-weight: 700; color: #DC2626; margin-right: 6px;">You:</span>
    #                     <span style="color: #000000;">{msg["content"]}</span>
    #                 </div>
    #             """
    #         else:
    #             messages_html += f"""
    #                 <div class="chat-message" style="font-size: 16px; line-height: 1.6; margin-bottom: 8px;">
    #                     <span style="font-weight: 700; color: #2563EB; margin-right: 6px;">Assistant:</span>
    #                     <span style="color: #000000;">{msg["content"]}</span>
    #                 </div>
    #                 <div style="height: 24px;"></div>
    #             """
    
    # # Display all messages in scrollable container using HTML component
    # chat_html = f"""
    # <style>
    # {open('').read() if False else '''
    # .chat-messages-container {{
    #     height: 500px;
    #     overflow-y: auto;
    #     overflow-x: hidden;
    #     padding: 20px;
    #     border: 2px solid #E5E7EB;
    #     border-radius: 12px;
    #     background: #FAFBFC;
    # }}
    # .chat-messages-container::-webkit-scrollbar {{
    #     width: 8px;
    # }}
    # .chat-messages-container::-webkit-scrollbar-track {{
    #     background: #f1f1f1;
    #     border-radius: 4px;
    # }}
    # .chat-messages-container::-webkit-scrollbar-thumb {{
    #     background: #888;
    #     border-radius: 4px;
    # }}
    # .chat-messages-container::-webkit-scrollbar-thumb:hover {{
    #     background: #555;
    # }}
    # .chat-message {{
    #     margin-bottom: 8px;
    #     line-height: 1.6;
    #     font-size: 15px;
    # }}
    # .message-label {{
    #     font-weight: 700;
    #     margin-right: 6px;
    # }}
    # .user-label {{
    #     color: #DC2626 !important;
    # }}
    # .assistant-label {{
    #     color: #2563EB !important;
    # }}
    # .message-text {{
    #     color: #000000 !important;
    # }}
    # .message-separator {{
    #     height: 16px;
    # }}
    # .welcome-message {{
    #     text-align: center;
    #     padding: 60px 20px;
    #     color: #6B7280;
    # }}
    # .welcome-message h3 {{
    #     color: #0F4C81;
    #     margin-bottom: 12px;
    #     font-size: 24px;
    # }}
    # '''}
    # </style>
    # <div class="chat-messages-container" id="chatMessages">
    #     {messages_html}
    # </div>
    # <script>
    # setTimeout(function() {{
    #     var chatContainer = document.getElementById('chatMessages');
    #     if (chatContainer) {{
    #         chatContainer.scrollTop = chatContainer.scrollHeight;
    #     }}
    # }}, 100);
    # </script>
    # """
    ## components.html(chat_html, height=550, scrolling=False)
    
    # Fixed input container at bottom
st.markdown(
    """
    <style>
      /* remove empty white space and box */
      .fixed-input-container { margin-top: 0; background: transparent; border: none; padding: 0; }
      .fixed-input-container .stForm { margin: 0; padding: 0; background: transparent; border: none; }
    </style>
    <div class="fixed-input-container"></div>
    """,
    unsafe_allow_html=True
)

# Input + Submit form
with st.form("chat_form", clear_on_submit=True):
    c_in, c_btn = st.columns([6, 1], gap="medium")
    with c_in:
        q = st.text_input(
            label="Type your message...",
            label_visibility="collapsed",
            key="question_box",
            placeholder="Ask about geotechnical construction..."
        )
    with c_btn:
        submitted = st.form_submit_button("Send", use_container_width=True)

# Handle message sending (ENTER or button)
if submitted and q.strip():
    st.session_state.messages.append({"role": "user", "content": q})
    convo = st.session_state.get("conversation")
    status = st.session_state.get("rag_status")
    if convo:
        try:
            with st.spinner("Thinking..."):
                result = convo({"question": q})
            answer = result.get("answer", "(no answer returned)")
        except Exception as e:
            answer = f"(RAG error: {e})"
    else:
        if status == "missing_key":
            answer = "⚠️ RAG unavailable: Please set your OPENAI_API_KEY in the Settings tab."
        elif status == "no_pdfs":
            answer = "⚠️ RAG unavailable: Please add PDFs in the Knowledge Base tab and rebuild the index."
        else:
            answer = "⚠️ RAG not ready. Please check Settings and Knowledge Base."
    formatted = tidy_response(answer)
    st.session_state.messages.append({"role": "assistant", "content": formatted})
    st.rerun()
    
    ## st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
    # col_in, col_btn = st.columns([6, 1], gap="medium")
    # with col_in:
    #     q = st.text_input(
    #         "Type your message...", 
    #         label_visibility="collapsed", 
    #         key=f"question_box_{st.session_state.input_key}",
    #         placeholder="Ask about geotechnical construction..."
    #     )
    # with col_btn:
    #     send = st.button("Send", use_container_width=True, key="send_btn", type="primary")
    # st.markdown('</div>', unsafe_allow_html=True)  # Close fixed-input-container
    
    # # Handle message sending
    # if send and q.strip():
    #     st.session_state.messages.append({"role": "user", "content": q})
    #     convo = st.session_state.get("conversation")
    #     status = st.session_state.get("rag_status")
    #     if convo:
    #         try:
    #             with st.spinner("Thinking..."):
    #                 result = convo({"question": q})
    #             answer = result.get("answer", "(no answer returned)")
    #         except Exception as e:
    #             answer = f"(RAG error: {e})"
    #     else:
    #         if status == "missing_key":
    #             answer = "⚠️ RAG unavailable: Please set your OPENAI_API_KEY in the Settings tab."
    #         elif status == "no_pdfs":
    #             answer = "⚠️ RAG unavailable: Please add PDFs in the Knowledge Base tab and rebuild the index."
    #         else:
    #             answer = "⚠️ RAG not ready. Please check Settings and Knowledge Base."
    #     st.session_state.messages.append({"role": "assistant", "content": answer})
    #     st.session_state.input_key += 1  # Increment key to reset input
    ##     st.rerun()

# --- Photos Tab ---
with tab_photos:
    st.markdown('<div class="app-card" style="padding:20px;">', unsafe_allow_html=True)
    
    # Gallery view
    st.subheader("Photo Gallery")
    st.caption("Geotechnical Construction Images")
    
    # --- Shallow Foundation Section ---
    st.markdown("---")
    st.markdown("### **Shallow Foundation**")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    shallow_imgs = list_images_from_dir(SHALLOW_FOUNDATION_DIR)
    if shallow_imgs:
        cols = st.columns(4)
        for i, img_path in enumerate(shallow_imgs):
            with cols[i % 4]:
                try:
                    img = Image.open(img_path)
                    img = fix_image_orientation(img)
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    else:
        st.info("No images available yet. Images will be added later.")
    
    # --- Deep Foundation Section ---
    st.markdown("---")
    st.markdown("### **Deep Foundation**")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    deep_imgs = list_images_from_dir(DEEP_FOUNDATION_DIR)
    if deep_imgs:
        cols = st.columns(4)
        for i, img_path in enumerate(deep_imgs):
            with cols[i % 4]:
                try:
                    img = Image.open(img_path)
                    img = fix_image_orientation(img)
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    else:
        st.info("No images found in the Deep Foundation directory.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Knowledge Base Tab ---
with tab_kb:
    st.markdown('<div class="app-card" style="padding:20px;">', unsafe_allow_html=True)
    st.subheader("Knowledge Base")
    st.caption("PDFs in the app folder are indexed for retrieval.")

    pdfs = list_pdfs_in_cwd()
    if not pdfs:
        st.info("No PDF files found in the app folder.")
    else:
        for f in pdfs:
            size_kb = os.path.getsize(f) / 1024
            st.markdown(f"• **{f}** — {size_kb:,.1f} KB")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Rebuild Knowledge Base", use_container_width=True):
            api_key = get_api_key()
            if not api_key:
                st.error("Set OPENAI_API_KEY in Settings first.")
            else:
                # clean old vs to force rebuild
                if os.path.isdir(VECTORSTORE_DIR):
                    shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
                vs = build_or_load_vectorstore(api_key)
                if vs:
                    st.session_state.conversation = make_chain(vs, api_key)
                    st.session_state.rag_status = "ready"
                    st.success("Knowledge base rebuilt.")
                else:
                    st.error("No PDFs found or failed to build vectorstore.")
    with c2:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Cleared.")

    # Vectorstore metadata
    if os.path.exists(METADATA_PATH):
        st.divider()
        st.caption("Index metadata")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        st.json(meta)

    st.markdown('</div>', unsafe_allow_html=True)

# --- References Tab ---
with tab_references:
    st.markdown('<div class="app-card" style="padding:20px;">', unsafe_allow_html=True)
    st.subheader("📚 References & Resources")
    st.caption("Useful links and resources for geotechnical construction")
    
    # Shallow Foundations
    st.markdown("### Shallow Foundations")
    st.markdown("""
    - [Shallow Foundations Explained - Dozr](https://dozr.com/blog/shallow-foundations-explained)
    - [Shallow Foundation Design - TexasCE](https://www.texasce.org/tce-news/shallow-foundation-design/)
    - [Types of Shallow Foundations - Geotech](https://www.geotech.hr/en/types-of-shallow-foundations/)
    - [Types of Foundations - Understand Construction](http://www.understandconstruction.com/types-of-foundations.html)
    - [Foundations and Their Types - S3DA Design](https://s3da-design.com/the-foundations-and-their-types-shallow-and-deep-foundations/)
    - [Shallow vs Deep Foundations - PileBuck](https://pilebuck.com/shallow-versus-deep-foundations-factors-consider-common-mistakes-pitfalls/)
    - [Different Types of Shallow Foundation - SlideShare](https://www.slideshare.net/slideshow/different-type-of-shallow-foundation/79438707)
    """)
    
    st.divider()
    
    # Deep Foundations
    st.markdown("### Deep Foundations")
    st.markdown("""
    - [Deep Foundations - Goettle](https://goettle.com/deep-foundations/)
    - [Augercast Piles - Goettle](https://goettle.com/project_category/augercast-piles/)
    - [Different Types of Deep Foundations - PileBuck](https://pilebuck.com/different-types-deep-foundations/)
    - [Types of Deep Foundation - SlideShare](https://www.slideshare.net/slideshow/types-of-deep-foundation-245265493/245265493)
    - [What is a Deep Foundation? - FNA Engineering](https://www.fnaengineering.com/what-is-a-deep-foundation/)
    - [Deep Foundation Video - YouTube](https://www.youtube.com/watch?v=GyIvYE27ZrI)
    """)
    
    st.divider()
    
    # Deep Foundation Challenges
    st.markdown("### Deep Foundation Challenges & Solutions")
    st.markdown("""
    - [3 Common Problems in Deep Foundation - Sinorock](https://www.sinorockco.com/news/industry-news/3-common-problems-and-countermeasures-in-deep-foundation.html)
    - [Deep Foundation Design Challenges - GCPAT](https://gcpat.com/en/about/news/blog/digging-deep-deep-foundation-design-challenges)
    """)
    
    st.divider()
    
    # Drilling Methods
    st.markdown("### Drilling Methods & Techniques")
    st.markdown("""
    - [Drill Shafts vs Driven Piles - Goliath Tech](https://www.goliathtechpiles.com/what-is-the-difference-between-drill-shafts-and-driven-piles)
    - [Drilled Shafts vs Driven Piles Comparison - Western Equipment](https://westernequipmentsolutions.com/comparing-drilling-methods-drilled-shafts-vs-driven-piles/)
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Settings Tab ---
with tab_settings:
    st.markdown('<div class="app-card" style="padding:20px;">', unsafe_allow_html=True)
    st.subheader("Settings")

    st.text_input(
        "OpenAI API Key",
        value=st.session_state.get("OPENAI_API_KEY_OVERRIDE", ""),
        type="password",
        help="Optional override; if empty the app uses the OPENAI_API_KEY environment variable.",
        key="OPENAI_API_KEY_OVERRIDE",
    )

    # colA, colB = st.columns(2)
    # with colA:
    #     if st.button("Initialize / Refresh RAG", use_container_width=True):
    #         st.session_state.conversation = None
    #         ensure_conversation()
    #         st.success(f"Status: {st.session_state.get('rag_status')}")
    # with colB:
    #     if st.button("Delete Vectorstore", use_container_width=True):
    #         if os.path.isdir(VECTORSTORE_DIR):
    #             shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
    #             st.session_state.conversation = None
    #             st.session_state.rag_status = "init"
    #             st.success("Deleted. Rebuild from Knowledge Base tab.")
    #         else:
    #             st.info("No vectorstore found.")

    st.caption("⚠️ The administrative options are disabled in the public version.")
    st.markdown('</div>', unsafe_allow_html=True)
