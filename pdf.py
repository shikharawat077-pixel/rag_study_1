import streamlit as st
from src.config import setup_gemini
from src.ui import initialize_session_state, display_chat_history, add_message
from src.document_processor import extract_text_from_pdf, chunk_text
from src.embedding import load_embedding_model, get_embeddings
from src.vector_store import create_index, search_index
from src.rag_pipeline import generate_answer

st.set_page_config(page_title="Study Buddy RAG", page_icon="📚", layout="wide")
st.title("📚 Study Buddy RAG")

# 1. Initialize State and Config
initialize_session_state()
setup_gemini()
embedding_model = load_embedding_model()

# 2. Sidebar UI
with st.sidebar:
    st.header("1. Upload Material")
    uploaded_file = st.file_uploader("Upload your Study Material (PDF)", type="pdf")
    
    if uploaded_file and st.button("Process PDF", type="primary", use_container_width=True):
        with st.spinner("Reading PDF..."):
            full_text = extract_text_from_pdf(uploaded_file)
            
        with st.spinner("Organizing notes..."):
            texts = chunk_text(full_text)
            if texts:
                embeddings = get_embeddings(texts, embedding_model)
                index = create_index(embeddings)
                
                # Save to session state
                st.session_state.texts = texts
                st.session_state.index = index
                st.success(f"✅ Ready! Extracted {len(texts)} concept chunks.")
            else:
                st.warning("No text found in the PDF.")
                
    st.divider()
    if st.session_state.index is not None:
        st.success("✅ Document is loaded and ready for questions.")
    else:
        st.info("Please upload and process a document first.")

# 3. Main Chat Interface
st.header("2. Chat with your Study Buddy")

display_chat_history()

if question := st.chat_input("Ask your Study Buddy a question..."):
    # Add User Message
    add_message("user", question)
    
    # Process Answer
    if st.session_state.index is None:
        warning_msg = "I'm ready to help, but please upload and process a PDF from the sidebar first!"
        st.warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve Context
                q_emb = get_embeddings([question], embedding_model)
                indices = search_index(st.session_state.index, q_emb, k=4)
                context = "\n\n".join([st.session_state.texts[i] for i in indices])
                
                # Generate Answer
                answer = generate_answer(question, context)
                if answer:
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    with st.expander("📄 View Source Context Used"):
                        st.write(context)
