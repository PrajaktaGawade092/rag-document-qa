import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A using RAG")
st.write("Upload a PDF and ask questions about it!")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Reading and processing your PDF..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success(f"PDF processed! {len(pages)} page(s), {len(chunks)} chunks created.")

    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Searching for answer..."):
            results = vectorstore.similarity_search(question, k=3)
            context = "\n".join([r.page_content for r in results])
            response = client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct-0905",
                messages=[
                    {"role": "system", "content": "Answer questions based only on the provided context. If the answer is not in the context, say 'I don't know based on the document'."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
            )
            answer = response.choices[0].message.content

        st.write("### Answer:")
        st.write(answer)

        with st.expander("See relevant chunks found"):
            for i, chunk in enumerate(results):
                st.write(f"**Chunk {i+1}:**")
                st.write(chunk.page_content)
                st.write("---")