📄 Document Q\&A using RAG (Retrieval-Augmented Generation)



An AI-powered web app that lets you upload any PDF and ask questions about it in natural language. Built with LangChain, FAISS, and Hugging Face — completely free to use!



🔗 Live Demo:[Click here to try the app](your-streamlit-url-here)



\---



🧠 How it Works



1.Upload a PDF — any document, resume, research paper, etc.

2.Text is split into chunks — using LangChain's RecursiveCharacterTextSplitter

3.Chunks are converted to embeddings — using sentence-transformers (all-MiniLM-L6-v2)

4.Stored in FAISS — a fast vector database for similarity search

5.You ask a question — it gets converted to an embedding too

6.FAISS finds relevant chunks — based on semantic similarity

7.LLM generates the answer — using only the relevant chunks as context



\---


 ✨ Features



 📤 Upload any PDF document

 🔍 Semantic search using FAISS vector database

 🤖 AI-powered answers using Kimi-K2-Instruct LLM

 👁️ Transparent — shows which chunks were used to generate the answer

 🆓 Completely free (Hugging Face free tier)



\---


 🛠️ Tech Stack



| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web interface |
| LangChain | RAG pipeline framework |
| FAISS | Vector database |
| sentence-transformers | Text embeddings |
| Hugging Face API | LLM (Kimi-K2-Instruct) |
| PyPDF | PDF reading |


\---



⚙️ Setup Instructions



1.Clone this repository

