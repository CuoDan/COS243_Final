####################
## Required Packages
####################

import os
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from dotenv import load_dotenv 

####################
## For ebooks upload
####################

import os
import tkinter as tk
from tkinter import filedialog
from ebooklib import epub
import fitz  # PyMuPDF

load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")

llm = Ollama(model="phi3.5:3.8b-mini-instruct-q8_0", api_key="OLLAMA_API_KEY")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

# Upload and Import Ebooks Function
def upload_and_import_ebook(file_path=None, save_directory="data"):
    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select an ebook or PDF file if no file path is provided
    if not file_path:
        file_path = filedialog.askopenfilename(
            title="Select an Ebook or PDF",
            filetypes=[("Ebook and PDF Files", "*.epub *.pdf")]
        )

    if file_path:
        content = ""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".epub":
            # Load the ebook
            book = epub.read_epub(file_path)
            
            # Extract the content
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content += item.get_body_content().decode('utf-8')
        
        elif file_extension == ".pdf":
            # Load the PDF
            pdf_document = fitz.open(file_path)
            
            # Extract the content
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                content += page.get_text()

        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the content to a file in the save directory
        save_path = os.path.join(save_directory, os.path.basename(file_path) + ".txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        return save_path
    else:
        return None

# Document Loader
ebook_path = upload_and_import_ebook()
if ebook_path:
    print(f"Ebook content imported successfully to {ebook_path}.")
    docs = SimpleDirectoryReader(input_files=[ebook_path]).load_data()
else:
    print("No ebook or PDF selected. Sample book will be used")
    ebook_path = """"data/The Theory That Would Not Die How Bayes Rule Cracked the Enigma Code, 
            Hunted Down Russian Submarines, and Emerged Triumphant from Two Centuries of Controversy 
            by Sharon Bertsch McGrayne.pdf.txt"""
    docs = SimpleDirectoryReader(input_files=[ebook_path]).load_data()


index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=5)

# Custom function to include citations in the response
def include_citations(response, docs):
    citations = []
    for doc in docs:
        if hasattr(doc, 'page_number'):
            citations.append(f"Page {doc.page_number}")
        elif hasattr(doc, 'section'):
            citations.append(f"Section {doc.section}")
    if citations:
        return f"{response}\n\nCitations: {', '.join(citations)}"
    else:
        return response

# Custom query engine tool with citation inclusion
class CitationQueryEngineTool(QueryEngineTool):
    def query(self, query):
        response, docs = super().query(query)
        return include_citations(response, docs)
    
# vector_tool = QueryEngineTool(
#     index.as_query_engine(),
#     metadata=ToolMetadata(
#         name="vector_search",
#         description="Useful for searching for specific facts.",
#     ),
# )

# Create tools with citation support
vector_tool = CitationQueryEngineTool(
    index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts with citations.",
    ),
)

# summary_tool = QueryEngineTool(
#     index.as_query_engine(response_mode="tree_summarize"),
#     metadata=ToolMetadata(
#         name="summary",
#         description="Useful for summarizing an entire document.",
#     ),
# )

summary_tool = CitationQueryEngineTool(
    index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document with citations.",
    ),
)

query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool], select_multi=False, verbose=True, llm=llm
)

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    index.as_retriever(),
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)