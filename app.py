####################
## Required Packages
####################

import os
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
import json

####################
## For ebooks upload
####################

import os
import tkinter as tk
from tkinter import filedialog
from ebooklib import epub
import fitz  # PyMuPDF


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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