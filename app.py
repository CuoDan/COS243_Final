####################
## Required Packages
####################

import os
import shutil
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# Initialize LLM and embedding model
# llm = Ollama(model="qwen2.5:14b", api_key="OLLAMA_API_KEY")
llm = Ollama(model="phi3.5:3.8b-mini-instruct-q8_0", request_timeout=60)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

# Create a custom query engine tool with citation inclusion
class CitationQueryEngineTool(QueryEngineTool):
    # Define the GetSources function to include citations
    def GetSources(response):
        sources = []
        for node in response.source_nodes:
            text_node = node.node
            source = text_node.metadata.get('file_name')
            page = text_node.metadata.get('page_label')
            section = text_node.metadata.get('section_name')
            sources.append(f"Source: {source[:30]}..., Page: {page}, Section: {section}")
        return "\n".join(sources)
    
    def query(self, query, timeout=100):
        response = super().query(query, timeout=timeout)
        sources = self.GetSources(response)
        return f"{response}\n\nCitations:\n{sources}"

# Function to upload and import ebook content
def upload_ebook(files):
    destination_path = "data"  # Ensure this path is correctly set
    for file in files:
        if os.path.isfile(file):  # Ensure the file is not a directory
            shutil.copy(file, destination_path)
    return "Ebook uploaded successfully"

def select_file():
    upload_dir = "data"
    files = os.listdir(upload_dir)

    if not files:
        file = """sample_books/The Theory That Would Not Die How Bayes Rule Cracked the Enigma Code, Hunted Down Russian Submarines, and Emerged Triumphant from Two Centuries of Controversy by Sharon Bertsch McGrayne.txt"""
    else:
        file = os.path.join(upload_dir, files[0])

    print(file)
    return file

def handle_file_upload(files):
    ebook_content = upload_ebook(files)
    return ebook_content

file = select_file()
docs = SimpleDirectoryReader(input_files=[file]).load_data()

# Build a vector index
vector_index = VectorStoreIndex.from_documents(docs)

# Build a summary index
summary_index = SummaryIndex.from_documents(docs)

# Create tools with citation support
vector_tool = CitationQueryEngineTool(
    vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts with citations.",
    ),
)
summary_tool = CitationQueryEngineTool(
    summary_index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document with citations.",
    ),
)

query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool], select_multi=False, verbose=True, llm=llm
)

# Gradio App Interface
def chat_with_llm(user_input):
    try:
        response = query_engine.query(user_input)
    except TimeoutError:
        print("The query timed out. Attempt to try again.")
        response = query_engine.query(user_input)
    except Exception as e:
        response = f"An error occurred: {e}"
    return response

with gr.Blocks() as demo:
    with gr.Tabs("Document Upload"):
        file_upload = gr.File(label="Upload Documents", file_count="single", type="filepath")
        file_output = gr.Textbox(label="Uploaded Content", lines=3, interactive=False)
        file_upload.upload(handle_file_upload, inputs=file_upload, outputs=file_output)

    with gr.Tabs("Chatbot"):
        user_input = gr.Textbox(label="Enter your question")
        chat_output = gr.Textbox(label="Response", lines=5, interactive=False)
        submit_btn = gr.Button("Submit")

        submit_btn.click(chat_with_llm, inputs=user_input, outputs=chat_output)

if __name__ == "__main__":
    demo.launch()