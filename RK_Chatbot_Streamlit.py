
# ============================================================================
# Setup
# ============================================================================

import os
from pyexpat import model
import uuid
import pickle
from langchain_ollama import OllamaEmbeddings
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.schema.document import Document
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.storage import InMemoryStore
from langchain_classic.vectorstores import Chroma
from pydantic.v1 import BaseModel, Field
from langchain_classic.schema.runnable import RunnableLambda, RunnablePassthrough
from langfuse.langchain import CallbackHandler
from langfuse import get_client, propagate_attributes
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from dataclasses import dataclass
from urllib.parse import quote
import requests
import json
from langchain.chat_models import init_chat_model
from os import getenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# # Set working directory.
# os.chdir(r"C:\Users\derek\Documents\Personal\Volunteer\1. Repair Kopitiam\1. Chatbot\2. Code")

# ============================================================================
# Set up OpenRouter API call
# ============================================================================

# # Initialize the model with OpenRouter's base URL
# model = init_chat_model(
#     model = "google/gemma-3-27b-it:free",
#     model_provider = "ollama",
#     base_url = "https://openrouter.ai/api/v1",
#     api_key = getenv("OPENROUTER_API_KEY")
#     # default_headers={
#     #     "HTTP-Referer": getenv("YOUR_SITE_URL"),  # Optional. Site URL for rankings on openrouter.ai.
#     #     "X-Title": getenv("YOUR_SITE_NAME"),  # Optional. Site title for rankings on openrouter.ai.
#     # }
# )

@st.cache_resource # Add the caching decorator
def load_model():
    """
    Load the ChatOpenAI model with OpenRouter configuration.
    """

    model = ChatOpenAI(
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="google/gemma-3-27b-it:free",
        temperature=0.0,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        extra_body={
            "provider": {
                "order": ["Amazon Bedrock", "Azure"],
                "sort": "latency"
            },
            "models": ["google/gemma-3-27b-it:free"]
        }
    )
    
    return model

# Load the model.
model = load_model()

# ============================================================================
# Prepare images
# ============================================================================

@st.cache_resource  # Add the caching decorator
def load_image_data(path: str):
    """
    Load image data from pickle file.

    :return: Dictionary containing image summaries, base64 images and image locations.
    """
    
    # Load the dictionary from the file using pickle.
    with open(path, 'rb') as f:
        RK_presentation_image_dict = pickle.load(f)
        
    return RK_presentation_image_dict

# Load image data from pickle file.
RK_presentation_image_dict = load_image_data(path = './1. Datasets/RK_presentation_image_dict')

# Extract image summaries, base64 images and image locations from the dictionary.
RK_presentation_image_summaries = [RK_presentation_image_dict[i]['summary'] for i in RK_presentation_image_dict]
RK_presentation_images_base_64_processed = [RK_presentation_image_dict[i]['image'] for i in RK_presentation_image_dict]
RK_presentation_image_locations = [RK_presentation_image_dict[i]['location'] for i in RK_presentation_image_dict]

# ============================================================================
# Build Multi-Vector Retriever for Semantic Image Search
# ============================================================================

@st.cache_resource # Add the caching decorator.
def create_multi_vector_retriever(vectorstore, image_summaries, images, image_locations):
    """
    Create retriever that indexes summaries, but returns raw images or text or image sources.

    :param vectorstore: Vectorstore to store embedded image sumamries
    :param image_summaries: Image summaries
    :param images: Base64 encoded images
    :param image_location: Location of images
    :return: Retriever
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"
    location = "location"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore = vectorstore,
        docstore = store,
        id_key = id_key,
        location = location
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(_retriever, doc_summaries, doc_contents, doc_locations):
        
        # Generate unique IDs for each document.
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        
        # Wrap summaries in Document objects for vectorstore, with ids and locations in metadata.
        summary_docs = [
            Document(page_content = s, metadata = {id_key: doc_ids[i], location: doc_locations[i]})
            for i, s in enumerate(doc_summaries)
        ]
        
        # Add documents to the vectorstore.
        retriever.vectorstore.add_documents(summary_docs)
        
        # Wrap content (images) in Document objects for docstore, with ids and locations in metadata.
        content_docs = [
            Document(page_content = content, metadata = {id_key: doc_ids[i], location: doc_locations[i]})
            for i, content in enumerate(doc_contents)
        ]
        
        # Store all images in the docstore, mapping each doc_id to its corresponding image Document object.
        retriever.docstore.mset(list(zip(doc_ids, content_docs)))
    
    # Add image summaries, images and image locations to the vectorstore and docstore.
    add_documents(retriever, image_summaries, images, image_locations)

    # Return the retriever.
    return retriever

@st.cache_resource  # Add the caching decorator
def initialize_multi_vector_retriever():
    
    # Create a Chroma vectorstore with "embeddinggemma" embeddings to index image summaries.
    vectorstore_mvr = Chroma(
        # collection_name = "multi-modal-rag-mv", embedding_function = OllamaEmbeddings(model = "embeddinggemma")
        collection_name = "multi-modal-rag-mv", embedding_function = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-small")
    )

    # # Clear the vectorstore collection to reset it before repopulating with new data.
    # vectorstore_mvr.delete_collection()

    # Create multi-vector retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore_mvr,
        RK_presentation_image_summaries,
        RK_presentation_images_base_64_processed,
        RK_presentation_image_locations
    )
    
    return retriever_multi_vector_img

# Create multi-vector retriever for images.
retriever_multi_vector_img = initialize_multi_vector_retriever()

# ============================================================================
# Image Processing and Multimodal Prompt Construction
# ============================================================================

# Define a schema to generate JSON output. This will help to extract structured data from the LLM response.
class Data(BaseModel):
    """An answer to a question about images, along with the source image filenames."""

    # Define the fields in the schema.
    response: str = Field(description = "The response to the user's question")
    image_location: list = Field(description = "The filename(s) of the image file. This is stored in the image_loc metadata. You MUST ALWAYS store each filename as a dict with the key 'filename'.")

@st.cache_data # Add the caching decorator.
def prepare_images(docs):
    """
    Prepare images for prompt

    :param docs: A list of base64-encoded images from retriever.
    :return: Dict containing a list of base64-encoded strings (representing images) and their locations.
    """
    
    # Initialize lists to hold base64 images and their locations. 
    b64_images = []
    b64_images_locations = []
    
    # Extract base64 images and their locations from the retrieved documents and 
    # store them in the respective lists.
    for doc in docs:
        if isinstance(doc, Document):
            
            # Extract base64 image from the Document object. 
            b64_image = doc.page_content
            
            # Extract image location from the Document metadata.
            b64_images_location = doc.metadata.get("location")
            
            # Append base64 image and location to the respective lists.
            b64_images.append(b64_image)
            b64_images_locations.append(b64_images_location)
            
    return {"images": b64_images, "locations": b64_images_locations}

@st.cache_data # Add the caching decorator.
def img_prompt_func(data_dict, num_images = 1):
    """
    GPT-4V prompt for image analysis.

    :param data_dict: A dict with images and a user-provided question.
    :param num_images: Number of images to include in the prompt.
    :return: A list containing message objects for each image and the text prompt.
    """
    
    # Initialize a list to hold message objects.
    messages = []
    
    # Check if there are any images in the context to process.
    # If no images are present, skip this block and proceed to create the text message.
    if data_dict["context"]["images"]:
        
        # Iterate through each retrieved image, limiting to num_images parameter (default is 2).
        # enumerate() provides the index (0, 1, 2...) and the base64-encoded image string.
        for index, image in enumerate(data_dict["context"]["images"][:num_images], start = 0):
            
            # Retrieve the filename/location of the current image from the locations list using the same index.
            # This ensures each image is paired with its corresponding source filename.
            location = data_dict["context"]["locations"][index]        
            
            # Create a message dictionary for the current image with the following structure:
            # - "type": Specifies this is an image URL message (used by multimodal LLMs)
            # - "image_url": Contains the base64-encoded JPEG image data (data URI format)
            # - "image_loc": Stores the source filename/location for reference
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                "image_loc": {"location": location}
            }
            
            # Append the image message dictionary to the messages list.
            # This message will be included in the multimodal prompt sent to the LLM.
            messages.append(image_message)
    
    # Create a formatted string of filenames separated by commas.
    # If the locations list is empty or None, default to "No files" message.
    # This string is used in the prompt to inform the LLM which source files are being referenced.
    filenames_str = ", ".join(data_dict['context']['locations']) if data_dict['context']['locations'] else "No files"
    
    # Create the text message dictionary with instructions and user question.
    # This message provides context to the LLM on how to use the images and what question to answer.
    text_message = {
        "type": "text",
        "text": (
            "You are a useful assistant tasked with answering questions.\n"
            "You will be given a set of image(s) from a slide deck / presentation.\n"
            "IMPORTANT: You MUST extract and list the source image filenames in your response.\n"
            f"Source filenames you are referencing: {filenames_str}\n"
            f"Use this information to answer the user question. At the end of your response, ALWAYS include a 'Sources:' section listing the filenames.\n"
            f"User-provided question: {data_dict['question']}\n\n"
        ),
    }
    
    # Append the text message to the messages list.
    messages.append(text_message)
    
    # Return the messages wrapped in a HumanMessage object.
    return [HumanMessage(content = messages)]

@st.cache_data # Add the caching decorator.
def multi_modal_rag_chain(retriever):
    """    
    Build Multimodal RAG Image Question Answering Chain
    """
    
    # # Instantiate the ChatOllama model with specified parameters.
    # model = ChatOllama(model = "llava", max_tokens = 1024, temperature = 0)

    # Define the retrieval-augmented generation (RAG) pipeline.
    # This chain orchestrates the flow of data through multiple processing steps:
    chain = (
        # Step 1: Create a dictionary with two keys for the prompt function:
        {
            # "context": Retrieve relevant documents using the retriever, then prepare them (convert to images dict)
            "context": retriever | RunnableLambda(prepare_images),
            # "question": Pass the user's question through unchanged
            "question": RunnablePassthrough(),
        }
        # Step 2: Format the context and question into a multimodal prompt with image messages
        | RunnableLambda(img_prompt_func)
        # Step 3: Send the prompt to the LLM model and parse the response into the Data schema
        # (ensures the output has "response" and "image_location" fields)
        | model.with_structured_output(schema = Data)
    )
    
    return chain

# Create RAG chain
chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

# ============================================================================
# Building the Conversational Agent with LangGraph
# ============================================================================

# Initialize Langfuse client.
langfuse = get_client()

# Initialize the Langfuse callback handler. 
# LangChain has its own callback system, and Langfuse listens to those callbacks to record what your chains and LLMs are doing.
langfuse_handler = CallbackHandler()

# Define the state of the graph, consisting of the shape or schema of the graph state as well as reducer functions. 
class BasicChatState(TypedDict): 
    # The state is a dictionary with a single key: messages. Messages have the type 'list'. 
    # The messages key is annotated with the add_messages reducer function,
    # which tells LangGraph to append new messages to the existing list, rather than overwriting it.
    messages: Annotated[list, add_messages]

@st.cache_data # Add the caching decorator.
def chatbot(state: BasicChatState):
    """
    LangGraph node that processes user queries through the multimodal RAG pipeline.
    Extracts the latest message, invokes the image retrieval and LLM chain, 
    and returns the AI response with source image citations.
    """
    
    # Extract the last user message from the state.
    last_message = state["messages"][-1]
    
    # Get the content of the last message, handling both HumanMessage and other types.
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Add the Langfuse callback handler to your chain. 
    # The Langfuse callback handler plugs into LangChain’s event system. 
    # Every time the chain runs or the LLM is called, LangChain emits events, and the handler turns those into traces and observations in Langfuse.
    result = chain_multimodal_rag.invoke(query, config = {"callbacks": [langfuse_handler]})
    
    # Convert the Data object (returned from RAG chain) into an AIMessage object for LangGraph.
    # LangGraph expects messages in its standard message format, so we transform the structured
    # Data response into an AIMessage with:
    ai_response = AIMessage(
        content = result.response,  # The LLM's text response to the user's question
        metadata = {"image_location": result.image_location}  # List of source image filenames referenced
    )
    
    return {"messages": [ai_response]}

graph = StateGraph(BasicChatState) # Define the graph with the BasicChatState schema.

# Every node we define will receive the current State as input and return a value that updates that state.
graph.add_node("chatbot", chatbot) # Add the chatbot node to the graph.

# Define the edges to specify the flow of execution in the graph.
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile the graph into a runnable object, with the familiar invoke and stream methods.
chat_agent = graph.compile()

# ============================================================================
# Integrating LangGraph with Streamlit
# ============================================================================

# Setting the configuration of the webpage. 
st.set_page_config(
    page_title = "AI Chatbot", 
    page_icon = "🤖")

# Create a title for the app
st.title("🤖 First AI Chatbot")

@dataclass
class Message:
    role: str
    content: str
    image_urls: list

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

# Initialize chat history if not already present
if MESSAGES not in st.session_state:
    # Set initial message from assistant.
    st.session_state[MESSAGES] = [Message(role = ASSISTANT, content = "Hi! How can I help you?", image_urls = [])]

# Display chat messages from history on app rerun
for message in st.session_state[MESSAGES]:
    with st.chat_message(message.role):
        # Display the text content of the message.
        st.markdown(message.content)
        
        # Display each image in the message container.
        for url in message.image_urls:
            st.image(url)

# Display a chat input widget so the user can enter a message.
# The returned value is the user's input, which is None if the user hasn't sent a message yet. 
# React  to user input.
if prompt := st.chat_input("Type your message..."):
    
    # Display user message in chat message container.
    with st.chat_message(USER):
        st.markdown(prompt)
    
    # Add user message to chat history. 
    st.session_state[MESSAGES].append(Message(role = USER, content = prompt, image_urls = []))
    
    # Get response from LangGraph chat agent.
    result = chat_agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    # Extract the AI response content from the result.
    response: str = result["messages"][-1].content
    
    # Extract image locations from the AI response metadata.
    response_image_filenames_raw = result["messages"][-1].metadata.get("image_location") if hasattr(result["messages"][-1], 'metadata') else None

    # Extract filenames from the list of dictionaries.
    response_image_filenames = [item['filename'] for item in response_image_filenames_raw]
    
    # Generate the corresponding image URLs by encoding the filenames to ensure they are URL-safe. 
    response_image_urls = ['https://storage.googleapis.com/rk_chatbot/img/' + quote(response_image_filename, safe='.-') for response_image_filename in response_image_filenames]
    
    # Add the AI response to chat history.
    st.session_state[MESSAGES].append(Message(role = ASSISTANT, content = response, image_urls = response_image_urls))
    
    # Insert a multi-element chat message container into the app, with the name of the message author.
    with st.chat_message(ASSISTANT):
        st.write(response) # Add a text response to the message container.
        
        # If there are image locations, process and display each image.
        if response_image_urls: 
            # Display each image in the message container, along with its URL.
            for response_image_url in response_image_urls:
                # st.write(response_image_url)
                st.image(response_image_url)
            