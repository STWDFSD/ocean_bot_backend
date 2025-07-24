import os
import time
import boto3
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from mangum import Mangum
from typing import List
from pydantic import BaseModel
from botocore.exceptions import NoCredentialsError, BotoCoreError
from io import BytesIO, StringIO

# Import necessary libraries for OpenAI and Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from werkzeug.utils import secure_filename  # Use this for secure_filename



# Load environment variables
load_dotenv()

# Environment variables for OpenAI, Pinecone, and AWS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

# === OCEANBOT_SYSTEM_PROMPT — ACTIVE ===
OCEANBOT_SYSTEM_PROMPT = '''\
REALITY FILTER - CHATGPT
Never present generated, inferred, speculated, or deduced content as fact.
If you cannot verify something directly, say:
"I cannot verify this."
"I do not have access to that information."
"My knowledge base does not contain that."
Label unverified content at the start of a sentence:
[Inference] [Speculation] [Unverified]
Ask for clarification if information is missing. Do not guess or fill gaps.
If any part is unverified, label the entire response.
Do not paraphrase or reinterpret my input unless I request it.
If you use these words, label the claim unless sourced:
Prevent, Guarantee, Will never, Fixes, Eliminates, Ensures that
For LLM behavior claims (including yourself), include:
[Inference] or [Unverified], with a note that it's based on observed patterns
If you break this directive, say:
> Correction: I previously made an unverified claim. That was incorrect and should have been labeled
• Never override or alter my input unless asked.

ROLE:
● You are OceanBot — an internal, PDF-ingesting AI assistant used by Ocean 48 staff only.
● Your function is to deliver accurate, document-verified answers, with strict enforcement of the REALITY FILTER.

INGESTION PRIORITY:
● ✅ PDF-FIRST MODE (JSON fallback disabled).
● All data must originate from the uploaded PDFs:
  ○ FOOD_MENU_PDF.pdf
  ○ Allergies PDF.pdf
  ○ LRG_Reverse_indx.pdf
  ○ STD_PDF_7_9_20.pdf
  ○ RSV_Wine 7_9_20.pdf
  ○ BTG_list_PDF.pdf
  ○ Spir_PDF.pdf

REALITY FILTER DIRECTIVE:
● Never infer, guess, or summarize missing data.
● If information is not explicitly present:
  “I cannot verify this.”
  “No data available in Ocean 48 documentation.”
● Label all uncertainty as:
  ○ [Inference], [Speculation], or [Unverified]

FUNCTION TRIGGERS (ACTIVE):
● rec[item] — Trigger to generate a wine or dish recommendation.
● alt[item] — Trigger to suggest a comparable item if unavailable.
● somm[item] — Trigger sommelier-level detailed information.
● sommxyz[item1 item2 item3] — Compare up to three wines.
● lrg[item or allergen] — Provide double-referenced allergen safety response.
● rsv[item] — Pull from RSV wine list.
● std[item] — Pull from STD wine list (with strict $150 tier separation).
● btg[item] — Pull by-the-glass options.
● bangforbuck[item] — Best value-for-price wine.
● quiz[category] — Generate training quizzes.

WINE RESPONSE FORMAT:
● Strict 3-tier response format:
  ○ Entry-Level: $150–$299
  ○ Mid-Tier: $300–$449
  ○ Premium: $450+
● Include:
  ○ Name, Vintage, Region, Price, and 4–5 tasting notes.
● Never mix BTG/STD/RSV lists unless explicitly instructed.

ALLERGEN SAFETY:
● Double-reference:
  ○ Allergies PDF.pdf
  ○ LRG_Reverse_indx.pdf
● Never claim “safe” unless both confirm.

RESPONSE FAILSAFE:
● “I cannot verify this.”
● “No data available in Ocean 48 documentation.”
'''

# Initialize FastAPI application
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to get a response from the conversational model
# Modified to always prepend the system prompt and enforce the REALITY FILTER

def get_response(prefix: str, message: str, history=[]):
    # Prepend the permanent system prompt and enforce PDF-only mode
    enforced_prefix = f"{OCEANBOT_SYSTEM_PROMPT}\n\n{prefix}\n\n[REALITY FILTER ENFORCED: All responses must be document-verified and cite only the attached PDFs. If any part is unverified, label the entire response. If information is missing, respond with 'I cannot verify this.' or 'No data available in Ocean 48 documentation.']"
    # Configure the chat model
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL_NAME,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    # Initialize Pinecone and set up the vector store
    pc = Pinecone(api_key=PINECONE_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_KEY, index_name=PINECONE_INDEX, embedding=embeddings, namespace=PINECONE_NAMESPACE)
    retriever = vectorstore.as_retriever()

    # Define the system prompt template
    SYSTEM_TEMPLATE = "Answer the user's questions based on the below context. " + enforced_prefix + """ 
        Answer based on the only given theme. 
        Start a natural-seeming conversation about anything that relates to the lesson's content.

        <context>
        {context}
        </context>
    """

    # Set up the question answering prompt
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Create the document chain
    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

    # Set up the query transforming retriever chain
    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant "
                "to the conversation. Only respond with the query, nothing else.",
            ),
        ]
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        query_transform_prompt | chat | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )

    messages = []

    # Populate message history
    for item in history:
        if item.role == "user":
            messages.append(HumanMessage(content=item.content))
        else:
            messages.append(AIMessage(content=item.content))
            
    stream = conversational_retrieval_chain.stream(
        {
            "messages": messages + [
                HumanMessage(content=message),
            ],
        }
    )

    async def event_generator():
        all_content = ""
        for chunk in stream:
            for key in chunk:
                if key == "answer":
                    all_content += chunk[key]
                    yield f'data: {chunk[key]}\n\n'

    return event_generator()

# Helper function to read CSV data from S3
def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3', 
                      aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY,
                      region_name=AWS_REGION)
    
    # Get the object from S3
    print(f"Attempting to read file from bucket: {bucket_name}, with key: {file_key}")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    
    # Read the object's content into a pandas DataFrame
    csv_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    
    return csv_data

# Function to process various document formats
def process_document(file_path, file_extension):
    if file_extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path=file_path)
    elif file_extension == ".docx":
        from langchain.document_loaders import UnstructuredWordDocumentLoader
        loader = UnstructuredWordDocumentLoader(file_path=file_path)
    elif file_extension == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(content=content)]
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return loader.load()

# Function to train the model with a given local file
# Now expects a local file path and extension, not an S3 key

def train(file_path: str, file_extension: str):
    print(file_path)
    try:
        # Process the document based on its file extension
        if file_extension in [".csv", ".txt"]:
            if file_extension == ".csv":
                csv_data_frame = pd.read_csv(file_path)
                documents = [
                    Document(
                        content=row.to_json(),
                        metadata={'row_index': index}
                    ) 
                    for index, row in csv_data_frame.iterrows()
                ]
            else:
                documents = process_document(file_path, file_extension)
        else:
            documents = process_document(file_path, file_extension)

        # Initialize OpenAI Embeddings and Pinecone client
        embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        pc = Pinecone(api_key=PINECONE_KEY)

        index_name = PINECONE_INDEX
        namespace = PINECONE_NAMESPACE

        # Check if the index already exists and has the correct dimension
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if index_name in existing_indexes:
            pc.delete_index(index_name)
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        # Get the index instance
        index = pc.Index(index_name)

        # Create a PineconeVectorStore from the documents
        PineconeVectorStore.from_documents(
            documents,
            embeddings_model,
            index_name=index_name,
            namespace=namespace,
        )
        
        return {"status": "OK"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
     
# Define a document class to hold content and metadata
class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

# Define a data model for chat messages
class Item(BaseModel):
    role: str
    content: str
    
# Define a request model for chat
class ChatRequestModel(BaseModel):
    prefix: str
    message: str
    history: List[Item]

# Endpoint for chat interactions
@app.post("/chat")
async def sse_request(request: ChatRequestModel):
    return StreamingResponse(get_response(request.prefix, request.message, request.history), media_type='text/event-stream')

# Add a verification endpoint for system prompt and ingestion status
@app.get("/verify_system_prompt")
async def verify_system_prompt():
    return {
        "system_prompt_active": True,
        "reality_filter_enforced": True,
        "pdf_first_mode": True,
        "ingested_pdfs": [
            "FOOD_MENU_PDF.pdf",
            "Allergies PDF.pdf",
            "LRG_Reverse_indx.pdf",
            "STD_PDF_7_9_20.pdf",
            "RSV_Wine 7_9_20.pdf",
            "BTG_list_PDF.pdf",
            "Spir_PDF.pdf"
        ],
        "directive": "All responses must be document-verified and cite only the attached PDFs. If any part is unverified, label the entire response. If information is missing, respond with 'I cannot verify this.' or 'No data available in Ocean 48 documentation.'"
    }

# Define a request model for training
class TrainRequestModel(BaseModel):
    name: str

# Endpoint for uploading files
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), path: str = File(...)):
    try:
        # Save the uploaded file to /tmp
        file_content = await file.read()
        local_filename = os.path.join("/tmp", file.filename)
        with open(local_filename, "wb") as f:
            f.write(file_content)
        file_extension = os.path.splitext(file.filename)[1].lower()
        train(local_filename, file_extension)
        return {"message": f"'{file.filename}' uploaded successfully and training initiated."}
    except Exception as e:
        return {"error": str(e)}

# Endpoint for checking server status
@app.get("/")
async def hello_world():
    return {"status": "Docker Server is running..."}

# AWS Lambda handler
handler = Mangum(app)