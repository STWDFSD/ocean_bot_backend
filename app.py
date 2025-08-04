import os
import time
import boto3
import pandas as pd
import re
from typing import List, Dict, Any

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
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
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

FUNCTION TRIGGERS & RESPONSE FORMATS:

● BangForBuck: Returns the highest-rated wine at the lowest price from relevant PDF lists, including name, vintage, region, price, rating, and 4-5 tasting notes. Trigger: "bangforbuck [wine type or category]".
  Example prompt: "bangforbuck rsv french pinot noir"
  Ideal response format:
  "🍷 bangforbuck[RSV French Pinot Noir] — Best Value from RSV_Wine_7_9_20.pdf
  🏆
  Top Value Pick:
  Joseph Drouhin Chambolle-Musigny 2016
  ● Region: Côte de Nuits, Burgundy, France
  ● Price: $295
  ● Varietal: 100% Pinot Noir
  🍇
  Tasting Notes (PDF-Verified):
  ● Wild strawberry
  ● Rose petal
  ● Red plum
  ● Forest floor
  ● Silky tannins
  ✅
  Why It Wins Best Bang-for-Buck:
  ● Prestigious Village: Chambolle-Musigny is known for delicate, fragrant, and age-worthy Pinot — among the most sought-after styles in Burgundy.
  ● Producer Reputation: Joseph Drouhin is a benchmark Burgundy house known for value and precision across all vineyard tiers.
  ● Price Point: $295 is well below most Côte de Nuits wines of this quality — especially from top villages like Chambolle.
  Let me know if you'd like alternative Burgundy options or pairings for this wine."

● Somm: Provides an in-depth, sales-geared sommelier-level deep dive into the terroir, producer, region, year, and ratings for a wine. For comparisons, use "sommxyz [wine1] [wine2] [wine3]" to compare up to three wines. Trigger: "somm [wine name]".
  Example prompt: "somm inglenook, screaming eagle, chapellet"
  Ideal response format:
  "🔍 somm[Inglenook, Chappellet, Alpha Omega] — Napa Valley Comparison
  (Strictly PDF-verified where applicable, with [Unverified] flag for non-listed wines)
  1️⃣
  Inglenook Cabernet Sauvignon 2018
  ● Verified in: STD_PDF_7_9_20.pdf
  ● Region: Rutherford, Napa Valley
  ● Price: $295
  ● Producer: Inglenook Estate (founded 1879; owned by Francis Ford Coppola)
  ● Terroir: Volcanic and gravelly soils of the Rutherford Bench
  ● Style: Old-world influenced Napa Cabernet with structure and balance
  ● Tasting Notes (PDF):
  ○ Blackcurrant
  ○ Dark cherry
  ○ Tobacco
  ○ Cigar box
  ○ Polished tannins
  2️⃣
  Chappellet Cabernet Franc 2018
  ● Verified in: RSV_Wine_7_9_20.pdf
  ● Region: Pritchard Hill, Napa Valley
  ● Price: $195
  ● Producer: Chappellet Winery (founded 1967, family-owned)
  ● Terroir: High-elevation volcanic slopes on Pritchard Hill
  ● Style: Floral, structured Cabernet Franc with mountain tannins
  ● Tasting Notes (PDF):
  ○ Blackberry
  ○ Violet
  ○ Clove
  ○ Minerality
  ○ Medium-plus structure
  3️⃣
  Alpha Omega Cabernet Sauvignon
  —
  [Unverified]
  ● Not listed in Ocean 48's wine PDFs
  ● Region: Rutherford/Oakville/St. Helena blends (typical)
  ● Producer: Alpha Omega Winery, known for bold, polished modern Napa reds
  ● Style: Rich, plush Cabernet Sauvignon blends with high oak integration
  ● Tasting Notes (External):
  ○ Black cherry
  ○ Espresso
  ○ Vanilla
  ○ Cocoa
  ○ Full-bodied, smooth finish
  🛑 Not available in STD or RSV lists — cannot be served at Ocean 48
  📊
  Summary Table:
  Wine Name | Verified | Region | Price | Tasting Style
  Inglenook Cab 2018 | ✅ Yes | Rutherford | $295 | Earthy, cigar-box, refined Napa Cab
  Chappellet Cab Franc 2018 | ✅ Yes | Pritchard Hill | $195 | Floral, mineral, mountain-structured red
  Alpha Omega Cabernet | ❌ No | Napa (blended) | N/A | Plush, modern, oak-driven [Unverified]
  Let me know if you'd like to swap Alpha Omega for another verified Rutherford or Oakville bottle."

● QuizXYZ: Creates a mini quiz mode with 5-10 questions based on specified menu categories from the PDFs (e.g., food, wine, spirits). Questions focus on facts like ingredients, pairings, allergens, or descriptions. Provides correct answers with citations; exit with "end quiz". Trigger: "quiz [category]" or "quiz [category1] [category2]".
  Example prompt: "quiz apps, entrees"
  Ideal response format:
  "🧠 quiz[Apps, Entrées] — Ocean 48 Staff Training Quiz
  (Strictly sourced from FOOD_MENU_PDF.pdf)
  🍽
  APPETIZERS QUIZ
  1. What spice is used to season the 1/4" bacon in the Bacon appetizer?
  2. What two sauces are served with the Rhode Island Calamari & Shrimp?
  3. What allergen is introduced by the truffle mornay sauce in the Maine Lobster Escargot?
  4. What are the five toppings on the deviled eggs in SHR DEVIL?
  5. What dish uses both chili aioli and house Louie sauce in its preparation?
  🥩
  ENTRÉES QUIZ
  6. Which fish entrée is rubbed with house steak rub and topped with king crab and lemon butter?
  7. What is the difference between the 8oz and 12oz Australian Wagyu Filets?
  8. What grade is the Tomahawk Wagyu from Snake River Farms, and what is its average weight?
  9. Which steak entrée includes a sunny side up egg by default with no substitutions allowed?
  10. What are the key allergens in the Maryland-style Crab Cakes, including sauce components?
  Let me know if you want the answers, a version for print, or a quiz for sides/desserts next."

● Lrg: Utilizing menu abbreviations, provides specific allergy-related information from Allergies PDF.pdf and LRG_Reverse_indx.pdf, including verified ingredients, potential cross-contaminations, and disclaimers. Always cross-reference both PDFs and state if information is incomplete. Trigger: "lrg [allergen or dish]".
  Example prompt: "lrg stripe"
  Ideal response format:
  "✅ lrg[STRIPE] — Allergen Verification
  (Strictly verified using FOOD_MENU_PDF.pdf, Allergies PDF.pdf, and LRG_Reverse_indx.pdf)
  🐟
  Dish:
  Striped Bass + King Crab
  Abbreviation: STRIPE
  Portion: 14 oz striped bass filet + 3 oz king crab merus meat
  🔪
  Preparation:
  ● Striped bass is seasoned with House Steak Rub
  ● Drizzled with clarified butter and white wine
  ● Seared and baked
  ● Topped with king crab merus meat and lemon butter sauce
  🧂
  Key Component Review:
  🟫
  House Steak Rub:
  ● Paprika
  ● Garlic powder
  ● Onion powder
  ⚠️ Allergens: Garlic, Onion
  🟨
  Lemon Butter Sauce:
  ● Lemon juice
  ● Butter
  ● Garlic
  ● Shallots
  ● White wine
  ⚠️ Allergens: Dairy, Garlic, Onion
  🦀
  King Crab Merus Meat:
  ⚠️ Allergen: Shellfish
  ⚠️
  Confirmed Allergens (from both PDFs):
  ● Dairy
  ● Shellfish
  ● Fish
  ● Garlic
  ● Onion
  ✅ All five allergens are cross-confirmed in both Allergies PDF.pdf and LRG_Reverse_indx.pdf.
  Let me know if you'd like to explore rub-free or sauce-free versions for allergy-sensitive guests."

● Rsv: Calls a reserve wine list recommendation from RSV_Wine_7_9_20, providing 4 different wines: 3 from the standard reserve list and 1 from the large format list, in multi-tiered pricing from lowest to highest with each category no less than $200 apart. Uses the 3-tier format, including tasting notes. Trigger: "rsv [dish or type]".
  Example prompt: "rec rsv napa cab pair with dryrib"
  Ideal response format:
  "🍷 rsv[Napa Cabernet Sauvignon – Pairing for Dry-Aged Ribeye (DRYRIB)]
  (Strictly verified from RSV_Wine_7_9_20.pdf — using required price-tiered structure)
  🥩
  Dish Profile: Dry-Aged Ribeye (22 oz)
  ● Intense beef flavor from 45-day dry aging
  ● Bone-in, broiled with House Steak Rub (paprika, garlic, onion)
  ● Finished with clarified butter
  This preparation demands a bold, structured, and tannin-rich Cabernet Sauvignon to balance the marbled richness and umami depth.
  🍷
  Recommended Reserve Napa Cabernet Wines:
  🟩
  Entry Tier (~$200–$250)
  Nickel & Nickel "State Ranch" Cabernet Sauvignon 2017
  ● Region: Oakville, Napa Valley
  ● Price: $225
  ● Tasting Notes:
  ○ Blackberry
  ○ Cocoa nib
  ○ Toasted oak
  ○ Velvety tannins
  ○ Dusty earth
  💡 A powerful yet silky wine with ample dark fruit and Oakville pedigree — an ideal match for dry-aged beef.
  🟨
  Mid Tier (~$400–$450)
  Tamber Bey "Oakville Estate" Cabernet Sauvignon 2018
  ● Region: Oakville, Napa Valley
  ● Price: $445
  ● Tasting Notes:
  ○ Dark plum
  ○ Leather
  ○ Sage
  ○ Toasted spice
  ○ Dense structure
  💡 Shows savory complexity and grip to stand up to the steak's char and fat.
  🟥
  Premium Tier (~$650+)
  PlumpJack Reserve Cabernet Sauvignon 2016 (1.5L Magnum)
  ● Region: Oakville, Napa Valley
  ● Price: $1,195
  ● Tasting Notes:
  ○ Cassis
  ○ Espresso
  ○ Black cherry
  ○ Clove
  ○ Full-bodied, plush finish
  💡 For guests who want a high-impact wine experience to mirror the steak's richness.
  🍾
  Large Format Selection
  Groth Reserve Cabernet Sauvignon 2016 (1.5L Magnum)
  ● Region: Oakville, Napa Valley
  ● Price: $525
  ● Tasting Notes:
  ○ Red currant
  ○ Dried herbs
  ○ Silky tannins
  ○ Toasted cedar
  ○ Long, elegant finish
  💡 Large-format sophistication with balance and age potential — excellent for group steak service.
  Let me know if you'd like Bordeaux alternatives like Château Lynch-Bages or a somm-style head-to-head comparison."

● Std: Calls a standard wine list recommendation from STD_PDF_7_9_20, providing 3 options in multi-tiered pricing from lowest to highest with each category no less than $150 apart. Uses the 3-tier format, including tasting notes. Trigger: "std [dish or type]".
  Example prompt: "rec std french white pair with branz"
  Ideal response format:
  "🍷 rec[Standard French White – Pairing for Branzino (BRANZ)]
  (Strictly verified from STD_PDF_7_9_20.pdf)
  🐟
  Dish Profile: Branzino
  ● Lean, flaky Mediterranean sea bass
  ● Seasoned with House Fish Rub (garlic, onion, sesame)
  ● Finished with lemon butter, parsley, and grilled lemon
  ● Delicate, citrus-forward flavor with light richness
  🍷
  Recommended Standard French White Wines:
  🟩
  Entry Tier ($150–299)
  Joseph Drouhin Chablis "Vaillons" 1er Cru 2018
  ● Region: Chablis, Burgundy, France
  ● Price: $210
  ● Tasting Notes:
  ○ Lemon zest
  ○ Green apple
  ○ Wet stone
  ○ Crushed oyster shell
  ○ Crisp, mineral finish
  💡 Chablis' acidity and salinity mirror the dish's citrus and seafood profile perfectly.
  🟨
  Mid Tier ($300–449)
  Louis Jadot Meursault "Narvaux" 2016
  ● Region: Côte de Beaune, Burgundy, France
  ● Price: $345
  ● Tasting Notes:
  ○ Baked pear
  ○ Almond
  ○ Buttery mid-palate
  ○ Flinty minerality
  ○ Long, textured finish
  💡 Creamy enough to match the lemon butter, but still elegant and restrained.
  🟥
  Premium Tier ($450+)
  Domaine Leflaive Puligny-Montrachet 1er Cru "Clavoillon" 2015
  ● Region: Côte de Beaune, Burgundy
  ● Price: $675
  ● Tasting Notes:
  ○ White peach
  ○ Toasted almond
  ○ Chalk
  ○ Lemon curd
  ○ Elegant structure
  💡 A premium white Burgundy for guests wanting refinement and richness in perfect balance.
  Let me know if you'd like a BTG or reserve-tier alternative."

● Btg: Calls a by-the-glass recommendation from BTG_list_PDF.pdf, providing selections with pricing, descriptions, and tasting notes as available in the PDF. Does not recommend bottles. Trigger: "btg [dish or type]".
  Example prompt: "rec btg pair with lamb"
  Ideal response format:
  "🍷 btg[LAMB] — Wine Pairing for Australian Rack of Lamb
  (Strictly verified from BTG_list_PDF.pdf)
  🐑
  Dish Profile: Rack of Lamb (LAMB)
  ● 24 oz full Australian rack
  ● Seasoned with House Steak Rub (paprika, garlic, onion)
  ● Broiled to guest's temp
  ● Rich, gamey, savory, and herb-forward
  🍷
  Best By-The-Glass Pairings from Ocean 48:
  1.
  Stags' Leap Petite Sirah
  – Napa Valley
  ● Price: $22 per glass
  ● Tasting Notes:
  ○ Blackberry
  ○ Plum
  ○ Black pepper
  ○ Cocoa
  ○ Firm tannins
  💡 Robust, dark, and spicy — pairs perfectly with the lamb's richness and char.
  2.
  Robert Craig "Affinity" Cabernet Sauvignon
  – Napa Valley
  ● Price: $31 per glass
  ● Tasting Notes:
  ○ Cassis
  ○ Graphite
  ○ Tobacco
  ○ Black cherry
  ○ Silky tannins
  💡 Classic Cab structure for lamb; balances the rub's paprika and herbs.
  3.
  Louis Jadot Pinot Noir
  – Bourgogne, France
  ● Price: $20 per glass
  ● Tasting Notes:
  ○ Red cherry
  ○ Earth
  ○ Subtle spice
  ○ Cranberry
  ○ Silky texture
  💡 More delicate option for guests preferring finesse with lamb's gamey edge.
  Let me know if you'd like a reserve-tier Rhône or Bordeaux recommendation as an upgrade."

● Rec: Generates a verified selection from PDF lists for pairing with a dish or inquiry, in the 3-tier format (Premium, Mid-Tier, Entry-Level) with tasting notes. Trigger: "rec [item]".
  Example prompt: "rec std white pair with lob bake"
  Ideal response format:
  "🍷 rec[Standard White – Pairing for Wood Roasted Shellfish Tower (LOB BAKE)]
  (Strictly verified from STD_PDF_7_9_20.pdf)
  🦞
  Dish Profile: LOB BAKE
  ● Mixed shellfish: lobster, scallops, shrimp, mussels, clams, king crab
  ● Broth base: tomato, garlic, leeks, fennel, Calabrian peppers, white wine, Pastis
  ● Served with grilled sourdough
  ● Rich, aromatic, briny, and herbal
  🍷
  Recommended White Wines from the Standard List:
  🟩
  Entry Tier ($150–299)
  Matanzas Creek Sauvignon Blanc 2021 – Sonoma County
  ● Price: $155
  ● Tasting Notes:
  ○ Lemon zest
  ○ Green apple
  ○ Gooseberry
  ○ Fresh herbs
  ○ Clean, mineral-driven finish
  💡 The acidity and herbaceous lift complement the fennel, seafood, and tomato complexity.
  🟨
  Mid Tier ($300–449)
  Cakebread Chardonnay 2020 – Napa Valley
  ● Price: $325
  ● Tasting Notes:
  ○ Ripe pear
  ○ Citrus blossom
  ○ Light vanilla oak
  ○ Creamy texture
  ○ Smooth finish
  💡 Balances shellfish richness and grilled sourdough with texture and lemon-butter compatibility.
  🟥
  Premium Tier ($450+)
  Far Niente Chardonnay 2019 – Napa Valley
  ● Price: $495
  ● Tasting Notes:
  ○ Baked apple
  ○ Meyer lemon
  ○ Toasted hazelnut
  ○ Buttery oak
  ○ Lingering citrus acidity
  💡 Luxurious pairing for lobster and king crab, yet bright enough for mussels and clams.
  Let me know if you'd like to see French reserve alternatives or a by-the-glass white pairing set."

● Alt: Suggests a comparable item if the requested one is unavailable, pulling only from PDFs. Trigger: "alt [item]".
  Example prompt: "alt herradura blanco tequila"
  Ideal response format:
  "🥃 alt[Herradura Blanco Tequila] — Closest Match from Verified Spirits List
  (Strictly from Spir_PDF.pdf)
  ✅
  Closest Available Match: Cazadores Blanco
  ● Category: Tequila — Blanco
  ● Region: Mexico
  ● Proof: 80
  ✅
  Justification:
  Herradura Blanco is a traditional lowland-style tequila, known for bright agave notes and subtle herbal complexity. While not listed on the Ocean 48 spirits menu, Cazadores Blanco offers the closest match based on:
  ● Same category: Blanco (unaged) tequila
  ● Agave-forward profile: Crisp, clean, citrus-laced finish
  ● Intended use: Excellent in cocktails or neat — just like Herradura Blanco
  ● Authenticity: Produced in Jalisco, Mexico, using similar methods
  Let me know if you'd prefer a reposado or añejo substitution instead."

● Bottle/Spirit Inquiries: Handles general bottle pairing or spirit inquiries in the 3-tier format, pulling from SpirPDF.pdf or wine PDFs with pricing, descriptions, region, and proof (if listed). Does not list unavailable brands.

● Tip Tracker: Allows users to log and track tips per individual, including parameters like gross sales, gross tips, net sales, net tips, table numbers, cover counts, PPA, and tip percentage. Provides summaries, trends, and correlations with upsells. Trigger: Commands like "log tip [amount] [table] [covers] [sales]" or "track [parameter]"

● 911 Lrg: Conveys the exact item description and allergies from source docs for an item utilizing designated abbreviations, breaking down sauces or garnishes on a component and allergy basis. Deconstructs the full dish into all components with potential allergens listed. Cross-references PDFs and includes disclaimers. Trigger: "911 lrg [item name]".
  Example prompt: "911 lrg stack"
  Ideal response format:
  "🛑 911 lrg[STACK] — Full Allergen Breakdown
  (Strictly verified using FOOD_MENU_PDF.pdf, Allergies PDF.pdf, and LRG_Reverse_indx.pdf)
  Disclaimer: Not medical advice.
  🦀
  Dish:
  Tomato and King Crab Stack
  Abbreviation: STACK
  🍽
  Description (Verified):
  ● Two thick slices of heirloom tomato
  ● Topped with ~3 oz of white balsamic–marinated king crab meat
  ● Drizzled with herb purée
  ● Garnished with scallions and avocado
  🧾
  Component Breakdown:
  🟩
  King Crab Meat (marinated):
  ● White balsamic marinade
  ● Primary allergen: Shellfish
  🟨
  Herb Purée:
  (No exact ingredient breakdown in PDFs — allergens derived from confirmed PDF listing)
  ● Confirmed to contain: Garlic, Onion
  🟪
  Garnishes:
  ● Avocado
  ● Scallions (onion family)
  ⚠️
  Confirmed Allergens (in both PDFs):
  ● Dairy
  ● Shellfish
  ● Onion
  ● Garlic
  ✅ All 4 allergens are consistently listed in both Allergies PDF.pdf and LRG_Reverse_indx.pdf.
  Let me know if you'd like to request a modified version without the herb purée or avocado for allergy-sensitive guests."

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
    print(f"Training with file: {file_path}")
    try:
        # Get filename for chunking strategy
        filename = os.path.basename(file_path)
        
        # Process the document based on its file extension
        if file_extension in [".csv", ".txt"]:
            if file_extension == ".csv":
                csv_data_frame = pd.read_csv(file_path)
                documents = [
                    Document(
                        content=row.to_json(),
                        metadata={'row_index': index, 'source': filename}
                    ) 
                    for index, row in csv_data_frame.iterrows()
                ]
            else:
                documents = process_document(file_path, file_extension)
                # Add source metadata to documents
                for doc in documents:
                    doc.metadata['source'] = filename
        else:
            documents = process_document(file_path, file_extension)
            # Add source metadata to documents
            for doc in documents:
                doc.metadata['source'] = filename

        # Apply specific chunking strategy for PDFs
        if file_extension == ".pdf":
            print(f"Applying specialized chunking for PDF: {filename}")
            documents = chunk_pdf_specifically(documents, filename)
            print(f"Created {len(documents)} chunks from PDF")
        else:
            # For non-PDF files, use general chunking if needed
            if file_extension == ".txt" and len(documents) == 1:
                # Split large text files
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                split_docs = []
                for doc in documents:
                    chunks = text_splitter.split_text(doc.page_content)
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            split_docs.append(Document(
                                content=chunk,
                                metadata={
                                    'source': filename,
                                    'chunk_type': 'text_split',
                                    'chunk_index': i,
                                    'total_chunks': len(chunks)
                                }
                            ))
                documents = split_docs

        # Initialize OpenAI Embeddings and Pinecone client
        embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        pc = Pinecone(api_key=PINECONE_KEY)

        index_name = PINECONE_INDEX
        namespace = PINECONE_NAMESPACE

        # Check if the index already exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if index_name not in existing_indexes:
            # Create new index only if it doesn't exist
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

        # Create a PineconeVectorStore from the chunked documents
        # This will add new documents to the existing index without deleting previous data
        PineconeVectorStore.from_documents(
            documents,
            embeddings_model,
            index_name=index_name,
            namespace=namespace,
        )
        
        return {
            "status": "OK",
            "filename": filename,
            "total_chunks": len(documents),
            "chunking_strategy": "specialized" if file_extension == ".pdf" else "general"
        }

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
     
# Define a document class to hold content and metadata
class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

def chunk_pdf_specifically(documents: List[Document], filename: str) -> List[Document]:
    """
    Advanced PDF chunking specifically designed for restaurant menu and wine list data.
    Uses different strategies based on the type of document.
    """
    chunked_docs = []
    
    # Determine document type based on filename
    filename_lower = filename.lower()
    
    if any(keyword in filename_lower for keyword in ['wine', 'rsv', 'std', 'btg']):
        # Wine list chunking strategy
        chunked_docs = chunk_wine_list(documents, filename)
    elif any(keyword in filename_lower for keyword in ['menu', 'food']):
        # Food menu chunking strategy
        chunked_docs = chunk_food_menu(documents, filename)
    elif any(keyword in filename_lower for keyword in ['allerg', 'lrg']):
        # Allergen information chunking strategy
        chunked_docs = chunk_allergen_info(documents, filename)
    elif any(keyword in filename_lower for keyword in ['spir']):
        # Spirits list chunking strategy
        chunked_docs = chunk_spirits_list(documents, filename)
    else:
        # Default chunking strategy
        chunked_docs = chunk_general_document(documents, filename)
    
    return chunked_docs

def chunk_wine_list(documents: List[Document], filename: str) -> List[Document]:
    """
    Specialized chunking for wine lists - preserves wine entries as complete units.
    """
    chunked_docs = []
    
    for doc in documents:
        content = doc.page_content
        page_num = doc.metadata.get('page', 0)
        
        # Split by wine entries (look for patterns like wine names, vintages, prices)
        # Common wine list patterns
        wine_patterns = [
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s+(\d{4})\s+([A-Za-z\s,]+)\s+\$(\d+)',  # Name Vintage Region $Price
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s+(\d{4})\s+\$(\d+)',  # Name Vintage $Price
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s+\$(\d+)',  # Name $Price
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s+(\d{4})',  # Name Vintage
        ]
        
        # Split content into sections
        sections = []
        current_section = ""
        
        lines = content.split('\n')
        for line in lines:
            # Check if this line starts a new wine entry
            is_wine_entry = any(re.match(pattern, '\n' + line) for pattern in wine_patterns)
            
            if is_wine_entry and current_section.strip():
                # Save current section and start new one
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line
            else:
                current_section += '\n' + line if current_section else line
        
        # Add the last section
        if current_section.strip():
            sections.append(current_section.strip())
        
        # Create chunks with overlap for context
        for i, section in enumerate(sections):
            if len(section) > 50:  # Only create chunks for substantial content
                chunked_docs.append(Document(
                    content=section,
                    metadata={
                        'source': filename,
                        'page': page_num,
                        'chunk_type': 'wine_entry',
                        'chunk_index': i,
                        'total_chunks': len(sections)
                    }
                ))
    
    return chunked_docs

def chunk_food_menu(documents: List[Document], filename: str) -> List[Document]:
    """
    Specialized chunking for food menus - preserves dish entries and categories.
    """
    chunked_docs = []
    
    for doc in documents:
        content = doc.page_content
        page_num = doc.metadata.get('page', 0)
        
        # Split by menu categories and dishes
        # Look for category headers (usually in caps or with special formatting)
        category_pattern = r'\n([A-Z][A-Z\s&]+)\n'
        
        # Split content by categories
        sections = re.split(category_pattern, content)
        
        current_category = ""
        for i, section in enumerate(sections):
            if i % 2 == 0:  # Even indices are content
                if section.strip() and current_category:
                    # Create chunk with category context
                    full_content = f"Category: {current_category}\n{section.strip()}"
                    chunked_docs.append(Document(
                        content=full_content,
                        metadata={
                            'source': filename,
                            'page': page_num,
                            'chunk_type': 'menu_item',
                            'category': current_category,
                            'chunk_index': i // 2
                        }
                    ))
            else:  # Odd indices are category names
                current_category = section.strip()
    
    return chunked_docs

def chunk_allergen_info(documents: List[Document], filename: str) -> List[Document]:
    """
    Specialized chunking for allergen information - preserves complete allergen entries.
    """
    chunked_docs = []
    
    for doc in documents:
        content = doc.page_content
        page_num = doc.metadata.get('page', 0)
        
        # Split by allergen entries (look for dish names followed by allergen info)
        # Pattern: Dish name followed by allergen codes or descriptions
        allergen_patterns = [
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s*[:\-]\s*([A-Za-z\s,]+)',  # Dish: Allergens
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s+([A-Z]{2,})',  # Dish ALLERGEN_CODES
        ]
        
        # Split content into allergen entries
        entries = []
        current_entry = ""
        
        lines = content.split('\n')
        for line in lines:
            # Check if this line starts a new allergen entry
            is_allergen_entry = any(re.match(pattern, '\n' + line) for pattern in allergen_patterns)
            
            if is_allergen_entry and current_entry.strip():
                if current_entry.strip():
                    entries.append(current_entry.strip())
                current_entry = line
            else:
                current_entry += '\n' + line if current_entry else line
        
        # Add the last entry
        if current_entry.strip():
            entries.append(current_entry.strip())
        
        # Create chunks
        for i, entry in enumerate(entries):
            if len(entry) > 20:  # Only create chunks for substantial content
                chunked_docs.append(Document(
                    content=entry,
                    metadata={
                        'source': filename,
                        'page': page_num,
                        'chunk_type': 'allergen_entry',
                        'chunk_index': i,
                        'total_chunks': len(entries)
                    }
                ))
    
    return chunked_docs

def chunk_spirits_list(documents: List[Document], filename: str) -> List[Document]:
    """
    Specialized chunking for spirits lists - preserves spirit entries with pricing.
    """
    chunked_docs = []
    
    for doc in documents:
        content = doc.page_content
        page_num = doc.metadata.get('page', 0)
        
        # Split by spirit entries (look for brand names, types, prices)
        spirit_patterns = [
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s+([A-Za-z\s]+)\s+\$(\d+)',  # Brand Type $Price
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s+\$(\d+)',  # Brand $Price
            r'\n([A-Z][A-Za-z\s&\.\-\']+)\s+(\d+%)',  # Brand Proof
        ]
        
        # Split content into spirit entries
        entries = []
        current_entry = ""
        
        lines = content.split('\n')
        for line in lines:
            # Check if this line starts a new spirit entry
            is_spirit_entry = any(re.match(pattern, '\n' + line) for pattern in spirit_patterns)
            
            if is_spirit_entry and current_entry.strip():
                if current_entry.strip():
                    entries.append(current_entry.strip())
                current_entry = line
            else:
                current_entry += '\n' + line if current_entry else line
        
        # Add the last entry
        if current_entry.strip():
            entries.append(current_entry.strip())
        
        # Create chunks
        for i, entry in enumerate(entries):
            if len(entry) > 30:  # Only create chunks for substantial content
                chunked_docs.append(Document(
                    content=entry,
                    metadata={
                        'source': filename,
                        'page': page_num,
                        'chunk_type': 'spirit_entry',
                        'chunk_index': i,
                        'total_chunks': len(entries)
                    }
                ))
    
    return chunked_docs

def chunk_general_document(documents: List[Document], filename: str) -> List[Document]:
    """
    General chunking strategy for other document types.
    """
    chunked_docs = []
    
    # Use RecursiveCharacterTextSplitter for general documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        page_num = doc.metadata.get('page', 0)
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                chunked_docs.append(Document(
                    content=chunk,
                    metadata={
                        'source': filename,
                        'page': page_num,
                        'chunk_type': 'general',
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                ))
    
    return chunked_docs

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
# @app.get("/verify_system_prompt")
# async def verify_system_prompt():
#     return {
#         "system_prompt_active": True,
#         "reality_filter_enforced": True,
#         "pdf_first_mode": True,
#         "ingested_pdfs": [
#             "FOOD_MENU_PDF.pdf",
#             "Allergies PDF.pdf",
#             "LRG_Reverse_indx.pdf",
#             "STD_PDF_7_9_20.pdf",
#             "RSV_Wine 7_9_20.pdf",
#             "BTG_list_PDF.pdf",
#             "Spir_PDF.pdf"
#         ],
#         "directive": "All responses must be document-verified and cite only the attached PDFs. If any part is unverified, label the entire response. If information is missing, respond with 'I cannot verify this.' or 'No data available in Ocean 48 documentation.'"
#     }

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
        
        # Train with chunking and get detailed results
        result = train(local_filename, file_extension)
        
        return {
            "message": f"'{file.filename}' uploaded successfully and training completed.",
            "training_details": result,
            "chunking_applied": result.get("chunking_strategy", "none"),
            "total_chunks_created": result.get("total_chunks", 0)
        }
    except Exception as e:
        return {"error": str(e)}

# Endpoint for checking server status
@app.get("/")
async def hello_world():
    return {"status": "Docker Server is running..."}

# Endpoint for getting chunking statistics and index information
@app.get("/chunking-stats")
async def get_chunking_stats():
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_KEY)
        index_name = PINECONE_INDEX
        namespace = PINECONE_NAMESPACE
        
        # Get index statistics
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        # Get namespace statistics
        namespace_stats = stats.get('namespaces', {}).get(namespace, {})
        
        return {
            "index_name": index_name,
            "namespace": namespace,
            "total_vectors": stats.get('total_vector_count', 0),
            "namespace_vectors": namespace_stats.get('vector_count', 0),
            "dimension": stats.get('dimension', 0),
            "metric": stats.get('metric', ''),
            "chunking_strategies": {
                "wine_list": "Preserves complete wine entries with pricing and region info",
                "food_menu": "Preserves dish entries with category context",
                "allergen_info": "Preserves complete allergen entries with dish associations",
                "spirits_list": "Preserves spirit entries with pricing and proof info",
                "general": "Uses RecursiveCharacterTextSplitter with 1000 char chunks"
            },
            "supported_file_types": {
                "pdf": "Specialized chunking based on content type",
                "csv": "Row-based chunking with JSON metadata",
                "txt": "General text splitting with overlap",
                "docx": "Document-based chunking"
            }
        }
    except Exception as e:
        return {"error": f"Failed to get chunking stats: {str(e)}"}

# AWS Lambda handler
handler = Mangum(app)