from flask import Flask, request, jsonify
from langchain_mistralai import ChatMistralAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings: deprecated
# from langchain.vectorstores import FAISS:  deprecated
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import PyPDF2 
import os

app = Flask(__name__)

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
    api_key=os.getenv("MISTRAL_API_KEY")
)

db_uri = "mysql+mysqlconnector://root:password@host:port/database_name"
db = SQLDatabase.from_uri(db_uri)
execute_query = QuerySQLDataBaseTool(db=db)
generate_query = create_sql_query_chain(llm, db)

def get_user_data(phone_number):
    """Fetch user data from the database based on the provided name."""
    query = generate_query.invoke({"question": f"Select all data for the user where the phone_number is '{phone_number}' "})
    context = execute_query.run(query)
    return context

pdfreader = PyPDF2.PdfReader('docs/RBL Retail-Collection-Policy.pdf')
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=50, length_function=len)
texts = text_splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
document_search = FAISS.from_texts(texts, embeddings)

def get_response(user_query, context, doc_context):
    docs = document_search.similarity_search(user_query)
    combined_context = context + "\n\nDocuments: " + "\n".join([doc.page_content for doc in docs])

    llm_query_res_template = """
        Answer the question based on the context below. If the question cannot be answered using the information provided, reply with "I don't know". Also, make sure to answer the following questions considering the history of the conversation:
        Instructions:
        1. Use precise financial language and ensure clear, accurate information.
        2. Facilitate payments: If needed, ask for payment details and guide the customer through the process.
        3. Offer solutions: If the customer is struggling, provide options like grace periods, payment restructuring, or deadline extensions.
        4. Keep responses short and to the point.
        5. Ensure confidentiality and remind the customer to keep their payment details secure.

        Context: {context}
        Question: {user_query}
        Doc context: {combined_context}
        Answer:
    """

    prompt_query_res_template = ChatPromptTemplate.from_template(llm_query_res_template)
    llm_chain = prompt_query_res_template | llm | StrOutputParser()

    return llm_chain.invoke({
        "user_query": user_query,
        "context": context,
        "combined_context": combined_context,
    })

# Define the Flask endpoint for user queries
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    phone_number = data.get('phone_number')
    user_query = data.get('user_query')
    context = get_user_data(phone_number)
    doc_context = document_search.similarity_search(user_query)
    response = get_response(user_query, context, doc_context)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
