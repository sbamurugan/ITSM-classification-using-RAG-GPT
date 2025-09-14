from flask import Flask, request, render_template
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
import json
import tiktoken

load_dotenv()
core42_key = os.getenv("CORE42_API_KEY")
print("CORE42_API_KEY =", core42_key)
os.environ["OPENAI_API_KEY"] = "xxx"  # dummy key for langchain SDK
app = Flask(__name__, template_folder="template")

# --- Embedding & Vector Store ---
embedding = OpenAIEmbeddings(model="text-embedding-3-large",api_key=core42_key, openai_api_base="https://api.core42.ai/v1")
vectordb = Chroma(
    persist_directory="./helpdesk4_chromadb_store",
    embedding_function=embedding,
    collection_name="helpdesk_requests4"
)

# --- Prompt using structured JSON context ---
custom_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are an IT service desk assistant. You are provided with a JSON array of past tickets (including metadata) and a new ticket query.

Instructions:
1. Always select **Support Group, Category, Subcategory, Priority, Service Category** only from the provided context. Do NOT invent any new values.
2. If a value is missing in the context, return "None".
3. Priority must be one of: Critical, High, Medium, Low. If missing, return "None".
4. Summarize the ticket in 1–2 lines using only the Query text.
5. Determine **Request Type** based on the query:
   - If the query indicates a service is not working, broken, or there is an outage, classify it as "Incident".
   - If the query is for a new service, access, or configuration change, classify it as "Service Request".
   - If unclear, return "Service Request".

Context JSON:
{context}

Query:
{query}

Return strictly in this format:
Support Group: <Group or None>
Priority: <Critical/High/Medium/Low or None>
Category: <Category or None>
Subcategory: <Subcategory or None>
Service Category: <Service Category or Others>
Request Type: <Incident/Service Request>
Summary: <Short summary of the query>

"""
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.1,api_key=core42_key, openai_api_base="https://api.core42.ai/v1" )
llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    result = ""
    sources = []

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            # --- Retrieve top 3 similar documents ---
            docs = vectordb.similarity_search(query, k=10)
            # --- Build structured JSON context from metadata ---
            context_list = []
            for doc in docs:
                md = doc.metadata
                context_list.append({
                    "Subject": md.get("text",""),
                    "Group": md.get("Group","None"),
                    "Category": md.get("Category","None"),
                    "Subcategory": md.get("Subcategory","None"),
                    "Priority": md.get("Priority","None"),
                    "Service Category": md.get("Service Category","None")
                })
            print(context_list)
            context_json = json.dumps(context_list, indent=2)
            encoding = tiktoken.encoding_for_model("gpt-4")

            # Count tokens
            tokens = encoding.encode(context_json)
            print("🔢 Token count:", len(tokens))
            # --- Run LLM ---
            response = llm_chain.run({"context": context_json, "query": query})
            result = response

            # --- Show sources with metadata ---
            sources = [
                f"{doc.metadata.get('text','')} ({doc.metadata.get('Category','None')} / {doc.metadata.get('Subcategory','None')})"
                for doc in docs
            ]

    return render_template("chat.html", query=query, result=result, sources=sources)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8990, debug=True)
