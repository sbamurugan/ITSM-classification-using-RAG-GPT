from flask import Flask, request, render_template
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from dotenv import load_dotenv
import tiktoken
import os

load_dotenv()

app = Flask(__name__, template_folder="template")

# -------------------------
# 🔹 Embedding & Vector Store
# -------------------------
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory="./helpdesk1chromadb_store",
    embedding_function=embedding
)

# -------------------------
# 🔹 LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4", temperature=0.1)

# -------------------------
# 🔹 Redundancy Filter
# -------------------------
redundancy_filter = EmbeddingsRedundantFilter(embeddings=embedding, similarity_threshold=0.7)

# -------------------------
# 🔹 Token counter helper
# -------------------------
def count_tokens(texts):
    enc = tiktoken.get_encoding("cl100k_base")  # same tokenizer GPT-4 uses
    return sum(len(enc.encode(t)) for t in texts)

# -------------------------
# 🔹 Custom Prompt for Issue Classification
# -------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are an IT service desk assistant. Given the following context from internal documents and a ticket description (query), classify and route the issue.

Perform the following:
1. Identify the relevant:
    - Support Group
    - Priority (Critical, High, Medium, Low)
2. Summarize the issue in 1–2 lines.

Use the given context to extract accurate values.

Return output in this format:
Support Group: <Group Name or None>
Priority: <Critical/High/Medium/Low or None>
Summary: <Short summary of issue>

Context:
{context}

Query:
{query}

Answer:
"""
)

llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
combine_docs_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

# -------------------------
# 🔹 Manual Reranking Function
# -------------------------
def rerank_docs(query, docs, llm_model):
    """
    Return documents sorted by relevance score (0-1) using the LLM.
    """
    scored_docs = []
    for doc in docs:
        prompt = f"""
Rate the relevance of the following document to the query on a scale from 0 (irrelevant) to 1 (highly relevant).
Return only a float number.

Query: {query}
Document: {doc.page_content}
"""
        score_text = llm_model.call_as_llm(prompt)
        try:
            score = float(score_text.strip())
        except:
            score = 0
        scored_docs.append((score, doc))
    # Sort descending by score
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored_docs]

# -------------------------
# 🔹 Flask App
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    result = ""
    sources = []
    token_stats = {}

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            # Step 1: Retrieve raw docs
            raw_docs = vectordb.similarity_search(query, k=8)
            raw_texts = [doc.page_content for doc in raw_docs]
            print(raw_docs)
            # Step 2: Rerank documents by relevance
            reranked_docs = rerank_docs(query, raw_docs, llm)
            print(reranked_docs)
            # Step 3: Apply redundancy filter
            filtered_docs = redundancy_filter.transform_documents(reranked_docs)
            filtered_texts = [doc.page_content for doc in filtered_docs]

            # Step 4: Count tokens
            before_tokens = count_tokens([doc.page_content for doc in reranked_docs])
            after_tokens = count_tokens(filtered_texts)
            token_stats = {
                "before": before_tokens,
                "after": after_tokens,
                "saved": before_tokens - after_tokens,
                "reduction_percent": round((1 - after_tokens / before_tokens) * 100, 2) if before_tokens else 0
            }
            print(token_stats)
            # Step 5: Build context & generate response
            context = "\n\n".join(filtered_texts)
            response = llm_chain.run({"context": context, "query": query})
            result = response
            sources = filtered_texts

    return render_template("chat.html", query=query, result=result, sources=sources, token_stats=token_stats)

if __name__ == "__main__":
    app.run(debug=True)
