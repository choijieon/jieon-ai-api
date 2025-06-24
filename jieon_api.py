import os
import json
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = OpenAI(api_key=OPENAI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Jieon API is running. Use the /ask endpoint with a POST request to ask questions."

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    chat_history = data.get("chat_history", [])

    question_embedding = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    results = supabase.table("jieon_chunks").select("*").execute().data

    ranked = sorted(
        results,
        key=lambda x: cosine_similarity(json.loads(x["embedding"]), question_embedding),
        reverse=True
    )
    top_chunks = [r["content"] for r in ranked[:5]]

    context = "\n\n".join(top_chunks)
    messages = chat_history + [
        {"role": "user", "content": f"Use the context below to answer the question. Be thoughtful and infer details if not explicit.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"}
    ]

    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages,
        temperature=1
    )
    return jsonify({"response": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
