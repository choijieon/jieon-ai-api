import os
from flask import Flask, request, jsonify
from openai import OpenAI
from supabase import create_client, Client
import numpy as np
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

conversation_history = []

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def ask_jieon(question):
    question_embedding = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    data = supabase.table("jieon_chunks").select("*").execute()
    results = data.data

    ranked = sorted(
        results,
        key=lambda x: cosine_similarity(json.loads(x["embedding"]), question_embedding),
        reverse=True
    )
    top_chunks = [r["content"] for r in ranked[:5]]

    context = "\n\n".join(top_chunks)
    conversation_history.append({"role": "user", "content": question})

    messages = [
        {"role": "system", "content": "You're an AI who knows Jieon Choi's background. Use context and reasoning."},
        {"role": "user", "content": f"Context:\n{context}"}
    ] + conversation_history

    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages
    )
    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})
    return answer

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    answer = ask_jieon(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
