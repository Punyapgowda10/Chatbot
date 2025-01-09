from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import time

app = Flask(__name__)

# Function to generate a chatbot response
def response_generator(prompt):
    client = InferenceClient(api_key="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")

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    completion = client.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct",
        messages=messages,
        max_tokens=500
    )

    # Extract the assistant's message and format the response
    response = completion.choices[0].message["content"]

    # Ensure the response is step-by-step and formatted
    response_lines = response.split("\n")  # Split response into lines
    formatted_response = ""
    for line in response_lines:
        if line.strip():  # Add non-empty lines with bullet points
            formatted_response += f"• {line.strip()}\n"
            time.sleep(0.5)

    return formatted_response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Generate a response
    assistant_response = response_generator(user_message)
    return jsonify({"response": assistant_response})

if __name__ == "__main__":
    app.run(debug=True)