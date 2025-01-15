from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D
from huggingface_hub import InferenceClient
import re

app = Flask(__name__)

# Directory to store flowchart images
FLOWCHART_DIR = "static/flowcharts"
os.makedirs(FLOWCHART_DIR, exist_ok=True)

# Initialize HuggingFace client
API_KEY = "API"  # Replace with your API key
client = InferenceClient(api_key=API_KEY)

# Function to generate a flowchart using matplotlib
def generate_flowchart(flowchart_text):
    # Split flowchart into lines and remove empty lines
    lines = [line.strip() for line in flowchart_text.split("\n") if line.strip()]

    fig, ax = plt.subplots(figsize=(10, len(lines) * 1.5))
    ax.axis("off")

    # Node positions and styles
    y = len(lines) - 0.5
    x = 0.5
    node_positions = []

    # Define color and shape rules
    def get_shape_and_color(line):
        if "start" in line.lower() or "end" in line.lower():
            return "ellipse", "#ffcccc"  # Start/End
        elif "is" in line.lower() or "?" in line:
            return "diamond", "#fff2cc"  # Decision
        else:
            return "rectangle", "#cce5ff"  # Process

    # Draw nodes
    for line in lines:
        shape, color = get_shape_and_color(line)
        if shape == "ellipse":
            node = Ellipse((x, y), width=0.8, height=0.5, color=color, ec="black")
        elif shape == "diamond":
            node = FancyBboxPatch((x - 0.4, y - 0.25), 0.8, 0.5, boxstyle="round,pad=0.2", facecolor=color, edgecolor="black")
        else:
            node = FancyBboxPatch((x - 0.4, y - 0.25), 0.8, 0.5, boxstyle="round,pad=0.2", facecolor=color, edgecolor="black")

        ax.add_patch(node)
        ax.text(x, y, line, ha="center", va="center", fontsize=10)
        node_positions.append((x, y))
        y -= 1

    # Draw arrows
    for i in range(len(node_positions) - 1):
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[i + 1]
        ax.add_line(Line2D([x1, x2], [y1 - 0.25, y2 + 0.25], color="black", lw=1.5, alpha=0.8, marker="o"))

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(lines))
    ax.set_aspect("equal")

    # Save the image
    timestamp = int(time.time())
    image_path = os.path.join(FLOWCHART_DIR, f"flowchart_{timestamp}.png")
    try:
        plt.savefig(image_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Flowchart saved to: {image_path}")
        return f"flowcharts/flowchart_{timestamp}.png"
    except Exception as e:
        print(f"Error saving flowchart: {e}")
        return None

# Function to generate chatbot response
def response_generator(prompt):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct",
        messages=messages,
        max_tokens=1000
    )
    response = completion.choices[0].message["content"]
    print(f"AI Response: {response}")

    # Attempt to detect textual flowchart representation
    flowchart_pattern = re.compile(r"```[\s\S]*?```")
    flowchart_match = flowchart_pattern.search(response)

    if flowchart_match:
        flowchart_text = flowchart_match.group(0).strip("`")
        print(f"Flowchart Detected:\n{flowchart_text}")

        # Generate flowchart
        image_path = generate_flowchart(flowchart_text)
        if image_path:
            return {
                "type": "flowchart",
                "content": response.replace(flowchart_match.group(0), "").strip(),
                "flowchart_path": url_for('static', filename=image_path)
            }
        else:
            return {"type": "error", "content": "Failed to generate flowchart."}
    else:
        return {"type": "text", "content": response}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Generate response
    assistant_response = response_generator(user_message)
    return jsonify(assistant_response)

if __name__ == "__main__":
    app.run(debug=True)
