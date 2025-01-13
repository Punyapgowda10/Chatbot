from flask import Flask, render_template, request, jsonify, send_file
import os
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D
from huggingface_hub import InferenceClient
import re

app = Flask(__name__)

# Directory to store flowchart images
FLOWCHART_DIR = "static/flowcharts"
os.makedirs(FLOWCHART_DIR, exist_ok=True)

# Initialize HuggingFace client
API_KEY = "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"  # Replace with your API key
client = InferenceClient(api_key=API_KEY)

# Function to generate a flowchart
def generate_flowchart(flowchart_text):
    lines = [line.strip() for line in flowchart_text.split("\n") if line.strip()]

    fig, ax = plt.subplots(figsize=(10, len(lines) * 2))
    ax.axis("off")

    node_positions = []
    y = len(lines) - 0.5
    x = 0.5

    def get_shape_and_color(line):
        if "start" in line.lower():
            return "ellipse", "#ffcccc"
        elif "end" in line.lower():
            return "ellipse", "#ffcccc"
        elif "is" in line.lower() or "?" in line:
            return "diamond", "#fff2cc"
        else:
            return "rectangle", "#cce5ff"

    for line in lines:
        shape, color = get_shape_and_color(line)
        if shape == "ellipse":
            node = Ellipse((x, y), width=3, height=1.2, color=color, ec="black")
        elif shape == "diamond":
            node = Polygon(
                [(x - 1.5, y), (x, y + 0.8), (x + 1.5, y), (x, y - 0.8)],
                color=color,
                ec="black",
            )
        else:
            node = FancyBboxPatch(
                (x - 1.5, y - 0.6), 3, 1.2, boxstyle="round,pad=0.3", facecolor=color, edgecolor="black"
            )

        ax.add_patch(node)
        ax.text(x, y, line, ha="center", va="center", fontsize=10, wrap=True)
        node_positions.append((x, y))
        y -= 2

    for i in range(len(node_positions) - 1):
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[i + 1]
        ax.add_line(Line2D([x1, x2], [y1 - 0.6, y2 + 0.6], color="black", lw=1.5))

    ax.set_xlim(-1, 2)
    ax.set_ylim(-2, len(lines) + 1)
    ax.set_aspect("equal")

    image_path = os.path.join(FLOWCHART_DIR, f"flowchart_{int(time.time())}.png")
    plt.savefig(image_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return image_path

def response_generator(prompt):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct",
        messages=messages,
        max_tokens=1000,
    )
    response = completion.choices[0].message["content"]

    flowchart_pattern = re.compile(r"```[\s\S]*?```")
    flowchart_match = flowchart_pattern.search(response)

    if flowchart_match:
        flowchart_text = flowchart_match.group(0).strip("`")
        image_path = generate_flowchart(flowchart_text)
        return {
            "type": "flowchart",
            "content": response.replace(flowchart_match.group(0), "").strip(),
            "flowchart_path": image_path,
        }
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

    assistant_response = response_generator(user_message)
    return jsonify(assistant_response)

@app.route("/flowchart/<filename>")
def serve_flowchart(filename):
    return send_file(os.path.join(FLOWCHART_DIR, filename), mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
