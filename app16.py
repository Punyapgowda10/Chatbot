from flask import Flask, render_template, request, jsonify, url_for
import os
import matplotlib.pyplot as plt
import networkx as nx
import textwrap
import uuid
import numpy as np
from huggingface_hub import InferenceClient
import re

app = Flask(__name__)

# Directory to store flowchart images
FLOWCHART_DIR = "static/flowcharts"
os.makedirs(FLOWCHART_DIR, exist_ok=True)

# Initialize HuggingFace client
API_KEY = "API"  # Replace with your API key
client = InferenceClient(api_key=API_KEY)

# Function to create a visually appealing flowchart
def create_flowchart(text_content):
    G = nx.DiGraph()
    lines = [line.strip() for line in text_content.split('\n') if line.strip() and not line.startswith("Here is the flowchart")]
    node_counter = 0
    positions = {}
    current_x, current_y = 0, 0
    level_height = 1.5  # Space between nodes adjusted for no gaps

    # Add start node
    start_node = f'node_{node_counter}'
    G.add_node(start_node, label='Start', node_type='terminal')
    positions[start_node] = (current_x, current_y)
    last_node = start_node
    node_counter += 1
    current_y -= level_height

    for line in lines:
        line = re.sub(r'[+\-|]+', '', line).strip()  # Remove unwanted ASCII symbols
        if not line:
            continue

        current_node = f'node_{node_counter}'
        wrapped_label = '\n'.join(textwrap.wrap(line, width=25))  # Adjust width for better readability

        if '?' in line:  # Decision node
            G.add_node(current_node, label=wrapped_label, node_type='decision')
        else:  # Regular process node
            G.add_node(current_node, label=wrapped_label, node_type='process')

        positions[current_node] = (current_x, current_y)
        G.add_edge(last_node, current_node)
        last_node = current_node
        node_counter += 1
        current_y -= level_height

    # Add end node
    end_node = f'node_{node_counter}'
    G.add_node(end_node, label='End', node_type='terminal')
    positions[end_node] = (current_x, current_y)
    G.add_edge(last_node, end_node)

    # Plot the flowchart
    plt.figure(figsize=(10, len(positions) * 1.5), dpi=150)
    ax = plt.gca()
    ax.set_facecolor('white')

    # Define node colors and shapes
    node_colors = {
        'decision': '#FFD966',  # Yellow
        'terminal': '#93C47D',  # Green
        'process': '#6FA8DC',  # Blue
    }
    node_shapes = {
        'decision': 'd',  # Diamond
        'terminal': 'o',  # Circle
        'process': 's',  # Square
    }

    for node_type in ['terminal', 'process', 'decision']:
        node_list = [node for node in G.nodes() if G.nodes[node].get('node_type') == node_type]
        nx.draw_networkx_nodes(
            G, positions, nodelist=node_list,
            node_color=node_colors[node_type],
            node_size=5000,  # Larger shapes
            node_shape=node_shapes[node_type],
            edgecolors='black'
        )

    nx.draw_networkx_edges(G, positions, arrowstyle='-|>', arrowsize=20, edge_color='black', width=2)
    nx.draw_networkx_labels(G, positions, labels=nx.get_node_attributes(G, 'label'), font_size=10, font_weight='bold')

    plt.axis('off')
    plt.tight_layout()

    # Save the flowchart
    os.makedirs(FLOWCHART_DIR, exist_ok=True)
    unique_id = str(uuid.uuid4())[:8]
    flowchart_path = os.path.join(FLOWCHART_DIR, f"flowchart_{unique_id}.png")
    plt.savefig(flowchart_path, bbox_inches='tight', dpi=300)
    plt.close()
    return f"/static/flowcharts/flowchart_{unique_id}.png"

# Function to generate chatbot response
def response_generator(prompt):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct",
        messages=messages,
        max_tokens=1000
    )
    response = completion.choices[0].message["content"]

    # Extract the flowchart content
    flowchart_match = re.search(r'(?s)### Flowchart.*?```(.*?)```', response)
    if flowchart_match:
        flowchart_text = flowchart_match.group(1).strip()
        print(f"Flowchart detected:\n{flowchart_text}")

        # Generate flowchart image
        image_path = create_flowchart(flowchart_text)
        if image_path:
            return {
                "type": "flowchart",
                "content": response.replace(flowchart_match.group(0), "").strip(),
                "flowchart_path": image_path
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

    assistant_response = response_generator(user_message)
    return jsonify(assistant_response)

if __name__ == "__main__":
    app.run(debug=True)
