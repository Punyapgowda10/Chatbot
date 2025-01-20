from flask import Flask, render_template, request, jsonify, url_for 
import os
import matplotlib.pyplot as plt
import networkx as nx
import textwrap
import uuid
from huggingface_hub import InferenceClient
import re

app = Flask(__name__)

# Directory to store flowchart images
FLOWCHART_DIR = "static/flowcharts"
os.makedirs(FLOWCHART_DIR, exist_ok=True)

# Initialize HuggingFace client
API_KEY = "API"  # Replace with your API key
client = InferenceClient(api_key=API_KEY)

# Function to calculate dynamic node size based on text dimensions
def calculate_node_size(label, node_type):
    char_width = 30  # Approximate width of a character in pixels
    line_height = 15  # Approximate height of a line in pixels
    padding = 60  # Padding for shapes to ensure text fits nicely

    # Split label into lines and calculate maximum dimensions
    lines = label.split("\n")
    max_line_length = max(len(line) for line in lines)
    width = max_line_length * char_width + padding
    height = len(lines) * line_height + padding

    # Return size based on node type (e.g., circle or square)
    if node_type == 'terminal':  # Circle (Start/End)
        diameter = max(width, height)
        return diameter * 32  # Scale for circular area
    elif node_type == 'decision':  # Diamond
        return max(width, height) * 8  # Scale for diamond area
    else:  # Process (Square/Rectangle)
        return width * height / 4  # Scale for rectangular area

# Function to create a visually appealing flowchart
def create_flowchart_from_steps(steps):
    G = nx.DiGraph()
    node_counter = 0
    positions = {}
    current_x, current_y = 0, 0
    level_height = 65.5  # Space between nodes

    # Ensure Start and End nodes exist exactly once
    steps = ["Start"] + [step for step in steps if step.lower() not in ["start", "end"]] + ["End"]

    # Add nodes and edges from the steps
    last_node = None
    for step in steps:
        current_node = f'node_{node_counter}'
        wrapped_label = '\n'.join(textwrap.wrap(step, width=30))  # Wrap text for readability

        # Node type logic for Start, End, Decision, and Process nodes
        if step.lower() == "start":
            G.add_node(current_node, label="Start", node_type="terminal")
        elif step.lower() == "end":
            G.add_node(current_node, label="End", node_type="terminal")
        elif '?' in step:
            G.add_node(current_node, label=wrapped_label, node_type="decision")
        else:
            G.add_node(current_node, label=wrapped_label, node_type="process")

        # Position the node
        positions[current_node] = (current_x, current_y)

        # Connect the node to the last one
        if last_node is not None:
            G.add_edge(last_node, current_node)

        # Update counters
        last_node = current_node
        node_counter += 1
        current_y -= level_height

    # Plot the flowchart
    plt.figure(figsize=(12, len(steps) * 2.5), dpi=150)
    ax = plt.gca()
    ax.set_facecolor('white')

    # Define node colors and shapes
    node_colors = {
        'terminal': '#93C47D',  # Green for Start and End
        'decision': '#FFD966',  # Yellow for Decision
        'process': '#6FA8DC',  # Blue for Process
    }
    node_shapes = {
        'terminal': 'o',  # Circle for Start and End
        'decision': 'd',  # Diamond for Decision
        'process': 's',  # Square for Process
    }

    for node_type in ['terminal', 'process', 'decision']:
        node_list = [node for node in G.nodes() if G.nodes[node].get('node_type') == node_type]
        node_sizes = [calculate_node_size(G.nodes[node]['label'], node_type) for node in node_list]

        nx.draw_networkx_nodes(
            G, positions, nodelist=node_list,
            node_color=node_colors[node_type],
            node_size=node_sizes,
            node_shape=node_shapes[node_type],
            edgecolors='black'
        )

    # Draw edges with arrowheads
    nx.draw_networkx_edges(
        G,
        positions,
        arrowstyle='-|>',
        arrowsize=20,
        edge_color='black',
        width=2
    )
    # Draw node labels
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

# Function to process the chatbot response and generate a flowchart
def response_generator(prompt):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct",
        messages=messages,
        max_tokens=2000
    )
    response = completion.choices[0].message["content"]

    # Detect if the response contains a flowchart description
    if "flowchart" in prompt.lower():
        # Extract numbered steps from the response
        response_cleaned = re.sub(r"\*\*\*", "", response)
        steps = re.findall(r'\d+\.\s(.*?)\n', response)
        if not steps:
            # If no numbered steps, fall back to line-based splitting
            steps = [line.strip() for line in response.split("\n") if line.strip()]

        if steps:
            print(f"Flowchart Steps Detected:\n{steps}")
            image_path = create_flowchart_from_steps(steps)
            return {
                "type": "flowchart",
                "content": response,
                "flowchart_path": image_path
            }

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
