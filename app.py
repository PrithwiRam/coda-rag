import os
from flask import Flask, request, jsonify, send_from_directory
from main import ACORAPipeline, sample_documents

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"), static_url_path="")

# Initialise pipeline once
pipeline = ACORAPipeline()
pipeline.initialize(sample_documents)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/query', methods=['POST'])
def query_api():
    data = request.get_json(force=True)
    prompt = data.get('prompt') or data.get('query', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    try:
        result = pipeline.query(prompt)
        return jsonify({
            "answer": result.get('answer', ''),
            "classification": result.get('classification', {}),
            "retrieval_calls": result.get('retrieval_calls', 0),
            "latency_ms": result.get('latency_ms', 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
