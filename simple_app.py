from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Microgrid IDS</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; }
        h1 { font-size: 24px; }
        .data-row { display: flex; margin: 10px 0; }
        .label { font-weight: bold; width: 200px; }
        button { padding: 10px; margin: 5px; background: #f0f0f0; border: 1px solid #ddd; cursor: pointer; }
        .result { margin-top: 20px; padding: 20px; border: 1px solid #ddd; }
        .attack { color: red; }
        .normal { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1># Microgrid IDS - Cyber Attack Detection System</h1>
        <div class="data-row"><span class="label">- **Node ID**:</span> <span id="nodeId">101</span></div>
        <div class="data-row"><span class="label">- **Hop Count**:</span> <span id="hopCount">3</span></div>
        <div class="data-row"><span class="label">- **Packet Loss Rate**:</span> <span id="packetLoss">0.05</span></div>
        <div class="data-row"><span class="label">- **Data Rate (kbps)**:</span> <span id="dataRate">50</span></div>

        <button onclick="detect()">🔍 Detect Attack</button>
        <button onclick="normal()">Normal Sample</button>
        <button onclick="attack()">Attack Sample</button>

        <div id="result" class="result">Ready</div>
    </div>

    <script>
        function normal() {
            document.getElementById('packetLoss').innerText = '0.05';
            document.getElementById('dataRate').innerText = '50';
        }
        function attack() {
            document.getElementById('packetLoss').innerText = '0.98';
            document.getElementById('dataRate').innerText = '300';
        }
        async function detect() {
            const data = {
                packet_loss_rate: parseFloat(document.getElementById('packetLoss').innerText),
                data_rate_kbps: parseFloat(document.getElementById('dataRate').innerText)
            };
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await res.json();
            document.getElementById('result').innerHTML = 
                `<h3 class="${result.is_attack ? 'attack' : 'normal'}">${result.message}</h3>`;
        }
    </script>
</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    loss = data.get('packet_loss_rate', 0)
    rate = data.get('data_rate_kbps', 0)

    if loss > 0.5:
        return jsonify({'is_attack': True, 'message': '⚠️ Blackhole Attack Detected!'})
    elif rate > 100:
        return jsonify({'is_attack': True, 'message': '⚠️ Flooding Attack Detected!'})
    else:
        return jsonify({'is_attack': False, 'message': '✅ Normal Traffic'})


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n🚀 Server running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)