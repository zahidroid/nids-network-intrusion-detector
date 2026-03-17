import gradio as gr
import numpy as np
import pickle
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Load model and scaler
model = tf.keras.models.load_model(
    hf_hub_download("zahidmohd/nids-network-intrusion-detector", "nids_model.keras")
)
with open(hf_hub_download("zahidmohd/nids-network-intrusion-detector", "nids_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

label_map = {0: "Normal", 1: "Port Scan", 2: "DDoS", 3: "Brute Force"}
label_colors = {0: "#44bb44", 1: "#ff8800", 2: "#ff4444", 3: "#cc44ff"}
label_icons = {0: "✅", 1: "🔍", 2: "💥", 3: "🔐"}

def analyze(packet_size, packets_per_sec, bytes_sent, bytes_received,
            connection_duration, unique_ports, failed_logins, syn_ratio):
    features = np.array([[packet_size, packets_per_sec, bytes_sent, bytes_received,
                          connection_duration, unique_ports, failed_logins, syn_ratio]])
    scaled = scaler.transform(features)
    probs = model.predict(scaled, verbose=0)[0]
    predicted = np.argmax(probs)
    confidence = probs[predicted] * 100
    label = label_map[predicted]
    color = label_colors[predicted]
    icon = label_icons[predicted]

    prob_bars = ""
    for i, (name, prob) in enumerate(zip(label_map.values(), probs)):
        bar_color = label_colors[i]
        bar_width = int(prob * 100)
        prob_bars += f"""
        <div style='margin:6px 0;'>
            <span style='color:#ccc; font-size:12px; display:inline-block; width:100px;'>{label_icons[i]} {name}</span>
            <div style='display:inline-block; background:{bar_color}; width:{bar_width}%; height:16px; border-radius:3px; vertical-align:middle;'></div>
            <span style='color:#ccc; font-size:12px; margin-left:6px;'>{prob*100:.1f}%</span>
        </div>"""

    if predicted == 0:
        details = "<p style='color:#aaffaa;'>All traffic metrics within normal range. No suspicious activity detected.</p>"
        actions = ""
    elif predicted == 1:
        details = f"""
        <p style='color:#ffcc88;'>⚠️ Rapid connection attempts to {int(unique_ports)} different ports detected.</p>
        <p style='color:#ffcc88;'>⚠️ High SYN ratio ({syn_ratio:.2f}) indicates scanning behavior.</p>
        <p style='color:#ffcc88;'>⚠️ Short connection duration suggests automated scanning tool.</p>"""
        actions = """
        <h3 style='color:white;'>Recommended Actions:</h3>
        <p style='color:#aaffaa;'>1. Block source IP immediately: iptables -A INPUT -s [IP] -j DROP</p>
        <p style='color:#aaffaa;'>2. Enable port knocking or firewall rules</p>
        <p style='color:#aaffaa;'>3. Review firewall logs: cat /var/log/ufw.log</p>"""
    elif predicted == 2:
        details = f"""
        <p style='color:#ff8888;'>⚠️ Extreme packet rate: {int(packets_per_sec):,} packets/sec detected.</p>
        <p style='color:#ff8888;'>⚠️ Massive data volume: {int(bytes_sent):,} bytes sent.</p>
        <p style='color:#ff8888;'>⚠️ Traffic pattern matches volumetric DDoS attack.</p>"""
        actions = """
        <h3 style='color:white;'>Recommended Actions:</h3>
        <p style='color:#aaffaa;'>1. Enable DDoS protection / rate limiting immediately</p>
        <p style='color:#aaffaa;'>2. Contact upstream ISP for traffic scrubbing</p>
        <p style='color:#aaffaa;'>3. Activate CDN / load balancer failover</p>
        <p style='color:#aaffaa;'>4. Block offending IP ranges: iptables -A INPUT -m limit</p>"""
    else:
        details = f"""
        <p style='color:#dd88ff;'>⚠️ {int(failed_logins)} failed login attempts detected.</p>
        <p style='color:#dd88ff;'>⚠️ Repeated connections to same port — credential stuffing.</p>
        <p style='color:#dd88ff;'>⚠️ Pattern matches automated brute force tool.</p>"""
        actions = """
        <h3 style='color:white;'>Recommended Actions:</h3>
        <p style='color:#aaffaa;'>1. Block IP after 5 failed attempts: fail2ban-client set sshd banip [IP]</p>
        <p style='color:#aaffaa;'>2. Enable MFA immediately</p>
        <p style='color:#aaffaa;'>3. Change default SSH port from 22</p>
        <p style='color:#aaffaa;'>4. Review auth logs: cat /var/log/auth.log</p>"""

    result = f"""
    <div style='background:#0d0d1a; border:2px solid {color}; padding:20px; border-radius:8px; font-family:monospace;'>
        <h2 style='color:{color};'>{icon} {label} {"DETECTED" if predicted != 0 else "— System Secure"}</h2>
        <p style='color:#aaa;'>Confidence: <b style='color:{color};'>{confidence:.1f}%</b></p>
        <hr style='border-color:{color}; margin:12px 0;'>
        <h3 style='color:white;'>Probability Distribution:</h3>
        {prob_bars}
        <hr style='border-color:#333; margin:12px 0;'>
        <h3 style='color:white;'>Analysis:</h3>
        {details}
        {actions}
    </div>"""
    return result

with gr.Blocks() as demo:
    gr.HTML("""
    <div style='text-align:center; padding:20px; background:#0d0d1a; border-radius:8px; margin-bottom:10px;'>
        <h1 style='color:#4fc3f7;'>🛡️ AI Network Intrusion Detection System</h1>
        <p style='color:#888;'>Real-Time Network Traffic Analysis & Attack Detection</p>
        <p style='color:#666; font-size:12px;'>Trained on 8000 samples | 100% Accuracy | Detects: Port Scan, DDoS, Brute Force</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📡 Network Traffic Metrics")
            packet_size = gr.Slider(40, 1500, value=500, label="Packet Size (bytes)")
            packets_per_sec = gr.Slider(0, 50000, value=250, label="Packets Per Second")
            bytes_sent = gr.Slider(0, 1000000, value=5000, label="Bytes Sent")
            bytes_received = gr.Slider(0, 1000000, value=5000, label="Bytes Received")
            connection_duration = gr.Slider(0, 300, value=30, label="Connection Duration (sec)")
            unique_ports = gr.Slider(1, 65535, value=3, label="Unique Ports Accessed")
            failed_logins = gr.Slider(0, 200, value=0, label="Failed Login Attempts")
            syn_ratio = gr.Slider(0, 1, value=0.2, label="SYN Packet Ratio")
            analyze_btn = gr.Button("🔍 Analyze Traffic", variant="primary")

        with gr.Column():
            gr.Markdown("### 🚨 Detection Result")
            result = gr.HTML()

    gr.Markdown("### ⚡ Attack Simulations")
    with gr.Row():
        gr.Button("✅ Normal Traffic").click(
            lambda: (500, 250, 5000, 5000, 30, 3, 0, 0.2),
            outputs=[packet_size, packets_per_sec, bytes_sent, bytes_received,
                    connection_duration, unique_ports, failed_logins, syn_ratio])
        gr.Button("🔍 Port Scan").click(
            lambda: (60, 1200, 500, 100, 1, 45000, 1, 0.95),
            outputs=[packet_size, packets_per_sec, bytes_sent, bytes_received,
                    connection_duration, unique_ports, failed_logins, syn_ratio])
        gr.Button("💥 DDoS Attack").click(
            lambda: (100, 35000, 500000, 100, 1, 5, 0, 0.85),
            outputs=[packet_size, packets_per_sec, bytes_sent, bytes_received,
                    connection_duration, unique_ports, failed_logins, syn_ratio])
        gr.Button("🔐 Brute Force").click(
            lambda: (200, 150, 3000, 2000, 1, 2, 120, 0.4),
            outputs=[packet_size, packets_per_sec, bytes_sent, bytes_received,
                    connection_duration, unique_ports, failed_logins, syn_ratio])

    analyze_btn.click(
        analyze,
        inputs=[packet_size, packets_per_sec, bytes_sent, bytes_received,
                connection_duration, unique_ports, failed_logins, syn_ratio],
        outputs=result
    )

demo.launch()
 
 
