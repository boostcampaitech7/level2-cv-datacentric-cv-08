from flask import Flask, request, jsonify
import psutil
import subprocess
import threading
import time

app = Flask(__name__)

#각 서버에 맞게 호출명 1~4로 변경해야함
@app.route('/server1-status', methods=['POST'])
def server_status():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    gpu_usage = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True).decode("utf-8").strip()

    response = {
        "response_type": "in_channel",
        "text": f"서버 상태: CPU 사용량 {cpu_usage}%, 메모리 사용량 {memory_usage}%, GPU 사용량 {gpu_usage}%"
    }
    return jsonify(response)

def run_flask():
    app.run(host='0.0.0.0', port=5002)

def run_cloudflared():
    tunnel = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:5002"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True  
    )

    # 여기서 나온 URL을 링크를 나한테 보내주면 슬랙봇에 추가가능
    for line in tunnel.stdout:
        print(line.strip())
        if "https://" in line:
            print("Cloudflare Tunnel URL:", line.strip())
        

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    cloudflared_thread = threading.Thread(target=run_cloudflared)

    flask_thread.start()
    cloudflared_thread.start()

    flask_thread.join()
    cloudflared_thread.join()
