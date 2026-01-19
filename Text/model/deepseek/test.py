from flask import Flask, request, jsonify, Response
from openai import OpenAI
import json

app = Flask(__name__)

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-72ff3b9692c245889a2e077b86dcb6fe",  # 建议从环境变量获取，不要硬编码
    base_url="https://api.deepseek.com"
)

@app.route('/api/chat', methods=['POST'])
def chat_completion():
    """
    DeepSeek聊天API端点
    请求格式:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ],
        "model": "deepseek-reasoner",  # 可选，默认为deepseek-reasoner
        "stream": false  # 可选，默认为false
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()

        # 验证必要参数
        if not data or 'messages' not in data:
            return jsonify({"error": "Missing required parameter: messages"}), 400

        # 设置默认值
        model = data.get('model', 'deepseek-reasoner')
        stream = data.get('stream', False)

        # 调用DeepSeek API
        response = client.chat.completions.create(
            model=model,
            messages=data['messages'],
            stream=stream
        )

        # 处理流式和非流式响应
        if stream:
            def generate():
                for chunk in response:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"

            return Response(generate(), mimetype='text/event-stream')
        else:
            return jsonify({
                "content": response.choices[0].message.content
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)