import requests
import base64
# 替换为您的API密钥
OPENAI_API_KEY = "sk-lui3w1QQftJEKZ51596cA4E603A54398B5C0B44a666dE685"
# 图像文件路径
image_path = "/chanxueyan/ano_people/datasets/Data-DeQA-Score/KONIQ/images/10020891105.jpg"
# 将图像转换为base64编码
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
# 调用参数
url = "http://192.168.9.180:8123/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": "qwen3_VL_235B",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请描述这张图片的内容"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
}
# 发送请求
response = requests.post(url, headers=headers, json=data)
# 打印响应结果
print("status:", response.status_code)
print("content-type:", response.headers.get("content-type"))
print("text head:", response.text[:2000])

print(response.json())