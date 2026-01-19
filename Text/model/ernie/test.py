import os
import qianfan


os.environ["QIANFAN_ACCESS_KEY"] = "ALTAKWkLQI0tjmjZgC69YkaiKX"
os.environ["QIANFAN_SECRET_KEY"] = "bf2cf6caafcf4cc782c37038ed2f47c0"

chat_comp = qianfan.ChatCompletion()
__name__ = "ernie"

resp = chat_comp.do(model="ERNIE-3.5-8K", messages=[{
    "role": "user",
    "content": "你好"
}])

print(resp["body"]["result"])
