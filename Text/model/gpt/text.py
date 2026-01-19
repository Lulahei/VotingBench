from openai import OpenAI
client = OpenAI(
    api_key= "sk-GqG86a30a2246f331aee15a1c982da61c3ab83b868fbQOlZ",
    base_url="https://api.gptsapi.net/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://gitee.com/lulahei/img/raw/master/images/202502281312256.png",
                    },

                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://gitee.com/lulahei/img/raw/master/images/202502281321540.png",
                    },

                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])