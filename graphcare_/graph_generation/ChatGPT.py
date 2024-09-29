import os
import openai

with open("../../resources/openai.key", 'r') as f:
    key = f.readlines()[0][:-1]

class ChatGPT:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        openai.api_key = "sk-LVqjx8TbyNiBLiWdEdF4E20eB62041938e4c745fB6E87f58"
        # API的url
        openai.api_base = "https://one.gptgod.work/v1"
        self.messages = []

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4o",
            messages=self.messages
        )
        # self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]