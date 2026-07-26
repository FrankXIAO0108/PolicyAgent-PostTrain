import os
from openai import OpenAI


class DeepSeekClient:


    def __init__(self):

        api_key = os.getenv(
            "DEEPSEEK_API_KEY"
        )


        if not api_key:

            raise ValueError(
                "Missing DEEPSEEK_API_KEY environment variable"
            )


        self.client = OpenAI(

            api_key=api_key,

            base_url=
            "https://api.deepseek.com"

        )



    def chat(
        self,
        prompt
    ):


        response = self.client.chat.completions.create(

            model="deepseek-v4-flash",

            messages=[

                {
                    "role":
                    "system",

                    "content":
                    "You are an expert AI agent verifier."
                },

                {
                    "role":
                    "user",

                    "content":
                    prompt
                }

            ],


            temperature=0

        )


        return (
            response
            .choices[0]
            .message
            .content
        )