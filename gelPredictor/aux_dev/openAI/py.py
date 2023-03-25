import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
# prompt="A barefoot heels sticking out from under the blanket"
prompt="A  social worker is chilling on the canape like on the paints of  Lucian Freud"
response = openai.Image.create(
  prompt=prompt,
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)