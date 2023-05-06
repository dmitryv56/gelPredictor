import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
# prompt="A barefoot heels sticking out from under the blanket"
#prompt="A  naked social worker is chilling on the canape like on the paints of  Lucian Freud"
#prompt="a picture in the spirit of Breakfast on the Grass by C. Manet "
prompt ="a picture of night watch  in Rembrandt style"
prompt ="a picture of night watch  in Rembrandt style"
response = openai.Image.create(
  prompt=prompt,
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)