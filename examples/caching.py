import os

from dotenv import load_dotenv

from gpt_form_filler.openai_client import CHEAPEST_MODEL, OpenAiClient

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

openai_client = OpenAiClient(open_ai_api_key=OPEN_AI_API_KEY)
test_prompt = "list neighborhoods in san francisco as json key: value where key is name and value is zip code"
sf_result_1 = openai_client.run_prompt(test_prompt, CHEAPEST_MODEL, print_prompt=True)
print(sf_result_1)
# run again, this should yield a cache hit
sf_result_2 = openai_client.run_prompt(test_prompt, CHEAPEST_MODEL, print_prompt=True)
assert sf_result_1 == sf_result_2
assert len(openai_client.all_prompts) == 1
