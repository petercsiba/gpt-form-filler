# gpt-form-filler
Programmatically fill in annoying intake forms with data dumps or knowledge base


## Motivation
Repeated Manual Data Entry is not what humans like to do. 

What if you can update your CRM or fill in large intakes just from a voice note or a data dump?
This tool is an opinionated attempt on solving this niche problem.

## How it works
It is a bit like [Guardrails.ai](https://github.com/guardrails-ai/guardrails),
or one of the many Chrome/Firefox/Safari auto-fill options,
just addressing a more specific use case when:
* You usually have at least one page of data
* You might want to fill in multiple entries at once (e.g. all your notes from a day)
* The form is somewhat specific / needs to be created programmatically

## Implementations
It was previously used to:
* Meeting Notes to Hubspot API Contact Intake: [preview - requires HubSpot account](https://app.hubspot.com/ecosystem/43920988/marketplace/apps/_preview/voxana-voice-data-entry-2150554)
* Push rows into Google Sheets API based of the sheet header. 

# Current State: WIP
I wanted to re-use and opensource these utils so for now just shamelessly copy pasted from my private repo.

## Features:
* Caching GPT request, useful for re-runs
* TODO: Better DevX for the form object
* TODO: Use JSON mode for OpenAI (this was written before it came out so there is custom monkeypatched code)

## Installation
Install GPT Form Filler Python Library using pip:

```shell
pip install git+https://github.com/petercsiba/gpt-form-filler.git
```

## Example Usage
```python
from dotenv import load_dotenv
import os

from gpt_form_filler.form import FormDefinition, FieldDefinition
from gpt_form_filler.openai_client import OpenAiClient, CHEAPEST_MODEL

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai_client = OpenAiClient(open_ai_api_key=OPEN_AI_API_KEY)
test_prompt = "list neighborhoods in san francisco as json key: value where key is name and value is zip code"
sf_result_1 = openai_client.run_prompt(
    test_prompt, CHEAPEST_MODEL, print_prompt=True
)
print(sf_result_1)
# {
#   "Sunset District": "94122",
#   "Mission District": "94110",
#   "Marina District": "94123",
#   "North Beach": "94133",
#   "Hayes Valley": "94102",
#   "The Castro": "94114",
#   "Haight-Ashbury": "94117",
#   "Chinatown": "94108",
#   "SoMa": "94103",
#   "Pacific Heights": "94115"
# }

# run again, this should yield a cache hit
sf_result_2 = openai_client.run_prompt(
    test_prompt, CHEAPEST_MODEL, print_prompt=True
)
assert sf_result_1 == sf_result_2
# assert it was served from cache
assert len(openai_client.all_prompts) == 1

# EXAMPLE Form, `description` is passed to GPT prompt to obtain result `name` of type == `field_type` 
FOOD_LOG_FIELDS = [
    FieldDefinition(
        name="ingredient",
        field_type="text",
        label="Ingredient",
        description="one food item like you would see on an ingredients list",
    ),
    FieldDefinition(
        name="amount",
        field_type="text",
        label="Amount",
        description=(
            "approximate amount of the ingredient taken, if not specified it can be just using 'a bit' or 'some"
        ),
    ),
    FieldDefinition(
        name="has_gluten",
        field_type="bool",
        label="Has Gluten?",
        description="does this ingredient have gluten",
    ),
]
test_form = FormDefinition(form_name="food_log", fields=FOOD_LOG_FIELDS)

datadump = """
For lunch I had this nice ricebowl with fair amount of sauted chicken over sesame oil and some smashed tortilla
chips as a topping.
"""
# mutli-entry form can yield a list of results kinda like named-=entity-extraction (works much better with GPT4)
food_log_result, test_err = openai_client.fill_in_multi_entry_form(
    form=test_form, text=datadump
)
assert test_err is None
for food_log_form_data in food_log_result:
    # display tuples will use `label` instead of `name`
    print(food_log_form_data.to_display_tuples())

assert len(food_log_result) == 3
# [('Ingredient', 'chicken'), ('Amount', 'fair amount'), ('Has Gluten?', False)]
# [('Ingredient', 'sesame oil'), ('Amount', 'None'), ('Has Gluten?', False)]
# [('Ingredient', 'tortilla chips'), ('Amount', 'some'), ('Has Gluten?', True)]
```