# A real-world test
import os

from dotenv import load_dotenv

from gpt_form_filler.form import FieldDefinition, FormDefinition
from gpt_form_filler.openai_client import OpenAiClient

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai_client = OpenAiClient(open_ai_api_key=OPEN_AI_API_KEY)

# TEST FORMS (doing it here to prevent circular dependency)
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
food_log_result, test_err = openai_client.fill_in_multi_entry_form(
    form=test_form, text=datadump
)
assert test_err is None
for food_log_form_data in food_log_result:
    print(food_log_form_data.to_display_tuples())

assert len(food_log_result) == 3
# [('Ingredient', 'chicken'), ('Amount', 'fair amount'), ('Has Gluten?', False)]
# [('Ingredient', 'sesame oil'), ('Amount', 'None'), ('Has Gluten?', False)]
# [('Ingredient', 'tortilla chips'), ('Amount', 'some'), ('Has Gluten?', True)]
