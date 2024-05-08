from gpt_form_filler.openai_client import gpt_response_to_json


# Test function for the first scenario
def test_json_with_extra_output():
    test_json_with_extra_output = """{
        "name": "Marco",
        "mnemonic": "Fashion Italy",
        "mnemonic_explanation": "Marco has an Italian name and he works in the fashion industry.",
        "industry": "Fashion",
        "role": "Unknown",
        "vibes": "Neutral",
        "priority": 2,
        "follow_ups": null,
        "needs": [
            "None mentioned."
        ]
    }"""
    test_res = gpt_response_to_json(test_json_with_extra_output)
    assert test_res["name"] == "Marco"


# Test function for the second scenario
def test_json_in_md_ticks():
    test_json_in_md_ticks = """```json
    {
        "pricing_tiers": "Free of charge trial, Fully functional for 30 days. After the trial, ...",
        "search_terms": "Filter data Google Sheets, multiple VLOOKUP, filter values, extract data,",
        "tags": "Data Processing, Google Sheets, Filtering, VLOOKUP, Data Extraction",
        "main_integrations": "Google Sheets",
        "overview_summary": "An advanced alternative to standard Google Sheets lookup functions, ..."
    }
    ```"""
    test_res_md = gpt_response_to_json(test_json_in_md_ticks)
    assert test_res_md["main_integrations"] == "Google Sheets"
