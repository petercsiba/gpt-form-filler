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
