# TODO: use some logging library
# TODO(P0, devx, quality): Update chat-gpt model usage, ideally use those functions too!
#  * https://platform.openai.com/docs/guides/gpt/function-calling
#   Define a function called
#       extract_people_data(people: [
#           {name: string, birthday: string, location: string}
#       ]), to extract all people mentioned in a Wikipedia article.
# TODO(P1, devx): This Haystack library looks quite good https://github.com/deepset-ai/haystack
# TODO(P3, research, fine-tune): TLDR; NOT worth it. Feels like for repeated tasks it would be great to
#  speed up and/or cost save https://platform.openai.com/docs/guides/fine-tuning/advanced-usage
import datetime
import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import openai
import pytz
import tiktoken
from dotenv import load_dotenv

# TODO(P1, features): Add Assistant API
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from gpt_form_filler.form import FieldDefinition, FormData, FormDefinition, Option

# TODO(P1, devx): Keep this updated
# TODO(P2, specify organization id): Header OpenAI-Organization
DEFAULT_MODEL = "gpt-3.5-turbo-0125"
BEST_MODEL = "gpt-4"
BETTER_MODEL = "gpt-4-turbo-2024-04-09"
CHEAPEST_MODEL = "gpt-3.5-turbo-0125"
# Sometimes seems the newest models experience downtime-so try to backup.
BACKUP_MODEL = "gpt-3.5-turbo"
BACKUP_MODEL_AFTER_NUM_RETRIES = 3


# TODO(P1, quality): Higher numbers only really work with GPT-4, as lower models might get confused and treat the
#   options as form fields.
GPT_MAX_NUM_OPTION_FIELDS = 5


@dataclass
class PromptStats:
    request_time_ms: int = 0  # in milliseconds
    prompt_tokens: int = 0
    completion_tokens: int = 0
    millionth_dollar_estimate: int = 0

    total_requests: int = 0
    total_tokens: int = 0

    # TODO(P2, fun): Might be cool to translate it to dollars, I guess one-day usage based billing.
    def pretty_print(self):
        return (
            f"{self.total_requests} queries to LLMs ({self.total_tokens} tokens, "
            f"input: {self.prompt_tokens} output: {self.completion_tokens}) \n"
            f"${self.millionth_dollar_estimate / 1000000} spent estimate\n"
            f"in {self.request_time_ms / 1000:.2f} seconds total query time."  # noqa: E231
        )


def _get_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


# NOTE: This somewhat evolved from text-based GPT prompts to also include other API calls
class PromptCacheEntry(BaseModel):
    model: str
    prompt: str
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    request_time_ms: Optional[int] = 0
    result: Optional[str] = None

    def total_tokens(self) -> int:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return 0
        return self.prompt_tokens + self.completion_tokens

    def prompt_hash(self) -> str:
        return _get_prompt_hash(self.prompt)


class CacheStoreBase(ABC):
    """Abstract base class for the caching mechanism."""

    @abstractmethod
    def get_or_create(self, prompt: str, model: str) -> PromptCacheEntry:
        pass

    @abstractmethod
    def write_cache(self, entry: PromptCacheEntry) -> None:
        pass


class InMemoryCacheStore(CacheStoreBase):
    def __init__(self):
        self.cache = {}

    @staticmethod
    def _cache_key(prompt, model) -> Tuple[str, str]:
        return _get_prompt_hash(prompt), model

    def get_or_create(self, prompt: str, model: str) -> PromptCacheEntry:
        cache_key = InMemoryCacheStore._cache_key(prompt, model)
        if cache_key not in self.cache:
            entry = PromptCacheEntry(
                model=model,
                prompt=prompt,
            )
            self.cache[cache_key] = entry
        return self.cache[cache_key]

    def write_cache(self, entry: PromptCacheEntry) -> None:
        cache_key = InMemoryCacheStore._cache_key(entry.prompt, entry.model)
        self.cache[cache_key] = entry
        print(
            f"In-memory cache: written {entry.model}:{entry.prompt_hash}"
        )  # noqa: E231


class PromptCache:
    def __init__(
        self, cache_store: CacheStoreBase, prompt: str, model: str, print_prompt: bool
    ):
        self.cache_store = cache_store
        self.cache_entry = PromptCacheEntry(prompt=prompt, model=model)

        # somewhat redundant to cache_entry
        self.prompt = prompt
        self.model = model

        self.print_prompt: bool = print_prompt
        self.cache_hit: bool = False
        self.start_time: Optional[float] = None

    def __enter__(self):
        if self.print_prompt:
            loggable_prompt = self.prompt.replace("\n", " ")
            print(f"Asking {self.model} for: {loggable_prompt}")

        self.cache_entry = self.cache_store.get_or_create(
            prompt=self.prompt, model=self.model
        )
        if bool(self.cache_entry.result):
            self.cache_hit = True
            print(
                f"prompt_log: serving from cache {self.model}:{self.cache_entry.prompt_hash()}"  # noqa: E231
            )
            if self.cache_entry.prompt != self.prompt:
                print(
                    f"ERROR: hash collision for {self.cache_entry.prompt_hash()} for prompt {self.prompt}"
                )

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cache_entry.request_time_ms = int(1000 * (time.time() - self.start_time))
        if self.print_prompt:
            seconds = self.cache_entry.request_time_ms / 1000
            print(
                f"{self.model}: {seconds} seconds used {self.cache_entry.total_tokens()}"
            )

        if self.cache_entry.result is not None and not self.cache_hit:
            self.cache_store.write_cache(self.cache_entry)


class OpenAiClient:
    def __init__(
        self,
        open_ai_api_key: str,
        cache_store=InMemoryCacheStore(),
        force_no_print_prompt=False,
    ):
        print("OpenAiClient init")
        self.client = openai.OpenAI(api_key=open_ai_api_key)

        self.cache_store = cache_store
        self.all_prompts: List[PromptCacheEntry] = []

        self.force_no_print_prompt = force_no_print_prompt

    def _should_print_prompt(self, print_prompt_arg: bool):
        if self.force_no_print_prompt:
            return False
        return print_prompt_arg

    def sum_up_prompt_stats(self) -> PromptStats:
        stats = PromptStats()
        for prompt_cache_entry in self.all_prompts:
            input_tok = prompt_cache_entry.prompt_tokens
            output_tok = prompt_cache_entry.completion_tokens
            stats.prompt_tokens += input_tok
            stats.completion_tokens += output_tok
            stats.request_time_ms += prompt_cache_entry.request_time_ms
            # https://openai.com/pricing
            if prompt_cache_entry.model.startswith("gpt-4-turbo"):
                stats.millionth_dollar_estimate += 10 * input_tok + 30 * output_tok
            elif prompt_cache_entry.model.startswith("gpt-4-32k"):
                stats.millionth_dollar_estimate += 60 * input_tok + 120 * output_tok
            elif prompt_cache_entry.model.startswith("gpt-4"):
                stats.millionth_dollar_estimate += 30 * input_tok + 60 * output_tok
            elif prompt_cache_entry.model.startswith("gpt-3.5-turbo-instruct"):
                stats.millionth_dollar_estimate += 1.5 * input_tok + 2 * output_tok
            elif prompt_cache_entry.model.startswith("gpt-3.5-turbo"):
                stats.millionth_dollar_estimate += 0.5 * input_tok + 1.5 * output_tok
            else:
                print(f"WARNING: unknown pricing for model {prompt_cache_entry.model}")

        stats.total_requests = len(self.all_prompts)
        stats.total_tokens = stats.prompt_tokens + stats.completion_tokens
        return stats

    def _run_prompt(
        self, prompt: str, model=DEFAULT_MODEL, retry_timeout=10, retry_num=0
    ) -> Optional[ChatCompletion]:
        # wait is too long so carry on
        if retry_timeout > 300:
            print("ERROR: waiting for prompt too long")
            return None
        if retry_num >= BACKUP_MODEL_AFTER_NUM_RETRIES and model != BACKUP_MODEL:
            print(
                f"WARNING: Changing model from {model} to {BACKUP_MODEL} after {retry_num} retries"
            )
            # The "cutting-edge" models experience more downtime.
            model = BACKUP_MODEL

        # TODO(P1, ux): My testing on gpt-4 through the browser gives better results
        #  - get access and use it on drafts.
        response = None
        should_retry = False
        try:
            # TODO(P2, devx): This can get stuck-ish, we should handle that somewhat nicely.
            # NOTE: openai.Completion is only for older models.
            response = self.client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": prompt}]
            )
        # openai.error.RateLimitError: That model is currently overloaded with other requests.
        # You can retry your request, or contact us through our help center at help.openai.com
        # if the error persists.
        # (Please include the request ID 7ed28a69c5cda5378f57266336539b7d in your message.)
        except (
            openai.RateLimitError,
            openai.Timeout,
            # TODO(P1, open-ai-migration): openai.TryAgain,
        ) as err:
            print(
                f"Got time-based {type(err)} error - sleeping for {retry_timeout} cause {err}"
            )
            should_retry = True
            time.sleep(retry_timeout)
        # Their fault
        # TODO(P1, open-ai-migration): openai.ServiceUnavailableError
        except (openai.APIError, openai.InternalServerError) as err:
            print(
                f"Got server-side {type(err)} error - sleeping for {retry_timeout} cause {err}"
            )
            should_retry = True
            time.sleep(retry_timeout)
        # Our fault
        except (
            openai.BadRequestError,
            openai.AuthenticationError,
            openai.NotFoundError,
            openai.PermissionDeniedError,
            openai.ConflictError,
        ) as err:
            print(
                f"Got client-side {type(err)} error - we messed up so lets rethrow this error {err}"
            )
            raise err
        except Exception as err:
            # Capture any unanticipated errors here
            print(
                f"Unexpected {type(err).__name__} error - "
                f"sleeping for {retry_timeout} seconds due to {err}"
            )
            raise err

        if should_retry:
            return self._run_prompt(
                prompt, model, 2 * retry_timeout, retry_num=retry_num + 1
            )  # exponential backoff

        return response

    # About 0.4 cents per request (about 2000 tokens). Using gpt-4 would be 15x more expensive :/
    # TODO(peter): Do sth about max prompt length (4096 tokens INCLUDING the generated response)
    # TODO(P1, devx): We should templatize the prompt into "function body" and "parameters";
    #   then we can re-use the "body" to "fine-tune" a model and have faster responses.
    def run_prompt(
        self,
        prompt: str,
        model=DEFAULT_MODEL,
        print_prompt=True,
    ):
        with PromptCache(
            cache_store=self.cache_store,
            prompt=prompt,
            model=model,
            print_prompt=self._should_print_prompt(print_prompt),
        ) as pcm:
            if pcm.cache_hit:
                return pcm.cache_entry.result

            response = self._run_prompt(prompt, model)
            if response is None:
                return None

            gpt_result = response.choices[0].message.content.strip()
            pcm.cache_entry.result = gpt_result

            token_usage: CompletionUsage = response.usage
            if token_usage:
                pcm.cache_entry.prompt_tokens = token_usage.prompt_tokens
                pcm.cache_entry.completion_tokens = token_usage.completion_tokens

            self.all_prompts.append(pcm.cache_entry)
            # `pcm.__exit__` will update the database

            return gpt_result

    # TODO(P1, devx): We should have a gpt_utils.py file to organize the logic from transformations.
    # Main reason to separate Definition from Values is that we can generate GPT prompts in a generic-ish way.
    # Sample output "industry": "which business area they specialize in professionally",
    @staticmethod
    def form_field_to_gpt_prompt(field: FieldDefinition) -> Optional[str]:
        if field.ignore_in_prompt:
            print(f"ignoring {field.name} for gpt prompt gen")
            return None

        result = f'"{field.name}": "{field.field_type} field representing {field.label}'

        if bool(field.description):
            result += f" described as {field.description}"
        if bool(field.options) and isinstance(field.options, list):
            if len(field.options) > GPT_MAX_NUM_OPTION_FIELDS:
                print(f"too many options, shortening to {GPT_MAX_NUM_OPTION_FIELDS}")
            options_slice: List[Option] = field.options[:GPT_MAX_NUM_OPTION_FIELDS]
            option_values = "\n".join(
                [f"{opt.value}: {opt.label}" for opt in options_slice]
            )
            result += (
                f" restricted to these options defined as a list of 'result: description' pairs. "
                f"Pick the most suitable option and only output the result part.\n"
                f"{option_values}"
                ""
            )
        result += '"'

        return result

    @staticmethod
    def form_to_gpt_prompt(form: FormDefinition):
        field_prompts = [
            OpenAiClient.form_field_to_gpt_prompt(field) for field in form.fields
        ]
        return ",\n".join([f for f in field_prompts if f is not None])

    @staticmethod
    def _maybe_add_current_time_to_prompt(use_current_time: bool) -> str:
        if not use_current_time:
            return ""

        # Get the current UTC time and then convert it to local time.
        local_now = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
        # We omit minutes for 1. UX and 2. better caching of especially local test/research runs.
        # 2023-10-03 02:30 PM PDT
        now_with_hours_and_tz = local_now.strftime("%Y-%m-%d %H:%M %Z")
        return f"\nGeneral context: Current time is {now_with_hours_and_tz}"

    def fill_in_form(
        self,
        form: FormDefinition,
        text: str,
        model=DEFAULT_MODEL,
        print_prompt=False,
        use_current_time: bool = False,
    ) -> Tuple[Optional[FormData], Optional[str]]:
        extra_context = OpenAiClient._maybe_add_current_time_to_prompt(
            use_current_time=use_current_time
        )
        form_prompt = OpenAiClient.form_to_gpt_prompt(form)

        gpt_query = """
            The following is a definition of form,
            given as a list of form field labels, description and type (or options of possible values):
            {form_prompt}
            Using this note:
            {note}
            Fill in the form.
            Return a valid JSON dictionary where keys are form labels, and values are filled in results.
            For unknown field values just use null.
            {extra_context}
            """.format(
            form_prompt=form_prompt,
            note=text,
            extra_context=extra_context,
        )
        raw_response = self.run_prompt(
            gpt_query, model=model, print_prompt=print_prompt
        )
        form_data_raw = gpt_response_to_json(raw_response)
        if print_prompt:
            print(f"fill_in_form response: {form_data_raw}")

        if not isinstance(form_data_raw, dict):
            err = f"gpt resulted form_data ain't a dict: {form_data_raw}"
            print(f"ERROR: {err}")
            return None, err

        form_data = FormData(form, form_data_raw, omit_unknown_fields=True)
        # TODO(P1, correctness): Would be nice to double-check if all GPT requested fields were actually returned.
        return form_data, None

    def fill_in_multi_entry_form(
        self,
        form: FormDefinition,
        text: str,
        print_prompt=False,
        use_current_time: bool = False,
    ) -> Tuple[List[FormData], Optional[str]]:
        extra_context = OpenAiClient._maybe_add_current_time_to_prompt(
            use_current_time=use_current_time
        )
        form_prompt = OpenAiClient.form_to_gpt_prompt(form)

        gpt_query = """
            The following is a definition of form,
            described as a list of form field labels, description and type (or options of possible values):
            {form_prompt}

            Below is the text which is one or multiple answers to the form.
            Fill in the form and return a valid JSON list of dictionaries. Only return that JSON list of dictionaries.
            For each form response, return one list item, every list item is represented as a dictionary,
            where keys are form labels and values are filled in results.
            For unknown field values just use null.
            If there are no form responses at all or the text is irrelevant, then just return an empty list.

            This the answer text:
            {note}

            {extra_context}
            """.format(
            form_prompt=form_prompt,
            note=text,
            extra_context=extra_context,
        )
        raw_response = self.run_prompt(gpt_query, print_prompt=print_prompt)
        form_data_raw = gpt_response_to_json(raw_response)
        if print_prompt:
            print(f"fill_in_form response: {form_data_raw}")

        if not isinstance(form_data_raw, list):
            err = f"gpt resulted form_data ain't a list: {form_data_raw}"
            print(f"ERROR: {err}")
            return [], err

        results = []
        for form_item in form_data_raw:
            if not isinstance(form_item, dict):
                # only warning as it is better to return something than nothing
                print(f"WARNING: form_item result is not a dict: {form_item}, skipping")
                continue
            # TODO(P1, correctness): Would be nice to double-check if all GPT requested fields were actually returned.
            results.append(
                FormData(form=form, data=form_item, omit_unknown_fields=True)
            )

        return results, None

    # TODO(P2, Facebook MMS): Better multi-language support, Slovak was OK, but it got some things quite wrong.
    #   * https://about.fb.com/news/2023/05/ai-massively-multilingual-speech-technology/
    #   We might need to run the above ourselves for now (BaseTen hosting?)
    #   For inspiration on how to run Whisper locally:
    #   https://towardsdatascience.com/whisper-transcribe-translate-audio-files-with-human-level-performance-df044499877
    # They claim to have WER <50% for these:
    # Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech,
    # Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic,
    # Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori,
    # Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili,
    # Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.
    # NOTE: I verified that for English there is no difference between "transcribe" and "translate",
    # by changing it locally and seeing the translate is "cached_prompt: serving out of cache".
    # TODO(P0, quality): Once it comes out use Whisper 3
    # TODO(P1, quality): For real world call transcription diarization is a must IMHO.
    #   https://community.openai.com/t/thoughts-on-whisper-3-announcement/475687/3
    def transcribe_audio(self, audio_filepath, model="whisper-1"):
        prompt_hint = "notes on my discussion from an in-person meeting or conference"

        # We mainly do caching
        with PromptCache(
            cache_store=self.cache_store,
            prompt=audio_filepath,
            model=model,
            print_prompt=self._should_print_prompt(True),
        ) as pcm:
            # We only use the cache for local runs to further speed up development (and reduce cost)
            # TODO(P1, devx): Fix this
            # if pcm.cache_hit and not is_running_in_aws():
            #     return pcm.prompt_log.result

            with open(audio_filepath, "rb") as audio_file:
                # TODO(P0, bug): Seems like empty audio files can get stuck here (maybe temperature=0 and backoff?).
                print(
                    f"Transcribing (and translating) {get_fileinfo(file_handle=audio_file)}"
                )
                # Data submitted through the API is no longer used for service improvements (including model training)
                #   unless the organization opts in
                # https://openai.com/blog/introducing-chatgpt-and-whisper-apis
                # (2023, May): File uploads are currently limited to 25 MB and the these file types are supported:
                #   mp3, mp4, mpeg, mpga, m4a, wav, and webm (m4a FAKE news). Confirmed that webm and ffmpeg mp4 work.
                # TODO(P2, feature); For longer inputs, we can use pydub to chunk it up
                #   https://platform.openai.com/docs/guides/speech-to-text/longer-inputs
                res = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="json",
                    # language="en",  # only for openai.Audio.transcribe
                    prompt=prompt_hint,
                    # If set to 0, the model will use log probability to automatically increase the temperature
                    #   until certain thresholds are hit (i.e. it can take longer).
                    # TODO(P1, open-ai-migration): temperatue=0
                )
                transcript = res.text
                print(f"audio transcript: {res}")
                pcm.cache_entry.result = transcript
                # `pcm.__exit__` will update the database
                return transcript


def _get_first_occurrence(s: str, list_of_chars: list):
    first_occurrence = len(s)  # initialize to length of the string

    for char in list_of_chars:
        index = s.find(char)
        if (
            index != -1 and index < first_occurrence
        ):  # update if char found and it's earlier
            first_occurrence = index

    if first_occurrence == len(s):
        return -1
    return first_occurrence


def _get_last_occurrence(s: str, list_of_chars: list):
    last_occurrence = -1  # initialize to -1 as a "not found" value

    for char in list_of_chars:
        index = s.rfind(char)
        if index > last_occurrence:  # update if char found and it's later
            last_occurrence = index

    return last_occurrence


def _try_decode_non_json(raw_response: str):
    # Sometimes it returns a list of strings in format of " -"
    lines = raw_response.split("\n")
    if len(lines) > 1:
        bullet_point_lines = sum(s.lstrip().startswith("-") for s in lines)
        if bullet_point_lines + 1 >= len(lines):
            print(
                f"Most of lines {bullet_point_lines} out of {len(lines)} start as a bullet point, assuming list"
            )
            return [s for s in lines if s.lstrip().startswith("-")]

    print("WARNING: Giving up on decoding")
    return None


# TODO(P0, implementation): Use the force json output on GPT prompts - now there is JSON mode, so just use that:
#   https://medium.com/@ralfelfving/openai-json-response-format-explained-with-example-dynamically-render-a-quiz-app-2050d1e719b0
def gpt_response_to_json(raw_response: Optional[str], debug=True) -> Optional[Any]:
    if raw_response is None:
        if debug:
            print("raw_response is None")
        return None
    wrong_input_responses = [
        "Sorry, it is not possible to create a json dict",
        "Sorry, as an AI language model",
        "The note does not mention any person",  # 's name or identifier
    ]
    if any(raw_response.startswith(s) for s in wrong_input_responses):
        if debug:
            print(
                f"WARNING: Likely provided wrong input as GPT is complaining with {raw_response}"
            )
        return None

    # Here do some black-magic regex postprocessing for all previously encountered problems (mostly with GPT 3.5).
    orig_response = raw_response
    # Output: ```json <text> ```
    raw_response = re.sub(r"```[a-z\s]*?(.*?)```", r"\1", raw_response, flags=re.DOTALL)

    # BEFORE DOING ALL THIS MAMBO-JAMBO, lets just try if GPT returned a valid JSON.
    # As some of these replacements actually mess up valid json.
    try:
        # The model might have just crafted a valid json object
        return json.loads(raw_response)
    except json.decoder.JSONDecodeError as err:
        if debug:
            print(
                f"WARNING: couldn't decode orig response cause {err}. Orig response {orig_response}"
            )

    # ========= ATTEMPT AUTO-CORRECTING WRONG JSON OUTPUT ==========
    # Double comma is seldom ok: ,  ,
    # Yes this happened, see 559719ecbb1012e05fc95d533295170dc4a339548ad602202a3c8dbac62138bc
    raw_response = re.sub(r",\s*,", ",", raw_response)
    # For "Expecting property name enclosed in double quotes"
    # Welp - OMG this from PPrint :facepalm:
    raw_response = (
        raw_response.replace("{'", '{"').replace("':", '":').replace(", '", ', "')
    )
    # NOTE: .replace("',", '",') can happen for example "part-time for 'leave', reached"
    raw_response = raw_response.replace(": '", ': "').replace("'}", '"}')
    # See test_gpt_response_to_json
    # raw_response = raw_response.replace(': ""', ': "').replace('""}', '"}')
    # Sometimes, it includes the input in the response. So only consider what is after "Output"
    match = re.search("(?i)output:", raw_response)
    if match:
        raw_response = raw_response[match.start() :]
    # Yeah, sometimes it does that lol
    #   **Output:**<br> ["Shervin: security startup guy from Maryland who wears a 1337/1338 shirt"]<br>
    raw_response = raw_response.replace("<br>\n", "\n")
    raw_response = raw_response.replace("<br />\n", "\n")
    # Sometimes GPT adds the extra comma, well, everyone is guilty of that leading to a production outage so :shrug:
    # Examples: """her so it was cool",    ],"""
    # TODO(P2, devx): Redundant character escape
    raw_response = re.sub(r'",\s*]', '"]', raw_response)
    raw_response = re.sub(r'",\s*}', '"}', raw_response)
    raw_response = re.sub(r"],\s*}", "]}", raw_response)
    raw_response = re.sub(r"],\s*]", "]]", raw_response)
    raw_response = re.sub(r"},\s*}", "}}", raw_response)
    raw_response = re.sub(r"},\s*]", "}]", raw_response)
    # To replace newlines inside JSON strings
    raw_response = re.sub(
        r'("(?:[^"\\]|\\.)*")', lambda m: m.group(1).replace("\n", "\\n"), raw_response
    )
    # Sometimes it just does do backslash to try making a newline (like in a shellscript)
    raw_response = re.sub(r"\\+\n\s*", "\\\\n", raw_response)
    # Happened with email upload vue79j44one1liatgmjf2kbbvgeqjebi95uutk01 - long output
    if "[" not in raw_response and "]" in raw_response:
        raw_response = raw_response.replace("]", "")

    # if debug:
    #     print(f"converted {orig_response}\n\nto\n\n{raw_response}")
    try:
        # The model might have just crafted a valid json object
        res = json.loads(raw_response)
    except json.decoder.JSONDecodeError as orig_err:
        # In case there is something before the actual json output like "Output:", "Here you go:", "Sure ..".
        start_index = _get_first_occurrence(raw_response, ["{", "["])
        last_index = _get_last_occurrence(raw_response, ["}", "]"])
        # -1 works
        raw_json = raw_response[start_index : last_index + 1]
        if debug and len(raw_json) * 2 < len(
            raw_response
        ):  # heuristic to determine that we shortened too much
            print(
                f"WARNING: likely the GPT response is NOT a JSON (shortened [{start_index}:{last_index}]):"  # noqa: 231
                f"\n{raw_json}\nresulted from\n{orig_response}"
            )
            return _try_decode_non_json(raw_response)
        try:
            res = json.loads(raw_json)
        except json.decoder.JSONDecodeError as sub_err:
            if debug:
                print(
                    f"Could NOT decode json cause SUB ERROR: {sub_err} for raw_response "
                    f"(note does a bunch of replaces) {raw_json}. ORIGINAL ERROR: {orig_err}"
                )
            return None
    return res


# ================ SMALL UTIL FUNCTIONS ================
def pretty_filesize_int(file_size: int) -> str:
    return f"{file_size / (1024 * 1024):.2f}MB"  # noqa: 231


def pretty_filesize_path(file_path: str) -> str:
    return pretty_filesize_int(os.path.getsize(file_path))


def get_fileinfo(file_handle):
    return f"File {file_handle.name} is {pretty_filesize_path(file_handle.name)}"


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    # Encoding name	OpenAI models
    # cl100k_base	gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    # p50k_base	Codex models, text-davinci-002, text-davinci-003
    # r50k_base (or gpt2)	GPT-3 models like davinci
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    test_json_with_extra_output = """Output:
    {
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

    load_dotenv()
    OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
    openai_client = OpenAiClient(open_ai_api_key=OPEN_AI_API_KEY)
    test_prompt = "list neighborhoods in san francisco as json key: value where key is name and value is zip code"
    sf_result_1 = openai_client.run_prompt(
        test_prompt, CHEAPEST_MODEL, print_prompt=True
    )
    print(sf_result_1)
    # run again, this should yield a cache hit
    sf_result_2 = openai_client.run_prompt(
        test_prompt, CHEAPEST_MODEL, print_prompt=True
    )
    assert sf_result_1 == sf_result_2
    assert len(openai_client.all_prompts) == 1

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
