import pytest

from gpt_form_filler.openai_client import (
    InMemoryCacheStore,
    OpenAiClient,
    PromptCache,
    PromptCacheEntry,
)


# Test function for the first scenario
def test_cache_hit_run_prompt():
    # TODO(P1, devx): Mock the OpenAI API to return a specific response,
    #   for now just test the cache hit
    api_key = "NO CALLS TO API"
    cache = InMemoryCacheStore()
    gpt_client = OpenAiClient(open_ai_api_key=api_key, cache_store=cache)

    model = "text-davinci-003"
    prompt = "This is a test prompt"
    cache_entry = PromptCacheEntry(
        model=model,
        prompt=prompt,
    )
    cache_entry.result = "This is the cached result"
    cache.write_cache(cache_entry)
    result = gpt_client.run_prompt(prompt, model)
    assert result == cache_entry.result


# Test function for the first scenario
def test_cache_hit_transcribe_audio():
    api_key = "NO CALLS TO API"
    cache = InMemoryCacheStore()
    gpt_client = OpenAiClient(open_ai_api_key=api_key, cache_store=cache)

    model = "whisper-1"
    audio_filepath = "tests/test_audio.wav"
    cache_entry = PromptCacheEntry(
        model=model,
        prompt=audio_filepath,
    )
    cache_entry.result = "This is the cached transcription"
    cache.write_cache(cache_entry)
    result = gpt_client.transcribe_audio(
        audio_filepath=audio_filepath, model=model, use_cache_hit=True
    )
    assert result == cache_entry.result

    with pytest.raises(FileNotFoundError) as exc_info:
        # use_cache_hit = False
        gpt_client.transcribe_audio(audio_filepath=audio_filepath, model=model)
        # Tries to transcribe the audio file, but it doesn't exist cause I am lazy to mock it
        assert exc_info.match(r"No such file or directory")


def test_cache_written():
    cache_store = InMemoryCacheStore()
    with PromptCache(
        cache_store=cache_store,
        prompt="This is a test prompt",
        model="text-davinci-003",
    ) as pcm:
        assert pcm.cache_hit is False
        pcm.cache_entry.result = "This is a valid result"

    cached_pcm = cache_store.maybe_get("This is a test prompt", "text-davinci-003")
    assert cached_pcm.result == "This is a valid result"


def test_cache_no_written_for_none():
    cache_store = InMemoryCacheStore()
    with PromptCache(
        cache_store=cache_store,
        prompt="This is a test prompt",
        model="text-davinci-003",
    ) as pcm:
        assert pcm.cache_hit is False
        pcm.cache_entry.result = None  # bad result

    assert len(cache_store.cache) == 0


def test_cache_no_written_for_empty_result():
    cache_store = InMemoryCacheStore()
    with PromptCache(
        cache_store=cache_store,
        prompt="This is a test prompt",
        model="text-davinci-003",
    ) as pcm:
        assert pcm.cache_hit is False
        pcm.cache_entry.result = ""  # can happen for audio transcription

    assert len(cache_store.cache) == 0
