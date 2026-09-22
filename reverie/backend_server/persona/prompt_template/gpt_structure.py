"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Shared text and embedding adapter for OpenAI and Ollama.
"""
import json
import random
import os
import openai
import time 

from utils import *

# Preserve utils.py credentials; an environment variable can override them.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
if LLM_PROVIDER not in ("openai", "ollama"):
  raise ValueError("LLM_PROVIDER must be 'openai' or 'ollama'")
TEXT_MODEL = (os.getenv("OLLAMA_MODEL", "llama3.1:8b") if LLM_PROVIDER == "ollama"
              else os.getenv("OPENAI_MODEL", "gpt-5.6-sol"))
EMBEDDING_MODEL = (os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
                   if LLM_PROVIDER == "ollama"
                   else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"))
EMBEDDING_IDENTITY = {"provider": LLM_PROVIDER, "model": EMBEDDING_MODEL}
_client = None


def get_client():
  global _client
  if _client is None:
    if LLM_PROVIDER == "ollama":
      base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
      if not base_url.endswith("/v1"):
        base_url += "/v1"
      _client = openai.OpenAI(
        base_url=base_url, api_key=os.getenv("OLLAMA_API_KEY") or "ollama",
        timeout=float(os.getenv("OLLAMA_TIMEOUT", "300")), max_retries=2)
    else:
      _client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or openai_api_key,
        timeout=90.0, max_retries=2)
  return _client


class ModelResponseError(RuntimeError):
  """The API returned no usable complete text."""


def _request_text(prompt, max_tokens=2048, continuation=False, stop=None):
  instructions = (
    "Return only the requested output, without commentary or Markdown fences. "
    "Follow the requested format exactly."
  )
  if continuation:
    instructions += (
      " Continue the supplied text from its final character. Return only the "
      "missing continuation; do not repeat the prompt or its final prefix."
    )
  stops = [stop] if isinstance(stop, str) else (stop or [])
  if "\n" in stops:
    instructions += " Output only the completion of the final line, on one line. Do not generate subsequent lines."
  if LLM_PROVIDER == "ollama":
    return _request_ollama_text(prompt, instructions, max_tokens, stops)
  # A legacy 5-50 token budget is too small for Responses. Retry only output
  # truncation, with a bounded budget; API failures still propagate directly.
  budget = max(1024, int(max_tokens))
  for attempt in range(3):
    response = get_client().responses.create(
      model=TEXT_MODEL, input=prompt, instructions=instructions,
      reasoning={"effort": "none"},
      max_output_tokens=budget, store=False)
    text = response.output_text or ""
    positions = [text.find(token) for token in stops if token and token in text]
    reason = getattr(response.incomplete_details, "reason", None) if response.status == "incomplete" else None
    # If the requested stop was reached, the needed prefix is complete even
    # when generation of unwanted subsequent lines exhausted the API budget.
    if positions and (response.status == "completed" or reason == "max_output_tokens"):
      text = text[:min(positions)]
      if text.strip():
        return text
    if response.status == "completed":
      if text.strip():
        return text
      raise ModelResponseError(f"{TEXT_MODEL} returned no text (empty or refused output).")
    if reason == "max_output_tokens" and attempt < 2:
      budget *= 2
      continue
    raise ModelResponseError(
      f"{TEXT_MODEL} response {response.status}: "
      f"{response.incomplete_details or response.error}")


def _request_ollama_text(prompt, instructions, max_tokens, stops):
  # Simulation reflection is an ordinary prompt; it does not require a
  # separate thinking trace for every small classification/completion call.
  effort = os.getenv("OLLAMA_REASONING_EFFORT")
  if effort is None and TEXT_MODEL.lower().split("/")[-1].startswith("qwen3"):
    effort = "none"
  options = {}
  if effort and effort != "default":
    if effort not in ("none", "low", "medium", "high", "max"):
      raise ValueError("OLLAMA_REASONING_EFFORT must be default, none, low, medium, high, or max")
    options["reasoning_effort"] = effort
  budget = max(1024, int(max_tokens))
  for attempt in range(3):
    print(f"[Ollama] {TEXT_MODEL}: requesting response "
          f"(reasoning={effort or 'default'}, max_tokens={budget}, attempt={attempt + 1})", flush=True)
    started = time.monotonic()
    try:
      response = get_client().chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "system", "content": instructions},
                  {"role": "user", "content": prompt}],
        max_tokens=budget, temperature=0, stream=False, **options)
    except openai.OpenAIError:
      print(f"[Ollama] Request failed after {time.monotonic() - started:.1f}s", flush=True)
      raise
    if not response.choices:
      raise ModelResponseError(f"{TEXT_MODEL} returned no choices.")
    choice = response.choices[0]
    print(f"[Ollama] Response after {time.monotonic() - started:.1f}s "
          f"(finish_reason={choice.finish_reason})", flush=True)
    text = choice.message.content or ""
    # Apply legacy stops locally so a leading newline cannot end generation.
    positions = [text.find(token) for token in stops if token and token in text]
    if positions and choice.finish_reason in ("stop", "length"):
      prefix = text[:min(positions)]
      if prefix.strip():
        return prefix
    if choice.finish_reason == "stop" and text.strip():
      return text
    if choice.finish_reason == "length" and attempt < 2:
      budget *= 2
      continue
    raise ModelResponseError(
      f"{TEXT_MODEL} returned empty or incomplete text "
      f"(finish_reason={choice.finish_reason}).")


def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt):
  temp_sleep()
  return _request_text(prompt)


def GPT4_request(prompt):
  """Compatibility name for callers; all text generation uses TEXT_MODEL."""
  return ChatGPT_single_request(prompt)


def ChatGPT_request(prompt):
  return ChatGPT_single_request(prompt)


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except (openai.OpenAIError, ModelResponseError):
      raise
    except (ValueError, TypeError, KeyError, IndexError):
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except (openai.OpenAIError, ModelResponseError):
      raise
    except (ValueError, TypeError, KeyError, IndexError):
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except (openai.OpenAIError, ModelResponseError):
      raise
    except (ValueError, TypeError, KeyError, IndexError):
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
  """Adapt legacy completion prompts to the configured text provider.

  Legacy engine and sampling/penalty fields are intentionally not forwarded.
  """
  temp_sleep()
  return _request_text(
    prompt, max_tokens=gpt_parameter.get("max_tokens", 2048),
    continuation=True, stop=gpt_parameter.get("stop"))


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    try:
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
    except (ValueError, TypeError, KeyError, IndexError):
      pass  # Retry malformed model output, never API failures.
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model=None):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return get_client().embeddings.create(
          input=[text], model=model or EMBEDDING_MODEL,
          encoding_format="float").data[0].embedding


def prepare_memory_embeddings(embeddings, saved_identity):
  """Rebuild vectors when changing models, even if dimensions happen to match."""
  legacy_identity = {"provider": "openai", "model": "text-embedding-ada-002"}
  if (saved_identity or legacy_identity) == EMBEDDING_IDENTITY:
    return embeddings
  if embeddings:
    print(f"Rebuilding {len(embeddings)} memory embeddings with {EMBEDDING_MODEL}...")
  return {text: get_embedding(text) for text in embeddings}


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)


















