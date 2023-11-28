#!/usr/bin/env python3
import sys
import requests
import argparse
import json
import openai
import os
import tiktoken
import re

SAMPLE_PROMPTS = {
    "user": "prompt/user.md",
    "command": "prompt/command.md",
    "format": "prompt/format.md",
    "title": "prompt/title.md",
    "unified_change": "prompt/unified_change.md",
}

GOOD_SAMPLE_RESPONSES = {"response": "response/response.md"}

COMPLETION_PROMPTS = {
    "user": "prompt/user.md",
    "command": "prompt/command.md",
    "format": "prompt/format.md",
    "title": "prompt/title.md",
}


def get_code_blocks(markdown_text):
    code_blocks = re.findall(
        r"```(.+?)\n([\s\S]+?)```|~~~(.+?)\n([\s\S]+?)~~~", markdown_text
    )
    clean_code_blocks = []

    for match in code_blocks:
        if match[0]:
            clean_code_blocks.append(match[1])
        else:
            clean_code_blocks.append(match[3])

    return clean_code_blocks


def separate_code_blocks(markdown_text):
    # Regular expression to match code blocks
    code_blocks = re.split(r"```[\w\W]*?```", markdown_text)

    # Regular expression to find all code blocks
    all_code_blocks = re.findall(r"```[\w\W]*?```", markdown_text)

    # Append non-code content and code blocks alternately
    separated_content = []
    for i, block in enumerate(code_blocks):
        separated_content.append(block)
        if i < len(all_code_blocks):
            separated_content.append(all_code_blocks[i])

    # Filter out any empty strings from the list
    separated_content = list(filter(None, separated_content))
    return separated_content


def markdown_to_string(file_path):
    result = ""
    with open(file_path, "r", encoding="utf-8") as file:
        markdown_content = file.read()

        for line in separate_code_blocks(markdown_content):
            code_block = re.findall(r"```[\w\W]*?```", line)
            if code_block:
                block = get_code_blocks(code_block[0])
                for code_block_line in block:
                    result += code_block_line
            else:
                result += line
    return result


def combine_prompt(prompts: dict):
    prompt = ""
    for key, value in prompts.items():
        prompt += markdown_to_string(value).strip() + "\n\n"
    return prompt


def get_first_word_in_quote(sentence):
    # Find the index of the first quote character
    start_quote_index = sentence.find('"')
    end_quote_index = sentence.find(
        '"', start_quote_index + 1
    )  # Find the matching end quote

    if start_quote_index != -1 and end_quote_index != -1:
        # Extract the substring between the quotes
        quote = sentence[start_quote_index + 1 : end_quote_index]

        # Split the quote into words and get the first word
        words = quote.split()
        if words:
            return words[0]  # Return the first word
    return None  # Return None if no quote is found or if it's empty


def replace_title(title_file, pull_request_title):
    title = markdown_to_string(title_file).strip()
    title_quote = get_first_word_in_quote(title)
    title = title.replace(title_quote, pull_request_title)
    return title


SAMPLE_PROMPT = combine_prompt(SAMPLE_PROMPTS)

GOOD_SAMPLE_RESPONSE = combine_prompt(GOOD_SAMPLE_RESPONSES)


def main():
    parser = argparse.ArgumentParser(
        description="Use ChatGPT to generate a description for a pull request."
    )
    parser.add_argument(
        "--github-api-url", type=str, required=True, help="The GitHub API URL"
    )
    parser.add_argument(
        "--github-repository", type=str, required=True, help="The GitHub repository"
    )
    parser.add_argument(
        "--pull-request-id",
        type=int,
        required=True,
        help="The pull request ID",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        required=True,
        help="The GitHub token",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        required=True,
        help="The OpenAI API key",
    )
    parser.add_argument(
        "--allowed-users",
        type=str,
        required=False,
        help="A comma-separated list of GitHub usernames that are allowed to trigger the action, empty or missing means all users are allowed",
    )
    args = parser.parse_args()

    github_api_url = args.github_api_url
    repo = args.github_repository
    github_token = args.github_token
    pull_request_id = args.pull_request_id
    openai_api_key = args.openai_api_key
    allowed_users = os.environ.get("INPUT_ALLOWED_USERS", "")
    if allowed_users:
        allowed_users = allowed_users.split(",")
    # open_ai_model
    open_ai_models = json.loads(os.environ.get("INPUT_OPENAI_MODELS"))
    # max_prompt_tokens = int(os.environ.get("INPUT_MAX_TOKENS", "1000"))
    max_response_tokens = int(os.environ.get("INPUT_MAX_RESPONSE_TOKENS"))
    model_temperature = float(os.environ.get("INPUT_TEMPERATURE"))
    model_sample_prompt = os.environ.get("INPUT_SAMPLE_PROMPT", SAMPLE_PROMPT)
    model_sample_response = os.environ.get(
        "INPUT_SAMPLE_RESPONSE", GOOD_SAMPLE_RESPONSE
    )
    file_types = os.environ.get("INPUT_FILE_TYPES", "").split(",")

    authorization_header = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token %s" % github_token,
    }

    status, completion_prompt = get_pull_request_description(
        allowed_users,
        github_api_url,
        repo,
        pull_request_id,
        authorization_header,
        file_types,
    )
    if status != 0:
        return 1
    else:
        if completion_prompt == "":
            return status

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who writes pull request descriptions",
        },
        {"role": "user", "content": model_sample_prompt},
        {"role": "assistant", "content": model_sample_response},
        {"role": "user", "content": completion_prompt},
    ]
    # calculate for model selection
    model, prompt_token = model_selection(open_ai_models, messages, max_response_tokens)
    if model == "":
        print("No model available for this prompt")
        return 1

    token_left = open_ai_models[model] - prompt_token - max_response_tokens
    if token_left < 0:
        print(f"Model {model} does not have enough token to generate response")
        return 1

    extend_response_token = int(max_response_tokens + token_left * 0.8)
    print(
        f"Using model {model} with {prompt_token} prompt tokens and reserve {extend_response_token} response token"
    )

    openai.api_key = openai_api_key

    while True:
        try:
            openai_response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=model_temperature,
                max_tokens=extend_response_token,
            )
            break
        except Exception as e:
            print(f"Exception: {e}")
            if "Connection aborted".lower() in str(e).lower():
                print("Retry")
            else:
                return 1

    try:
        usage = openai_response.usage
        print(f"OpenAI API usage this request: {usage}")
    except:
        pass
    generated_pr_description = openai_response.choices[0].message.content
    redundant_prefix = "This pull request "
    if generated_pr_description.startswith(redundant_prefix):
        generated_pr_description = generated_pr_description[len(redundant_prefix) :]
        generated_pr_description = (
            generated_pr_description[0].upper() + generated_pr_description[1:]
        )
    print(f"Generated pull request description: \n'{generated_pr_description}'")
    issues_url = "%s/repos/%s/issues/%s" % (
        github_api_url,
        repo,
        pull_request_id,
    )
    update_pr_description_result = requests.patch(
        issues_url,
        headers=authorization_header,
        json={"body": generated_pr_description},
        timeout=30,
    )

    if update_pr_description_result.status_code != requests.codes.ok:
        status = "".join(
            ["Request to update pull request description failed: "],
            str(update_pr_description_result.status_code),
        )
        print(status)
        print("Response: " + update_pr_description_result.text)
        return 1


def get_pull_request_description(
    allowed_users,
    github_api_url,
    repo,
    pull_request_id,
    authorization_header,
    file_types,
):
    pull_request_url = f"{github_api_url}/repos/{repo}/pulls/{pull_request_id}"
    pull_request_result = requests.get(
        pull_request_url,
        headers=authorization_header,
        timeout=30,
    )
    if pull_request_result.status_code != requests.codes.ok:
        status = "".join(
            [
                "Request to get pull request data failed: ",
                str(pull_request_result.status_code),
            ]
        )
        print(status)
        return 1, ""
    pull_request_data = json.loads(pull_request_result.text)

    if pull_request_data["body"]:
        print("Pull request already has a description, skipping")
        return 0, ""

    if allowed_users:
        pr_author = pull_request_data["user"]["login"]
        if pr_author not in allowed_users:
            print(
                f"Pull request author {pr_author} is not allowed to trigger this action"
            )
            return 0, ""

    pull_request_title = pull_request_data["title"]

    pull_request_files = []
    # Request a maximum of 30 pages (900 files)
    for page_num in range(1, 31):
        pull_files_url = f"{pull_request_url}/files?page={page_num}&per_page=30"
        pull_files_result = requests.get(
            pull_files_url,
            headers=authorization_header,
            timeout=30,
        )

        if pull_files_result.status_code != requests.codes.ok:
            status = "".join(
                [
                    "Request to get list of files failed with error code: ",
                    str(pull_files_result.status_code),
                ]
            )
            print(status)
            return 1, ""

        pull_files_chunk = json.loads(pull_files_result.text)

        if len(pull_files_chunk) == 0:
            break

        pull_request_files.extend(pull_files_chunk)

        title_promt = replace_title(COMPLETION_PROMPTS["title"], pull_request_title)
        COMPLETION_PROMPTS.pop("title")
        completion_prompt = combine_prompt(COMPLETION_PROMPTS)
        completion_prompt += title_promt

    is_any_file_type_matched = False
    for pull_request_file in pull_request_files:
        # Not all PR file metadata entries may contain a patch section
        # For example, entries rquoelated to removed binary files may not contain it
        if "patch" not in pull_request_file:
            continue

        filename = pull_request_file["filename"]
        patch = pull_request_file["patch"]

        if not check_file_type(filename, file_types):
            print(f"skip file {filename}")
            continue

        is_any_file_type_matched = True
        completion_prompt += f"Changes in file {filename}: \n{patch}\n"

    if not is_any_file_type_matched:
        print("No file type matched")
        return 0, ""
    print(f"Completion_prompt: \n{completion_prompt}")
    return 0, completion_prompt


def check_file_type(filename, file_types):
    for file_type in file_types:
        if filename.endswith(file_type):
            return True
    return False


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def model_selection(models, messages, max_response_tokens):
    candidate = []
    for model in models:
        max_token = models[model]
        max_prompt_tokens = max_token - max_response_tokens
        if max_prompt_tokens < 0:
            continue
        prompt_tokens = num_tokens_from_messages(messages, model)
        if prompt_tokens > max_prompt_tokens:
            continue
        print(
            f"May using model {model} with {prompt_tokens} prompt tokens and reserve {max_response_tokens} response token"
        )
        candidate.append([model, models[model], prompt_tokens])
    if len(candidate) == 0:
        return "", 0
    # sort by max_token
    candidate.sort(key=lambda x: x[1])
    print(f"Using model {candidate[0][0]}")
    return candidate[0][0], candidate[0][2]


if __name__ == "__main__":
    sys.exit(main())
