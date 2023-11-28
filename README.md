# `platisd/openai-pr-description` GitHub Action

Autofill the description of your pull requests with the power of OpenAI!

![openai-pr-description-screenshot](media/openai-pr-description-screenshot.png)

## What does it do?

`Amstelz/openai-pr-description` is forked from
```
`platisd/openai-pr-description` which is a GitHub Action that looks at the title as well as the contents
of your pull request and uses the [OpenAI API](https://openai.com/blog/openai-api) to automatically
fill up the description of your pull request. Just like ChatGPT would! 🎉<br>
The Action tries to focus on **why** the changes are needed rather on **what** they are,
like any proper pull request description should.

The GitHub Action will only run when a PR description is not already provided.
In other words it will not accidentally overwrite your existing description.
The idea is this Action will save you the time and trouble of writing **meaningful** pull request descriptions.<br>
You can customize it in different ways. One of them allows the Action to only run on pull requests started
by specific users, e.g. the main maintainers of the repository.
Keep in mind the OpenAI API is not free to use. That being said, so far it's been rather cheap,
i.e. around ~$0.10 for 15-20 pull requests so far.
```

## New Feature
- model selection base on change which will choose bigger model if pull request if big.
- terminated if there aren't any avialable model for pull request if pull request if too big.
- specific file type. so some trash file doesn't included.


## How can you use it?

1. Create an account on OpenAI, set up a payment method and get your [OpenAI API key].
2. Add the OpenAI API key as a [secret] in your repository's settings.
3. Create a workflow YAML file, e.g. `.github/workflows/openai-pr-description.yml` with the following contents:

### Example workflow
``` yaml
name: Autofill PR description

on:
  pull_request:
    branches:
      - dev
      - main
      - master

jobs:
  openai-pr-description:
    runs-on: ubuntu-22.04
    timeout-minutes: 10
    steps:
      - uses: chalermporn17/openai-pr-description@dev
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
```

| Input                | Description                                                  | Required | Default                                               |
|----------------------|--------------------------------------------------------------|----------|-------------------------------------------------------|
| `github_token`       | The GitHub token used for the Action                         | Yes      |                                                       |
| `openai_api_key`     | The [OpenAI API key] to use, ensure it remains hidden        | Yes      |                                                       |
| `pull_request_id`    | The ID of the pull request to be used                        | No       | Extracted from metadata                               |
| `openai_models`      | The [OpenAI model] to use, contains a JSON key indicating the model name and value as the max tokens for the model | No       | `{ "gpt-3.5-turbo": 4096, "gpt-3.5-turbo-16k": 16384}`|
| `max_response_tokens`| The maximum number of **response tokens** to be used         | No       | `2048`                                                |
| `temperature`        | Higher values increase model creativity (range: 0-2)        | No       | `0.6`                                                 |
| `sample_prompt`      | The prompt used to provide context to the model              | No       | See `SAMPLE_PROMPT`                                   |
| `sample_response`    | A sample response used to provide context to the model       | No       | See `GOOD_SAMPLE_RESPONSE`                            |
| `file_types` |   A comma separated list of file types to include in the prompt. | No       | See `File types that will include in prompt` |



[OpenAI API key]: https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key
[OpenAI model]: https://platform.openai.com/docs/models
[secret]: https://docs.github.com/en/actions/security-guides/encrypted-secrets

**example pull request** https://github.com/chalermporn17/openai-pr-description/pull/2

## File types that will include in prompt
`.java,.py,.js,.ts,.cpp,.c,.cs,.rb,.go,.php,.swift,.html,.htm,.css,.scss,.sass,.less,.xml,.json,.yaml,.yml,.sh,.bat,.ps1,.cfg,.conf,.ini,.md,.txt,.rtf,.pdf,.sql,.db,.sqlite,.xaml,.plist`

too include or exclude file type, you can edit `file_types` in action argument.

## Default sample prompt
- SAMPLE_PROMPT
```
Write a pull request description focusing on the motivation behind the change and why it improves the project.
Go straight to the point.

The title of the pull request is "Enable valgrind on CI" and the following changes took place:

Changes in file .github/workflows/build-ut-coverage.yml: @@ -24,6 +24,7 @@ jobs:
         run: |
           sudo apt-get update
           sudo apt-get install -y lcov
+          sudo apt-get install -y valgrind
           sudo apt-get install -y ${{ matrix.compiler.cc }}
           sudo apt-get install -y ${{ matrix.compiler.cxx }}
       - name: Checkout repository
@@ -48,3 +49,7 @@ jobs:
         with:
           files: coverage.info
           fail_ci_if_error: true
+      - name: Run valgrind
+        run: |
+          valgrind --tool=memcheck --leak-check=full --leak-resolution=med \
+            --track-origins=yes --vgdb=no --error-exitcode=1 ${build_dir}/test/command_parser_test
Changes in file test/CommandParserTest.cpp: @@ -566,7 +566,7 @@ TEST(CommandParserTest, ParsedCommandImpl_WhenArgumentIsSupportedNumericTypeWill
     unsigned long long expectedUnsignedLongLong { std::numeric_limits<unsigned long long>::max() };
     float expectedFloat { -164223.123f }; // std::to_string does not play well with floating point min()
     double expectedDouble { std::numeric_limits<double>::max() };
-    long double expectedLongDouble { std::numeric_limits<long double>::max() };
+    long double expectedLongDouble { 123455678912349.1245678912349L };

     auto command = UnparsedCommand::create(expectedCommand, "dummyDescription"s)
                        .withArgs<int, long, unsigned long, long long, unsigned long long, float, double, long double>();
```
- GOOD_SAMPLE_RESPONSE
```
This pull request aims to integrate Valgrind into our Continuous Integration (CI) process to enhance memory leak detection and code reliability.

## Changes Made
    - **CI Workflow Update**: Included Valgrind installation and configured it to run with our test suite.
    - **Test Suite Adjustments**: Made minor adjustments to the test suite for better compatibility with Valgrind and improved test coverage.

## Impact
    - **Enhanced Code Quality**: Integrating Valgrind allows for automatic detection of memory leaks and access errors, leading to a more robust codebase.
    - **Improved Testing**: The adjustments in the test suite ensure comprehensive testing and accuracy.
```

## Model Selection Process
1. Calculate the total number of tokens used in the model prompt (`prompt_tokens`), which includes the tokens from the `example prompt`, `example response`, and the `prompt generated from the pull request`.

2. Select an appropriate model based on the following criteria:
    - Ensure that the model's maximum token capacity (`model_max_token`) is greater than the sum of `prompt_tokens` and `max_response_tokens`.
    - If multiple models satisfy this condition, select the one with the smallest maximum token capacity.


## `403` error when updating the PR description

If you get a `403` error when trying to update the PR description, it's most likely because
the GitHub Action is not allowed to do so.
The easiest way forward is to grant the necessary permissions to the `GITHUB_TOKEN` secret
at `<your_repo_url>/settings/actions` under `Workflow permissions`.
