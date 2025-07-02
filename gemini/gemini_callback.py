from google import genai
from google.genai import types
import json

client = genai.Client(api_key="AIzaSyA7SbKNMH1FWqgu232oTEAzjg_yisa4bWw")

def process_command_with_gemini(sentence):
    """Processes a user command using Gemini function calling with response_schema."""

    model = "gemini-2.0-flash"

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "description": "Determines the best browser command to execute based on the user's request.",
            "properties": {
                "function": {
                    "type": "STRING",
                    "description": "The function to be executed in the browser (e.g., 'open_website', 'click_element', 'scroll')."
                },
                "parameters": {
                    "type": "ARRAY",
                    "description": "List of parameters required for the selected function.",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "name": {
                                "type": "STRING",
                                "description": "The name of the parameter."
                            },
                            "value": {
                                "type": "STRING",
                                "description": "The value of the parameter."
                            }
                        }
                    }
                }
            },
            "required": ["function"]
        }
    )

    prompt = f"""
    You are an intelligent assistant controlling a web browser. You receive natural
    language commands from the user and translate them into browser actions.

    You have access to the following functions within the 'BrowserCLI' class:

    - `open_website(url: str)`: Opens a website using url.
    - `web_search(text: str)`: Performs a web search in omnibox.
    - `new_tab(url: str)`: Opens a new tab with the given URL.
    - `switch_tab(index: int)`: Switches to a specific tab by index.
    - `close_tab()`: Closes the current tab.
    - `scroll(direction: str, amount: int)`: Scrolls up or down by a given amount.
    - `type_text(text: str)`: Types text into an active input field.
    - `click_element(description: str)`: Clicks an element based on it's description.
    - `exit_browser()`: Closes the browser.

    Guidelines:
    - Select only ONE function per request.
    - Provide the necessary parameters for the selected function if required.

    User Request: {sentence}
    """

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=config
    ).text

    try:
        parsed_data = json.loads(response)

        function_name = parsed_data.get("function")
        parameters = parsed_data.get("parameters", {})

        print(f"Function Name: {function_name}")
        print(f"Parameters: {parameters}")

        # if function_name in commands:
        #     func = commands[function_name]
        #     if parameters:
        #         result = func(**parameters)
        #         if function_name == "extract_xpath":
        #             xpath = run_external_xpath_script(result)
        #             commands["click_element"](xpath)
        #     else:
        #         func()
        # else:
        #     return f"Unknown function: {function_name}"

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Returning text: {response}")
        return response


# process_command_with_gemini("open the leetcode.com website") # needs to add https:// at starting
# process_command_with_gemini("open the first link in google search page") # need to give more description
# process_command_with_gemini("search for 'python programming'")
# process_command_with_gemini("click search bar present in the page")
# process_command_with_gemini("scroll down by 500")
# process_command_with_gemini("type 'hello world'")
# process_command_with_gemini("click the first issues link for current repositary")
process_command_with_gemini("exit the browser")
