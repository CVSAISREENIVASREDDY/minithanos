import base64
import os
from google import genai
from google.genai import types


def generate(prompt):
    client = genai.Client(
        api_key="AIzaSyA7SbKNMH1FWqgu232oTEAzjg_yisa4bWw",
    )

    files = [
        # Make the file available in local system working directory
        client.files.upload(file="files.txt"),
    ]
    model = "gemini-2.0-flash-thinking-exp-01-21"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text="""I am gonna give you tasks. You have to return the command for them. Let's start!),

Command Format:
- I am using bash.
- Analyze the uploaded file for a reference to the file paths
- ONLY the command, no formattings, no explanations, no extra words.
- Remember the files can contain SPACES in their names.
RETURN LINUX COMMANDS ONLY.
"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""echo okay"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""open CSE Bros"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""start '' '/c/Users/santh/Downloads/CSE_BROS.csv'"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Create a javascript file and a python file in videos and show them in the file explorer"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""cd '/c/Users/santh/Videos/' && echo \"\" > script.js && echo \"\" > script.py && start ."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""{prompt}"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""Return only command"""),
        ],
    )

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    ).text
        

if __name__ == "__main__":
    generate("find the largest file in the downloads folder and open it")