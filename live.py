import asyncio
from google import genai
import pyaudio
from RealtimeSTT import AudioToTextRecorder
from agents.all_functions import all_functions
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"),
                      http_options={'api_version': 'v1alpha'})
model = "gemini-2.0-flash-exp"
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    tools=list(all_functions.values()) + [{"google_search": {}}],
    system_instruction=types.Content(
        parts=[
            types.Part(
                text="You are a helpful assistant 'Alexa' and answer in a friendly tone. Don't give long responses. Respond like if you were talking to a friend. Give only SHORT responses.",
            )
        ]
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
        )
    ),
)


async def async_enumerate(it):
    n = 0
    async for item in it:
        yield n, item
        n += 1

p = pyaudio.PyAudio()


async def handle_tool_call(session, tool_call):
    for fc in tool_call.function_calls:
        f = all_functions[fc.name]
        tool_response = types.LiveClientToolResponse(
            function_responses=[
                types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response=f(**fc.args),
                )
            ]
        )
    await session.send(input=tool_response)
    

async def main():
    while True:
        print("SESSION STARTED!")
        async with client.aio.live.connect(model=model, config=config) as session:
            while True:
                try:
                    stream = p.open(format=p.get_format_from_width(2),
                                    channels=1,
                                    rate=24000,
                                    output=True)

                    message = recorder.text()
                    print(message)
                    recorder.stop()
                    if 'alexa' in message.lower():
                        await session.send(input=message, end_of_turn=True)
                        async for idx, response in async_enumerate(session.receive()):
                            if response.tool_call is not None:
                                await handle_tool_call(session, response.tool_call)
                            if response.data is not None:
                                stream.write(response.data)
                    stream.close()
                except Exception as e:
                    print(e)
                    break

if __name__ == "__main__":
    recorder = AudioToTextRecorder(language="en", spinner=True, ensure_sentence_ends_with_period=True)
    while True:
        try:
            asyncio.run(main())
        except Exception as e:
            print(e)
        
