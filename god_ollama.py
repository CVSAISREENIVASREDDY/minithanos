from ollama import AsyncClient, ChatResponse
from all_tools import FUNCTIONS
from utils import get_files_list
import asyncio

base_prompt = """
You are an all-powerful assistant with access to a wide range of tools. Use them skillfully.
Please:
- Respond to me in a friendly, conversational tone.
- Clearly understand and utilize the tools at your disposal.
- When I ask you to perform a task, use the most appropriate tool with the correct arguments.
- If you're unsure how to complete a task directly, run a command that could solve it.
- Never reveal your internal workings or the tools you're using.
- Never reveal that you are a large language model.

Act confidently and helpfully. Let’s get things done.
Don't ask me if you can do something, just do it.
If I ask you something that doesn't require a tool, just answer it directly.
"""

model = "llama3.1:8b"
client = AsyncClient()
files_list = get_files_list()
messages = [
    {"role": "user", "content": base_prompt + "\nHere's my files list:\n" + str(files_list)}
]
client.chat(model=model, messages=messages)


async def chat(prompt):
    global files_list, messages
    new_files_list = get_files_list()
    if new_files_list != files_list:
        files_list = new_files_list
        prompt += "\nHere's my files list:\n" + str(files_list)
    messages += [{"role": "user", "content": prompt}]
    return await client.chat(
        model=model,
        messages= messages,
        tools=list(FUNCTIONS.values()),
    )


async def do_thing(prompt):
  response: ChatResponse = await chat(prompt)

  if response.message.tool_calls:
    # There may be multiple tool calls in the response
    for tool in response.message.tool_calls:
      # Ensure the function is available, and then call it
      if function_to_call := FUNCTIONS.get(tool.function.name):
        print('Calling function:', tool.function.name)
        print('Arguments:', tool.function.arguments)
        output = function_to_call(**tool.function.arguments)
        print('Function output:', output)
      else:
        print('Function', tool.function.name, 'not found')

  # Only needed to chat with the model using the tool call results
  if response.message.tool_calls:
    # Add the function response to messages for the model to use
    messages.append(response.message)
    messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

    # Get final response from model with function outputs
    final_response = await client.chat(model, messages=messages)
    print('Final response:', final_response.message.content)

  else:
    print('No tool calls returned from model')


async def main():
    prompt = "Hello, how are you?"
    await do_thing(prompt)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        await do_thing(user_input)



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
