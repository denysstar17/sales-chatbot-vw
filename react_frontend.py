import gradio as gr
from react_backend import run_react_response, search_show_car_tool, get_car_info_tool, get_general_info_tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import pandas as pd
import os

from shared import *


tools = [search_show_car_tool, get_car_info_tool, get_general_info_tool]
prompt_react = hub.pull("hwchase17/react")
react_agent = create_react_agent(model, tools=tools, prompt=prompt_react)
react_agent_executor = AgentExecutor(
    agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
)


def chat_function(self, message, history):
    global conversation
    if len(history) == 0:
        conversation = []

    response, conversation = run_react_response(message, react_agent_executor)

    if conversation[-1]['shown_car_id'] is not None:
        response = response + f"\n\n{cars_df.at[conversation[-1]['shown_car_id'], 'link']} <img src='data:image/png;base64,{cars_df.at[conversation[-1]['shown_car_id'], 'image_base64']}'>"
    return response


theme=gr.themes.Default(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.gray)

app = gr.ChatInterface(
    chat_function,
    theme=theme,
    title="Volkswagen Group",
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, auth=(os.environ["GRADIO_USERNAME"], os.environ["GRADIO_PASSWORD"]))