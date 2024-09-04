import gradio as gr
import os
from llm_backend import run_response
from shared import *



def chat_function(message, history):
    global conversation

    if len(history) == 0:
        conversation = []
    response, conversation = run_response(message)

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