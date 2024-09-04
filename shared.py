from utils import load_models
import pandas as pd
import ast


def convert_to_list(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def clean_list_values(val):
    if isinstance(val, list):
        if len(val) == 0:
            return None
        else:
            return val[0]
    return val

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#model_id = "mistralai/Mistral-7B-Instruct-v0.3"
#model_id = "google/gemma-2-2b-it"
#model_guidance_id = "google/gemma-2-2b-it"

model, tokenizer, model_guidance = load_models(model_id)

global cars_df
cars_df = pd.read_csv('data/cars.csv')
cars_info_df = pd.read_csv('data/cars_info.csv')

conversation = []

cars_df = cars_df.map(convert_to_list)
cars_df = cars_df.map(clean_list_values)