from guidance import select, gen
from ast import literal_eval
import pandas as pd
from shared import *


def run_generation(conversation, model, tokenizer):
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1026,
        eos_token_id=terminators,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)


def node_1(query, conversation, model_guidance):  # Do we need to show a car or just answer the question
    if len(conversation) == 0:
        conv = f'''[
{{"role": "user", "content": {query}}},
]'''
    elif len(conversation) == 2:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''
    else:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {conversation[2]['content']}}},
{{"role": "assistant", "content": {conversation[3]['content']}, "shown_car": {conversation[3]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''

    lm = model_guidance + f'''You are a text classifier. Judging from the conversation say wether user wants to see a car or just asking questions. Say "show_car" if the user wants to see an exact car or just states some car specification, say "current_car_question" if user just asks a question about the car that is currently displayed, say "general_question" if user just asks a question about Volkswagen cars or says anythin else. If the user says just a description of the car without asking anything - say "show_car".

User: [
{{"role": "user", "content": "Can you show me a red Volkswagen Golf?"}},
]
Assistant: show_car

User: [
{{"role": "user", "content": "What cars do you have?"}},
]
Assistant: general_question

User: [
{{"role": "user", "content": "The fastest car"}},
]
Assistant: show_car

User: [
{{'role': 'user', 'content': 'used vw golf gti'}},
{{'role': 'assistant', 'content': "Good morning/afternoon.  We have a used Golf GTI with 35,132 miles covered in excellent condition. It's a 2020 model with automatic DSG transmission, black exterior and black cloth interior, and is powered by a 1.984-liter petrol engine with a 245bhp. \n", 'shown_car': '2020 used Volkswagen Golf GTI in black color. Price is: 23775.0.'}},
{{"role": "user", "content": "How many previous owners does it have?"}},
]
Assistant: current_car_question

User: [
{{"role": "user", "content": "Hi, how are you?"}},
]
Assistant: general_question

User: [
{{"role": "user", "content": "Like this but in blue."}},
]
Assistant: show_car

User: [
{{"role": "user", "content": "Is this car electric?"}},
]
Assistant: current_car_question

User: [
{{"role": "user", "content": "Show me the fastest car."}},
]
Assistant: show_car

User: [
{{"role": "user", "content": "hybrid Passat"}},
]
Assistant: show_car

User: [
{{"role": "user", "content": "What is the fastest SUV you have?"}},
]
Assistant: general_question

User: {conv}
Assistant: ''' + select(['show_car', 'general_question', 'current_car_question'], name='answer')

    return lm['answer']


def node_2(query, conversation, model_guidance):  # Information about which models do we need?
    if len(conversation) == 0:
        conv = f'''[
{{"role": "user", "content": {query}}},
]'''
    elif len(conversation) == 2:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''
    else:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {conversation[2]['content']}}},
{{"role": "assistant", "content": {conversation[3]['content']}, "shown_car": {conversation[3]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''

    lm = model_guidance + f'''You are a helpful assistant. User will give you a conversation and you must classify which Volkswagen models the given question is about in the last user query in the conversation. Output the list of model names. If the information about cars is not needed to answer the user - output an empty list "[]". If the question is about all cars - output "['all']". Take model names from the dictionary:

volkswagen_models = {{
    "Golf": {{
        "engine_types": ["gasoline", "diesel", "hybrid"],
        "body_types": ["hatchback"]
    }},
    "Golf_GTI": {{
        "engine_types": ["gasoline"],
        "body_types": ["hatchback"]
    }},
    "Golf_R": {{
        "engine_types": ["gasoline"],
        "body_types": ["hatchback"]
    }},
    "Polo": {{
        "engine_types": ["gasoline"],
        "body_types": ["hatchback"]
    }},
    "Passat": {{
        "engine_types": ["gasoline", "diesel", "hybrid"],
        "body_types": ["sedan"]
    }},
    "Arteon": {{
        "engine_types": ["gasoline", "hybrid"],
        "body_types": ["sedan"]
    }},
    "Arteon_Shooting_Brake": {{
        "engine_types": ["gasoline", "hybrid"],
        "body_types": ["wagon"]
    }},
    "Touareg": {{
        "engine_types": ["gasoline", "diesel", "hybrid"],
        "body_types": ["SUV"]
    }},
    "T-Roc": {{
        "engine_types": ["gasoline", "diesel"],
        "body_types": ["SUV"]
    }},
    "T-Cross": {{
        "engine_types": ["gasoline"],
        "body_types": ["SUV"]
    }},
    }},
    "ID.3": {{
        "engine_types": ["electric"],
        "body_types": ["hatchback"]
    }},
    "ID.4": {{
        "engine_types": ["electric"],
        "body_types": ["SUV"]
    }},
    "ID.5": {{
        "engine_types": ["electric"],
        "body_types": ["SUV"]
    }},
    "ID.7": {{
        "engine_types": ["electric"],
        "body_types": ["sedan"]
    }},
    "ID.BUZZ": {{
        "engine_types": ["electric"],
        "body_types": ["van"]
    }}
}}

User: [
{{"role": "user", "content": "Hi there!"}},
]
Assistant: []

User: [
{{"role": "user", "content": "Which car would you reccomend for a family?"}},
]
Assistant: ['all']

User: [
{{"role": "user", "content": "What are the different paint options for the Passat?"}},
]
Assistant: ['Passat']

User: [
{{"role": "user", "content": "What SUVs do you offer?"}},
]
Assistant: ['Touareg', 'T-Roc', 'T-Cross', 'ID.4', 'ID.5']

User: [
{{"role": "user", "content": "What is the fastest hybrid car?"}},
]
Assistant: ['Golf', 'Passat', 'Arteon', 'Arteon_Shooting_Brake', 'Touareg']

User: [
{{"role": "user", "content": "Can I buy A VW car in London?"}},
]
Assistant: []

User: {conv}
Assistant: ''' + gen(stop=']', name='answer')
    return literal_eval(lm['answer'] + ']')


def node_3(query, conversation, model_guidance):  # Which part of model info do we need?
    if len(conversation) == 0:
        conv = f'''[
{{"role": "user", "content": {query}}},
]'''
    elif len(conversation) == 2:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''
    else:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {conversation[2]['content']}}},
{{"role": "assistant", "content": {conversation[3]['content']}, "shown_car": {conversation[3]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''

    lm = model_guidance + f'''You are a helpful assistant. User will give you a conversation and you must classify - information about which attributes of the car is needed to answer the last user query of the conversation. Output the list of attributes. If the information about attributes is not needed to answer the user - output an empty list "[]". If the question is about all attributes of the cars - output "['all']". Take attributes names from the dictionary:

attributes = {{
    "general_information": ["model name", "description of the car"],
    "performance": ["speed", "acceleration", "fuel efficiency", "other performance stats"],
    "size": ["size of the car"],
    "addons": ["addons and features available for the car"],
}}


User: [
{{"role": "user", "content": "Hi there!"}},
]
Assistant: []

User: [
{{"role": "user", "content": "Which car would you reccomend for a family?"}},
]
Assistant: ['general_information']

User: [
{{"role": "user", "content": "Does ID.3 have LED headlights?"}},
]
Assistant: ['addons']

User: [
{{"role": "user", "content": "Is Golf or Polo larger?"}},
]
Assistant: ['size']

User: [
{{"role": "user", "content": "Tell me everything about Volkswagen Passat."}},
]
Assistant: ['all']

User: {conv}
Assistant: ''' + gen(stop=']', name='answer')
    return literal_eval(lm['answer'] + ']')


def gather_inf(models, topics, cars_info_df):  # Gather needed information from car specs
    if len(models) == 0 or len(topics) == 0:
        return []
    if models[0] == "all":
        models = ['ID.3', 'ID.4', 'ID.7', 'ID.5', 'ID.Buzz', 'Polo', 'Golf', 'Touareg', 'T-Roc', 'Arteon', 'Arteon_Shooting_Brake', 'Passat', 'T-Cross', 'Golf_GTE', 'Golf_GTI', 'Golf_R']
    if topics[0] == "all":
        topics = ['general_information', 'performance', 'size', 'addons']
        
    filtered_df = cars_info_df[cars_info_df['model'].isin(models)]

    information = '; '.join(
        filtered_df.apply(
            lambda row: ', '.join([f"{row['model']}, {col}: {row[col]}" for col in topics]),
            axis=1
        )
    )
    return information


def node_4(query, conversation, information, model, tokenizer):  # Answer the question

    if len(conversation) == 2:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
]'''
    elif len(conversation) == 4:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {conversation[2]['content']}}},
{{"role": "assistant", "content": {conversation[3]['content']}, "shown_car": {conversation[3]['shown_car']}}},
]'''

    if len(conversation) == 0:
        if len(information) > 0:
            prompt = f"""You are a Volkswagen customer assistant. You help customers choose and configure Volkswagen cars. Use this Information to answer questions:

Information:
{information}

The question is: {query}"""
        else:
            prompt = f"""You are a Volkswagen customer assistant. You help customers choose and configure Volkswagen cars. Answer their questions politely. Be concise. Just answer the question, do not ask user any questions.

The question is: {query}"""

        conversation_for_gen = [{"role": "user", "content": prompt},]
    else:
        if len(information) > 0:
            prompt = f"""You are a Volkswagen customer assistant. You help customers choose and configure Volkswagen cars. Use this Information to answer questions:

Information:
{information}

The previous conversation is: {conv}

The question is: {query}"""
        else:
            prompt = f"""You are a Volkswagen customer assistant. You help customers choose and configure Volkswagen cars. Answer their questions politely. Be concise. Just answer the question, do not ask user any questions.

The previous conversation is: {conv}

The question is: {query}"""

        conversation_for_gen = [{"role": "user", "content": prompt},]
    
    return run_generation(conversation_for_gen, model, tokenizer)


def node_5(query, conversation, model_guidance):  # Does user ask to change the current shown car or a completely new car
    if len(conversation) == 0:
        conv = f'''[
{{"role": "user", "content": {query}}},
]'''
    elif len(conversation) == 2:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''
    else:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {conversation[2]['content']}}},
{{"role": "assistant", "content": {conversation[3]['content']}, "shown_car": {conversation[3]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''

    lm = model_guidance + f'''You are a text classifier. Judging from the last user query in the conversation say if the user wants to see changes to a shown car or wants to see a completely new car. Say "show_new" if the user wants to see a new car, say "change_current" in case user wants to add or change something about the car that is already shown.

User: [
{{"role": "user", "content": "Can you show me a red Volkswagen Golf?"}},
]
Assistant: show_new

User: [
{{'role': 'user', 'content': 'what is the fastest car you have?'}},
{{'role': 'assistant', 'content': 'The fastest car in our lineup is the **Golf R**. With its 2.0-liter TSI engine producing up to **320 hp**, it can accelerate from 0 to 100 km/h in just **4.7 seconds**.\n', 'shown_car': None}},
{{"role": "user", "content": "Give me it in blue."}},
]
Assistant: show_new

User: [
{{'role': 'user', 'content': 'diesel touareg'}},
{{'role': 'assistant', 'content': 'Welcome to the world of the Volkswagen Touareg Black Edition.  This powerful and stylish SUV boasts a 3.0-liter diesel engine, a smooth automatic transmission, and all-wheel drive for exceptional handling. Its luxurious interior features premium leather seating and advanced technologies for an unmatched driving experience. \n', 'shown_car': '2024 new Volkswagen Touareg Black_Edition in beige color. Price is: 68065.0.'}},
{{"role": "user", "content": "This is too expensive for me."}},
]
Assistant: change_current

User: [
{{"role": "user", "content": "family car"}},
]
Assistant: show_new

User: [
{{"role": "user", "content": "This is too expensive for me."}},
]
Assistant: change_current

User: [
{{"role": "user", "content": "Passat with 18 inch wheels."}},
]
Assistant: show_new

User: {conv}
Assistant: ''' + select(['show_new', 'change_current'], name='answer')

    return lm['answer']



def node_6(query, conversation, cars_df, model_guidance):

    if len(conversation) == 0:
        conv = f'''[
{{"role": "user", "content": {query}}},
]'''
    elif len(conversation) == 2:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''
    else:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {conversation[2]['content']}}},
{{"role": "assistant", "content": {conversation[3]['content']}, "shown_car": {conversation[3]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''

    lm = model_guidance + f'''You are a sales assistant for Volkswagen who fills up json files according to conversation with user. You must return a search json according to users' last query. User will tell you which car he wants, according to this you must fill up the json. If in the given field you have an "int | none," choise you mmust choose none if the user did not imply to change this option and int if he said to change it. Set "none" everywere by default. Only set any int or categorical values if the user explicitly ask for them. If the user asks you to show him the car he was told previously in conversation only set "model" and "version" unles user (not assistant) ask for anything else. Set none to every numerical value unless the user (not assistant) gives a number. Only output the given json.

json format:
{{
    "model": ID.3 | ID.4 | ID.5 | ID.7 | ID.Buzz | Polo | Golf | T-Cross | T-Roc | Tiguan | Touareg | Passat | Arteon | Arteon_Shooting_Brake | none,
    "new_used": new | used | none,
    "body_type": "hatchback" | "SUV" | "sedan" | "van" | "wagon" | "none",
    "fuel_type": petrol | plug_in_hybrid | diesel | electric | none,
    "paint_color": beige | black | blue | brown | green | grey | orange | pink | purple | red | silver | white | yellow | none,
    "engine_size_cc(less_than)": int | none,
    "engine_size_cc(more_than)": int | none,
    "engine_size_cc(min)": True | none,
    "engine_size_cc(max)": True | none,
    "power_bhp(less_than)": int | none,
    "power_bhp(more_than)": int | none,
    "power_bhp(min)": True | none,
    "power_bhp(max)": True | none,
    "mileage(less_than)": int | none,
    "mileage(more_than)": int | none,
    "mileage(min)": True | none,
    "mileage(max)": True | none,
    "price(less_than)": int | none,
    "price(more_than)": int | none,
    "price(min)": True | none,
    "price(max)": True | none,
    "registration_year(less_than)": int | none,
    "registration_year(more_than)": int | none,
    "registration_year(min)": True | none,
    "registration_year(max)": True | none,
    "number_of_previous_owners": int,
    "number_of_previous_owners(less_than)": int | none,
    "number_of_previous_owners(more_than)": int | none,
    "ABS": True | none,
    "sunroof": True | none,
    "lane_assist": True | none,
    "car_info_display": True | none,
    "automatic_headlights": True | none,
    "interion_ambient_lighting": True | none,
    "adaptive_cruise_control": True | none,
    "roof_rails": True | none,
    "keyless_start": True | none,
    "cabriolet": True | none
}}


User: [
{{"role": "user", "content": "Green Passat Style with diesel engine. It must have ABS. The price should be no more than 24000. It mast have the least mileage possible."}},
]
Assistant: {{
    "model": Passat,
    "new_used": none,
    "body_type" none,
    "fuel_type": diesel,
    "paint_color": green,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": none,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": True,
    "mileage(max)": none,
    "price(less_than)": 24000,
    "price(more_than)": none,
    "price(min)": none,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": True,
    "sunroof": none,
    "lane_assist": none,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}


User: [
{{"role": "user", "content": "Show me the most powerful used cabriolet that costs at least 15000 and has lane assist"}},
]
Assistant: {{
    "model": none,
    "new_used": used,
    "body_type" none,
    "fuel_type": none,
    "paint_color": none,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": True,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": none,
    "mileage(max)": none,
    "price(less_than)": none,
    "price(more_than)": 15000,
    "price(min)": none,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": none,
    "sunroof": none,
    "lane_assist": True,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}


User: [
{{"role": "user", "content": "Show me the cheapest car"}},
]
Assistant: {{
    "model": none,
    "new_used": none,
    "body_type" none,
    "fuel_type": none,
    "paint_color": none,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": none,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": none,
    "mileage(max)": none,
    "price(less_than)": none,
    "price(more_than)": none,
    "price(min)": True,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": none,
    "sunroof": none,
    "lane_assist": none,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}


User: [
{{'role': 'user', 'content': 'what is the fastest car you have?'}},
{{'role': 'assistant', 'content': "Our fastest car is the **Golf R**. It's powered by a 2.0-liter TSI engine that produces up to 320 hp, enabling it to accelerate from 0 to 100 km/h in just 4.7 seconds and reach a top speed electronically limited to 250 km/h.", 'shown_car': None}},
{{'role': 'user', 'content': 'show me it'}}
]
Assistant: {{
    "model": Golf,
    "new_used": none,
    "body_type" none,
    "fuel_type": none,
    "paint_color": none,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": none,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": none,
    "mileage(max)": none,
    "price(less_than)": none,
    "price(more_than)": none,
    "price(min)": none,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": none,
    "sunroof": none,
    "lane_assist": none,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}


User: [
{{'role': 'user', 'content': 'what is the most powerfull vw suv?'}},
{{'role': 'assistant', 'content': 'The most powerful Volkswagen SUV is the **Touareg**.\n\nIt comes equipped with a 3.0-liter V6 TDI engine that can reach up to 282 hp.  \n', 'shown_car': None}},
{{'role': 'user', 'content': 'show me it'}}
]
Assistant: {{
    "model": Touareg,
    "new_used": none,
    "body_type" none,
    "fuel_type": none,
    "paint_color": none,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": none,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": none,
    "mileage(max)": none,
    "price(less_than)": none,
    "price(more_than)": none,
    "price(min)": none,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": none,
    "sunroof": none,
    "lane_assist": none,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}


User: {conv}
Assisstant: {{
    "model": '''+ select(["none", "ID.3", "ID.4", "ID.5", "ID.7", "ID.Buzz", "Polo", "Golf", "T-Cross", "T-Roc", "Tiguan", "Touareg", "Passat", "Arteon", "Arteon_Shooting_Brake"], name='model')+''',
    "new_used": '''+ select(["new", "used", "none"], name='new_used')+''',
    "body_type": '''+ select(["hatchback", "SUV", "sedan", "van", "wagon", "none"], name='body_type')+''',
    "fuel_type": '''+ select(["petrol", "plug_in_hybrid", "diesel", "electric", "none"], name='fuel_type')+''',
    "paint_color": '''+ select(["beige", "black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "silver", "white", "yellow", "none"], name='paint_color')+''',
    "engine_size_cc(less_than)":  '''+ gen(stop=',', name='engine_size_cc(less_than)')+''',
    "engine_size_cc(more_than)":  '''+ gen(stop=',', name='engine_size_cc(more_than)')+''',
    "engine_size_cc(min)": '''+ select(["True", "none"], name='engine_size_cc(min)')+''',
    "engine_size_cc(max)": '''+ select(["True", "none"], name='engine_size_cc(max)')+''',
    "power_bhp(less_than)": '''+ gen(stop=',', name='power_bhp(less_than)')+''',
    "power_bhp(more_than)": '''+ gen(stop=',', name='power_bhp(more_than)')+''',
    "power_bhp(min)": '''+ select(["True", "none"], name='power_bhp(min)')+''',
    "power_bhp(max)": '''+ select(["True", "none"], name='power_bhp(max)')+''',
    "mileage(less_than)": '''+ gen(stop=',', name='mileage(less_than)')+''',
    "mileage(more_than)": '''+ gen(stop=',', name='mileage(more_than)')+''',
    "mileage(min)": '''+ select(["True", "none"], name='mileage(min)')+''',
    "mileage(max)": '''+ select(["True", "none"], name='mileage(max)')+''',
    "price(less_than)": '''+ gen(stop=',', name='price(less_than)')+''',
    "price(more_than)": '''+ gen(stop=',', name='price(more_than)')+''',
    "price(min)": '''+ select(["True", "none"], name='price(min)')+''',
    "price(max)": '''+ select(["True", "none"], name='price(max)')+''',
    "registration_year(less_than)": '''+ gen(stop=',', name='registration_year(less_than)')+''',
    "registration_year(more_than)": '''+ gen(stop=',', name='registration_year(more_than)')+''',
    "registration_year(min)": '''+ select(["True", "none"], name='registration_year(min)')+''',
    "registration_year(max)": '''+ select(["True", "none"], name='registration_year(max)')+''',
    "number_of_previous_owners": '''+ gen(stop=',', name='number_of_previous_owners')+''',
    "number_of_previous_owners(less_than)": '''+ gen(stop=',', name='number_of_previous_owners(less_than)')+''',
    "number_of_previous_owners(more_than)": '''+ gen(stop=',', name='number_of_previous_owners(more_than)')+''',
    "ABS": '''+ select(["True", "none"], name='ABS')+''',
    "sunroof": '''+ select(["True", "none"], name='sunroof')+''',
    "lane_assist": '''+ select(["True", "none"], name='lane_assist')+''',
    "car_info_display": '''+ select(["True", "none"], name='car_info_display')+''',
    "automatic_headlights": '''+ select(["True", "none"], name='automatic_headlights')+''',
    "interion_ambient_lighting": '''+ select(["True", "none"], name='interion_ambient_lighting')+''',
    "adaptive_cruise_control": '''+ select(["True", "none"], name='adaptive_cruise_control')+''',
    "roof_rails": '''+ select(["True", "none"], name='roof_rails')+''',
    "keyless_start": '''+ select(["True", "none"], name='keyless_start')+''',
    "cabriolet": '''+ select(["True", "none"], name='cabriolet')+''',
}}
'''
    search_dict = {
        "model": lm['model'],
        "new_used": lm['new_used'],
        "body_type": lm['body_type'],
        "fuel_type": lm['fuel_type'],
        "paint_color": lm['paint_color'],
        "engine_size_cc(less_than)": lm['engine_size_cc(less_than)'],
        "engine_size_cc(more_than)": lm['engine_size_cc(more_than)'],
        "engine_size_cc(min)": lm['engine_size_cc(min)'],
        "engine_size_cc(max)": lm['engine_size_cc(max)'],
        "power_bhp(less_than)": lm['power_bhp(less_than)'],
        "power_bhp(more_than)": lm['power_bhp(more_than)'],
        "power_bhp(min)": lm['power_bhp(min)'],
        "power_bhp(max)": lm['power_bhp(max)'],
        "mileage(less_than)": lm['mileage(less_than)'],
        "mileage(more_than)": lm['mileage(more_than)'],
        "mileage(min)": lm['mileage(min)'],
        "mileage(max)": lm['mileage(max)'],
        "price(less_than)": lm['price(less_than)'],
        "price(more_than)": lm['price(more_than)'],
        "price(min)": lm['price(min)'],
        "price(max)": lm['price(max)'],
        "registration_year(less_than)": lm['registration_year(less_than)'],
        "registration_year(more_than)": lm['registration_year(more_than)'],
        "registration_year(min)": lm['registration_year(min)'],
        "registration_year(max)": lm['registration_year(max)'],
        "number_of_previous_owners": lm['number_of_previous_owners'],
        "number_of_previous_owners(less_than)": lm['number_of_previous_owners(less_than)'],
        "number_of_previous_owners(more_than)": lm['number_of_previous_owners(more_than)'],
        "ABS": lm['ABS'],
        "sunroof": lm['sunroof'],
        "lane_assist": lm['lane_assist'],
        "car_info_display": lm['car_info_display'],
        "automatic_headlights": lm['automatic_headlights'],
        "interion_ambient_lighting": lm['interion_ambient_lighting'],
        "adaptive_cruise_control": lm['adaptive_cruise_control'],
        "roof_rails": lm['roof_rails'],
        "keyless_start": lm['keyless_start'],
        "cabriolet": lm['cabriolet'],
    }
    
    if not any(char.isdigit() for char in query):  # TODO: do it not like this
        for key in ["engine_size_cc(less_than)", "engine_size_cc(more_than)", "power_bhp(less_than)", "power_bhp(more_than)", "mileage(less_than)", "mileage(more_than)", "price(less_than)", "price(more_than)", "registration_year(less_than)", "registration_year(more_than)"]:
            if key in search_dict:
                search_dict[key] = 'none'

    filtered_list = filter_cars_df(search_dict, cars_df)

    if len(filtered_list) > 0:
        correct_car = True
        return min(filtered_list), search_dict, correct_car
    else:
        correct_car = False
        return None, search_dict, correct_car


def node_7(query, last_search_dict, cars_df, model_guidance):

    lm = model_guidance + f'''You are a sales assistant for Volkswagen who fills up search json files according to the query and the previous search json. You must return a search json according to users' last query. User will tell you which car he wants, according to this you must fill up the json. If in the given field you have an "int | none," choise you mmust choose none if the user did not imply to change this option and int if he said to change it. Set "none" everywere by default. Only set any int or categorical values if the user explicitly ask for them. You must apply changes to the json according to the user query.

json format:
{{
    "model": ID.3 | ID.4 | ID.5 | ID.7 | ID.Buzz | Polo | Golf | T-Cross | T-Roc | Tiguan | Touareg | Passat | Arteon | Arteon_Shooting_Brake | none,
    "new_used": new | used | none,
    "body_type": "hatchback" | "SUV" | "sedan" | "van" | "wagon" | "none",
    "fuel_type": petrol | plug_in_hybrid | diesel | electric | none,
    "paint_color": beige | black | blue | brown | green | grey | orange | pink | purple | red | silver | white | yellow | none,
    "engine_size_cc(less_than)": int | none,
    "engine_size_cc(more_than)": int | none,
    "engine_size_cc(min)": True | none,
    "engine_size_cc(max)": True | none,
    "power_bhp(less_than)": int | none,
    "power_bhp(more_than)": int | none,
    "power_bhp(min)": True | none,
    "power_bhp(max)": True | none,
    "mileage(less_than)": int | none,
    "mileage(more_than)": int | none,
    "mileage(min)": True | none,
    "mileage(max)": True | none,
    "price(less_than)": int | none,
    "price(more_than)": int | none,
    "price(min)": True | none,
    "price(max)": True | none,
    "registration_year(less_than)": int | none,
    "registration_year(more_than)": int | none,
    "registration_year(min)": True | none,
    "registration_year(max)": True | none,
    "number_of_previous_owners": int,
    "number_of_previous_owners(less_than)": int | none,
    "number_of_previous_owners(more_than)": int | none,
    "ABS": True | none,
    "sunroof": True | none,
    "lane_assist": True | none,
    "car_info_display": True | none,
    "automatic_headlights": True | none,
    "interion_ambient_lighting": True | none,
    "adaptive_cruise_control": True | none,
    "roof_rails": True | none,
    "keyless_start": True | none,
    "cabriolet": True | none
}}


User: Give me a used one.
Last search json: {{
    "model": Passat,
    "new_used": none,
    "body_type" none,
    "fuel_type": diesel,
    "paint_color": green,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": none,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": True,
    "mileage(max)": none,
    "price(less_than)": 24000,
    "price(more_than)": none,
    "price(min)": none,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": True,
    "sunroof": none,
    "lane_assist": none,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}
Assistant: {{
    "model": Passat,
    "new_used": used,
    "body_type" none,
    "fuel_type": diesel,
    "paint_color": green,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": none,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": True,
    "mileage(max)": none,
    "price(less_than)": 24000,
    "price(more_than)": none,
    "price(min)": none,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": True,
    "sunroof": none,
    "lane_assist": none,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}


User: I need a cheaper Golf
Last search json: {{
    "model": Golf,
    "new_used": used,
    "body_type" none,
    "fuel_type": none,
    "paint_color": none,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": True,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": none,
    "mileage(max)": none,
    "price(less_than)": none,
    "price(more_than)": none,
    "price(min)": none,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": none,
    "sunroof": none,
    "lane_assist": True,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}
Assistant: {{
    "model": Golf,
    "new_used": used,
    "body_type" none,
    "fuel_type": none,
    "paint_color": none,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": True,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": none,
    "mileage(max)": none,
    "price(less_than)": none,
    "price(more_than)": none,
    "price(min)": True,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": none,
    "sunroof": none,
    "lane_assist": True,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}


User: Show me it in blue
Last search json: {{
    "model": none,
    "new_used": none,
    "body_type" none,
    "fuel_type": none,
    "paint_color": blue,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": none,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": none,
    "mileage(max)": none,
    "price(less_than)": none,
    "price(more_than)": none,
    "price(min)": True,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": none,
    "sunroof": none,
    "lane_assist": none,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}
Assistant: {{
    "model": none,
    "new_used": none,
    "body_type" none,
    "fuel_type": none,
    "paint_color": none,
    "engine_size_cc(less_than)": none,
    "engine_size_cc(more_than)": none,
    "engine_size_cc(min)": none,
    "engine_size_cc(max)": none,
    "power_bhp(less_than)": none,
    "power_bhp(more_than)": none,
    "power_bhp(min)": none,
    "power_bhp(max)": none,
    "mileage(less_than)": none,
    "mileage(more_than)": none,
    "mileage(min)": none,
    "mileage(max)": none,
    "price(less_than)": none,
    "price(more_than)": none,
    "price(min)": True,
    "price(max)": none,
    "registration_year(less_than)": none,
    "registration_year(more_than)": none,
    "registration_year(min)": none,
    "registration_year(max)": none,
    "number_of_previous_owners": none,
    "number_of_previous_owners(less_than)": none,
    "number_of_previous_owners(more_than)": none,
    "ABS": none,
    "sunroof": none,
    "lane_assist": none,
    "car_info_display": none,
    "automatic_headlights": none,
    "interion_ambient_lighting": none,
    "adaptive_cruise_control": none,
    "roof_rails": none,
    "keyless_start": none,
    "cabriolet": none
}}


User: {query}
Assisstant: {{
    "model": '''+ select(["none", "ID.3", "ID.4", "ID.5", "ID.7", "ID.Buzz", "Polo", "Golf", "T-Cross", "T-Roc", "Tiguan", "Touareg", "Passat", "Arteon", "Arteon_Shooting_Brake"], name='model')+''',
    "new_used": '''+ select(["new", "used", "none"], name='new_used')+''',
    "body_type": '''+ select(["hatchback", "SUV", "sedan", "van", "wagon", "none"], name='body_type')+''',
    "fuel_type": '''+ select(["petrol", "plug_in_hybrid", "diesel", "electric", "none"], name='fuel_type')+''',
    "paint_color": '''+ select(["beige", "black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "silver", "white", "yellow", "none"], name='paint_color')+''',
    "engine_size_cc(less_than)":  '''+ gen(stop=',', name='engine_size_cc(less_than)')+''',
    "engine_size_cc(more_than)":  '''+ gen(stop=',', name='engine_size_cc(more_than)')+''',
    "engine_size_cc(min)": '''+ select(["True", "none"], name='engine_size_cc(min)')+''',
    "engine_size_cc(max)": '''+ select(["True", "none"], name='engine_size_cc(max)')+''',
    "power_bhp(less_than)": '''+ gen(stop=',', name='power_bhp(less_than)')+''',
    "power_bhp(more_than)": '''+ gen(stop=',', name='power_bhp(more_than)')+''',
    "power_bhp(min)": '''+ select(["True", "none"], name='power_bhp(min)')+''',
    "power_bhp(max)": '''+ select(["True", "none"], name='power_bhp(max)')+''',
    "mileage(less_than)": '''+ gen(stop=',', name='mileage(less_than)')+''',
    "mileage(more_than)": '''+ gen(stop=',', name='mileage(more_than)')+''',
    "mileage(min)": '''+ select(["True", "none"], name='mileage(min)')+''',
    "mileage(max)": '''+ select(["True", "none"], name='mileage(max)')+''',
    "price(less_than)": '''+ gen(stop=',', name='price(less_than)')+''',
    "price(more_than)": '''+ gen(stop=',', name='price(more_than)')+''',
    "price(min)": '''+ select(["True", "none"], name='price(min)')+''',
    "price(max)": '''+ select(["True", "none"], name='price(max)')+''',
    "registration_year(less_than)": '''+ gen(stop=',', name='registration_year(less_than)')+''',
    "registration_year(more_than)": '''+ gen(stop=',', name='registration_year(more_than)')+''',
    "registration_year(min)": '''+ select(["True", "none"], name='registration_year(min)')+''',
    "registration_year(max)": '''+ select(["True", "none"], name='registration_year(max)')+''',
    "number_of_previous_owners": '''+ gen(stop=',', name='number_of_previous_owners')+''',
    "number_of_previous_owners(less_than)": '''+ gen(stop=',', name='number_of_previous_owners(less_than)')+''',
    "number_of_previous_owners(more_than)": '''+ gen(stop=',', name='number_of_previous_owners(more_than)')+''',
    "ABS": '''+ select(["True", "none"], name='ABS')+''',
    "sunroof": '''+ select(["True", "none"], name='sunroof')+''',
    "lane_assist": '''+ select(["True", "none"], name='lane_assist')+''',
    "car_info_display": '''+ select(["True", "none"], name='car_info_display')+''',
    "automatic_headlights": '''+ select(["True", "none"], name='automatic_headlights')+''',
    "interion_ambient_lighting": '''+ select(["True", "none"], name='interion_ambient_lighting')+''',
    "adaptive_cruise_control": '''+ select(["True", "none"], name='adaptive_cruise_control')+''',
    "roof_rails": '''+ select(["True", "none"], name='roof_rails')+''',
    "keyless_start": '''+ select(["True", "none"], name='keyless_start')+''',
    "cabriolet": '''+ select(["True", "none"], name='cabriolet')+''',
}}
'''
    search_dict = {
        "model": lm['model'],
        "new_used": lm['new_used'],
        "body_type": lm['body_type'],
        "fuel_type": lm['fuel_type'],
        "paint_color": lm['paint_color'],
        "engine_size_cc(less_than)": lm['engine_size_cc(less_than)'],
        "engine_size_cc(more_than)": lm['engine_size_cc(more_than)'],
        "engine_size_cc(min)": lm['engine_size_cc(min)'],
        "engine_size_cc(max)": lm['engine_size_cc(max)'],
        "power_bhp(less_than)": lm['power_bhp(less_than)'],
        "power_bhp(more_than)": lm['power_bhp(more_than)'],
        "power_bhp(min)": lm['power_bhp(min)'],
        "power_bhp(max)": lm['power_bhp(max)'],
        "mileage(less_than)": lm['mileage(less_than)'],
        "mileage(more_than)": lm['mileage(more_than)'],
        "mileage(min)": lm['mileage(min)'],
        "mileage(max)": lm['mileage(max)'],
        "price(less_than)": lm['price(less_than)'],
        "price(more_than)": lm['price(more_than)'],
        "price(min)": lm['price(min)'],
        "price(max)": lm['price(max)'],
        "registration_year(less_than)": lm['registration_year(less_than)'],
        "registration_year(more_than)": lm['registration_year(more_than)'],
        "registration_year(min)": lm['registration_year(min)'],
        "registration_year(max)": lm['registration_year(max)'],
        "number_of_previous_owners": lm['number_of_previous_owners'],
        "number_of_previous_owners(less_than)": lm['number_of_previous_owners(less_than)'],
        "number_of_previous_owners(more_than)": lm['number_of_previous_owners(more_than)'],
        "ABS": lm['ABS'],
        "sunroof": lm['sunroof'],
        "lane_assist": lm['lane_assist'],
        "car_info_display": lm['car_info_display'],
        "automatic_headlights": lm['automatic_headlights'],
        "interion_ambient_lighting": lm['interion_ambient_lighting'],
        "adaptive_cruise_control": lm['adaptive_cruise_control'],
        "roof_rails": lm['roof_rails'],
        "keyless_start": lm['keyless_start'],
        "cabriolet": lm['cabriolet'],
    }
    
    if not any(char.isdigit() for char in query):  # TODO: do it not like this
        for key in ["engine_size_cc(less_than)", "engine_size_cc(more_than)", "power_bhp(less_than)", "power_bhp(more_than)", "mileage(less_than)", "mileage(more_than)", "price(less_than)", "price(more_than)", "registration_year(less_than)", "registration_year(more_than)"]:
            if key in search_dict:
                search_dict[key] = 'none'

    filtered_list = filter_cars_df(search_dict, cars_df)

    if len(filtered_list) > 0:
        correct_car = True
        return min(filtered_list), search_dict, correct_car
    else:
        correct_car = False
        return None, search_dict, correct_car
    


def filter_cars_df(search_dict, cars_df):
    print(search_dict)
    mask = pd.Series([True] * len(cars_df))
    # First pass: filter by standard criteria (excluding min/max)
    for key, value in search_dict.items():
        if value == 'none':
            continue

        if '(less_than)' in key:
            column = key.replace('(less_than)', '')
            mask &= cars_df[column].notna() & (cars_df[column] < float(value))

        elif '(more_than)' in key:
            column = key.replace('(more_than)', '')
            mask &= cars_df[column].notna() & (cars_df[column] > float(value))

        elif '(min)' not in key and '(max)' not in key:
            # For boolean flags (True or none)
            if value == 'True':
                mask &= cars_df[key].notna() & (cars_df[key] == True)
            
            # For categorical or string matches
            elif key in cars_df.columns:
                mask &= cars_df[key].notna() & (cars_df[key] == value)

    # Second pass: filter by min/max criteria
    for key, value in search_dict.items():
        if value == 'none':
            continue

        if '(min)' in key:
            column = key.replace('(min)', '')
            if value == 'True':
                min_value = cars_df[mask][column].min()  # Only consider rows that passed the first filter
                mask &= cars_df[column].notna() & (cars_df[column] == min_value)

        elif '(max)' in key:
            column = key.replace('(max)', '')
            if value == 'True':
                max_value = cars_df[mask][column].max()  # Only consider rows that passed the first filter
                mask &= cars_df[column].notna() & (cars_df[column] == max_value)

    # Filter the DataFrame based on the final mask
    filtered_indices = cars_df[mask].index.tolist()

    return filtered_indices


def node_8(query, conversation, chosen_car_id, cars_df, model, tokenizer):  # Justify correct car

    if len(conversation) == 0:
        conv = f'''[
{{"role": "user", "content": {query}}},
]'''
    elif len(conversation) == 2:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''
    else:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {conversation[2]['content']}}},
{{"role": "assistant", "content": {conversation[3]['content']}, "shown_car": {conversation[3]['shown_car']}}},
{{"role": "user", "content": {query}}},
]'''

    car_dict = f"{cars_df.loc[chosen_car_id, ['model', 'version', 'engine_size_cc', 'co2_emissions_g/km', 'power_bhp', 'capacity_kWh', 'range_miles', 'fuel_type', 'transmission', 'drive', 'mileage', 'price', 'monthly_payment', 'warranty_month', 'registration_year', 'number_of_previous_owners', 'doors', 'interior_colors', 'interior_material', 'paint_color', 'metallic_paint', 'ABS', 'sunroof', 'lane_assist', 'car_info_display', 'automatic_headlights', 'interion_ambient_lighting', 'adaptive_cruise_control', 'roof_rails', 'keyless_start', 'cabriolet', 'new_used', 'body_type']].to_dict()}"
    prompt = f"""You are a Volkswagen customer assistant. You will get a coversation and a car that was chosen according to customers' request in the conversation, you need to tell the customer about the car. Be concise, do not ask user any questions. Do not offer user anything.

Previous conversation:
{conv}

Chosen car dictionary:
{car_dict}

The query is: {query}"""

    conversation_for_gen = []
    for i in range(len(conversation)):
        conversation_for_gen.append({"role": conversation[i]['role'], "content": conversation[i]['content']})
    conversation_for_gen.append({"role": "user", "content": prompt})
    return run_generation(conversation_for_gen, model, tokenizer)


def node_11(query, conversation, last_shown_car_id, cars_df, model, tokenizer):  # qa for the last shown car
    if len(conversation) == 2:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
]'''
    elif len(conversation) == 4:
        conv = f'''[
{{"role": "user", "content": {conversation[0]['content']}}},
{{"role": "assistant", "content": {conversation[1]['content']}, "shown_car": {conversation[1]['shown_car']}}},
{{"role": "user", "content": {conversation[2]['content']}}},
{{"role": "assistant", "content": {conversation[3]['content']}, "shown_car": {conversation[3]['shown_car']}}},
]'''
        
    car_dict = f"{cars_df.loc[last_shown_car_id, ['model', 'version', 'engine_size_cc', 'co2_emissions_g/km', 'power_bhp', 'capacity_kWh', 'range_miles', 'fuel_type', 'transmission', 'drive', 'mileage', 'price', 'monthly_payment', 'warranty_month', 'registration_year', 'number_of_previous_owners', 'doors', 'interior_colors', 'interior_material', 'paint_color', 'metallic_paint', 'ABS', 'sunroof', 'lane_assist', 'car_info_display', 'automatic_headlights', 'interion_ambient_lighting', 'adaptive_cruise_control', 'roof_rails', 'keyless_start', 'cabriolet', 'new_used', 'body_type']].to_dict()}"

    prompt = f"""You are a polite Volkswagen customer assistant. You answer questions about cars according to the shown car information. Just answer the question, do not say where you got the information from.

Shown car information:
{car_dict}

The previous conversation is: {conv}

The question is: {query}"""


    conversation_for_gen = [{"role": "user", "content": prompt},]
    
    return run_generation(conversation_for_gen, model, tokenizer)


def car_summury_from_id(cars_df, car_id):
    car_summary = f"{cars_df['registration_year'][car_id]} {cars_df['new_used'][car_id]} Volkswagen {cars_df['model'][car_id]} {cars_df['version'][car_id]} in {cars_df['paint_color'][car_id]} color. Price is: {cars_df['price'][car_id]}."
    return car_summary


def run_response(query):
    global conversation
    if len(conversation) > 4:
        conversation = conversation[-4:]
    last_shown_car_id = None
    for assistant_response in conversation[::-2]:
        if assistant_response["shown_car_id"] is not None:
            last_shown_car_id = assistant_response["shown_car_id"]
        break

    last_search_dict = None
    if len(conversation) > 0  and conversation[-1]["search_dict"] is not None:
        last_search_dict = conversation[-1]["search_dict"]
            
    node_1_responce = node_1(query, conversation, model_guidance)


    if node_1_responce == "current_car_question" and last_shown_car_id is not None:
        node_11_responce = node_11(query, conversation, last_shown_car_id, cars_df, model, tokenizer)
        conversation.append({"role": "user", "content": query})
        conversation.append({"role": "assistant", "content": node_11_responce, "shown_car": None, "shown_car_id": None, "search_dict": None})
        return node_11_responce, conversation


    elif node_1_responce == "show_car":
        node_5_responce = node_5(query, conversation, model_guidance)
        if node_5_responce == "change_current" and last_search_dict is not None:
            chosen_car_id, search_dict, correct_car = node_7(query, last_search_dict, cars_df, model_guidance)
        else:
            chosen_car_id, search_dict, correct_car = node_6(query, conversation, cars_df, model_guidance)

        if correct_car == True:
            node_8_responce = node_8(query, conversation, chosen_car_id, cars_df, model, tokenizer)
            conversation.append({"role": "user", "content": query})
            conversation.append({"role": "assistant", "content": node_8_responce, "shown_car": car_summury_from_id(cars_df, chosen_car_id), "shown_car_id": chosen_car_id, "search_dict": search_dict})
            return node_8_responce, conversation
        elif correct_car == "almost_true":  # So far not working
            node_9_responce = "justify not correct car"
            conversation.append({"role": "user", "content": query})
            conversation.append({"role": "assistant", "content": node_9_responce, "shown_car": car_summury_from_id(cars_df, chosen_car_id), "shown_car_id": chosen_car_id, "search_dict": search_dict})
            return node_9_responce, conversation
        else:
            node_10_responce = "Unfortunately such car is currently unavailable. How else can I help you?"
            conversation.append({"role": "user", "content": query})
            conversation.append({"role": "assistant", "content": node_10_responce, "shown_car": None, "shown_car_id": None, "search_dict": search_dict})
            return node_10_responce, conversation


    else:  # it is not a direct request to show a car or a question abou the shown car
        vw_models = node_2(query, conversation, model_guidance)  # find info about which cars is neeeded
        if len(vw_models) > 0:  # if we found info about which cars
            vw_models_topics = node_3(query, conversation, model_guidance)  # looking for topics of the question
            information = gather_inf(vw_models, vw_models_topics, cars_info_df)  # retrieve acctual info
        else:
            information = []
        node_4_responce = node_4(query, conversation, information, model, tokenizer)  # qa node
        conversation.append({"role": "user", "content": query})
        conversation.append({"role": "assistant", "content": node_4_responce, "shown_car": None, "shown_car_id": None, "search_dict": None})
        return node_4_responce, conversation