from llm_backend import node_2, node_3, gather_inf, node_5, node_6, node_7
import pandas as pd
import sys
from langchain.agents import Tool
from io import StringIO


def search_show_car(query, conversation, cars_df, model_guidance):
    node_5_responce = node_5(query, conversation, model_guidance)
    if node_5_responce == "change_current" and last_search_dict is not None:
        chosen_car_id, search_dict, correct_car = node_7(query, last_search_dict, cars_df, model_guidance)
    else:
        chosen_car_id, search_dict, correct_car = node_6(query, conversation, cars_df, model_guidance)
    last_search_dict = search_dict

    if chosen_car_id == None:
        return "Did not find a car for this request."
    return f"Found car id: {chosen_car_id}"


def get_car_info(car_id, cars_df):
    car_dict = f"{cars_df.loc[car_id, ['model', 'version', 'engine_size_cc', 'co2_emissions_g/km', 'power_bhp', 'capacity_kWh', 'range_miles', 'fuel_type', 'transmission', 'drive', 'mileage', 'price', 'monthly_payment', 'warranty_month', 'registration_year', 'number_of_previous_owners', 'doors', 'interior_colors', 'interior_material', 'paint_color', 'metallic_paint', 'ABS', 'sunroof', 'lane_assist', 'car_info_display', 'automatic_headlights', 'interion_ambient_lighting', 'adaptive_cruise_control', 'roof_rails', 'keyless_start', 'cabriolet', 'new_used', 'body_type']].to_dict()}"
    return car_dict


def get_general_info(query, conversation, cars_info_df, model_guidance):
    information = None
    vw_models = node_2(query, conversation, model_guidance)  # find info about which cars is neeeded
    if len(vw_models) > 0:  # if we found info about which cars
        vw_models_topics = node_3(query, conversation, model_guidance)  # looking for topics of the question
        information = gather_inf(vw_models, vw_models_topics, cars_info_df)  # retrieve acctual info

    if information == None:
        return "No information found."
    return information

search_show_car_tool = Tool(
    name="Search And Show A Car",
    func=search_show_car,
    description="Useful for searching for a car in the database and showing it. Input should be the name and/or description of the car."
)


get_car_info_tool = Tool(
    name="Get Information About Shown Car",
    func=get_car_info,
    description="Useful for getting full information about a car. Input should be the id of the car in the database."
)

get_general_info_tool = Tool(
    name="Get General Info",
    func=get_general_info,
    description="Useful for getting relevant information about Volkswagen cars. Input should be the question."
)


def run_react_response(query: str, conversation:list[dict], react_agent_executor, model, tokenizer, model_guidance):
    global cars_df


    react_agent_executor.invoke({"input": query})
    with StringIO() as text_output:
        sys.stdout = text_output
        completion = react_agent_executor.invoke({"input": query})
        sys.stdout = sys.__stdout__ 
        
        text_output_str = text_output.getvalue()

    return text_output_str, 

