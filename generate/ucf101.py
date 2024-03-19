import openai
import json
from imagenet_prompts import ucf_101
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""

all_json_dict = {}
all_responses = {}
root_folder = 'mpvr'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")
vowel_list = ['A', 'E', 'I', 'O', 'U']

category_list_all = {
    'UCF101': ucf_101}

for k, v in category_list_all.items():

    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}.json"

    if Path(json_name_all).is_file():
        raise ValueError("File already exists")


    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        # Dataset Name: UCF101
        # Description: UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. The action categories can be divided into five types: 1)Human-Object Interaction 2) Body-Motion Only 3) Human-Human Interaction 4) Playing Musical Instruments 5) Sports.

        prompts = []

        # 1-5
        prompts.append("Describe how the action of " + category + " appears in an image.")
        prompts.append("How do you distinguish and recognize the action of " + category + " in an image?")
        prompts.append("Provide a detailed description of what the action of " + category + " looks like in an image.")
        prompts.append(
            "Explore an internet image featuring a person engaged in the action of " + category + " and describe it.")
        prompts.append("In what ways can you identify the action of " + category + " when performed by a person?")

        prompts.append("Elaborate on the visual characteristics of the action class " + category + " in an image.")
        prompts.append("What visual cues enable the recognition of the action of " + category + " in an image?")
        prompts.append("Discuss the appearance and features defining the action of '" + category + "' in an image.")
        prompts.append(
            "Provide insights into an internet image depicting someone performing the action of " + category + ".")
        prompts.append("How can one discern and identify the action of " + category + " when performed by a person?")

        prompts.append("Break down the visual representation of the action category " + category + " in an image.")
        prompts.append(
            "What are the key features facilitating the recognition of the action of " + category + " in an image?")
        prompts.append("Detail the visual elements that compose the action of " + category + " in a still frame.")
        prompts.append("Analyze an internet image portraying the action of " + category + " and provide a description.")
        prompts.append("How can one spot and identify the action of " + category + " when performed by a person?")

        prompts.append("Examine the appearance and traits that define the action class " + category + " in an image.")
        prompts.append("What visual cues contribute to identifying the action of " + category + " in an image?")
        prompts.append("Discuss the visual aspects that make the action of " + category + "' recognizable in an image.")
        prompts.append("Explore a specific internet image illustrating the action of " + category + " and describe it.")
        prompts.append("How can one recognize and identify the action of " + category + " when performed by a person?")

        prompts.append(
            "Provide an analysis of the visual characteristics distinguishing the action of " + category + " in an image.")
        prompts.append(
            "Explain the recognizable features aiding in the identification of the action of " + category + " in an image.")
        prompts.append("Describe the key visual indicators signifying the action category " + category + " in images.")
        prompts.append(
            "Elaborate on an online image portraying the action of " + category + " and highlight its features.")
        prompts.append("How can one discern and recognize the action of " + category + " when performed by a person?")

        res_ = {}

        for curr_prompt in prompts:
            all_result = []

            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=curr_prompt,
                temperature=.99,
                max_tokens=50,
                n=10,
            )

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n\n", "") + ".")

            res_[curr_prompt] = all_result

        all_responses[category] = res_
        with open(json_name_all, 'w') as f:
            json.dump(all_responses, f, indent=4)
