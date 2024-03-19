import openai
import json
from imagenet_prompts import sun_397
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""

all_json_dict = {}
all_responses = {}
root_folder = 'mpvr'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")

sun_397__ = list()
for c in sun_397:
    if '_' in c:
        c = c.replace('_', ' ')
    sun_397__.append(c)

sun_397 = sun_397__


vowel_list = ['A', 'E', 'I', 'O', 'U']


category_list_all = {
    'SUN397': sun_397}

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

        # Dataset Name: SUN397
        # Description: The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of 397 categories (e.g., abbey, airplane cabin, athletic field outdoor, atrium public, basilica, canyon, creek, etc).

        prompts = []

        prompts.append("Describe how the scene of " + category + " appears.")
        prompts.append("How do you distinguish and recognize the scene of " + category + "?")
        prompts.append("Provide a detailed description of what the scene " + category + " looks like.")
        prompts.append("Explore an internet image featuring the scene of " + category + " and describe it.")
        prompts.append("In what ways can you identify the scene of " + category + ":")

        prompts.append("Elaborate on the visual characteristics of the scene category " + category + ".")
        prompts.append("What features enable the recognition of the scene of " + category + "?")
        prompts.append("Discuss the appearance and elements defining the scene of " + category + ".")
        prompts.append("Provide insights into an internet image depicting the scene of " + category + ".")
        prompts.append("How can one discern and identify the scene of " + category + ":?")

        prompts.append("Break down the visual representation of the scene " + category + ".")
        prompts.append("What are the key elements facilitating the recognition of the scene of " + category + "?")
        prompts.append("Detail the visual aspects that compose the scene of " + category + ".")
        prompts.append(
            "Analyze an internet image portraying the scene of " + category + " and provide a description.")
        prompts.append("How can one spot and identify the scene of " + category + ":?")

        prompts.append("Examine the appearance and traits that define the scene " + category + ".")
        prompts.append("What visual cues contribute to identifying the scene of " + category + "?")
        prompts.append("Discuss the visual aspects that make the scene of " + category + " recognizable.")
        prompts.append(
            "Explore a specific internet image illustrating the scene of " + category + " and describe it.")
        prompts.append("How can one recognize and identify the scene of " + category + ":?")

        prompts.append(
            "Provide an analysis of the visual characteristics distinguishing the scene of " + category + ".")
        prompts.append(
            "Explain the recognizable features aiding in the identification of the scene of " + category + ".")
        prompts.append("Describe the key visual indicators signifying the scene category " + category + ".")
        prompts.append(
            "Elaborate on an online image portraying the scene of " + category + " and highlight its features.")
        prompts.append("How can one discern and recognize the scene of " + category + ":?")

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
