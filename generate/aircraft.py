import openai
import json
from imagenet_prompts import fgvc_aircraft
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""

all_json_dict = {}
all_responses = {}
root_folder = 'gpt'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")
vowel_list = ['A', 'E', 'I', 'O', 'U']
category_list_all = {
    'FGVCAircraft': fgvc_aircraft}

for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')
    json_name_all = f"{root_folder}/{k}.json"
    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"
        prompts = []

        # 1-5
        prompts.append("Describe how does the aircraft of type " + category + " looks like.")
        prompts.append("How can you recognize the aircraft of type " + category + "?")
        prompts.append("What does the aircraft of type " + category + " look like?")
        prompts.append("Describe an image from the internet of the aircraft of type " + category + ".")
        prompts.append("How can you identify the aircraft of type " + category + ":")

        # 6-10
        prompts.append(
            "Imagine you encounter an aircraft of type " + category + ". How would you describe its appearance?")
        prompts.append("What features distinguish the aircraft of type " + category + " from others?")
        prompts.append("Describe a visual representation of the aircraft " + category + ".")
        prompts.append(
            "If you were to guide someone in recognizing the aircraft of type " + category + ", what details would you emphasize?")
        prompts.append("Capture the essence of the aircraft of type " + category + " in words.")

        # 11-15
        prompts.append(
            "Envision exploring a scene with diverse aircraft. How would you describe the one of type " + category + "?")
        prompts.append("What specific qualities would you look for to identify the aircraft of type " + category + "?")
        prompts.append("Provide a narrative description of an image featuring the aircraft of type " + category + ".")
        prompts.append(
            "If you were to write a caption for a photograph highlighting the aircraft of type " + category + ", what would it say?")
        prompts.append("Highlight the defining characteristics of the aircraft of type " + category + ".")

        # 16-20
        prompts.append("Imagine encountering the aircraft of type " + category + ". How would you depict it?")
        prompts.append("What features make the aircraft of type " + category + " distinct?")
        prompts.append("Describe a visual representation of the aircraft category " + category + ".")
        prompts.append("If you had to describe the aircraft of type " + category + " to someone, what would you say?")
        prompts.append("Explore the visual nuances of the aircraft of type " + category + ".")

        # 21-25
        prompts.append(
            "Visualize an image that perfectly represents the aircraft of type " + category + ". Describe it.")
        prompts.append("What words would you use to characterize the aircraft of type " + category + "?")
        prompts.append("Provide an in-depth description of the appearance of the aircraft of type " + category + ".")
        prompts.append("Compose a caption for an image highlighting the aircraft of type " + category + ":")
        prompts.append("Define the aircraft " + category + " through its visual attributes.")

        # 26-30
        prompts.append(
            "Envision an image that perfectly represents the aircraft of type " + category + ". Describe it.")
        prompts.append("What words would you use to characterize the aircraft of type " + category + "?")
        prompts.append("Provide an in-depth description of the appearance of the aircraft of type " + category + ".")
        prompts.append("Compose a caption for an image highlighting the aircraft of type " + category + ":")
        prompts.append("Define the aircraft category " + category + " through its visual attributes.")

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
