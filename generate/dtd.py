import openai
import json
from imagenet_prompts import dtd
from tqdm import tqdm
from pathlib import Path
from openai.error import APIError

openai.api_key = ""

all_json_dict = {}
all_responses = {}
root_folder = 'gpt'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")
category_list_all = {
    'DescribableTextures': dtd}

vowel_list = ['A', 'E', 'I', 'O', 'U']

for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}.json"

    for i, category in enumerate(tqdm(v[:])):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"
        prompts = []

        # 1-5
        prompts.append("Describe the visual characteristics of the texture labeled as " + category + ".")
        prompts.append("How would you recognize the texture labeled as " + category + "?")
        prompts.append("What are the key features of the texture labeled as " + category + "?")
        prompts.append("Provide a detailed description of the appearance of the texture labeled as " + category + ".")
        prompts.append("If you see an image with the texture labeled as " + category + ", what would stand out to you?")
        # 6-10
        prompts.append(
            "Imagine you encounter a surface with the texture labeled as " + category + ". How would you describe it?")
        prompts.append("What visual attributes define the texture category " + category + "?")
        prompts.append("Describe an image featuring the texture labeled as " + category + ".")
        prompts.append("Create a caption for an image showcasing the texture labeled as " + category + ":")
        prompts.append(
            "Detail the unique aspects that distinguish the texture labeled as " + category + " from others.")
        # 11-15
        prompts.append(
            "Envision a scenario where you encounter the texture labeled as " + category + ". How would you articulate its appearance?")
        prompts.append(
            "What specific qualities would you look for to identify the texture labeled as " + category + "?")
        prompts.append("Provide a narrative description of an image featuring the texture labeled as " + category + ".")
        prompts.append(
            "If you were to write a caption for a photograph highlighting the texture labeled as " + category + ", what would it say?")
        prompts.append("Highlight the defining characteristics of the texture labeled as " + category + ".")
        # 16-20
        prompts.append(
            "Imagine exploring a scene with diverse textures. How would you describe the one labeled as " + category + "?")
        prompts.append("What features make the texture labeled as " + category + " distinct?")
        prompts.append("Describe a visual representation of the texture category " + category + ".")
        prompts.append(
            "If you were to guide someone in recognizing the texture labeled as " + category + ", what details would you emphasize?")
        prompts.append("Capture the essence of the texture labeled as " + category + " in words.")
        # 21-25
        prompts.append("Visualize encountering the texture labeled as " + category + ". How would you depict it?")
        prompts.append("Outline the distinguishing visual elements of the texture labeled as " + category + ".")
        prompts.append("Craft a narrative about an image showcasing the texture labeled as " + category + ".")
        prompts.append("If you had to describe the texture labeled as " + category + " to someone, what would you say?")
        prompts.append("Explore the visual nuances of the texture labeled as " + category + ".")
        # 26-30
        prompts.append(
            "Envision an image that perfectly represents the texture labeled as " + category + ". Describe it.")
        prompts.append("What words would you use to characterize the texture labeled as " + category + "?")
        prompts.append("Provide an in-depth description of the appearance of the texture labeled as " + category + ".")
        prompts.append("Compose a caption for an image highlighting the texture labeled as " + category + ":")
        prompts.append("Define the texture category " + category + " through its visual attributes.")

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
