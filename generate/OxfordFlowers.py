import openai
import json
from imagenet_prompts import flowers
from tqdm import tqdm
from pathlib import Path

openai.api_key = "" # only for eccv

all_json_dict = {}
all_responses = {}
root_folder = 'mpvr'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")
category_list_all = {
    'OxfordFlowers': flowers}


vowel_list = ['A', 'E', 'I', 'O', 'U']



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

        # Dataset Name: Oxford Flowers Dataset
        # Description: Oxford Flowers consists of 102 flower categories. The flowers chosen to be flowers commonly occur in the United Kingdom.

        # Prompts
        prompts = []
        prompts.append("Describe the visual characteristics of the " + category + " flower.")
        prompts.append("How would you identify the " + category + " flower based on its appearance?")
        prompts.append("What are the distinctive features of the " + category + " flower?")
        prompts.append("Provide a detailed description of the appearance of the " + category + " flower.")
        prompts.append("Explain how one can visually recognize the " + category + " flower.")
        prompts.append("What unique traits distinguish the " + category + " flower from others?")
        prompts.append("Describe an image depicting the " + category + " flower found online.")
        prompts.append("How can you distinguish the " + category + " flower from other flowers?")
        prompts.append("Provide visual details that characterize the " + category + " flower.")
        prompts.append("What visual cues help in identifying the " + category + " flower?")
        prompts.append("Describe the appearance of a typical " + category + " flower.")
        prompts.append("How does the appearance of the " + category + " flower vary from other flowers?")
        prompts.append("What features stand out when observing the " + category + " flower?")
        prompts.append("Discuss the visual attributes that define the " + category + " flower.")
        prompts.append("How can one visually differentiate the " + category + " flower from the rest?")
        prompts.append("Provide a detailed visual description of the " + category + " flower.")
        prompts.append("Enumerate the visual characteristics that distinguish the " + category + " flower.")
        prompts.append("What unique visual patterns are associated with the " + category + " flower?")
        prompts.append("Describe the appearance of " + category + " flowers in general.")
        prompts.append("How can one visually classify the " + category + " flower category?")
        prompts.append("Discuss the key visual elements that define the " + category + " flower.")
        prompts.append("What specific visual traits help identify the " + category + " flower?")
        prompts.append("Describe the visual features that make the " + category + " flower distinctive.")
        prompts.append("How does the visual appearance of the " + category + " flower contribute to its recognition?")
        prompts.append("Elaborate on the visual cues that aid in recognizing the " + category + " flower.")
        prompts.append("What are the prominent visual characteristics of the " + category + " flower?")
        prompts.append("How can you visually distinguish between the " + category + " flower and other flowers?")
        prompts.append("Provide a detailed visual account of the " + category + " flower.")
        prompts.append("What visual aspects play a crucial role in identifying the " + category + " flower?")

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
