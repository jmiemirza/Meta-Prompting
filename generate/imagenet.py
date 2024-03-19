import openai
import json
from imagenet_prompts import imagenet_classes as imagenet_classes
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""

all_json_dict = {}
root_folder = 'mpvr'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")

vowel_list = ['A', 'E', 'I', 'O', 'U']


category_list_all = {
    'ImageNet': imagenet_classes}
all_responses = {}

for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}.json"

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"
        prompts = []
        prompts.append("Describe the characteristics of the " + category + ".")
        prompts.append("What are the distinguishing features of the " + category + "?")
        prompts.append("Illustrate the appearance of the " + category + " in detail.")
        prompts.append("Explain how one can recognize the " + category + " within an image.")
        prompts.append("Provide a detailed description of the " + category + ".")
        prompts.append("How would you identify the " + category + " based on its features?")
        prompts.append("Discuss the visual attributes that define the " + category + ".")
        prompts.append("Describe the key elements that make up the " + category + ".")
        prompts.append("What visual cues help in identifying the " + category + "?")
        prompts.append("Provide an in-depth description of the appearance of the " + category + ".")
        prompts.append("How can you distinguish between different instances of the " + category + "?")
        prompts.append("Discuss the unique characteristics associated with the " + category + ".")
        prompts.append("What are the typical visual patterns found in the " + category + "?")
        prompts.append("Describe the defining features of the " + category + ".")
        prompts.append("Explain the visual properties that set the " + category + " apart.")
        prompts.append("How does the " + category + " differ from other categories in appearance?")
        prompts.append("Describe a background and context in which " + category + " can appear?")
        prompts.append("Provide a detailed account of the visual elements in the " + category + ".")
        prompts.append("Discuss the perceptual attributes associated with the " + category + ".")
        prompts.append("What specific details define the visual representation of the " + category + "?")
        prompts.append("Describe the unique patterns and textures present in the " + category + ".")
        prompts.append("How would you recognize the " + category + " based on its visual attributes?")
        prompts.append("Provide insights into the distinctive visual traits of the " + category + ".")
        prompts.append("What are the key visual components that characterize the " + category + "?")
        prompts.append("Discuss the visual characteristics that make the " + category + " easily identifiable.")
        prompts.append("Explain the features that contribute to the visual identity of the " + category + ".")
        prompts.append("Describe the variations in visual appearance among instances of the " + category + ".")
        prompts.append("How can one identify the " + category + " based on visual cues?")
        prompts.append("Provide a detailed description of the visual attributes that define the " + category + ".")
        prompts.append("Discuss the visual elements that play a crucial role in identifying the " + category + ".")
        prompts.append("What unique visual patterns and structures are associated with the " + category + "?")

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
