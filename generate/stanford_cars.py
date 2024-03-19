import openai
import json
from imagenet_prompts import stanford_cars
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""

all_json_dict = {}
all_responses = {}
root_folder = 'mpvr'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")


vowel_list = ['A', 'E', 'I', 'O', 'U']

in_process = False

category_list_all = {
    'stanford_cars': stanford_cars}

for k, v in category_list_all.items():

    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}.json"

    if Path(json_name_all).is_file() and not in_process:
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
        prompts.append("Describe the distinctive features of the " + category + " car model.")
        prompts.append("How would you identify a " + category + " car among other models?")
        prompts.append("Paint a visual picture of the " + category + " car's appearance.")
        prompts.append("What makes the " + category + " car unique in terms of its visual characteristics?")
        prompts.append("Provide a detailed description of the exterior of a " + category + " car.")
        prompts.append("Envision the design elements that define the " + category + " car.")
        prompts.append("How does the visual appearance of a " + category + " car differ from other models?")
        prompts.append("Capture the essence of the " + category + " car's aesthetic attributes.")
        prompts.append("Detail the visual cues that distinguish a " + category + " car.")
        prompts.append("What are the key visual components that set the " + category + " car apart?")
        prompts.append("Illustrate the exterior details that help recognize a " + category + " car.")
        prompts.append("Describe the overall visual impression of the " + category + " car.")
        prompts.append("What specific visual aspects define the " + category + " car model?")
        prompts.append("How can you visually differentiate the " + category + " car from others?")
        prompts.append("Paint a mental image of a " + category + " car's exterior.")
        prompts.append("Enumerate the visual attributes that characterize the " + category + " car.")
        prompts.append("What visual features would you look for to identify a " + category + " car?")
        prompts.append("Describe the exterior aesthetics that define the " + category + " car.")
        prompts.append("How do the visual elements of the " + category + " car contribute to its identification?")
        prompts.append("Enlist the visual traits that make the " + category + " car recognizable.")
        prompts.append("What unique visual characteristics distinguish the " + category + " car?")
        prompts.append("Paint a detailed picture of the " + category + " car's external appearance.")
        prompts.append("How would you describe the exterior features of the " + category + " car?")
        prompts.append("What visual cues are essential for recognizing a " + category + " car?")
        prompts.append("Describe the visual attributes that set the " + category + " car class apart.")
        prompts.append("How can you visually classify the " + category + " car based on its appearance?")
        prompts.append("Provide a vivid depiction of the " + category + " car's visual elements.")
        prompts.append("What are the defining visual elements of the " + category + " car model?")
        prompts.append("Detail the visual characteristics that make the " + category + " car easily identifiable.")
        prompts.append("Envision the specific visual features that distinguish the " + category + " car model.")

        all_result = []
        for curr_prompt in prompts:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=curr_prompt,
                temperature=.99,
                max_tokens=50,
                n=10,
                stop="."
            )

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n\n", "") + ".")

        all_responses[category] = all_result

        # if i % 10 == 0:
        with open(json_name_all, 'w') as f:
            json.dump(all_responses, f, indent=4)
