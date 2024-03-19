import openai
import json
from imagenet_prompts import cubs
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""

all_json_dict = {}
root_folder = 'gpt'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")
vowel_list = ['A', 'E', 'I', 'O', 'U']
category_list_all = {
    'CUBS200': cubs}

all_responses = {}

for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}.json"

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        prompts= []
        prompts.append("Describe the distinctive features of the bird: " + category + ".")
        prompts.append("What sets apart the " + category + " bird from other birds visually?")
        prompts.append("How would you distinguish the appearance of a " + category + " bird?")
        prompts.append("Discuss the key visual characteristics of the " + category + " bird.")
        prompts.append("Provide details on the visual attributes that define the " + category + " bird.")
        prompts.append("Describe the plumage and markings of the " + category + " bird.")
        prompts.append("Explain the unique visual traits that help identify a " + category + " bird.")
        prompts.append("What are the visual cues that indicate a " + category + " bird in an image?")
        prompts.append("Illustrate the specific features that make the " + category + " bird visually distinctive.")
        prompts.append(
            "Discuss how the visual characteristics of the " + category + " bird contribute to fine-grained categorization.")
        prompts.append("Describe the color patterns and physical attributes of the " + category + " bird.")
        prompts.append("How can you visually recognize a bird belonging to the " + category + " category?")
        prompts.append("Elaborate on the unique visual elements of the " + category + " bird.")
        prompts.append("What visual cues help in differentiating the " + category + " bird from other bird categories?")
        prompts.append("Describe the appearance of a bird categorized as " + category + ".")
        prompts.append("How do the visual features of the " + category + " bird aid in categorization?")
        prompts.append(
            "Discuss the visual characteristics that play a crucial role in identifying the " + category + " bird.")
        prompts.append("Provide insights into the visual traits that distinguish the " + category + " bird category.")
        prompts.append("What visual attributes are significant for recognizing a " + category + " bird?")
        prompts.append("Explain the role of visual features in categorizing the " + category + " bird.")
        prompts.append(
            "Describe the physical characteristics that make the " + category + " bird category recognizable.")
        prompts.append("How do the visual elements of the " + category + " bird contribute to its categorization?")
        prompts.append(
            "Discuss the role of visual attributes in fine-grained categorization of the " + category + " bird.")
        prompts.append("What visual cues can be used to identify a bird from the " + category + " category?")
        prompts.append("Describe the visual traits that distinguish the " + category + " bird from others.")
        prompts.append("How would you visually differentiate the " + category + " bird from birds in other categories?")
        prompts.append("Explain the visual characteristics that aid in recognizing the " + category + " bird.")
        prompts.append(
            "Discuss the visual features that are key to identifying a bird categorized as " + category + ".")
        prompts.append("What visual elements play a crucial role in the categorization of the " + category + " bird?")

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
