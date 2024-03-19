import openai
import json
from imagenet_prompts import food101
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
    'Food101': food101}

for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}.json"

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        prompts = []

        # Generate prompts for Food101 dataset
        prompts.append("Describe the appearance of the food: " + category + ".")
        prompts.append("What distinguishes the " + category + " dish from others in the Food101 dataset?")
        prompts.append("Provide details about the visual characteristics of the " + category + " food.")
        prompts.append("How would you recognize a dish belonging to the " + category + " category?")
        prompts.append("Describe the key features that define the " + category + " dish.")
        prompts.append("What visual cues help in identifying the " + category + " food category?")
        prompts.append("Provide a detailed description of the visual attributes of the " + category + " dish.")
        prompts.append("How does the " + category + " food category stand out if we want to classify it?")
        prompts.append("Describe the specific visual elements that characterize the " + category + " dish.")
        prompts.append("What unique traits can be observed in images belonging to the " + category + " category?")
        prompts.append(
            "How can you differentiate between images of the " + category + " dish and other food categories?")
        prompts.append("Describe the typical visual composition of the " + category + " food.")
        prompts.append("What features are commonly associated with the " + category + " dish?")
        prompts.append("Provide a detailed account of the visual patterns found in the " + category + " food images.")
        prompts.append("How would you describe the overall visual aesthetics of the " + category + " dish?")
        prompts.append("What makes the " + category + " food category visually distinct in the Food101 dataset?")
        prompts.append("Describe the key elements that define the " + category + " dish visually.")
        prompts.append("How can one visually identify a dish belonging to the " + category + " category in Food101?")
        prompts.append("Provide insights into the visual characteristics that define the " + category + " food.")
        prompts.append("What visual patterns are commonly observed in the " + category + " dish images?")
        prompts.append("Describe the distinguishing visual features of the " + category + " food category.")
        prompts.append("How does the " + category + " dish visually differ from other categories in Food101?")
        prompts.append(
            "What visual cues would help in categorizing an image as belonging to the " + category + " food category?")
        prompts.append("Describe the unique visual attributes that characterize the " + category + " dish.")
        prompts.append("How can one visually recognize the " + category + " food category in the Food101 dataset?")
        prompts.append("Provide a detailed description of the visual traits specific to the " + category + " dish.")
        prompts.append("What visual elements contribute to the recognition of the " + category + " food category?")
        prompts.append("Describe the visual aspects that make the " + category + " dish stand out in Food101.")
        prompts.append(
            "How can the visual features of the " + category + " food be distinguished from other categories in the dataset?")
        prompts.append(
            "Provide insights into the visual characteristics that define the " + category + " dish in Food101.")

        res_ = {}

        for curr_prompt in prompts:
            all_result = []

            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=curr_prompt,
                temperature=.99,
                max_tokens=40,
                n=10,
            )

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n\n", "") + ".")

            res_[curr_prompt] = all_result

        all_responses[category] = res_
        with open(json_name_all, 'w') as f:
            json.dump(all_responses, f, indent=4)
