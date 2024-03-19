import openai
import json
from imagenet_prompts import oxford_pets
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""
all_json_dict = {}
root_folder = 'mpvr'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")

vowel_list = ['A', 'E', 'I', 'O', 'U']

category_list_all = {
        'OxfordPets': oxford_pets}

all_responses = {}

for k, v in category_list_all.items():

    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}_cls.json"

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"
        prompts = []

        prompts.append(f"Describe the features of the {category} pet.")
        prompts.append(f"What makes the {category} breed distinct in appearance?")
        prompts.append(f"Provide characteristics that define the {category} animal.")
        prompts.append(f"How would you identify a {category} from other pets?")
        prompts.append(f"Explain the key visual traits of a {category}.")

        prompts.append(f"Share details about the physical appearance of a {category} pet.")
        prompts.append(f"Discuss the distinctive markings or colors of the {category} breed.")
        prompts.append(f"What are the defining characteristics of the {category} animal?")
        prompts.append(f"How does the appearance of a {category} differ from other pets?")
        prompts.append(f"Provide a detailed description of a {category} pet.")

        prompts.append(f"Imagine you encounter a {category} pet. How would you describe its appearance?")
        prompts.append(f"What features stand out in the {category} breed?")
        prompts.append(f"Discuss the visual aspects that distinguish the {category} animal.")
        prompts.append(f"How can you visually recognize a {category} from different pets?")
        prompts.append(f"Describe the physical traits that make up the {category} pet.")

        prompts.append(f"Paint a picture with words of the {category} pet's appearance.")
        prompts.append(f"What visual elements are characteristic of the {category} breed?")
        prompts.append(f"How would you identify the {category} animal based on its appearance?")
        prompts.append(f"Detail the physical characteristics that define the {category} pet.")
        prompts.append(f"Provide a visual description of the {category} pet's features.")

        prompts.append("Discuss the unique features that make " + category + " pets easily recognizable.")
        prompts.append("Describe the distinguishing traits of the " + category + " pet images.")
        prompts.append("How would you characterize the visual aspects of the pet" + category + "?")
        prompts.append("What are the defining attributes of the " + category + " pet class in the Oxford Pets Dataset?")
        prompts.append("Examine and detail the appearance of the " + category + " pets in the dataset.")

        prompts.append("Provide insights into the visual representation of " + category + " pets.")
        prompts.append("Describe the key elements that help identify the " + category + " pet category.")
        prompts.append("What specific features make the " + category + " pets easily distinguishable?")
        prompts.append("Discuss the visual characteristics that define the " + category + " pet class.")
        prompts.append("Explain the distinctive traits that set apart " + category + " pets in the dataset.")

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
