import openai
import json
from imagenet_prompts import caltech_101
from tqdm import tqdm
from pathlib import Path

openai.api_key = "" # only for eccv

all_json_dict = {}
root_folder = 'gpt'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")

vowel_list = ['A', 'E', 'I', 'O', 'U']

category_list_all = {
    'Caltech101': caltech_101}
all_responses = {}

for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}.json"

    for i, category in enumerate(tqdm(v[:])):

        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        caltech_101_prompts = []

        # 1-5
        caltech_101_prompts.append(f"Describe {article} " + category + " in fine detail.")
        caltech_101_prompts.append(f"What are the distinguishing features of {article} " + category + "?")
        caltech_101_prompts.append(f"Provide details about the appearance of {article} " + category + ".")
        caltech_101_prompts.append(f"Discuss how {article} " + category + " category can be identified.")
        caltech_101_prompts.append(f"Explain the visual characteristics of {article} image of a " + category + ".")
        caltech_101_prompts.append(f"Share information about {article} " + category + " object.")
        caltech_101_prompts.append(f"How would you describe the images of {article} " + category + "?")
        caltech_101_prompts.append(f"What distinguishes {article} " + category + " from other objects in Caltech-101?")
        caltech_101_prompts.append(f"Elaborate on the visual properties of {article} " + category + " category.")
        caltech_101_prompts.append(f"Provide a detailed description of {article} " + category + ".")
        caltech_101_prompts.append(f"Discuss the characteristics that define {article} " + category + ".")
        caltech_101_prompts.append(f"Describe how {article} " + category + " object looks.")
        caltech_101_prompts.append(f"What features are notable in the images of {article} " + category + "?")
        caltech_101_prompts.append(f"Examine and describe {article} " + category + ".")
        caltech_101_prompts.append(f"How can you distinguish {article} " + category + " from other classes in a set of images?")
        caltech_101_prompts.append(f"Provide details about the visual aspects of {article} " + category + ".")
        caltech_101_prompts.append(f"Discuss the key attributes of {article} " + category + " in Caltech-101.")
        caltech_101_prompts.append(f"What visual cues are important for recognizing {article} " + category + "?")
        caltech_101_prompts.append(f"Describe an image featuring {article} " + category + ".")
        caltech_101_prompts.append(f"Explain how {article} " + category + " is visually represented.")
        caltech_101_prompts.append(f"What specific details can be used to identify {article} " + category + "?")
        caltech_101_prompts.append(f"Discuss the unique characteristics of {article} " + category + " in the dataset.")
        caltech_101_prompts.append(f"Describe the appearance and features of {article} " + category + ".")
        caltech_101_prompts.append(f"How would you characterize {article} " + category + "?")
        caltech_101_prompts.append(f"Examine and detail the visual aspects of {article} " + category + ".")
        caltech_101_prompts.append(f"What makes {article} " + category + " discriminative for object classification?")
        caltech_101_prompts.append(f"Describe how the images of {article} " + category + " differ from others.")
        caltech_101_prompts.append(f"How can {article} " + category + " be identified based on visual traits?")
        caltech_101_prompts.append(f"Provide a comprehensive description of {article} " + category + ".")
        caltech_101_prompts.append(f"Discuss the visual elements that define {article} " + category + " in Caltech-101.")

        res_ = {}

        for curr_prompt in caltech_101_prompts:
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
