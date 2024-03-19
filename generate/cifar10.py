import openai
import json
from imagenet_prompts import cifar10_classes
from tqdm import tqdm
from pathlib import Path

openai.api_key = "" # only for eccv

all_json_dict = {}
# all_responses = {}
root_folder = 'tap'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")



vowel_list = ['A', 'E', 'I', 'O', 'U']
# category_list_all = {
#     'CUBS200': cubs}
# with open('tap/CUBS200.json') as f:
#     data = json.load(f)
#
# classes_already_done = list(data.keys())
#
# cubs = [x for x in cubs if x not in classes_already_done]

category_list_all = {
    'CIFAR10_local': cifar10_classes}
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
        prompts.append(f"Describe what a {category} looks like")
        prompts.append(f"How can you identify a {category}?")
        prompts.append(f"What does a {category} look like?")
        prompts.append(f"Describe an image from the dataset of a {category}")
        prompts.append(f"Explain the distinguishing features of a {category}")
        prompts.append(f"Discuss the characteristics of {category}")
        prompts.append(f"Provide details about the appearance of {category}")
        prompts.append(f"Imagine a scenario involving a {category}")
        prompts.append(f"Elaborate on the key attributes of a {category}")
        prompts.append(f"How might one differentiate between different instances of {category}?")
        prompts.append(f"Discuss any unique patterns or textures associated with {category}")
        prompts.append(f"Can you recognize a {category} in a complex background?")
        prompts.append(f"What are the common environments where you would find a {category}?")
        prompts.append(f"Highlight the main visual cues that define a {category}")
        prompts.append(f"Explore the diversity in appearances among different {category}s")
        prompts.append(f"What are some potential challenges in accurately classifying a {category}?")
        prompts.append(f"Provide examples of objects that could be confused with a {category}")
        prompts.append(f"Discuss any variations or subtypes within the category of {category}")
        prompts.append(f"Describe the color palette typically associated with {category}")
        prompts.append(f"How would you describe the behavior or movement of a {category}?")
        prompts.append(f"Imagine a creative context where a {category} plays a significant role")
        prompts.append(f"What are the possible uses or applications of a {category}?")
        prompts.append(f"Examine the context in which a {category} might be captured in an image")
        prompts.append(f"Discuss the historical or cultural significance of {category}")
        prompts.append(f"Can you suggest potential improvements in the dataset's representation of {category}s?")
        prompts.append(f"How might advancements in technology impact the perception of {category}s?")
        prompts.append(f"Imagine a futuristic scenario involving {category} and describe it")
        prompts.append(f"What role do {category}s play in the ecosystem they inhabit?")
        prompts.append(f"Explore the emotions or sentiments associated with encountering a {category}")
        prompts.append(f"Discuss any ethical considerations related to the study or use of {category}s")

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
