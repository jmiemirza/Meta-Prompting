import openai
import json
from imagenet_prompts import cifar100_classes
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""

all_json_dict = {}
root_folder = 'gpt'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")

category_list_all = {
        'CIFAR100_local': cifar100_classes}

vowel_list = ['A', 'E', 'I', 'O', 'U']



for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}_30_prompts.json"

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        prompts = []
        prompts.append(f"Describe what a {category} looks like")
        prompts.append(f"How can you identify a {category}?")
        prompts.append(f"What does a {category} look like?")
        prompts.append(f"Describe an image from the internet of a {category}")
        prompts.append(f"Explain the characteristics of a {category}")
        prompts.append(f"What are the distinguishing features of a {category}?")
        prompts.append(f"Provide details about the appearance of a {category}")
        prompts.append(f"Discuss the typical habitat of a {category}")
        prompts.append(f"Imagine a scenario involving a {category}")
        prompts.append(f"Describe the behavior of a {category}")
        prompts.append(f"Highlight any unique markings on a {category}")
        prompts.append(f"Discuss the color variations in a {category}")
        prompts.append(f"How does a {category} interact with its environment?")
        prompts.append(f"Elaborate on the diet of a {category}")
        prompts.append(f"Describe the lifecycle of a {category}")
        prompts.append(f"Discuss the cultural significance of a {category}")
        prompts.append(f"Explain any interesting facts about a {category}")
        prompts.append(f"What are some common misconceptions about a {category}?")
        prompts.append(f"Create a story involving a {category}")
        prompts.append(f"Compare and contrast a {category} with another category from the dataset")
        prompts.append(f"Discuss the historical importance of a {category}")
        prompts.append(f"Imagine a world without {category} and its impact")
        prompts.append(f"Explain the role of a {category} in its ecosystem")
        prompts.append(f"What emotions does a {category} evoke?")
        prompts.append(f"Discuss any conservation efforts related to {category}")
        prompts.append(f"Describe the sensory experiences associated with a {category}")
        prompts.append(f"Imagine a day in the life of a {category}")
        prompts.append(f"Discuss the role of a {category} in human culture")
        prompts.append(f"Provide interesting anecdotes related to {category}")

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
