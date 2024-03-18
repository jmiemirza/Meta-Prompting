import openai
import json
from imagenet_prompts import places365
from tqdm import tqdm
from pathlib import Path

openai.api_key = "" # only for eccv

places365 = [x.replace('_', ' ') for x in places365]


all_json_dict = {}
all_responses = {}
root_folder = 'mpvr'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")


vowel_list = ['A', 'E', 'I', 'O', 'U']


category_list_all = {
    'places365': places365}

for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}.json"

    for i, category in enumerate(tqdm(v[:])):

        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        places365_prompts = []

        if '_' in category:
            category = category.replace('_', ' ')

        # 1-5
        places365_prompts.append("Describe the scene of " + category + ".")
        places365_prompts.append("What are the distinctive features of the " + category + " scene?")
        places365_prompts.append("Provide details about the appearance of the " + category + " environment.")
        places365_prompts.append("Discuss how the " + category + " scene can be recognized.")
        places365_prompts.append("Explain the visual characteristics of the " + category + " scene.")
        places365_prompts.append("Share information about the " + category + " location in Places365.")
        places365_prompts.append("How would you describe images featuring the " + category + " scene?")
        places365_prompts.append("What distinguishes the " + category + " scene from other scenes in Places365?")
        places365_prompts.append("Elaborate on the visual properties of the " + category + " environment.")
        places365_prompts.append("Provide a detailed description of the " + category + " scene in the dataset.")
        places365_prompts.append("Discuss the characteristics that define the " + category + " scene.")
        places365_prompts.append("Describe how the " + category + " scene looks in Places365.")
        places365_prompts.append("What features are notable in images of the " + category + " scene?")
        places365_prompts.append("Examine and describe the " + category + " scene in the dataset.")
        places365_prompts.append("How can you distinguish the " + category + " scene from other scenes?")
        places365_prompts.append("Provide details about the visual aspects of the " + category + " scene.")
        places365_prompts.append("Discuss the key attributes of the " + category + " scene in Places365.")
        places365_prompts.append("What visual cues are important for recognizing the " + category + " scene?")
        places365_prompts.append("Describe an image featuring the " + category + " scene in Places365.")
        places365_prompts.append("Explain how the " + category + " scene is visually represented.")
        places365_prompts.append("What specific details can be used to identify the " + category + " scene?")
        places365_prompts.append("Discuss the unique characteristics of the " + category + " scene in the dataset.")
        places365_prompts.append("Describe the appearance and features of the " + category + " scene.")
        places365_prompts.append("How would you characterize the " + category + " scene in Places365?")
        places365_prompts.append("Examine and detail the visual aspects of the " + category + " scene.")
        places365_prompts.append("What makes the " + category + " scene distinctive in Places365?")
        places365_prompts.append("Describe how images of the " + category + " scene differ from others.")
        places365_prompts.append("How can the " + category + " scene be identified based on visual traits?")
        places365_prompts.append("Provide a comprehensive description of the " + category + " scene.")
        places365_prompts.append("Discuss the visual elements that define the " + category + " scene in Places365.")

        res_ = {}

        for curr_prompt in places365_prompts:
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
