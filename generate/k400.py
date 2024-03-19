import openai
import json
from imagenet_prompts import k400
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
    'k400': k400}

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

        # Dataset Name: UCF101
        # Description: UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. The action categories can be divided into five types: 1)Human-Object Interaction 2) Body-Motion Only 3) Human-Human Interaction 4) Playing Musical Instruments 5) Sports.

        prompts = []

        # Generate additional diverse prompts for the Kinetics 400 dataset
        prompts.append("Describe how the " + category + " action looks in an image.")
        prompts.append("How can you identify the " + category + " action?")
        prompts.append("What does the " + category + " action look like in an image?")
        prompts.append("Describe an image from the internet illustrating the " + category + " action.")
        prompts.append("How can you recognize the " + category + " action?")
        prompts.append("What are the defining features of the " + category + " action?")
        prompts.append("Provide a detailed visual description of the " + category + " action.")
        prompts.append("Discuss the unique characteristics of the " + category + " action.")
        prompts.append("Explain how the " + category + " action can be visually identified.")
        prompts.append("Illustrate the visual appearance of the " + category + " action in an image.")
        prompts.append("How would you recognize the " + category + " action based on visual cues?")
        prompts.append(
            "Enumerate the key elements that define the visual representation of the " + category + " action.")
        prompts.append("Describe an image portraying the characteristic features of the " + category + " action.")
        prompts.append("What visual aspects contribute to the recognition of the " + category + " action?")
        prompts.append("Examine an image from the internet showcasing the " + category + " action.")
        prompts.append("How can one visually distinguish the " + category + " action from other actions?")
        prompts.append("Discuss the specific details that help in identifying the " + category + " action visually.")
        prompts.append("Describe an instance of the " + category + " action captured in an image.")
        prompts.append("What are the visual cues that aid in recognizing the " + category + " action?")
        prompts.append("Explain the unique visual characteristics associated with the " + category + " action.")
        prompts.append("Provide a visual depiction of the action of " + category + " in an image.")
        prompts.append("How can the action of " + category + " be visually distinguished?")
        prompts.append("Describe the visual attributes that characterize the " + category + " action in an image.")
        prompts.append("What visual cues can be used to recognize the " + category + " action?")
        prompts.append("Detail the appearance of an image depicting the " + category + " action.")
        prompts.append("Discuss the elements that make up the " + category + " action.")
        prompts.append("What distinguishes the " + category + " action from other actions?")
        prompts.append("Describe the typical scenario in which the " + category + " action occurs.")
        prompts.append("How would you describe the body movements in the " + category + " action?")
        prompts.append(
            "Provide an account of the observable features in the " + category + " action.")
        prompts.append("What are the defining characteristics of the " + category + " action?")
        prompts.append("Describe any accessories or equipment typically associated with the " + category + " action.")
        prompts.append(
            "How does the environment influence the execution of the " + category + " action?")

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
