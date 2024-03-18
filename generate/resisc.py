import openai
import json
from imagenet_prompts import resisc
from tqdm import tqdm
from pathlib import Path

openai.api_key = "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" # only for eccv

all_json_dict = {}
all_responses = {}

category_list_all = {
    'RESISC45': resisc
}

vowel_list = ['A', 'E', 'I', 'O', 'U']

root_folder = 'mpvr'

if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")

for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"{root_folder}/{k}_arial.json"

    if Path(json_name_all).is_file():
        raise ValueError("File already exists")

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        prompts = []

        prompts.append("Describe how does the " + category + " looks like from a satellite.")
        prompts.append("How can you recognize the " + category + " from a satellite?")
        prompts.append("What does the satellite photo of " + category + " look like?")
        prompts.append("What does the aerial photo of " + category + " look like?")
        prompts.append("Describe the satellite photo from the internet of " + category + ".")
        prompts.append("How can you identify the " + category + " from a satellite?")
        prompts.append("Explain the geographical features visible in the satellite image of " + category + ".")
        prompts.append("Highlight the distinguishing characteristics of " + category + " in satellite imagery.")
        prompts.append("What land cover details are discernible in the satellite snapshot of " + category + "?")
        prompts.append("Discuss the unique patterns captured in the satellite photo of " + category + ".")
        prompts.append("Elaborate on the topographical variations showcased in the satellite view of " + category + ".")
        prompts.append("Examine the aerial representation of " + category + " and identify notable landmarks.")
        prompts.append("What insights does the satellite data provide about the ecological dynamics of " + category + "?")
        prompts.append("Explore any seasonal changes depicted in the aerial imagery of " + category + ".")
        prompts.append("Describe the color palette present in the aerial picture of " + category + ".")
        prompts.append("In what ways can the satellite view aid in monitoring changes in " + category + "?")
        prompts.append("Discuss the spatial arrangements visible in the aerial photograph of " + category + ".")
        prompts.append("Examine the satellite image of " + category + " and comment on any infrastructural features.")
        prompts.append("How does the satellite perspective reveal the land use patterns of " + category + "?")
        prompts.append("Point out any water bodies or natural landmarks visible in the satellite view of " + category + ".")
        prompts.append("Describe the overall composition and layout captured in the aerial photo of " + category + ".")
        prompts.append("Discuss the significance of " + category + " in the context of environmental monitoring using satellite data.")
        prompts.append("What human-made structures stand out in the aerial imagery of " + category + "?")
        prompts.append("Analyze the satellite photo of " + category + " to identify changes over time.")
        prompts.append("Explain how the satellite data contributes to understanding the land cover types of " + category + ".")
        prompts.append("Detail the unique characteristics that distinguish " + category + " in satellite views.")
        prompts.append("Describe the aerial imagery features that help recognize " + category + " accurately.")
        prompts.append("Discuss any challenges in identifying " + category + " from aerial imagery data.")
        prompts.append("Examine the satellite image of " + category + " and comment on the diversity of vegetation.")
        prompts.append("How does the satellite perspective aid in mapping the extent of " + category + "?")
        prompts.append("Highlight any notable changes or developments observed in the satellite imagery of " + category + ".")
        prompts.append("What role does the aerial data play in assessing the land cover dynamics of " + category + "?")
        prompts.append("Discuss the spatial distribution patterns evident in the satellite photograph of " + category + ".")
        prompts.append("Explain the implications of " + category + " in land-use planning based on satellite observations.")
        prompts.append("Provide insights into the geographical context revealed by the satellite photo of " + category + ".")

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
