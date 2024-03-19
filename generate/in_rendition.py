import openai
import json
from imagenet_prompts import imagenet_rendition
from tqdm import tqdm
from pathlib import Path

openai.api_key = ""

all_json_dict = {}
all_responses = {}
root_folder = 'mpvr'
if not Path(root_folder).is_dir():
    raise ValueError("Folder does not exist")
category_list_all = {
    'ImageNetR': imagenet_rendition}


vowel_list = ['A', 'E', 'I', 'O', 'U']



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

        # Dataset Name: ImageNet-R(endition)
        # Description: ImageNet-R(endition) contains art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos, toys, and video game renditions of ImageNet classes.

        prompts = []

        prompts.append("Describe the artistic representation of the " + category + ".")
        prompts.append("How would you visually recognize the " + category + " class in art or cartoons?")
        prompts.append("Provide a detailed description of the graffiti or street art related to " + category + ".")
        prompts.append("What are the distinctive visual features of the embroidery depicting the " + category + "?")
        prompts.append("Describe the graphics that represent the visual essence of the " + category + " class.")
        prompts.append("Illustrate the origami models inspired by the " + category + ".")
        prompts.append("Explain the characteristics of paintings that portray the " + category + ".")
        prompts.append("Detail the visual patterns associated with the " + category + " class.")
        prompts.append("Describe the visual appearance of plastic objects related to the " + category + ".")
        prompts.append("What makes plush objects visually unique in representing the " + category + " class?")
        prompts.append("Describe sculptures visually inspired by the " + category + " category.")
        prompts.append("Provide a detailed visual description of sketches related to the " + category + ".")
        prompts.append("Elaborate on the visual aspects of tattoos that depict the " + category + ".")
        prompts.append("How do toys visually represent the characteristics of the " + category + "?")
        prompts.append("Describe video game renditions of the " + category + " class visually.")
        prompts.append("In what ways do art and cartoons visually capture the essence of the " + category + " category?")
        prompts.append("Provide visual insights into the graffiti and embroidery related to " + category + ".")
        prompts.append("What are the unique visual aspects of the graphics depicting the " + category + "?")
        prompts.append("Describe the origami models visually inspired by the " + category + ".")
        prompts.append("Explain the visual features of paintings and patterns associated with the " + category + ".")
        prompts.append("Detail the visual representation of plastic objects and plush objects for the " + category + ".")
        prompts.append("Describe the visual aspects of sculptures and video game renditions representing the " + category + ".")
        prompts.append("In what ways do sketches and tattoos visually capture the essence of the " + category + " class?")
        prompts.append("Provide visual insights into the origami models, paintings, and patterns related to " + category + ".")
        prompts.append("What are the distinctive visual features of plastic objects and plush objects depicting the " + category + "?")
        prompts.append("Describe the visual characteristics of sculptures and video game renditions representing the " + category + ".")
        prompts.append("Explain the visual characteristics of art, cartoons, and deviantart related to the " + category + ".")
        prompts.append("Detail the visual aspects of graffiti, embroidery, and graphics associated with the " + category + ".")

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
