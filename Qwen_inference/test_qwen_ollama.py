import ollama
import base64
import os
from tqdm import tqdm
import argparse
import json


parser = argparse.ArgumentParser()

parser.add_argument(
    "--image_path",
    type=str, default = "/Users/qianyuhang/Projects/Datasets/LAKERED_DATASET/train/images_annotated",
    help = "path to the image folder"
)

parser.add_argument(
    "--output_json",
    type=str, default="processed_gemma3_mask_camed.json",
    help = "output json file that stores the vlm reponses."
)

args = parser.parse_args()



def image_to_base64(image_path):
    """Convert the image files to Base64 encoded strings"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_vlm_conversation(image_path, model_name='gemma3:4b', system_prompt=""):
    """
    run a 4 round VLM-VQA via Ollama (works on mac)

    :param image_path: the path of local images
    :param model_name: the model name you want to use in Ollama
    """
    if not os.path.exists(image_path):
        print(f"Error: Unable to find the image file '{image_path}'")
        return

    try:
        # 1. encode all images to Base64
        b64_image = image_to_base64(image_path)

        # 2. define your 4-round VQA
        questions = [
            "Describe the camouflaged object/objects outlined by the contour and explain how it is/are camouflaged in the surroundings.",
            "Describe the environment outside the contour, and explain how it can camouflage the object/objects successfully.",
            "According to the conversation history, describe the image in detail.",
            "According to the conversation history, summarize this image in one sentence. Your answer should be formatted like this: *article* *object description* in the *environment description*. (don't generate *)"
        ]

        # 3. Initialize the conversation history
        # for continuous VQA requirements, we need to send the context(image included) to the model
        # we can establish a message list to realize this in Ollama package
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        #print(f"--- Begin using '{model_name}' to dialogue. ---\n")

        # 4. process each question via a loop
        for i, question in enumerate(questions):
            # print(f"--- Round {i+1} ---")
            # print(f"Question: {question}\n")

            # Establish the info of the current round
            current_message = {'role': 'user', 'content': question}

            # In round 1, we need to attach the image info to the messages
            if i == 0:
                current_message['images'] = [b64_image]

            # adding the current user questions to the conversation history
            messages.append(current_message)

            # call the Ollama API
            # stream=False means receiving the entire response in a time
            response = ollama.chat(
                model=model_name,
                messages=messages,
                stream=False
            )

            # obtain and print the answer of the model
            assistant_response = response['message']['content']
            # print(f"Answer: {assistant_response}\n")

            # Adding the model response to the conversation history, serving as the context for question in the next round
            messages.append({'role': 'assistant', 'content': assistant_response})
        
        return messages

    except Exception as e:
        print(f"Error when interacting with Ollama: {e}")
        return "Processed Failed."

# --- Main script entrance ---
if __name__ == "__main__":
    # --- Please modify here ---
    MODEL_TO_USE = 'qwen2.5vl:7b'
    SYSTEM_PROMPT = "You are a highly advanced vision-language model with exceptional capabilities in image \
        understanding and description. Your task is to carefully analyze the image and provide detailed, \
        accurate, and helpful responses about what you see. For camouflaged objects, pay special \
        attention to subtle patterns, colors, and shapes that might indicate hidden elements in the scene. \
        Your response is up to five sentences in default. \
        Do not add any extra commentary, greetings, or introductory phrases. Just provide the answer. \
        Important: The annotated outline is only a visual aid, never mention it or acknowledge its existance in your response."
    
    processed_imgs = set()

    if not os.path.exists(args.output_json):
        with open(args.output_json, 'w', encoding='utf-8') as f:
            pass
    else:
        with open(args.output_json, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_imgs.add(data["image"])

    img_lists = os.listdir(args.image_path)
    wrong_examples = set()
    for img in tqdm(img_lists, desc= "generating captions via gemma3..."):
        img_name = img
        if img_name in processed_imgs:
            continue

        mask_name = img.split(".")[0] + ".png"
        img_path = os.path.join(args.image_path, img)


        results = run_vlm_conversation(img_path, MODEL_TO_USE, SYSTEM_PROMPT)
        image_result = {
            "mask": mask_name,
            "image": img_name,
            "detail_prompt": "",
            "prompt": ""               
        }
        if results == "Processed Failed.":
            wrong_examples.add(img)
            image_result["detail_prompt"] = "Failed."
            image_result["prompt"] = "Failed."
        else:
            results[1]["images"] = img_path       #round 1 after sys info has images
            answers = [item for item in results if item['role'] == 'assistant']
            detail_prompt = answers[2]['content']
            prompt = answers[3]['content']
            
            image_result["detail_prompt"] = detail_prompt
            image_result["prompt"] = prompt
                

        with open(args.output_json, 'a', encoding='utf-8') as f:
            f.write(json.dumps(image_result, ensure_ascii=False) + '\n')
    print("num of wrong numbers:", len(wrong_examples))
    print(wrong_examples)