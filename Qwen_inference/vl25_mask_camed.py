import os
import json
from tqdm import tqdm
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# post-process, which ensures the sentences can ended correctly
def ensure_complete_sentence(text):
    # look for the end-of-sentence marker
    end_markers = ['. ', '! ', '? ', '."', '!"', '?"', '.', '!', '?']
    last_end_pos = -1
    
    for marker in end_markers:
        pos = text.rfind(marker)
        if pos > last_end_pos:
            last_end_pos = pos + len(marker) - (1 if marker[-1] in ['"', ' '] else 0)
    
    # truncate the text if end-of-sentence marker is found
    if last_end_pos > 0:
        return text[:last_end_pos]
    return text

# define the system info
system_message = {
    "role": "system",
    "content": "You are a highly advanced vision-language model with exceptional capabilities in image \
        understanding and description. Your task is to carefully analyze the image and provide detailed, \
        accurate, and helpful responses about what you see. For camouflaged objects, pay special \
        attention to subtle patterns, colors, and shapes that might indicate hidden elements in the scene. \
        Important: The annotated outline is only a visual aid, never mention it or acknowledge its existence in your response."
}

cam_image_folder = "/home/ubuntu/Projects/Datasets/LAKERED_DATASET/train/images_annotated"
cam_image_list = [cam_image_folder + '/' + image_name for image_name in os.listdir(cam_image_folder)]
# define the multi-round VQA
questions = [
    "Describe the camouflaged object/objects outlined by the contour and explain how it is/are camouflaged in the surroundings.\
    Your response is up to five sentences.",
    "Describe the environment outside the contour, and explain how it can camouflage the object/objects successfully.\
    Your response is up to five sentences.",
    "Describe the image in detail, your response is up to five sentences.",
    "According to the conversation history, summarize this image in one sentence. Your answer should be formatted \
    like this: *article* *object description* in the *environment description*. (annotation * not included)\
    "
]

output_json_path = "/home/ubuntu/Projects/Qwen2-VL/exps/exp_res/vl25_mask_camed.jsonl"
processed_img_names = set()

# Check if the output json file exists, if so, read the image names that are already processed
if os.path.exists(output_json_path):
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'image_name' in data:
                        processed_img_names.add(data['image_name'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {output_json_path}: {line.strip()}")
    except Exception as e:
        print(f"Error reading {output_json_path}: {e}. Starting with an empty set of processed images.")
else:
    # if the file doesn't exist then make an empty one
    with open(output_json_path, 'w', encoding='utf-8') as f:
        pass # create an empty file
# if not os.path.exists(output_json_path):
#     with open(output_json_path, 'w') as f:
#         pass    

for img_path in tqdm(cam_image_list, desc="Generating captions"):
    img_name = img_path.split('/')[-1]
    # check whether the image is processed
    if img_name in processed_img_names:
        print(f"Skipping already processed image: {img_name}")
        continue  # skip the images that already processed (support resume inference)
    
    print(f"Processing image: {img_name}")
    img_results = {
        "image_name": img_name,
        "conversation": []
    }

    # initialize the conversation history, start with only system message
    conversation_history = [system_message]

    # perform multi-round VQA
    for round_idx, question in enumerate(questions, 1):
        #the 1st round question contains images
        if round_idx == 1:
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        else:
            # the later rounds only contains text
            user_message = {
                "role": "user",
                "content": question
            }
        # add the user instruction to the conversation history
        conversation_history.append(user_message)

        # record the user question
        if round_idx == 1:
            # special operation on round1 because it contains images
            img_results["conversation"].append({
                "role": "user",
                "content": question  # only records text
            })
        else:
            img_results["conversation"].append(user_message)



        # Preparation for inference
        text = processor.apply_chat_template(
            conversation_history, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation_history)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        max_retries = 1
        retry_count = 0
        output_text = ""

        while retry_count <= max_retries:
            try:
                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, 
                                            max_new_tokens=128,
                                            temperature=0.9,
                                            top_p=0.9)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                break
            except Exception as e:
                retry_count += 1
                print(f"Error occurred when generating answers for {img_name} in round {round_idx}: {str(e)}")
                if retry_count > max_retries:
                    print(f"Still fail after retrying for {max_retries} times, log an empty value")
                    output_text = "Failed to generate captions."
                else:
                    print(f"Making the {retry_count}th retry...")
        
        # apply post-processing
        processed_output = ensure_complete_sentence(output_text)

        # apply the model output to the conversation history
        assistant_response = {
            "role": "assistant",
            "content": processed_output
        }
        conversation_history.append(assistant_response)
        # save the model answer
        img_results["conversation"].append(assistant_response)

    with open(output_json_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(img_results, ensure_ascii=False) + '\n')

print("Done!")