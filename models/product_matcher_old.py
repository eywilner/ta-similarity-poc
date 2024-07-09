import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .schemas import ProductPairWithAmazon
import tempfile
import os
import requests

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()



tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

generation_config = dict(
    num_beams=1,
    max_new_tokens=100,
    do_sample=False,
)

def download_image(image_url1, image_url2):
    # Create temporary files
    tmp_file1 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tmp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')

    try:
        # Download the first image with a timeout of 3 seconds
        response1 = requests.get(image_url1, timeout=3)
        if response1.status_code == 200:
            tmp_file1.write(response1.content)
            tmp_file1.close()
        else:
            tmp_file1.close()
            os.unlink(tmp_file1.name)
            raise Exception(f"Failed to download image from {image_url1}")

        # Download the second image with a timeout of 3 seconds
        response2 = requests.get(image_url2, timeout=3)
        if response2.status_code == 200:
            tmp_file2.write(response2.content)
            tmp_file2.close()
        else:
            tmp_file2.close()
            os.unlink(tmp_file2.name)
            raise Exception(f"Failed to download image from {image_url2}")

    except requests.RequestException as e:
        # Clean up in case of a request error
        tmp_file1.close()
        os.unlink(tmp_file1.name)
        tmp_file2.close()
        os.unlink(tmp_file2.name)
        raise Exception(f"An error occurred: {e}")

    # Return the paths to the downloaded images
    return tmp_file1.name, tmp_file2.name


def parse_ai_response(response):
    # Remove the "Assistant: " part from the response
    if response.startswith("Assistant: "):
        response = response[len("Assistant: "):]
    
    # Split the response at the '/' to separate match status and reason
    parts = response.split('/', 1)
    
    # Check if the response is in the expected format
    if len(parts) != 2:
        raise ValueError("Response format is incorrect")
    
    # Extract the match status and reason
    match_status = parts[0].strip().lower()
    reason = parts[1].strip()
    
    # Determine the boolean value for ai_match
    ai_match = match_status == "true"
    
    return {'ai_match': ai_match, 'ai_reason': reason}


def product_match_model(data: ProductPairWithAmazon):
    source_title = data.source_title
    source_image_url = data.source_image_url
    amazon_title = data.amazon_title
    amazon_image_url = data.amazon_image_url

    source_image_path, amazon_image_path = download_image(image_url1=source_image_url, image_url2=amazon_image_url)

    source_pixel_value = load_image(source_image_path, max_num=6).to(torch.bfloat16).cuda()
    amazon_pixel_value = load_image(amazon_image_path, max_num=6).to(torch.bfloat16).cuda()

    pixel_values = torch.cat((source_pixel_value, amazon_pixel_value), dim=0)
    num_patches_list = [source_pixel_value.size(0), amazon_pixel_value.size(0)]

    question = f"""
    'Image-1: <image>\nImage-2: <image>\n
    You are giving two images. One is a source product and one is amazon product. 
    Also, here are thier product titles: 
    source product title {source_title}\n
    amason product title {amazon_title}
    Are these two products the same? No match should be considered if they are different by 
    size, volume, quantity or they are totally differet.
"""
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=None, return_history=True)

    question = """
    Are these two products same or not, based on what youve found?
    Return you response exactly in a this format:
    True or False / "Detailed reason for the result specific to the products being compared"
"""
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=history, return_history=True)


    return parse_ai_response(response=response)
    










