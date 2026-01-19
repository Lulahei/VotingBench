import json
import re
import os
import math
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_score_each(true_img, guess_list, select_list, narrator_idx):
    score_list = None
    if true_img not in guess_list or all([True if true_img == i else False for i in guess_list]):
        # all_true | all_false
        score_list = [2] * (len(guess_list) + 1)
        score_list[narrator_idx] = 0
        return score_list
    else:
        # Partially correct
        score_list = []
        for guess_img in guess_list:
            if true_img == guess_img:
                score_list.append(3)
            else:
                score_list.append(0)

        # extra socres
        for i, select_img in enumerate(select_list):
            if select_img in guess_list:
                score_list[i] += guess_list.count(select_img)
        
        score_list.insert(narrator_idx, 3)
        
    return score_list

def get_random_pair_dict(image_path, n, num_per_player, output_path):
    all_imgs = os.listdir(image_path)

    all_pairs = []
    for idx, img in enumerate(all_imgs):
        remaining = all_imgs[:]
        remaining.remove(img)
        one = {}
        one['ori_img'] = os.path.join("data/images", img)
        one['player_hands'] = []
        for i in range(n-1):
            picked = random.sample(remaining, num_per_player)
            picked = [os.path.join("data/images", i) for i in picked]
            one['player_hands'].append(picked)
            remaining = [x for x in remaining if x not in picked]
        
        all_pairs.append(one)

    json.dump(all_pairs, open(f"{output_path}/all_pairs_{n}_player_{num_per_player}_hands.json", 'w', encoding='utf-8'), indent=2)
    return all_pairs

def get_idx_from_raw(raw_text, images):
    # 匹配 <idx> 标签内的数字
    card_pattern1 = r'<pad>(\d+)</pad>'
    card_pattern2 = r'<pad>Card(\d+)</pad>'
    # 使用正则表达式搜索卡牌序号
    card_match1 = re.search(card_pattern1, raw_text)
    card_match2 = re.search(card_pattern2, raw_text)
    if card_match1:
        idx = int(card_match1.group(1))  # 卡牌序号
    elif card_match2:
        idx = int(card_match2.group(1))  # 卡牌序号
    elif "卡牌" in raw_text:
        try:
            idx = int(raw_text[raw_text.find("卡牌") + 2])
        except:
            raw_text = raw_text[raw_text.find("卡牌") + 2:]
            idx = int(raw_text[raw_text.find("卡牌") + 2])
    elif len(raw_text) == 1 and raw_text.isdigit():
        idx = int(raw_text)
    elif raw_text[1].isdigit():
        idx = int(raw_text[1])
    elif re.search(r'<(\d+)>', raw_text):
        idx = int(re.search(r'<(\d+)>', raw_text).group(1))
    else:
        raise ValueError(f"原始文本格式错误： {raw_text}")
    return images[idx-1]

def get_description_from_raw(raw_text):
    description = None
    # 正则表达式模式
    # 匹配双引号内的文本或 <content> 标签内的文本
    description_pattern = r'“(.*?)”|"(.*?)"|\<content>(.*?)\</content>'

    
    # 使用正则表达式搜索描述文本
    description_match = re.search(description_pattern, raw_text)

    if description_match:
        # 由于描述文本可能被双引号或<content>标签包裹，我们需要检查哪个分组匹配了
        if description_match.group(1):    # “”分组匹配
            description = description_match.group(1)  
        elif description_match.group(2):  # ""分组匹配
            description = description_match.group(2)  
        elif description_match.group(3):  # <content> 标签分组匹配
            description = description_match.group(3) 
    else:
        print("未找到描述文本")
    
    if description == None:
        raise ValueError(f"原始文本格式错误： {raw_text}")
    return description

def get_piece_idx(seq_list, key):
    for i, piece in enumerate(seq_list):
        if piece['ori_img'] == key:
            return i
    return -100

def get_player_idx(narrator, voter, model_list):
    model_list_copy = model_list[:]
    model_list_copy.remove(narrator)
    for i, piece in enumerate(model_list_copy):
        if piece == voter:
            return i
    return -100

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def prepare_model_tokenizer(model_name, gpu_device_ids):
    model_path = {
        "Qwen": "../models/Qwen",
        "Qwen2": "../models/Qwen2-VL-7B-Instruct",
        "minicpm": "../models/MiniCPM-V-2_6",
        "GLM4V": "../models/glm-4v-9b",
        "InternVL2": "../models/InternVL2-8B",
        "LLama": "../models/Llama-3.2-11B-Vision-Instruct",
        "DeepSeekVL": "../models/deepseek-vl-7b-chat"
    }
    ckpt_path = model_path[model_name]
    model = None
    tokenizer = None
    if model_name == "Qwen":
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map='cuda:0', trust_remote_code=True).eval()
        # model.generation_config.do_sample = False
        # model.generation_config.temperature=None
        # model.generation_config.top_p=None
        # model.generation_config.top_k=None
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    elif model_name == 'Qwen2':
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )
        # default processer
        model.eval()
        tokenizer = AutoProcessor.from_pretrained(ckpt_path)
    elif model_name == "GLM4V":
        device_map = {'transformer.embedding': gpu_device_ids[1][0], 'transformer.rotary_pos_emb': gpu_device_ids[1][0], 'transformer.encoder.layers.0': gpu_device_ids[1][0], 'transformer.encoder.layers.1': gpu_device_ids[1][0], 'transformer.encoder.layers.2': gpu_device_ids[1][0], 'transformer.encoder.layers.3': gpu_device_ids[1][0], 'transformer.encoder.layers.4': gpu_device_ids[1][0], 'transformer.encoder.layers.5': gpu_device_ids[1][0], 'transformer.encoder.layers.6': gpu_device_ids[1][0], 'transformer.encoder.layers.7': gpu_device_ids[1][0], 'transformer.encoder.layers.8': gpu_device_ids[1][0], 'transformer.encoder.layers.9': gpu_device_ids[1][0], 'transformer.encoder.layers.10': gpu_device_ids[1][0], 'transformer.encoder.layers.11': gpu_device_ids[1][0], 'transformer.encoder.layers.12': gpu_device_ids[1][0], 'transformer.encoder.layers.13': gpu_device_ids[1][0], 'transformer.encoder.layers.14': gpu_device_ids[1][0], 'transformer.encoder.layers.15': gpu_device_ids[1][0], 'transformer.encoder.layers.16': gpu_device_ids[1][0], 'transformer.encoder.layers.17': gpu_device_ids[1][0], 'transformer.encoder.layers.18': gpu_device_ids[1][0], 'transformer.encoder.layers.19': gpu_device_ids[1][0], 'transformer.encoder.layers.20': gpu_device_ids[1][0], 'transformer.encoder.layers.21': gpu_device_ids[1][0], 'transformer.encoder.layers.22': gpu_device_ids[1][0], 'transformer.encoder.layers.23': gpu_device_ids[1][0], 'transformer.encoder.layers.24': gpu_device_ids[1][0], 'transformer.encoder.layers.25': gpu_device_ids[1][0], 'transformer.encoder.layers.26': gpu_device_ids[1][0], 'transformer.encoder.layers.27': gpu_device_ids[1][0], 'transformer.encoder.layers.28': gpu_device_ids[1][0], 'transformer.encoder.layers.29': gpu_device_ids[1][0], 'transformer.encoder.layers.30': gpu_device_ids[1][1], 'transformer.encoder.layers.31': gpu_device_ids[1][1], 'transformer.encoder.layers.32': gpu_device_ids[1][1], 'transformer.encoder.layers.33': gpu_device_ids[1][1], 'transformer.encoder.layers.34': gpu_device_ids[1][1], 'transformer.encoder.layers.35': gpu_device_ids[1][1], 'transformer.encoder.layers.36': gpu_device_ids[1][1], 'transformer.encoder.layers.37': gpu_device_ids[1][1], 'transformer.encoder.layers.38': gpu_device_ids[1][1], 'transformer.encoder.layers.39': gpu_device_ids[1][1], 'transformer.encoder.final_layernorm': gpu_device_ids[1][1], 'transformer.output_layer': gpu_device_ids[1][0], 'transformer.vision': gpu_device_ids[1][1]}
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map= device_map
                )
        model.eval()
    elif model_name == "InternVL2":
        # if OOM in single GPU, try it
        device_map = split_model('InternVL2-8B')
        model = AutoModel.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True, use_fast=False)
    elif model_name == "LLama":
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        model = MllamaForConditionalGeneration.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoProcessor.from_pretrained(ckpt_path)
    elif model_name == "DeepSeekVL":
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(ckpt_path)
        tokenizer = vl_chat_processor

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = vl_gpt.eval()

    return model, tokenizer


def get_random_nplayer_images(n, image_path, num_per_player):
    all_imgs = os.listdir(image_path)
    remaining = all_imgs[:]
    result = []
    
    for _ in range(n):
        if len(remaining) < n:  # 检查剩余元素是否足够
            raise ValueError("Not enough elements left to pick without repetition.")
        picked = random.sample(remaining, num_per_player)
        result.append(picked)
        remaining = [x for x in remaining if x not in picked]

    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] =os.path.join("data/images", result[i][j])

    return result  