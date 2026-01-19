from .eval_utils import *
from PIL import Image
import copy
from transformers import AutoProcessor
from openai import OpenAI
import base64

system_prompt = """
你是一个AI助手，现在要参与游玩一个叫“DIXIT”的游戏。以下是游戏规则的概要：

1. 游戏目标：玩家们轮流成为讲述者，通过简短的文字、动作或歌曲来描述自己手中的卡牌，挑战其他玩家的想象力和直觉。
2. 游戏准备：每位玩家抽取6张卡牌作为手牌，并将84张游戏卡牌洗混后放置在游戏区域中间作为抽牌堆。
3. 游戏流程：
   - 讲述者（当前回合的玩家）从手牌中选择一张卡牌，并用一个词、短语、句子或声音来描述这张卡牌，不能让其他玩家看到卡牌图案。
   - 其他玩家从自己的手牌中选择一张与讲述者描述相呼应的卡牌，背面朝上交给讲述者。
   - 讲述者将所有卡牌混在一起，背面朝上翻开，并编号。
   - 其他玩家猜测哪张是讲述者的卡牌，并进行投票。
4. 计分规则：
   - 如果所有玩家都猜中或都没猜中，讲述者不得分，其他玩家各得2分。
   - 如果有玩家猜中，讲述者和猜中的玩家各得3分，其他玩家每获得一票得1分。
5. 游戏结束：当抽牌堆的卡牌被抽完时，游戏结束，得分最多的玩家获胜。

请记住，讲述者的描述既不能太准确也不能太模糊，要恰到好处地激发其他玩家的想象力。游戏的乐趣在于语义的模糊性和想象力的发挥。
"""

class player():
    def __init__(self, model_name, image_list, running_type="local") -> None:
        self.hand = image_list
        if running_type == 'local':
            self.model, self.tokenizer = prepare_model_tokenizer(model_name, [[0,1], [0,1]])
    
    def update_hand(self, image_list):
        self.hand = image_list
    

class player4Qwen2VL(player):
    def __init__(self, model_name, image_list=None, run_type="local") -> None:
        super().__init__(model_name, image_list, run_type)
        # self.model = model
        # self.tokenizer = tokenizer
        # self.hand = image_list
    
    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        true_image = random.choice(self.hand)
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        true_img = Image.open(true_image).convert('RGB')
        if description_type == "rule":
            prompt = [
                {"role": "system", "content": [{"type": "text","text": system_prompt}]},
                {"role": "user", "content": [
                        {"type": "image", "image": true_img},
                        {"type": "text","text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。\n请给出你描述的文本（用\"<content></content>\"包裹）。"},
                    ]
                }
            ]
        else:
            prompt = [
                {"role": "system", "content": [{"type": "text","text": system_prompt}]},
                {"role": "user", "content": [
                        {"type": "image", "image": true_img},
                        {"type": "text","text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。你的描述应较为抽象但不脱离图片实际。\n请给出你描述的文本（用\"<content></content>\"包裹）。"},
                    ]
                }
            ]

        return prompt, true_image
    
    def prepare_select_input(self, description):
        prompt = [
            {"role": "system", "content": [
                    {"type": "text","text": system_prompt},
                ]
            },
            {"role": "user", "content": [
                    {"type": "text","text": "现在，你需要从你手中的6张卡牌中选择出与提供的描述最相近的一张。你手中的6张卡牌分别为：\n\n"},
                ]
            }
        ]

        for i, pics in enumerate(self.hand):
            image = Image.open(pics)
            prompt[1]['content'].extend([
                {"type": "text","text": f"卡牌{i+1}：\n"},
                {"type": "image","image": image}
        ])

        prompt[1]['content'].extend([
            {"type": "text","text": f"讲述者的描述内容为：{description}\n"},
            {"type": "text","text": f"\n请从你的{len(self.hand)}张手牌中选出一张与该描述最相符的卡牌。\n请给出你选择的卡牌的序号（用\"<idx></idx>包裹\"）。"},
        ])

        return prompt
    
    def prepare_vote_input(self, description, select_list):
        prompt = [
            {"role": "system", "content": [
                    {"type": "text","text": system_prompt},
                ]
            },
            {"role": "user", "content": [
                    {"type": "text","text": f"现在，游戏来到了投票环节，你需要从展示的{len(select_list)-1}张卡牌中选择出与提供的描述最相近的一张。展示的{len(select_list)-1}张卡牌分别为：\n\n"},
                ]
            }
        ]

        card_cnt = 1
        for i, pics in enumerate(select_list):
            if pics in self.hand:
                continue
            image = Image.open(pics)
            prompt[1]['content'].extend([
                {"type": "text","text": f"卡牌{card_cnt}：\n"},
                {"type": "image","image": image}
            ])
            card_cnt += 1

        prompt[1]['content'].extend([
            {"type": "text","text": f"讲述者的描述内容为：{description}\n"},
            {"type": "text","text": f"\n请从展示的{len(select_list)-1}张卡牌中选出一张与该描述最相符的卡牌。\n请给出你选择的卡牌的序号（用\"<idx></idx>包裹\"）。"},
        ])

        return prompt

    def select_image_ppl(self, description, images=None):
        from qwen_vl_utils import process_vision_info
        if images is None:
            images = self.hand
        
        all_ppl = []
        for image in images:
            image = Image.open(image)
            prompt_no_ans = [
                {"role": "user", "content": [
                        {"type": "image","image": image},
                        {"type": "text","text": "请对给出的图片进行描述。"},
                    ]
                }
            ]

            prompt = copy.deepcopy(prompt_no_ans)
            prompt.append({
                "role": "assistant", 
                "content": [
                        {"type": "text","text": f"描述：\n{description}"},
                    ]
            })

            text_no_ans = self.tokenizer.apply_chat_template(
                prompt_no_ans, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(prompt_no_ans)
            inputs_no_ans = self.tokenizer(
                text=[text_no_ans],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            question_lens = len(inputs_no_ans.input_ids[0])

            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(prompt)
            inputs = self.tokenizer(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            target_ids = inputs.input_ids.clone().to(self.model.device)
            target_ids[:, :question_lens] = -100
            inputs['labels'] = target_ids
            # print(target_ids[:, -100:])
            with torch.no_grad():
                outputs = self.model(**inputs)
                all_ppl.append(torch.exp(outputs.loss))
            
        min_idx = all_ppl.index(min(all_ppl))

        return images[min_idx]


    def generate_response(self, query_list):
        from qwen_vl_utils import process_vision_info
        # Preparation for inference
        text = self.tokenizer.apply_chat_template(
            query_list, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(query_list)
        inputs = self.tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


class player4GLM4V(player):
    def __init__(self, model_name, image_list=None, run_type="local") -> None:
        super().__init__(model_name, image_list, run_type)

    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        true_image = random.choice(self.hand)
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        true_img = Image.open(true_image).convert('RGB')
        if description_type == "rule":
           prompt = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "image": true_img,
                    "content": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。\n请给出你描述的文本（用\"<content></content>\"包裹）。"
                }
            ]
        else:
            prompt = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "image": true_img,
                    "content": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。你的描述应较为抽象但不脱离图片实际。\n请给出你描述的文本（用\"<content></content>\"包裹）。"
                }
            ]
        

        return prompt, true_image
    
    def select_image_ppl(self, description, images=None):
        if images is None:
            images = self.hand
        
        all_ppl = []
        for image in images:
            image = Image.open(image).convert('RGB')
            prompt_no_ans = [
                {
                    "role": "user", 
                    "image": image, 
                    "content": "请对给出的图片进行描述。"
                },
            ]

            prompt = copy.deepcopy(prompt_no_ans)
            prompt.append(
                {
                    "role": "assistant",
                    "content": f"描述：\n{description}",
                }
            )
            
            inputs_no_ans = self.tokenizer.apply_chat_template(prompt_no_ans, add_generation_prompt=True, tokenize=True, 
                                                    return_tensors="pt",return_dict=True)  # chat mode 
           
            question_lens = len(inputs_no_ans.input_ids[0]) 

            inputs = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=True, 
                                                    return_tensors="pt",return_dict=True).to(self.model.device)
            target_ids = inputs.input_ids.clone().to(self.model.device)
            target_ids[:, :question_lens] = -100
            inputs['labels'] = target_ids
            # print(target_ids[:, -100:])
            with torch.no_grad():
                outputs = self.model(**inputs)
                all_ppl.append(torch.exp(outputs.loss))
            
        min_idx = all_ppl.index(min(all_ppl))

        return images[min_idx]

    def generate_response(self, query_list):
        inputs = self.tokenizer.apply_chat_template(query_list,add_generation_prompt=True, tokenize=True, 
                                                    return_tensors="pt",return_dict=True)  # chat mode 
           
        inputs = inputs.to(self.model.device)

        gen_kwargs = {"max_length": 1024, "do_sample": False}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(outputs[0])

class player4QwenVL(player):
    def __init__(self, model_name, image_list=None, run_type="local") -> None:
        super().__init__(model_name, image_list, run_type)

    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        true_image = random.choice(self.hand)
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        if description_type == "rule":
            prompt = [
                {"role": "system", "content": system_prompt},
                {"image": true_image},
                {"text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。\n请给出你描述的文本（请用\"<content></content>包裹你描述的文本\"）。"}
            ]
        else:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"image": true_image},
                {"text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。你的描述应较为抽象但不脱离图片实际。\n请给出你描述的文本（请用\"<content></content>包裹你描述的文本\"）。"}
            ]

        return prompt, true_image
    
    def select_image_ppl(self, description, images=None):
        if images is None:
            images = self.hand
        
        all_ppl = []
        for image in images:
            prompt_no_ans = [
                {"text":" 请对给出的图片进行描述。"},
                {"image": image}, 
                {"text": "\n描述：\n"},
            ]

            prompt = copy.deepcopy(prompt_no_ans)
            prompt[-1]["text"] += description
            
            text_no_ans = self.tokenizer.from_list_format(prompt_no_ans) 
            inputs_no_ans = self.tokenizer(text_no_ans, return_tensors='pt')           
            question_lens = len(inputs_no_ans.input_ids[0])

            input_text = self.tokenizer.from_list_format(prompt)
            inputs = self.tokenizer(input_text, return_tensors='pt')     
            
            input_ids = inputs.input_ids.to(self.model.device)
            target_ids = input_ids.clone()
            target_ids[:, :question_lens] = -100
            # print(target_ids[:, -100:])
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                all_ppl.append(torch.exp(outputs.loss))
            
        min_idx = all_ppl.index(min(all_ppl))

        return images[min_idx]

    def generate_response(self, query_list):
        system = ''
        if 'role' in query_list[0] and query_list[0]['role'] == "system":
            sytem_msg = query_list.pop(0)
            system = "\n".join(sytem_msg['content'])
        query = self.tokenizer.from_list_format(query_list)
        # print(query)
        response, history = self.model.chat(self.tokenizer, query=query, system=system, history=None)
        return response

class player4InternVL2(player):
    def __init__(self, model_name, image_list=None, run_type="local") -> None:
        super().__init__(model_name, image_list, run_type)
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        true_image = random.choice(self.hand)
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        
        if description_type == "rule":
            appendix = ""
        else:
            appendix = "你的描述应较为抽象但不脱离图片实际。"
        question = f"<image>\n现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。{appendix}\n请给出你描述的文本（用\"<content></content>\"包裹）。"

        return ([true_image, system_prompt, question], true_image)
    
    def select_image_ppl(self, description, images=None):
        if images is None:
            images = self.hand
        
        all_ppl = []
        for image in images:
            pixel_values = load_image(image, max_num=12).to(torch.bfloat16).to(self.model.device)
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

            text_no_ans = "<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|>" + \
                                    "<|im_start|>user\n<image>\n请对给出的图片进行描述。<|im_end|>"
            
            for num_patches in num_patches_list:
                image_tokens = '<img>' + '<IMG_CONTEXT>' * 256 * num_patches + '</img>'
                text_no_ans = text_no_ans.replace('<image>', image_tokens, 1)
            
            inputs_no_ans = self.tokenizer(text_no_ans, return_tensors='pt')
            question_lens = len(inputs_no_ans.input_ids[0])

            input_text = "<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|>" + \
                                    f"<|im_start|>user\n<image>\n请对给出的图片进行描述。<|im_end|><|im_start|>assistant\n'描述：\n{description}"
            
            for num_patches in num_patches_list:
                image_tokens = '<img>' + '<IMG_CONTEXT>' * 256 * num_patches + '</img>'
                input_text = input_text.replace('<image>', image_tokens, 1)

            inputs = self.tokenizer(input_text, return_tensors='pt')
            input_ids = inputs.input_ids.to(self.model.device)
            target_ids = input_ids.clone().to(self.model.device)
            target_ids[:, :question_lens] = -100
            attention_mask = inputs['attention_mask'].to(self.model.device)
            image_flags = torch.ones(pixel_values.shape[0])
            # print(target_ids[:, -30:])
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, image_flags=image_flags, labels=target_ids)
                all_ppl.append(torch.exp(outputs.loss))
            
        min_idx = all_ppl.index(min(all_ppl))

        return images[min_idx]

    def generate_response(self, inputs):
        img, system, question = inputs
        pixel_values = load_image(img, max_num=12).to(torch.bfloat16).to(self.model.device)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        if system:
            self.model.system_message = system
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)

        return response

class player4LLama(player):
    def __init__(self, model_name, image_list=None, run_type="local") -> None:
        super().__init__(model_name, image_list, run_type)
        self.processor = self.tokenizer

    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        true_image = random.choice(self.hand)
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        if description_type == 'rule':
            messages = [
                {"role": "system", "content": [
                    {"type": "text","text": system_prompt},
                ]},
                {"role": "user", "content": [
                    {"type": "image", "image": true_image},
                    {"type": "text","text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。\n请给出你描述的文本（用\"<content></content>\"包裹）。"},
                ]}
            ]
        else:
            messages = [
                {"role": "system", "content": [
                    {"type": "text","text": system_prompt},
                ]},
                {"role": "user", "content": [
                    {"type": "image",  "image": true_image},
                    {"type": "text","text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。你的描述应较为抽象但不脱离图片实际。\n请给出你描述的文本（用\"<content></content>\"包裹）。"},
                ]}
            ]
        

        return messages, true_image
    
    def select_image_ppl(self, description, images=None):
        if images is None:
            images = self.hand
        
        all_ppl = []
        for image in images:
            image = Image.open(image)
            prompt_no_ans = [
                {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text","text": "请对给出的图片进行描述。"},
                    ]
                }
            ]

            prompt = copy.deepcopy(prompt_no_ans)
            prompt.append(
                {"role": "assistant", "content": [
                        {"type": "text","text": f"描述：\n{description}"},
                    ]
                }
            )

            text_no_ans = self.processor.apply_chat_template(
                prompt_no_ans, tokenize=False, add_generation_prompt=True
            )
            inputs_no_ans = self.processor(
                image,
                text_no_ans,
                add_special_tokens=False,
                return_tensors="pt"
            )
            question_lens = len(inputs_no_ans.input_ids[0])

            text = self.processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=False
            )
            inputs = self.processor(
                image,
                text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
            target_ids = inputs.input_ids.clone().to(self.model.device)
            target_ids[:, :question_lens] = -100
            inputs['labels'] = target_ids
            # print(target_ids[:, -100:])
            with torch.no_grad():
                outputs = self.model(**inputs)
                all_ppl.append(torch.exp(outputs.loss))
            
        min_idx = all_ppl.index(min(all_ppl))

        return images[min_idx]

    def generate_response(self, inputs):
        user_message = [m for m in inputs if m["role"] == "user"][0]
        image = Image.open(user_message["content"][0]["image"])

        input_text = self.processor.apply_chat_template(inputs, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=1024)
        output = output[:, inputs['input_ids'].shape[1]:]
        response = self.processor.decode(output[0])
        return response

class player4DeepSeekVL(player):
    def __init__(self, model_name, image_list=None, run_type="local") -> None:
        super().__init__(model_name, image_list, run_type)
        self.vl_chat_processor = self.tokenizer
        self.tokenizer = self.vl_chat_processor.tokenizer
        

    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        true_image = random.choice(self.hand)
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        
        if description_type == 'rule':
            prompt = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{system_prompt}\n\n现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。\n请给出你描述的文本（用\"<content></content>\"包裹）。",
                    "images": [true_image],
                },
                {"role": "Assistant", "content": ""},
            ]
        else:
            prompt = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{system_prompt}\n\n现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。你的描述应较为抽象但不脱离图片实际。\n请给出你描述的文本（用\"<content></content>\"包裹）。",
                    "images": [true_image],
                },
                {"role": "Assistant", "content": ""},
            ]

        return prompt, true_image
    
    def select_image_ppl(self, description, images=None):
        from deepseek_vl.utils.io import load_pil_images
        if images is None:
            images = self.hand
        
        all_ppl = []
        for image in images:
            prompt_no_ans = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>请对给出的图片进行描述。",
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]
            pil_images = load_pil_images(prompt_no_ans)

            inputs_no_ans = self.vl_chat_processor(
                conversations=prompt_no_ans,
                images=pil_images,
                force_batchify=True
            )
            question_lens = len(inputs_no_ans.input_ids[0])


            prompt = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>请对给出的图片进行描述。",
                    "images": [image],
                },
                {"role": "Assistant", "content": f"{description}"},
            ]

            inputs = self.vl_chat_processor(
                conversations=prompt,
                images=pil_images,
                force_batchify=True
            ).to(self.model.device)
            target_ids = inputs.input_ids.clone().to(self.model.device)
            target_ids[:, :question_lens] = -100
            
            # print(target_ids[:, -30:])
            inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
            with torch.no_grad():
                outputs = self.model.language_model(
                    inputs_embeds=inputs_embeds, 
                    attention_mask=inputs.attention_mask, 
                    labels=target_ids
                )
                all_ppl.append(torch.exp(outputs.loss))
            
        min_idx = all_ppl.index(min(all_ppl))

        return images[min_idx]

    def generate_response(self, inputs):
        from deepseek_vl.utils.io import load_pil_images
        # load images and prepare for inputs
        pil_images = load_pil_images(inputs)
        prepare_inputs = self.vl_chat_processor(
            conversations=inputs,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)

        return response

class player4STEP1V(player):
    def __init__(self, model_name, image_list=None, run_type="api") -> None:
        super().__init__(model_name, image_list, run_type)
        # self.model = model
        # self.tokenizer = tokenizer
        # self.hand = image_list
    
    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        
        true_image = random.choice(self.hand)
        # deprecated
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        
        # Base64 Convert
        true_img_base64 = base64.b64encode(open(true_image, 'rb').read()).decode('ascii')
        if description_type == "rule":
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                # 在对话中传入图片，来实现基于图片的理解
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。\n请给出你描述的文本（用\"<content></content>\"包裹）。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{true_img_base64}"},
                        },
                    ],
                },
            ]
        else:
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                # 在对话中传入图片，来实现基于图片的理解
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。你的描述应较为抽象但不脱离图片实际。\n请给出你描述的文本（用\"<content></content>\"包裹）。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{true_img_base64}"},
                        },
                    ],
                },
            ]

        return prompt, true_image
    
    def prepare_select_input(self, description):
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                    {"type": "text","text": f"现在，你需要从你手中的{len(self.hand)}张卡牌中选择出与提供的描述最相近的一张。你手中的6张卡牌分别为：\n\n"},
                ]
            }
        ]

        for i, pics in enumerate(self.hand):
            img_base64 = base64.b64encode(open(pics, 'rb').read()).decode('ascii')
            prompt[1]['content'].extend([
                {"type": "text","text": f"卡牌{i+1}：\n"},
                {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
        ])

        prompt[1]['content'].extend([
            {
                "type": "text",
                "text": f"\n讲述者的描述内容为：{description}。\n请从你的{len(self.hand)}张手牌中选择出与提供的描述最相近的一张，如果没有，请随机抽取一张作为回答。\n请给出你选择的卡牌的序号,并只回答我<pad>i</pad>，表示选择卡牌i。"
            },
        ])

        return prompt
    
    def prepare_vote_input(self, description, select_list):
        prompt = [
            {"role": "system", "content": [
                    {"type": "text","text": system_prompt},
                ]
            },
            {"role": "user", "content": [
                    {"type": "text","text": f"现在，游戏来到了投票环节，你需要从展示的{len(select_list)}张卡牌中选择出与提供的描述最相近的一张。展示的{len(select_list)}张卡牌分别为：\n\n"},
                ]
            }
        ]

        for i, pics in enumerate(select_list):
            img_base64 = base64.b64encode(open(pics, 'rb').read()).decode('ascii')
            prompt[1]['content'].extend([
                {"type": "text","text": f"卡牌{i+1}：\n"},
                {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
        ])

        prompt[1]['content'].extend([
            {"type": "text","text": f"讲述者的描述内容为：{description}。\n请从展示的{len(select_list)-1}张卡牌中选出一张与该描述最相符的卡牌。\n请给出你选择的卡牌的序号,并只回答我<pad>i</pad>，表示选择卡牌i。"},
        ])

        return prompt

    def generate_response(self, query_list):
        API_KEY= os.getenv("STEP_API_KEY")
        client = OpenAI(api_key=API_KEY, base_url="https://api.stepfun.com/v1")
 
        completion = client.chat.completions.create(
            model="step-1v-8k",
            messages=query_list
        )
        
        return completion.choices[0].message.content

class player4YiVisionV2(player):
    def __init__(self, model_name, image_list=None, run_type="api") -> None:
        super().__init__(model_name, image_list, run_type)
    
    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        true_image = random.choice(self.hand)
        # deprecated
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        
        # Base64 Convert
        true_img_base64 = base64.b64encode(open(true_image, 'rb').read()).decode('utf-8')
        
        if description_type == "rule":
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                # 在对话中传入图片，来实现基于图片的理解
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。\n请给出你描述的文本（用\"<content></content>\"包裹）。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{true_img_base64}"},
                        },
                    ],
                },
            ]
        else:
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                # 在对话中传入图片，来实现基于图片的理解
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。你的描述应较为抽象但不脱离图片实际。\n请给出你描述的文本（用\"<content></content>\"包裹）。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{true_img_base64}"},
                        },
                    ],
                },
            ]

        return prompt, true_image
    
    def prepare_select_input(self, description):
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                    {"type": "text","text": f"现在，你需要从你手中的{len(self.hand)}张卡牌中选择出与提供的描述最相近的一张。\n提供的描述为：{description}。\n请给出你选择的卡牌的序号，并只回答我<pad>i</pad>，表示选择卡牌i。\n你手中的6张卡牌分别为：\n\n"},
                ]
            }
        ]

        for i, pics in enumerate(self.hand):
            img_base64 = base64.b64encode(open(pics, 'rb').read()).decode('utf-8')
            prompt[1]['content'].append(
                {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
            )

        return prompt
    
    def prepare_vote_input(self, description, select_list):
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                    {"type": "text","text": f"现在，游戏来到了投票环节，你需要从展示的{len(select_list)}张卡牌中选择出与提供的描述最相近的一张。\n提供的描述为：{description}。\n请给出你选择的卡牌的序号，并只回答我<pad>i</pad>，表示选择卡牌i。\n展示的{len(select_list)}张卡牌(卡牌的序号为1-{len(select_list)})分别为：\n\n"},
                ]
            }
        ]

        for i, pics in enumerate(select_list):
            img_base64 = base64.b64encode(open(pics, 'rb').read()).decode('utf-8')
            prompt[1]['content'].append(
                {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
            )

        return prompt


    def generate_response(self, query_list):
        API_KEY= os.getenv("ZERONE_API_KEY")
        client = OpenAI(api_key=API_KEY, base_url="https://api.lingyiwanwu.com/v1")
 
        completion = client.chat.completions.create(
            model="yi-vision-v2",
            messages=query_list
        )
        
        return completion.choices[0].message.content


class player4Human(player):
    def __init__(self, model_name, image_list=None, run_type="human") -> None:
        super().__init__(model_name, image_list, run_type)
    
    def prepare_narrator_say_input(self, description_type, fixed_img=None, idx=None):
        true_image = random.choice(self.hand)
        if isinstance(fixed_img, list) and idx:
            true_image = fixed_img[idx]
        if description_type == "rule":
            prompt = f"{system_prompt}\n\n现在，你被选为当作讲述者。\n请对这张图片用不超过五个字进行描述。\n请给出你描述的文本（用\"<content></content>\"包裹）。"
        else:
            prompt = f"{system_prompt}\n\n现在，你被选为当作讲述者。请对这张图片用不超过五个字进行描述。你的描述应较为抽象但不脱离图片实际。\n请给出你描述的文本（用\"<content></content>\"包裹）。"

        new_im = Image.open(true_image)
        merged_image_path = "tmp_imgs/merged_image.png"
        new_im.save(merged_image_path)
        return ([prompt, true_image], merged_image_path)
    
    def prepare_select_input(self, description):
        prompt = f"{system_prompt}\n\n现在，你需要从你手中的{len(self.hand)}张卡牌中选择出与提供的描述最相近的一张。\n提供的描述为：{description}。\n请给出你选择的卡牌的序号，并只回答我<pad>i</pad>，表示选择卡牌i。"

        # image concat
        pil_image_list = [Image.open(i).resize((100, 175)) for i in self.hand]
        total_width = sum(img.width for img in pil_image_list)
        max_height = max(img.height for img in pil_image_list)

        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in pil_image_list:
            new_im.paste(img, (x_offset, 0))
            x_offset += img.width

        # save image after concat
        merged_image_path = "tmp_imgs/merged_image.png"
        new_im.save(merged_image_path)

        return [prompt, merged_image_path]
    
    def prepare_vote_input(self, description, select_list):
        prompt = f"{system_prompt}\n\n现在，游戏来到了投票环节，你需要从展示的{len(select_list)}张卡牌中选择出与提供的描述最相近的一张。\n提供的描述为：{description}。\n请给出你选择的卡牌的序号，并只回答我<pad>i</pad>，表示选择卡牌i。"

        # image concat
        pil_image_list = [Image.open(i).resize((100, 175)) for i in select_list]
        total_width = sum(img.width for img in pil_image_list)
        max_height = max(img.height for img in pil_image_list)

        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in pil_image_list:
            new_im.paste(img, (x_offset, 0))
            x_offset += img.width

        # save image after concat
        merged_image_path = "tmp_imgs/merged_image.png"
        new_im.save(merged_image_path)

        return [prompt, merged_image_path]
    
    def generate_response(self, query_list):
        query, true_image = query_list
        print("-----------------------------------------")
        print(query)
        print(f"上述文本中提到的图片（卡牌）地址：{true_image}")
        response = input("请输入你的回答：\n")
        print("-----------------------------------------")
        
        return response