from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.judge import is_legal_output,calculate_rouge2
from tools.utils import log_with_timestamp
import time
import re

class local_model:
    def __init__(self,model_name,device_map):
        self.model_name = model_name
        self.device_map = f"cuda:{device_map}"
        self.model,self.tokenizer = self.load_model()
    
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
        print("模型与分词器加载完成")
        return model,tokenizer
    
    def chat_with_model(self,user_input):
        
        messages = [
            {"role":"user","content":user_input}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text],return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


def play_round(model,player,history,log,field,language,is_trick):
    previous_outputs = []

    system_rule = [entry["content"] for entry in history if entry["role"] == "system"]
    system_rule_text = "\n".join(system_rule) if system_rule else ""
    visible_history = [entry for entry in history if entry["role"] != player.player_id and entry["role"] != "system"]
    if language == "cn":
        if visible_history:
            visible_history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in visible_history]) if visible_history else ""
            if is_trick:
                other_players_description = f"，观察其他玩家的描述，推测他们的物品，并使你的描述能够体现你的物品与他们物品的共同点：\n{visible_history_text}\n"
            else:
                other_players_description = f"，以下是其他人的描述：\n{visible_history_text}\n"
        else:
            other_players_description = ""

        prompt = (
            f"{system_rule_text}\n"
            f"{other_players_description}"
            f"你是 {player.player_id},当前词汇领域为{field}.你的任务是根据 {field} 的具体领域，用一句话描述“{player.word}”而不直接提及这个词汇。直接描述词汇即可。\n"
        )
    elif language == "en":
        if visible_history:
            visible_history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in visible_history]) if visible_history else ""
            if is_trick:
                other_players_description = f", observe other players' descriptions, infer their words, and make your description reflect the commonalities between your word and theirs:\n{visible_history_text}\n"
            else:
                other_players_description = f", here are the descriptions from others:\n{visible_history_text}\n"
        else:
            other_players_description = ""
        prompt = (
            f"{system_rule_text}\n"
            f"{other_players_description}"
            f"You are {player.player_id} and the current vocabulary domain is {field}. your task is to describe the word '{player.word}' based on the specific domain {field}, in one sentence without directly mentioning the word. Just output the content which describe the word directly.\n"
        )

    description = model.chat_with_model(prompt)
    
    if not is_legal_output(description, player.word):
        log_with_timestamp(log,f"{player.player_id} 输出非法！包含了分配的单词：{player.word}，描述：{description}")
        print(f"非法输出: {description}")
        return False  

    rouge_scores = calculate_rouge2(description, previous_outputs)
    max_rouge2 = max(rouge_scores.values(), default=0)
    if max_rouge2 > 0.55:  # 假设相似性阈值为 0.7
        log_with_timestamp(log,f"{player.player_id} 输出过于相似，可能为抄袭！ROUGE-2 分值: {max_rouge2}，描述：{description}")
        print(f"输出过于相似: {description}")
        return False  
    
    history.append({"role": player.player_id, "content": description})
    log_with_timestamp(log,f"{player.player_id} 生成合法输出: {description}")
    previous_outputs.append(description)  

    time.sleep(1)
    return True

def voting_round(model,player,history,log,language):
    vote_details = {}

    visible_history = [entry for entry in history if entry["role"] != player.player_id and entry["role"] != "system"]
    visible_history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in visible_history])

    if language == "cn":
        prompt = (
                f"你是 {player.player_id}，根据以下对话内容选择一个怀疑的玩家进行投票。\n"
                "对话内容：\n" + visible_history_text +
                "\n你怀疑谁是卧底？回答格式按以下格式“我怀疑playerx是卧底”" +
                f"请记住你是{player.player_id}，不要投票给{player.player_id}"
        )
    elif language == "en":
        prompt = (
                f"You are {player.player_id}, based on the following conversation content, choose a suspicious player to vote for.\n"
                "Conversation content:\n" + visible_history_text +
                "\nWho do you suspect is the spy? Answer in the following format: 'I suspect playerx is the spy'." +
                f"Remember you are {player.player_id}, do not vote for {player.player_id}."
        )

    response = model.chat_with_model(prompt)
    if language == "cn":
        match = re.search(r'我怀疑(player\d+)是卧底', response)
    elif language == "en":
        match = re.search(r'I suspect (player\d+) is the spy', response)
    
    if match:
        target = match.group(1)
    else:
        player_index = int(player.player_id.replace('player', ''))  # 从 "player1" 到 "playerN" 中提取数字部分
        next_player_index = (player_index+1) % (len(visible_history)+1) # 取模加1保证不越界
        if next_player_index == 0:
            next_player_index = len(visible_history)+1
        target = f"player{next_player_index}"
        print("投票混乱")

    vote_details[player.player_id] = target
    
    # 记录投票日志
    vote_log = f"{player.player_id} 投票给 {target},投票详情：{response}"
    log_with_timestamp(log,vote_log)

    return vote_details


# # 加载模型和分词器
# def localmodel_load(model_name):
#     # model_name = "/data/mjl/model_zoo/Qwen2.5-7B-Instruct"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="auto",
#         trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

# def chat_with_model(user_input):
#     # 构建对话模板
#     messages = [
#         {"role": "system", "content": "*你正在游玩·谁是卧底·的游戏。每轮游戏会有一个词汇对，大多数人会得到其中一个词汇，他们被称为‘平民’，有1个人会获得另一个词汇，他被称为‘卧底’。平民的任务是通过描述和推理找出唯一的卧底，卧底的任务是隐瞒自己与其他人不同的词语，混淆视听避免被发现。"},
#         {"role": "user", "content": user_input}
#     ]
    
#     # 使用分词器处理输入
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     # 生成响应
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response



# model = local_model("/data/mjl/model_zoo/Qwen2.5-7B-Instruct",3)
# log = []
# history = []
# class Player:
#     def __init__(self,player_id,word,model) :
#             self.player_id = player_id
#             self.word = word
#             self.model = model
    
#     def __str__(self):
#           return f"玩家信息：(player_id={self.player_id},word={self.word},分到的模型是{self.model})"
    
# player1 = Player("player1","风筝","local")
# play_round(model,player1,history,log)

