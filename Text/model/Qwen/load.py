import os
import random
import time
from openai import OpenAI
import re
from tools.judge import is_legal_output,calculate_rouge2
from tools.utils import log_with_timestamp
import pdb
client = OpenAI(
    api_key= os.environ.get("Qwen_apikey"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

__name__ = "Qwen"

# 每轮角色发言，使用 API 生成描述，角色只能看到其他人的发言和自己的词汇
def play_round(player, history, log,field,language,is_trick):
    previous_outputs = []  # 用于存储所有合法的之前输出
    # pdb.set_trace()

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
            # f"{other_players_description}"
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
    # pdb.set_trace()
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    description = completion.choices[0].message.content.strip()

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

def voting_round(player, history, log,language):

    vote_details = {}
    
    visible_history = [entry for entry in history if entry["role"] != player.player_id and entry["role"] != "system"]
    visible_history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in visible_history])

    if language == "cn":
        prompt = (
                f"你是 {player.player_id}，请根据对话记录投票选出卧底。\n"
                "---对话记录---\n" + visible_history_text + "\n"
                "**必须严格按以下格式回答**：\n"
                "'我怀疑playerX是卧底,\n"
                "推理:<content>\n"
                f"注意：不能投票给自己({player.player_id})"
        )
    elif language == "en":
        prompt = (
                f"You are {player.player_id}, vote for the spy based on the conversation.\n"
                "---Conversation---\n" + visible_history_text + "\n"
                "**Strictly use this format**:\n"
                "'I suspect playerX is the spy,\n"
                "Reasoning:<content>\n"
                f"Note: Do NOT vote for yourself({player.player_id})"
        )

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    response = completion.choices[0].message.content.strip()
    # pdb.set_trace()
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