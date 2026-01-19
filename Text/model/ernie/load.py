import os
import qianfan
import time
import re
from tools.judge import is_legal_output,calculate_rouge2
from tools.utils import log_with_timestamp



chat_comp = qianfan.ChatCompletion()
__name__ = "ernie"
# 指定特定模型
# resp = chat_comp.do(model="ERNIE-3.5-8K", messages=[{
#     "role": "user",
#     "content": "你好"
# }])
#
# print(resp["body"]["result"])

def play_round(player, history, log,field,language,is_trick):
    previous_outputs = []

    # 提取系统规则和其他玩家的历史记录
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
    # 使用千帆模型生成描述
    chat_comp = qianfan.ChatCompletion()
    resp = chat_comp.do(
        model="ERNIE-3.5-8K",  # 指定模型
        messages=[{"role": "user", "content": prompt}]
    )
    description = resp["body"]["result"].strip()

    # 检查输出是否合法
    if not is_legal_output(description, player.word):
        log_with_timestamp(log, f"{player.player_id} 输出非法！包含了分配的单词：{player.word}，描述：{description}")
        print(f"非法输出: {description}")
        return False

    # 检查输出是否与其他玩家的描述过于相似
    rouge_scores = calculate_rouge2(description, previous_outputs)
    max_rouge2 = max(rouge_scores.values(), default=0)
    if max_rouge2 > 0.55:  # 假设相似性阈值为 0.55
        log_with_timestamp(log, f"{player.player_id} 输出过于相似，可能为抄袭！ROUGE-2 分值: {max_rouge2}，描述：{description}")
        print(f"输出过于相似: {description}")
        return False

    # 将合法输出添加到历史记录中
    history.append({"role": player.player_id, "content": description})
    log_with_timestamp(log, f"{player.player_id} 生成合法输出: {description}")
    previous_outputs.append(description)

    # 等待 1 秒，避免频繁调用
    time.sleep(1)

    return True

def voting_round(player, history, log,language):
    global prompt
    vote_details = {}

    # 提取可见的历史记录（排除当前玩家和系统消息）
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

    # 使用千帆模型生成投票结果
    chat_comp = qianfan.ChatCompletion()
    resp = chat_comp.do(
        model="ERNIE-3.5-8K",  # 指定模型
        messages=[{"role": "user", "content": prompt}]
    )
    response = resp["body"]["result"].strip()

    # 解析投票结果
    if language == "cn":
        match = re.search(r'我怀疑(player\d+)是卧底', response)
    elif language == "en":
        match = re.search(r'I suspect (player\d+) is the spy', response)
    if match:
        target = match.group(1)  # 提取被怀疑的玩家
    else:
        # 如果模型返回的格式不正确，默认投票给下一个玩家
        player_index = int(player.player_id.replace('player', ''))  # 从 "player1" 到 "playerN" 中提取数字部分
        next_player_index = (player_index + 1) % (len(visible_history) + 1)  # 取模加1保证不越界
        if next_player_index == 0:
            next_player_index = len(visible_history) + 1
        target = f"player{next_player_index}"
        print("投票混乱")

    # 记录投票结果
    vote_details[player.player_id] = target

    # 记录投票日志
    vote_log = f"{player.player_id} 投票给 {target}, 投票详情：{response}"
    log_with_timestamp(log, vote_log)

    return vote_details