import importlib
import os
import pdb
import random
import time
random.seed(42)
from Dao.player import local_model, Player
from tools.utils import read_word_pairs, read_json, log_with_timestamp, save_log_to_file
from tqdm import tqdm
import json
all_player_stats = {}
model_stats = {}
local_models = {}
def get_model_api(model_name):
    try:
        model_module = importlib.import_module(f"model.{model_name}.load")
        return model_module
    except ModuleNotFoundError:
        raise Exception(f"模型{model_name}的api模块不存在，请确认模型的名称正确")

##为player实体分配model
def assign_models(players,model_names):
    model_count = len(model_names)

    for i,player in enumerate(players):
        player.model = model_names[i % model_count]

def human_play_round(player,history,log):

    visible_history = [entry for entry in history if entry["role"] != player.player_id and entry["role"] != "system"]
    visible_history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in visible_history]) if visible_history else ""
    print(f"\n你的回合 单词: {player.word} ")
    print("其他人的描述:")
    print(visible_history_text)
    while True:
        try:
            desc = input("请输入你的描述:")
            for char in desc:
                if ord(char) >= 0xD800 and ord(char) <= 0xDFFF:
                    raise ValueError("输入包含无效的 Unicode 字符，请重新输入。")
            break
        except Exception as e:
            print(e)
    history.append({"role":player.player_id,"content":desc})
    log_with_timestamp(log,f"{player.player_id} 生成合法输出: {desc}")

    return True

def human_voting_round(player,history,log):
    vote_details = {}
    visible_history = [entry for entry in history if entry["role"] != player.player_id and entry["role"] != "system"]
    visible_history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in visible_history])
    print("以下是对话内容:")
    print(visible_history_text)
    num = input("请输入数字来决定谁是卧底")
    target = f"player{num}"
    vote_details[player.player_id] = target
    vote_log = f"{player.player_id} 投票给 {target}"
    log_with_timestamp(log,vote_log)

    return vote_details
def main(csv_file,num_players,model_names,log_file_path,field,language,is_trick,word_num):
    pairs = read_word_pairs(csv_file)
    pairs_to_play = pairs[0:word_num]
    players = []
    for i in range(num_players):
        players.append(Player(player_id=f"player{i+1}",word="",model=model_names[i]))
    for player in players:
        model_stats[player.model] = model_stats.get(player.model, {
            "civilian_count": 0,
            "spy_count": 0,
            "correct_votes": 0,
            "spy_wins": 0,
            "illegial_times":0
        })
    for civilian_word,spy_word in tqdm(pairs_to_play,desc="游戏进行中",unit="轮"):

        participating_models = list(set(model_names))
        for model in participating_models:
            for player in players:
                player.word = civilian_word
            spy_player = [player for player in players if player.model == model]
            spy_player = random.choice(spy_player)
            spy_player.word = spy_word

            log = []
            log_with_timestamp(log, "------ 游戏开始 ------")
            log_with_timestamp(log, f"玩家分配如下：")
            for player in players:
                log_with_timestamp(log, f"{player}")
                times = 0
            save_log_to_file(log,log_file_path)
            if language == 'cn':
                history = [{"role": "system", "content": "你正在游玩·谁是卧底·的游戏。每轮游戏会有一个词汇对，大多数人会得到其中一个词汇，他们被称为‘平民’，有1个人会获得另一个词汇，他被称为‘卧底’。平民的任务是通过描述和推理找出唯一的卧底，卧底的任务是隐瞒自己与其他人不同的词语，混淆视听避免被发现。"}]
            else:
                history = [{"role": "system", "content": "You are playing the game 'Who is Spy'. In each round, there is a pair of words. Most players will receive one of the words and are called 'civilians'. One player will receive the other word and is called the 'spy'. The civilians' task is to identify the spy through descriptions and reasoning, while the spy's task is to conceal their different word and avoid being discovered by confusing others."}]
            random.shuffle(players)
            for player in players:
                if player.model=="user":
                    human_play_round(player,history,log)
                else:
                    while True:
                        model_api = get_model_api(player.model)
                        flag = model_api.play_round(player, history, log,field,language,is_trick)
                        if not flag:
                            model_stats[player.model]["illegial_times"] += 1
                            log_with_timestamp(log, "------ 检测到非法输出，重新执行玩家描述 ------")
                            continue
                        break
            all_vote_details ={}
            for player in players:
                if player.model == "user":
                    vote_details = human_voting_round(player,history,log)
                    all_vote_details[player.player_id] = vote_details[player.player_id]
                else:
                    model_api = get_model_api(player.model)
                    vote_details = model_api.voting_round(player, history, log,language)
                    all_vote_details[player.player_id] = vote_details[player.player_id]
            total_votes = {player.player_id: 0 for player in players}
            save_log_to_file(log,log_file_path)
            log = []
            for player_id, target in all_vote_details.items():
                total_votes[target] += 1

            max_votes = max(total_votes.values())
            most_voted = [k for k, v in total_votes.items() if v == max_votes]
            elimination = random.choice(most_voted)  # 随机选择一个被淘汰的玩家
            log_with_timestamp(log, f"玩家 {elimination} 被淘汰")

            spy_player = [player for player in players if player.word == spy_word][0]
            if elimination == spy_player.player_id:
                result = "卧底失败，平民胜利！"
            else:
                result = "平民失败，卧底胜利！"
            print(result)
            log_with_timestamp(log, result)
            save_log_to_file(log,log_file_path)
            for player in players:
                if player.player_id not in all_player_stats:
                    all_player_stats[player.player_id] = {
                        "vote_correct_count": 0,
                        "is_spy_count": 0,
                        "spy_win_count": 0,
                        "is_civilian_count": 0,
                        "civilian_win_count": 0,
                    }
                if player == spy_player:
                    all_player_stats[player.player_id]["is_spy_count"] += 1
                    if result == "平民失败，卧底胜利！":
                        all_player_stats[player.player_id]["spy_win_count"] += 1
                else:
                    all_player_stats[player.player_id]["is_civilian_count"] += 1
                    if result == "卧底失败，平民胜利！":
                        all_player_stats[player.player_id]["civilian_win_count"] += 1

                # 记录到日志中
            log_with_timestamp(log, "------ 游戏结束 ------")
            save_log_to_file(log, log_file_path)
            log = []
    log_with_timestamp(log, "------ 游戏统计结果 ------")

    for player_id, stats in all_player_stats.items():
        log_with_timestamp(log, f"{player_id} 详细统计:")
        log_with_timestamp(log, f"  - 被分配为卧底的次数: {stats['is_spy_count']}")
        log_with_timestamp(log, f"  - 卧底胜利的次数: {stats['spy_win_count']}")
        log_with_timestamp(log, f"  - 被分配为平民的次数: {stats['is_civilian_count']}")
        log_with_timestamp(log, f"  - 平民胜利的次数: {stats['civilian_win_count']}")
        model_name = next(player.model for player in players if player.player_id == player_id)
        model_stats[model_name]["spy_count"] += stats["is_spy_count"]
        model_stats[model_name]["civilian_count"] += stats["is_civilian_count"]
        model_stats[model_name]["spy_wins"] += stats["spy_win_count"]
        model_stats[model_name]["correct_votes"] += stats["civilian_win_count"]
    print(all_player_stats)
    print(model_stats)
    # 输出按模型统计的结果
    log_with_timestamp(log, "------ 按模型统计的结果 ------")
    for model, stats in model_stats.items():
        civilian_rate = stats["correct_votes"] / stats["civilian_count"] if stats["civilian_count"] > 0 else 0
        spy_win_rate = stats["spy_wins"] / stats["spy_count"] if stats["spy_count"] > 0 else 0
        log_with_timestamp(log, f"模型 {model}:")
        log_with_timestamp(log, f"  - 平民投票命中率: {civilian_rate:.4f}")
        log_with_timestamp(log, f"  - 卧底获胜率: {spy_win_rate:.4f}")
        log_with_timestamp(log, f"  - 非法输出次数: {stats['illegial_times']}")
    # pdb.set_trace()
    # 保存最终的日志到文件
    save_log_to_file(log,log_file_path)
if __name__ == "__main__":
    csv_file = "data/word_日常_cn.csv"
    file_name = os.path.splitext(os.path.basename(csv_file))[0]
    parts = file_name.split('_')
    field = parts[1]
    language = parts[2]
    is_trick = True
    model_names = ["deepseek","glm","Qwen","gpt","ernie","user"]
    num_players = len(model_names)
    log_file = "Result/test.txt"
    main(csv_file,num_players,model_names,log_file,field,language,is_trick,50)