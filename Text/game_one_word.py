import os
import time

from Dao.player import Player,local_model
import random
import importlib
from tools.utils import read_word_pairs,log_with_timestamp,save_log_to_file,read_json
import pdb
from tqdm import tqdm
import random
all_player_stats = {}
model_stats = {}
local_models = {}
##动态载入model api
def get_model_api(model_name):
    try:
        model_module = importlib.import_module(f"model.{model_name}.load")
        return model_module
    except ModuleNotFoundError:
        raise Exception(f"模型{model_name} 的api模块不存在，请确认模型的名称正确")

##为player实体分配model
def assign_models(players,model_names):
    model_count = len(model_names)

    for i,player in enumerate(players):
        player.model = model_names[i % model_count]

def main(csv_file,num_players,model_names,log_file_path,num_pairs_to_select,field,language,is_trick):
    #第一步获得词汇对
    pairs =  read_word_pairs(csv_file)
    random.shuffle(pairs)
    models_path = read_json("model/all_model.json")
    index = 7
    for name in model_names:
        if name.startswith("local"):
            name = name[5:]
            # pdb.set_trace()
            if name in models_path:
                model_path = models_path[name]["model_path"]
                # 加载本地模型和分词器
                model = local_model(model_path, index)
                # pdb.set_trace()
                # 将本地模型和分词器保存到全局字典中
                local_models[name] = {"model": model}
                index = index+1
            else:
                raise Exception(f"模型配置文件中没有找到{name}的配置")
    # pdb.set_trace()
    pairs_to_play = random.sample(pairs, num_pairs_to_select)
    #词汇循环
    for civilian_word, spy_word in tqdm(pairs_to_play, desc="游戏进行中", unit="轮"):

        retry_round = True
        while retry_round:
            try:
                players = []

                # 创建模型参与本轮游戏的列表
                participating_models = list(set(model_names))  # 去重，确保每个模型仅参与一次
                random.shuffle(participating_models)  # 打乱模型顺序

                # 创建玩家对象并初始化
                for i in range(num_players):
                    players.append(Player(player_id=f"player{i+1}", word=civilian_word, model=None))

                # 分配模型
                assign_models(players, model_names)

                # 第二层循环：遍历每个模型并分配卧底词
                for model_name in participating_models:
                    for player in players:
                        player.word = civilian_word
                    # 找到该模型名下的玩家
                    model_players = [player for player in players if player.model == model_name]

                    # 随机选择该模型下的一个玩家作为卧底
                    spy_player = random.choice(model_players)
                    spy_player.word = spy_word  # 将该玩家的词汇设置为卧底词

                    # 记录该模型已分配卧底词
                    model_stats[model_name] = model_stats.get(model_name, {
                        "civilian_count": 0,
                        "spy_count": 0,
                        "correct_votes": 0,
                        "spy_wins": 0,
                        "illegial_times":0
                    })

                    # 游戏过程：该模型下的玩家开始进行游戏
                    log = []
                    log_with_timestamp(log, "------ 游戏开始 ------")
                    log_with_timestamp(log, f"玩家分配如下：")
                    for player in players:
                        log_with_timestamp(log, f"{player}")
                    times = 0

                    while True:
                        # 玩家描述
                        if language == 'cn':
                            history = [{"role": "system", "content": "你正在游玩·谁是卧底·的游戏。每轮游戏会有一个词汇对，大多数人会得到其中一个词汇，他们被称为‘平民’，有1个人会获得另一个词汇，他被称为‘卧底’。平民的任务是通过描述和推理找出唯一的卧底，卧底的任务是隐瞒自己与其他人不同的词语，混淆视听避免被发现。"}]
                        else:
                            history = [{"role": "system", "content": "You are playing the game 'Who is Spy'. In each round, there is a pair of words. Most players will receive one of the words and are called 'civilians'. One player will receive the other word and is called the 'spy'. The civilians' task is to identify the spy through descriptions and reasoning, while the spy's task is to conceal their different word and avoid being discovered by confusing others."}]
                        # 判断是否有非法输出
                        flag = True
                        for player in players:
                            if player.model.startswith("local"):
                                model = local_models[player.model[5:]]["model"]
                                model_api = get_model_api(player.model[:5])
                                flag = model_api.play_round(model, player, history, log,field,language,is_trick)
                            else:
                                model_api = get_model_api(player.model)
                                flag = model_api.play_round(player, history, log,field,language,is_trick)
                                if not flag:
                                    model_stats[player.model]["illegial_times"] += 1
                                    break
                        if not flag:
                            log_with_timestamp(log, "------ 检测到非法输出，重新执行玩家描述 ------")
                            times += 1
                            if times >=10:
                                break
                            continue
                        else:
                            break

                    # 投票环节
                    all_vote_details = {}  # 用来存储所有玩家的投票细节
                    for player in players:
                        if player.model.startswith("local"):
                            model = local_models[player.model[5:]]["model"]
                            model_api = get_model_api(player.model[:5])
                            vote_details = model_api.voting_round(model, player, history, log,language)
                            all_vote_details[player.player_id] = vote_details[player.player_id]
                        else:
                            model_api = get_model_api(player.model)
                            vote_details = model_api.voting_round(player, history, log,language)
                            all_vote_details[player.player_id] = vote_details[player.player_id]

                    total_votes = {player.player_id: 0 for player in players}
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

                    log_with_timestamp(log, result)

                    # 更新玩家统计信息
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
                retry_round = False
            except Exception as e:
                # 捕获异常，记录日志
                log_with_timestamp(log, f"------ 游戏出错 ------")
                log_with_timestamp(log, f"错误信息: {str(e)}")
                save_log_to_file(log, log_file_path)
                retry_round = True
                time.sleep(10)
                continue
    #此处词汇循环结束
    # pdb.set_trace()
    # 游戏结束后记录总的玩家统计信息
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
    model_names = ["doubao","deepseek","glm","Qwen","gpt","ernie"]
    num_players = len(model_names)
    log_file = "Result/6model_battle_260112_test.txt"
    main(csv_file,num_players,model_names,log_file,50,field,language,is_trick)