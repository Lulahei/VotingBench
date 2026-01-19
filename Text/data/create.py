import random
import json
import csv

# 从CSV文件中加载词汇对
def load_word_pairs_from_csv(filename):
    word_pairs = []
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            word_pairs.append((row[0], row[1]))
    return word_pairs

# 读取CSV文件并加载词汇对
low_difficulty_pairs = load_word_pairs_from_csv("word_日常_cn.csv")
medium_difficulty_pairs = [("crane", "stork"), ("sapphire", "ruby"), ("pebble", "stone"), ("owl", "hawk")]
high_difficulty_pairs = [("parapet", "battlement"), ("juggernaut", "behemoth"), ("accolade", "praise"), ("abyss", "chasm")]

# 配置生成的词汇对数量
low_difficulty_count = 1000
medium_difficulty_count = 0
high_difficulty_count = 0

# 生成每个任务的JSON对象，并随机分配卧底模型
def generate_task(test_id, difficulty, similarity_type, rarity_type, pair):
    non_spy_word, spy_word = pair
    
    # 随机选择一个模型位置作为卧底
    model_ids = ["model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8"]
    spy_model_id = random.choice(model_ids)
    
    # 构建模型分配列表
    models = []
    for model_id in model_ids:
        word = spy_word if model_id == spy_model_id else non_spy_word
        models.append({"model_id": model_id, "word": word})
    
    # 构建任务字典
    task = {
        "test_id": test_id,
        "difficulty": difficulty,
        "similarity_type": similarity_type,
        "rarity_type": rarity_type,
        "non_spy_word": non_spy_word,
        "spy_word": spy_word,
        "models": models
    }
    return task

# 生成任务列表
def generate_evaluation_set():
    evaluation_set = []
    test_id = 1

    # 生成低难度任务
    for _ in range(low_difficulty_count):
        pair = random.choice(low_difficulty_pairs)
        task = generate_task(test_id, "low", "low", "common", pair)
        evaluation_set.append(task)
        test_id += 1

    # 生成中等难度任务
    for _ in range(medium_difficulty_count):
        pair = random.choice(medium_difficulty_pairs)
        task = generate_task(test_id, "medium", "medium", "uncommon", pair)
        evaluation_set.append(task)
        test_id += 1

    # 生成高难度任务
    for _ in range(high_difficulty_count):
        pair = random.choice(high_difficulty_pairs)
        task = generate_task(test_id, "high", "high", "rare", pair)
        evaluation_set.append(task)
        test_id += 1

    return evaluation_set

# 保存评测集为 JSON 文件
def save_evaluation_set(evaluation_set, filename="evaluation_set.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump({"evaluation_set": evaluation_set}, file, indent=4, ensure_ascii=False)

# 主程序执行
evaluation_set = generate_evaluation_set()
save_evaluation_set(evaluation_set)
print("评测集已生成并保存为 evaluation_set.json")
