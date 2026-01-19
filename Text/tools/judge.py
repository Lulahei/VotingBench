from rouge_score import rouge_scorer
from openai import OpenAI
def is_legal_output(output, assigned_word):
    """
    检查模型输出的语句是否包含其分配到的单词。

    Args:
        output (str): 模型生成的输出语句。
        assigned_word (str): 模型分配到的单词。

    Returns:
        bool: 如果输出不包含分配单词，返回 True；否则返回 False。
    """
    if assigned_word.lower() in output.lower():
        return False
    return True

def calculate_rouge2(output, previous_outputs):
    """
    计算模型输出与之前所有输出之间的 ROUGE-2 分值。

    Args:
        output (str): 当前模型的输出语句。
        previous_outputs (list of str): 之前模型生成的输出语句列表。

    Returns:
        dict: 包含每个之前输出的 ROUGE-2 分值。
    """
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    rouge_scores = {}

    for idx, prev_output in enumerate(previous_outputs):
        score = scorer.score(output, prev_output)
        rouge_scores[f"previous_output_{idx + 1}"] = score['rouge2'].fmeasure

    return rouge_scores

def is_related(output, word):
    client = OpenAI(api_key="sk-9a2ae9e5ed1740d7bc35f361c6fd872f", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )

    
    return response
