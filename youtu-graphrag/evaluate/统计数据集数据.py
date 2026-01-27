import json
import os


def count_stats(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_chunks = len(data)
    # 粗略估算 Token 数 (中文字符数 * 1.5 或直接统计字符)
    # 这里统计字符数作为参考
    total_chars = sum([len(item.get('text', '')) for item in data])
    # 假设 1个中文词 ≈ 1.5 tokens (视模型而定，这里用字符数展示也行)

    return num_chunks, total_chars


# 你的文件路径
file_corpus = 'data/uploaded/aviation_仅第二章/corpus.json'  # 第二章
file_aviation = 'data/uploaded/aviation_第四章/aviation.json'  # 第四章

n1, t1 = count_stats(file_corpus)
n2, t2 = count_stats(file_aviation)

print(f"=== 表 3-1 填空数据 ===")
print(f"Dataset A (第二章): {n1} Chunks, 约 {t1} 字符")
print(f"Dataset B (第四章): {n2} Chunks, 约 {t2} 字符")
print(f"Total: {n1 + n2} Chunks, 约 {t1 + t2} 字符")