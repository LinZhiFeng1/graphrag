import pandas as pd
import os
import numpy as np

# 设置多个文件路径
folder_paths = [
    r"evaluate/老正确率结果",
    r"evaluate/双路径",
    r"evaluate/传统RAG"
]

# 获取所有 CSV文件并按文件名排序
csv_files = []
for folder_path in folder_paths:
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        # 添加路径前缀以区分不同文件夹的文件
        csv_files.extend([(folder_path, f) for f in sorted(files)])
    else:
        print(f"警告：路径 {folder_path} 不存在")

# 存储结果
results = {}

print("正在分析所有 CSV文件中的节点召回率和正确率数据...\n")
print(f"搜索路径：{', '.join(folder_paths)}")
print(f"找到 {len(csv_files)} 个 CSV文件\n")

# 处理每个 CSV文件
for folder_path, csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    try:
        # 读取 CSV文件
        df = pd.read_csv(file_path)

        # 初始化统计变量
        recall_stats = {'avg_recall': 0, 'valid_count': 0, 'total_count': len(df)}
        accuracy_stats = {'correct_count': 0, 'total_valid': 0, 'accuracy': 0}

        # 处理节点召回率列
        if '节点召回率' in df.columns:
            # 提取有值的节点召回率数据（非空且非 NaN）
            recall_rates = df['节点召回率'].dropna()

            # 过滤掉空字符串
            recall_rates = recall_rates[recall_rates != '']

            # 转换为数值类型
            recall_rates = pd.to_numeric(recall_rates, errors='coerce')

            # 再次去除 NaN 值
            recall_rates = recall_rates.dropna()

            if len(recall_rates) > 0:
                # 计算平均值
                avg_recall = recall_rates.mean()
                valid_count = len(recall_rates)

                recall_stats = {
                    'avg_recall': round(avg_recall, 4),
                    'valid_count': valid_count,
                    'total_count': len(df)
                }
            else:
                recall_stats = {
                    'avg_recall': 0,
                    'valid_count': 0,
                    'total_count': len(df)
                }

        # 处理是否正确列
        if '是否正确' in df.columns:
            # 提取"是否正确"列的数据
            correctness = df['是否正确'].dropna()

            # 过滤掉空字符串
            correctness = correctness[correctness != '']

            # 转换为数值类型
            correctness = pd.to_numeric(correctness, errors='coerce')

            # 只保留值为 0 或 1 的行
            valid_correctness = correctness[(correctness == 0) | (correctness == 1)]

            if len(valid_correctness) > 0:
                correct_count = sum(valid_correctness == 1)
                total_valid = len(valid_correctness)
                accuracy = correct_count / total_valid if total_valid > 0 else 0

                accuracy_stats = {
                    'correct_count': int(correct_count),
                    'total_valid': int(total_valid),
                    'accuracy': round(accuracy, 4)
                }
            else:
                accuracy_stats = {
                    'correct_count': 0,
                    'total_valid': 0,
                    'accuracy': 0
                }

        # 存储结果（使用相对路径作为键）
        relative_key = os.path.join(os.path.basename(folder_path), csv_file)
        results[relative_key] = {
            '节点召回率': recall_stats,
            '正确率': accuracy_stats,
            'full_path': file_path
        }

        # 打印详细信息
        print(f"文件：{relative_key}")
        print(
            f"  - 节点召回率：{recall_stats['avg_recall']} ({recall_stats['valid_count']}/{recall_stats['total_count']}行)")
        print(
            f"  - 正确率：{accuracy_stats['accuracy']:.2%} ({accuracy_stats['correct_count']}/{accuracy_stats['total_valid']})")
        print()

    except Exception as e:
        print(f"处理文件 {relative_key} 时出错：{str(e)}")

# 输出汇总结果（按文件名顺序）
print("=" * 60)
print("汇总结果 (按文件名顺序):")
print("=" * 60)

for file_name, stats in results.items():
    recall_info = f"{stats['节点召回率']['avg_recall']} ({stats['节点召回率']['valid_count']}/{stats['节点召回率']['total_count']}行)"
    accuracy_info = f"{stats['正确率']['accuracy']:.2%} ({stats['正确率']['correct_count']}/{stats['正确率']['total_valid']})"
    print(f"\n文件：{file_name}")
    print(f"  节点召回率平均值：{recall_info}")
    print(f"  正确率：{accuracy_info}")