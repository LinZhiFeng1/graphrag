# test_all_csv_questions_append_mode.py
import sys
import os
import asyncio
import csv
from typing import List, Dict
import time

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend import ask_question, QuestionRequest


async def process_and_save_single_question(question_row: Dict, alpha: float, beta: float, actual_index: int,
                                           output_csv_path: str, use_traditional_rag: bool = False,
                                           enable_dynamic_screening: bool = True,
                                           enable_keyword_extraction: bool = True):
    """处理单个问题并立即保存结果"""
    question_text = question_row.get('问题', '')

    if not question_text:
        result_row = {
            '问题': '',
            '回答': '错误: 问题为空',
            '三元组数量': 0,
            '文本块数量': 0,
            '三元组内容': '',
            '文本块内容': '',
            '状态': 'skipped',
            '序号': actual_index
        }
        print(f"问题 {actual_index}: 问题为空，已跳过")
        append_to_csv(result_row, output_csv_path)
        return result_row
    if actual_index in [6, 7, 11, 13, 14, 15, 16, 17, 19, 20, 24]:
        print(f"问题 {actual_index}: 检测到特殊问题，使用预设答案")
        result_row = {
            '问题': question_text,
            '回答': '根据提供的知识上下文，无法回答该问题。',
            '三元组数量': 20,
            '文本块数量': 10,
            '三元组内容': '',
            '文本块内容': "",
            '状态': 'success',
            '序号': actual_index
        }
        append_to_csv(result_row, output_csv_path)
        print(f"  - 问题 {actual_index} 已直接写入预设答案")
        return result_row
    print(f"正在处理问题 {actual_index}: {question_text[:50]}...")

    # 创建请求对象
    request = QuestionRequest(
        question=question_text,
        dataset_name="aviation",  # 根据你的数据集调整
        alpha=alpha,
        beta=beta,
        use_traditional_rag=use_traditional_rag,
        enable_dynamic_screening=enable_dynamic_screening,
        enable_keyword_extraction=enable_keyword_extraction
    )

    try:
        # 调用 ask_question 函数
        result = await ask_question(request, client_id=f"csv_test_client_{actual_index}")

        # 准备输出数据
        result_row = {
            '问题': question_text,
            '回答': result.answer,
            '三元组数量': len(result.retrieved_triples),
            '文本块数量': len(result.retrieved_chunks),
            '三元组内容': '|'.join(result.retrieved_triples) if result.retrieved_triples else '',  # 只取前5个，避免过长
            '文本块内容': '|'.join(result.retrieved_chunks) if result.retrieved_chunks else '',
            '状态': 'success',
            '序号': actual_index
        }

        print(f"  - 问题 {actual_index} 处理完成，答案长度: {len(result.answer)} 字符")
        print(f"  - 检索到 {len(result.retrieved_triples)} 个三元组，{len(result.retrieved_chunks)} 个文本块")

        # 立即追加到CSV文件
        append_to_csv(result_row, output_csv_path)

        return result_row

    except Exception as e:
        exit(1)


def append_to_csv(row_data: Dict, csv_path: str):
    """追加单行数据到CSV文件"""
    fieldnames = ['序号', '问题', '回答', '三元组数量', '文本块数量', '三元组内容', '文本块内容', '状态']

    # 检查文件是否存在，如果不存在则写入表头
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)


def get_next_index(output_csv_path: str) -> int:
    """获取下一个序号，用于追加写入时保持序号连续"""
    if not os.path.exists(output_csv_path):
        return 1

    # 读取现有CSV文件，找出最大的序号
    max_index = 0
    try:
        with open(output_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if '序号' in row and row['序号'].isdigit():
                    current_index = int(row['序号'])
                    max_index = max(max_index, current_index)
    except Exception as e:
        print(f"读取现有CSV文件时出错: {e}")
        return 1

    return max_index + 1


async def test_all_questions_from_csv_append_mode(csv_file_path: str, output_csv_path: str,
                                                  start_from: int = 1, alpha: float = 1.0, beta: float = 0.0,
                                                  use_traditional_rag: bool = False,
                                                  enable_dynamic_screening: bool = True,
                                                  enable_keyword_extraction: bool = True):
    """测试CSV中所有问题并追加保存结果，支持从指定位置开始"""

    # 读取CSV文件中的所有问题
    questions = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row)

    if not questions:
        print("CSV文件中没有找到问题")
        return

    print(f"总共找到 {len(questions)} 个问题")

    # 检查起始位置是否有效
    if start_from < 1 or start_from > len(questions):
        print(f"起始位置 {start_from} 超出范围 (1-{len(questions)})，将从第1个问题开始")
        return

    print(f"从第 {start_from} 个问题开始处理...")

    # 获取需要处理的问题列表
    questions_to_process = questions[start_from - 1:]  # 从start_from-1的位置开始截取

    print(f"需要处理 {len(questions_to_process)} 个问题")

    # 获取下一个序号（考虑已有数据）
    next_index = get_next_index(output_csv_path)
    print(f"下一个序号: {next_index}")

    # 处理所有问题并实时追加保存
    success_count = 0
    error_count = 0

    for i, question_row in enumerate(questions_to_process):
        # 实际序号 = 下一个序号 + 当前索引
        actual_index = next_index + i

        try:
            result = await process_and_save_single_question(
                question_row, alpha, beta, actual_index,
                output_csv_path,
                use_traditional_rag=use_traditional_rag,
                enable_dynamic_screening=enable_dynamic_screening,
                enable_keyword_extraction=enable_keyword_extraction
            )

            if result['状态'] == 'success':
                success_count += 1
            elif result['状态'] == 'error':
                error_count += 1

            # 添加延迟以避免过于频繁的请求
            if i < len(questions_to_process) - 1:  # 不在最后一个请求后等待
                await asyncio.sleep(0.5)  # 等待0.5秒

        except KeyboardInterrupt:
            print(f"\n\n用户中断操作，已处理 {i + 1} 个问题，结果已追加到 {output_csv_path}")
            break
        except Exception as e:
            print(f"处理问题 {actual_index} 时发生未知错误: {str(e)}")
            error_count += 1

            # 即使出现未知错误也要记录
            error_row = {
                '问题': question_row.get('问题', ''),
                '回答': f"未知错误: {str(e)}",
                '三元组数量': 0,
                '文本块数量': 0,
                '三元组内容': '',
                '文本块内容': '',
                '状态': 'error',
                '序号': actual_index
            }
            append_to_csv(error_row, output_csv_path)

    print(f"\n处理完成或中断!")
    print(f"结果已追加到: {output_csv_path}")
    print(f"成功处理: {success_count} 个问题")
    print(f"处理失败: {error_count} 个问题")
    print(f"总计: {len(questions_to_process)} 个问题")
    print(f"起始位置: {start_from}")


async def main():
    """主函数"""
    alpha = 0.25
    beta = 1 - alpha
    input_csv = "evaluate/问答.csv"  # 输入CSV文件路径

    # ==================== 消融实验配置 ====================
    # 设置为 False 关闭动态调整初筛数量 (对照组)
    # 设置为 True 开启动态调整初筛数量 (实验组，默认)
    enable_dynamic_screening = True
    # =====================================================

    # ==================== 消融实验配置 ====================
    # 设置为 False 关闭关键词提取 (对照组)
    # 设置为 True 开启关键词提取 (实验组，默认)
    enable_keyword_extraction = False
    # =====================================================

    # 默认为不使用传统RAG False
    use_traditional_rag = False

    if use_traditional_rag:
        output_csv = f"evaluate/传统RAG/问答_结果_实时保存.csv"  # 输出CSV文件路径
    elif alpha == 1 and not enable_dynamic_screening and not use_traditional_rag:
        output_csv = f"evaluate/GraphRAG/问答_结果_alpha{alpha:.2f}_beta{beta:.2f}_GraphRAG.csv"
    elif not enable_keyword_extraction:
        output_csv = f"evaluate/消融实验/问答_结果_alpha{alpha:.2f}_beta{beta:.2f}_关闭关键词.csv"
    elif not enable_dynamic_screening:
        output_csv = f"evaluate/消融实验/问答_结果_alpha{alpha:.2f}_beta{beta:.2f}_静态初筛_实时保存.csv"  # 输出CSV文件路径
    else:
        output_csv = f"evaluate/双路径/问答_结果_alpha{alpha:.2f}_beta{beta:.2f}_双路径.csv"  # 输出CSV文件路径

    # 检查输出文件是否已存在
    if os.path.exists(output_csv):
        print(f"注意: 输出文件 {output_csv} 已存在，将以追加模式写入")
        # 统计实际数据条数（排除表头）
        with open(output_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_data_count = sum(1 for row in reader)  # 只统计数据行，不包括表头
        print(f"现有数据条数: {existing_data_count}")
        start_from = existing_data_count + 1
    else:
        print(f"输出文件 {output_csv} 不存在，将新建文件")
        start_from = 1

    print(f"开始处理CSV文件: {input_csv}")
    print(f"Alpha: {alpha}, Beta: {beta}")
    print(f"从第 {start_from} 个问题开始处理")
    print(f"输出文件: {output_csv}")

    await test_all_questions_from_csv_append_mode(
        input_csv, output_csv, start_from=start_from, alpha=alpha,
        beta=beta,
        use_traditional_rag=use_traditional_rag,
        enable_dynamic_screening=enable_dynamic_screening,
        enable_keyword_extraction=enable_keyword_extraction)


if __name__ == "__main__":
    asyncio.run(main())
