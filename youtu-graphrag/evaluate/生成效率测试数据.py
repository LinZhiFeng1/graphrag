import json
import os
import math

# === 配置文件名 ===
FILE_CH2 = "D:\桌面文件\Program\youtu-graphrag\youtu-graphrag\data\\uploaded\\aviation_仅第二章\corpus.json"  # 你的第二章文件（基座）
FILE_CH4 = "D:\桌面文件\Program\youtu-graphrag\youtu-graphrag\data\\uploaded\\aviation_第四章\\aviation.json"  # 你的第四章文件（增量源）
OUTPUT_DIR = "experiment_datasets"  # 输出目录


def main():
    # 1. 检查并读取两个源文件
    if not os.path.exists(FILE_CH2) or not os.path.exists(FILE_CH4):
        print(f"❌ 错误：请确保当前目录下存在 {FILE_CH2} 和 {FILE_CH4}")
        return

    print(f"📖 正在读取源文件...")
    with open(FILE_CH2, 'r', encoding='utf-8') as f:
        data_ch2 = json.load(f)
    with open(FILE_CH4, 'r', encoding='utf-8') as f:
        data_ch4 = json.load(f)

    print(f"   - 第二章 (基座): {len(data_ch2)} 条数据")
    print(f"   - 第四章 (增量): {len(data_ch4)} 条数据")

    # 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 生成梯度数据 (20% -> 100%)
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    print("\n🚀 开始生成实验数据文件...")

    for ratio in ratios:
        pct = int(ratio * 100)

        # 计算第四章的切片长度 (向上取整)
        # 比如 28条 * 0.2 = 5.6 -> 取前6条
        count = math.ceil(len(data_ch4) * ratio)

        # === 核心逻辑 ===
        # 切片：取第四章的前 count 条
        inc_slice = data_ch4[:count]

        # 拼接：第二章完整版 + 第四章切片
        full_combined = data_ch2 + inc_slice

        # === A组：增量构建专用文件 (Incremental) ===
        # 场景：你已经跑完了第二章，现在只想单独上传这一小部分增量
        # 文件名示例: Inc_Only_20pct.json
        filename_inc = f"Inc_Only_{pct}pct.json"
        path_inc = os.path.join(OUTPUT_DIR, filename_inc)
        with open(path_inc, 'w', encoding='utf-8') as f:
            json.dump(inc_slice, f, ensure_ascii=False, indent=2)

        # === B组：全量构建专用文件 (Full Rebuild) ===
        # 场景：你把以前的图谱全删了，想把两章内容一次性跑完
        # 文件名示例: Full_Combined_20pct.json
        filename_full = f"Full_Combined_{pct}pct.json"
        path_full = os.path.join(OUTPUT_DIR, filename_full)
        with open(path_full, 'w', encoding='utf-8') as f:
            json.dump(full_combined, f, ensure_ascii=False, indent=2)

        print(f"  [进度 {pct}%] 增量包: {len(inc_slice)}条 | 全量包: {len(full_combined)}条")

    print(f"\n✅ 全部完成！文件已保存在 '{OUTPUT_DIR}' 文件夹中。")
    print("👉 接下来去前端界面，按顺序上传这些文件进行测试即可。")


if __name__ == "__main__":
    main()