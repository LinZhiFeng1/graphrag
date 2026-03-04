import matplotlib.pyplot as plt

# 数据
labels = ['逻辑与推理', '事实与参数', '结构与属性']
sizes = [48.0, 28.0, 24.0] # Based on the count: 12, 7, 6
colors = ['#5c6bc0', '#26a69a', '#ef5350'] # 科研常用冷暖色调搭配
explode = (0.02, 0.02, 0.02)  # 突出显示占比最大的部分

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                  shadow=False, startangle=140, colors=colors,
                                  textprops={'color':"black"}, pctdistance=0.85)

# # 绘制中心圆，做成甜甜圈图，看起来更现代
# centre_circle = plt.Circle((0,0),0.60,fc='white')
# fig.gca().add_artist(centre_circle)


# 优化字体显示
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

ax.axis('equal')
# plt.title("Distribution of Question Types in Aviation-Maintenance Bench", y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()

# 保存
plt.savefig('question_type_distribution.png', dpi=300, bbox_inches='tight')
print("Pie chart generated.")