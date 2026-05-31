import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

x = np.linspace(-6, 6, 500)

silu = x * (1 / (1 + np.exp(-x)))
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, silu, 'b-', linewidth=2.5, label='SiLU = x · sigmoid(x)')
ax.plot(x, relu, 'r--', linewidth=1.8, alpha=0.6, label='ReLU = max(0, x)')
ax.plot(x, sigmoid, 'g:', linewidth=1.8, alpha=0.6, label='sigmoid(x)')

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

ax.annotate('负值区域\n小凹陷 ≈ -0.28',
            xy=(-1.28, -0.28), xytext=(-4.5, -1.2),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            color='blue')

ax.annotate('平滑过渡\n没有硬折角',
            xy=(0, 0), xytext=(3, -1.2),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            color='blue')

ax.annotate('ReLU 硬折角',
            xy=(0, 0), xytext=(3, -2),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            color='red')

ax.set_xlim(-6, 6)
ax.set_ylim(-2.5, 6)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('y', fontsize=13)
ax.set_title('SiLU (x · sigmoid(x)) vs ReLU vs sigmoid', fontsize=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:/Users/63258/Desktop/learning-note/learning-docs/llm/silu_plot.png', dpi=150, bbox_inches='tight')
print("图片已保存")
