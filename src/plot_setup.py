"""matplotlib 全局设置 —— 让图中中文正常显示。

在所有绘图脚本中:
    import matplotlib
    matplotlib.use("Agg")
    import plot_setup   # 必须在 import matplotlib.pyplot 之前
"""
import matplotlib

# Windows 常见中文字体,按优先级取第一个可用的
for font in ("Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [font] + list(matplotlib.rcParams["font.sans-serif"])
        break
    except KeyError:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
