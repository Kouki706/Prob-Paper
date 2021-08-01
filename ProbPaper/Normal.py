# %%
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 回帰式
def func(x, a, b):
    f = a * x + b
    return f

# データの読み込み
Qin = np.loadtxt('./Data.csv', delimiter=',', usecols=[0])
Qin = np.sort(Qin)

# データサイズの取得
max = Qin.size
print("データ数", max)

# メジアンランク法
Pin = np.empty(max)
for i in range(max):
    Pin[i] = (i + 0.7) / (max + 0.4)

# 重複する値の判定
for i in range(max - 2, 0, -1):
    if(Qin[i] == Qin[i + 1]):
        Pin[i] = Pin[i + 1]

# 重複する値を除く
Pin = np.unique(Pin)
Qin = np.unique(Qin)

# データをファイル出力する
Data = [Qin, Pin]
Data = np.array(Data).T
np.savetxt("./tmp/Prob.dat", Data, delimiter="\t")

# 有効データ数のデータサイズの取得
max = Qin.size
print("有効データ数 = ", max)

# 正規分布の値を取得
ppp = norm.ppf(Pin, loc=0, scale=1)
qqq = Qin

# 回帰直線
popt, pcov = curve_fit(func, qqq, ppp)
rr = np.corrcoef(qqq, ppp)
aa = popt[0]
bb = popt[1]

# 決定係数
residuals = ppp - func(qqq, popt[0], popt[1])
rss = np.sum(residuals**2)
tss = np.sum((ppp - np.mean(ppp))**2)
r_squared = 1 - (rss / tss)

# 図の書式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
fig = plt.figure(figsize=(4, 3))  # Figure
ax = fig.add_subplot()  # Axes
ax.patch.set_facecolor('lavender')  # subplotの背景色
ax.patch.set_alpha(0.2)  # subplotの背景透明度
ax.spines['top'].set_linewidth(0.1)
ax.spines['right'].set_linewidth(0.1)
ax.spines['left'].set_linewidth(0.1)
ax.spines['bottom'].set_linewidth(0.1)

# x軸の最大・最小
xmin = qqq[0] - (qqq[max - 1] - qqq[0]) / 100
xmax = qqq[max - 1] + (qqq[max - 1] - qqq[0]) / 100

# y軸の最大・最小
ymin = norm.ppf(0.0001, loc=0, scale=1)
ymax = norm.ppf(0.9999, loc=0, scale=1)

# 図の描画範囲
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# 図の表示書式
ax.tick_params(direction="inout", length=2, width=0.1)
ax.tick_params(direction="inout", axis="x", which="minor", length=2, width=0.1)
ax.tick_params(direction="inout", axis="y", which="minor", length=2, width=0.1)

# y軸目盛用
_dy = np.array([0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999])
dy = norm.ppf(_dy, loc=0, scale=1)
_dy = _dy * 100

# 水平軸の描画用
_dy_tick = np.array([0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999])
dy_tick = norm.ppf(_dy_tick, loc=0, scale=1)
# 水平軸の描画
ax.hlines(dy_tick, xmin, xmax, color='mediumpurple', linewidth=0.1)
ax.hlines(dy_tick[7], xmin, xmax, color='black', linewidth=0.1)
# 水平副目盛
_dy_tick_sub = np.array([0.0005, 0.005, 0.05, 0.3, 0.4, 0.6, 0.7])
dy_tick_sub = norm.ppf(_dy_tick_sub, loc=0, scale=1)

# x軸の目盛
_dx = np.empty(7)
_dx[0] = qqq[0]
_dx[6] = qqq[max - 1]

# x軸の表示目盛の計算
ddx = (_dx[6] - _dx[0]) / 6
for i in range(1, 6, 1):
    _dx[i] = _dx[0] + ddx * i

# 鉛直軸の描画
ax.vlines(_dx, ymin, ymax, color='mediumpurple', linewidth=0.1)

# x軸目盛
ax.get_xaxis().set_tick_params(pad=1)
ax.set_xticks(_dx)
ax.set_xticklabels(np.round(_dx, 1), fontsize=5)

# y軸目盛の値
ax.get_yaxis().set_tick_params(pad=1)
ax.set_yticks(dy)
ax.set_yticklabels(_dy, fontsize=5)
# 副目盛表示
ax.set_yticks(dy_tick_sub, minor=True)

# 右側目盛の値
ax_ = ax.twinx()
ax_.spines['top'].set_linewidth(0)
ax_.spines['right'].set_linewidth(0)
ax_.spines['left'].set_linewidth(0)
ax_.spines['bottom'].set_linewidth(0)
ax_.set_ylim([ymin, ymax])
ax_.tick_params(direction="inout", length=2, width=0.1)
# 平均+-の値
_dy_right = np.array([0.00135, 0.02275, 0.15865, 0.5, 0.84135, 0.97725, 0.99865])
dy_right = norm.ppf(_dy_right, loc=0, scale=1)
# 軸の表示
ax_.get_yaxis().set_tick_params(pad=1)
ax_.set_yticks(dy_right)
ax_.set_yticklabels(["$\mu-3\sigma$", "$\mu-2\sigma$", "$\mu-\sigma$", "$\mu$", "$\mu+\sigma$", "$\mu+2\sigma$", "$\mu+3\sigma$"], fontsize=4)

# 値のプロット
ax.scatter(qqq, ppp, s=2, alpha=0.7, linewidths=0.2, c="mediumslateblue", ec="navy", zorder=10)
ax.plot([qqq[0], qqq[max - 1]], [aa * qqq[0] + bb, aa * qqq[max - 1] + bb], color='navy', linestyle='-', linewidth=0.3, zorder=9)

# 文字のプロット
ax.text(xmin - (xmax - xmin) / 13, ymax + (ymax - ymin) / 50, "　　　メジアンランク法\nF(t)　(%)", ha='left', va='bottom', font="IPAexGothic", fontsize=4.5)

# 平均・標準偏差
var = 1 / aa
mean = -bb / aa
print('平均 = {mean:10.6f}'.format(**locals()))
print('標準偏差 = {var:10.6f}'.format(**locals()))

# 図中に値のプロット
boxdic = {
    "facecolor": "white",
    "edgecolor": "navy",
    "boxstyle": "square",
    "linewidth": 0.15,
}
_gamma = 0.0
ax.text(xmin + (xmax - xmin) / 45, ymax - (ymax - ymin) / 8.5,
        "\t $\mu$ ={mean:15.4f}\n\t $\sigma$ ={var:15.4f}\n\t $\gamma$ ={_gamma:15.4f}\nMTTF ={mean:15.4f}".format(**locals()), fontsize=4, bbox=boxdic)

# 相関係数の表示
print('相関係数 = {rr[0][1]:10.6f}'.format(**locals()))
print('決定係数 = {r_squared:10.6f}'.format(**locals()))

# Jupyterの表示
plt.show()

# %%
pdf = PdfPages('./img/Normal.pdf')
pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
pdf.close()
