# %%
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Weibull逆関数
def Weibull(x):
    return np.log(np.log(1 / (1 - x)))

# 回帰式
def func(x, a, b):
    f = a * x + b
    return f

# データの読み込み
Qin = np.loadtxt('./Data.csv', delimiter=',', usecols=[0])

# データサイズの取得
max = Qin.size
print("0以下を含むデータ数", max)

# 0以下の削除(対数のため)
Qin = Qin[Qin > 0]
Qin = np.sort(Qin)

# データサイズの取得
max = Qin.size
print("データ数", max)

# メジアンランク法
Pin = np.empty(max)
for i in range(max):
    Pin[i] = (i + 0.7) / (max + 0.4)

# 重複する値の設定
for i in range(max - 2, 0, -1):
    if(Qin[i] == Qin[i + 1]):
        Pin[i] = Pin[i + 1]

# 重複する値の削除
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
ppp = Weibull(Pin)
qqq = np.log(Qin)

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
for i in range(-5, 5):
    if(10**i < Qin[0] < 10**(i + 1)):
        _xmin = (10**i)

    if(10**i < Qin[max - 1] < 10**(i + 1)):
        _xmax = (10**(i + 1))

xmin = np.log(_xmin) - (np.log(_xmax) - np.log(_xmin)) / 100
xmax = np.log(_xmax) + (np.log(_xmax) - np.log(_xmin)) / 100

# y軸の最大・最小
if(Pin[0] < 0.001):
    ymin = Weibull(0.0001)
else:
    ymin = Weibull(0.001)

ymax = Weibull(0.999)

# 図の描画範囲
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# 図の表示書式
ax.tick_params(direction="inout", length=2, width=0.1)
ax.tick_params(direction="inout", axis="x", which="minor", length=2, width=0.1)
ax.tick_params(direction="inout", axis="y", which="minor", length=2, width=0.1)

# y軸目盛用
if(Pin[0] < 0.001):
    _dy = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.999])
else:
    _dy = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.999])

dy = Weibull(_dy)
_dy = _dy * 100

# y軸副目盛用
if(Pin[0] < 0.001):
    _dy_tick_sub = np.array([0.0003, 0.0004, 0.003, 0.004, 0.03, 0.04, 0.3, 0.4])
else:
    _dy_tick_sub = np.array([0.003, 0.004, 0.03, 0.04, 0.3, 0.4])

dy_tick_sub = Weibull(_dy_tick_sub)

# 水平軸の描画
plt.hlines(dy, xmin, xmax, color='mediumpurple', linewidth=0.1)
plt.hlines(0, xmin, xmax, color='black', linewidth=0.1)

# x軸の目盛ラベル
_dx = np.array([_xmin], dtype="float")
for i in range(1, 10):
    if(_xmin * 10**i <= _xmax):
        _dx = np.append(_dx, _xmin * 10**i)

dx = np.log(_dx)

# 鉛直軸
# 鉛直軸の準備
_dx_tick_pre = np.array([_xmin, _xmin * 5])
# 実際の表示用配列
_dx_tick = np.array([_xmin, _xmin * 5])
for i in range(1, 10):
    if(_xmin * 10**i < _xmax):
        _dx_tick = np.append(_dx_tick, _dx_tick_pre * 10**i)
    # xmaxのみ一つ追加
    if(_xmin * 10**i == _xmax):
        _dx_tick = np.append(_dx_tick, _xmin * 10**i)

# 副目盛
_dx_tick_pre = np.array([_xmin * 2, _xmin * 3, _xmin * 4, _xmin * 5])
# 実際の表示用配列
_dx_tick_sub = np.array([_xmin * 2, _xmin * 3, _xmin * 4, _xmin * 5])
for i in range(1, 10):
    if(_xmin * 10**i < _xmax):
        _dx_tick_sub = np.append(_dx_tick_sub, _dx_tick_pre * 10**i)

dx_tick = np.log(_dx_tick)
dx_tick_sub = np.log(_dx_tick_sub)

# 鉛直軸の描画
plt.vlines(dx_tick, ymin, ymax, color='mediumpurple', linewidth=0.1)
plt.vlines(0.0, ymin, ymax, color='black', linewidth=0.15)

# x軸目盛
ax.get_xaxis().set_tick_params(pad=1)
ax.set_xticks(dx)
ax.set_xticklabels(np.round(_dx, 2), fontsize=5)
# 副目盛表示
ax.set_xticks(dx_tick_sub, minor=True)

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
if(Pin[0] < 0.001):
    _dy_right = np.arange(-9, 2)
else:
    _dy_right = np.arange(-6, 2)
# 目盛の描画
ax_.get_yaxis().set_tick_params(pad=1)
ax_.set_yticks(_dy_right)
ax_.set_yticklabels(_dy_right, fontsize=4)

# 上側目盛の値
secax = ax.secondary_xaxis('top')
secax.spines['top'].set_linewidth(0)
secax.tick_params(direction="inout", length=2, width=0.1)
_dx_top = np.arange(-10, 10)
# 目盛の描画
secax.get_xaxis().set_tick_params(pad=1)
secax.set_xticks(_dx_top)
secax.set_xticklabels(_dx_top, fontsize=4)
# 副目盛表示
secax.tick_params(direction="inout", axis="x", which="minor", length=1, width=0.1)
_dx_top_sub = np.arange(-9.5, 10.5)
secax.set_xticks(_dx_top_sub, minor=True)

# 値のプロット
ax.scatter(qqq, ppp, s=2, alpha=0.7, linewidths=0.2, c="mediumslateblue", ec="navy", zorder=10)
ax.plot([qqq[0], qqq[max - 1]], [aa * qqq[0] + bb, aa * qqq[max - 1] + bb], color='navy', linestyle='-', linewidth=0.3, zorder=9)

# 文字のプロット
ax.text(xmin - (xmax - xmin) / 13, ymax + (ymax - ymin) / 50, "　　　メジアンランク法\nF(t)　(%)", ha='left', va='bottom', font="IPAexGothic", fontsize=4.5)

# 統計量計算
m = aa
n = math.exp(-bb / aa)
mean = n * math.gamma(1 + 1 / m)
var = math.sqrt(n * n * (math.gamma(1 + 2 / m) - (math.gamma(1 + 1 / m))**2))
# 統計量表示
print('形状パラメータ={m:10.6f}'.format(**locals()))
print('尺度パラメータ={n:10.6f}'.format(**locals()))
print('平均={mean:10.6f}'.format(**locals()))
print('標準偏差={var:10.6f}'.format(**locals()))

#　Boxのプロット
boxdic = {
    "facecolor": "white",
    "edgecolor": "navy",
    "boxstyle": "square",
    "linewidth": 0.15,
}
_gamma = 0.0
ax.text(xmin + (xmax - xmin) / 45, ymax - (ymax - ymin) / 8.5,
        "\t$m$ ={m:10.4f}\n\t $\eta$ ={n:10.4f}\n\t $\gamma$ ={_gamma:10.4f}\nMTTF ={mean:10.4f}".format(**locals()), fontsize=4, bbox=boxdic)

# Weibullの軸
wei_line = 0 - (xmax - xmin) / 10
plt.hlines(-m, -m / aa + wei_line, xmax, color='black', linewidth=0.15)
cc = -aa * wei_line
ax.plot([-m / aa + wei_line, wei_line], [-m, 0.0], color='black', linestyle='-', linewidth=0.15)

# 相関距離の表示
print('相関係数={rr[0][1]:10.6f}'.format(**locals()))
print('決定係数={r_squared:10.6f}'.format(**locals()))
plt.show()

# %%
pdf = PdfPages('./img/Weibull.pdf')
pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
pdf.close()
