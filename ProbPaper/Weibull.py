# %%
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Weibull逆関数
def Weibull(x):
    return np.log(np.log(1/(1-x)))

# 回帰式
def func(x, a, b):
    f = a*x + b
    return f

# データの読み込み
Qin = np.loadtxt('./Data.csv', delimiter=',', usecols=[0])
# 0以下の削除
Qin = Qin[Qin > 0]
Qin = np.sort(Qin)

# データサイズの取得
max = Qin.size
print("データ数", max)

# メジアンランク法
Pin = np.empty(max)
for i in range(max):
    Pin[i] = (i+0.7) / (max+0.4)

# 回帰分析用
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
tss = np.sum((ppp-np.mean(ppp))**2)
r_squared = 1 - (rss / tss)

# 重複する値を除く
for i in range(max-2, 0, -1):
    if(Qin[i] == Qin[i+1]):
        Pin[i] = Pin[i+1]

# データをファイル出力する
Data = [Qin, Pin]
Data = np.array(Data).T
np.savetxt("./tmp/Prob.dat", Data, delimiter="\t")

# 正規分布の値を取得
ppp = Weibull(Pin)
qqq = np.log(Qin)

# 図の書式
plt.rcParams['font.family'] = 'Times New Roman'
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
    if(10**i < Qin[0] < 10**(i+1)):
        _xmin = (10**i)

    if(10**i < Qin[max-1] < 10**(i+1)):
        _xmax = (10**(i+1))

xmin = np.log(_xmin) - (np.log(_xmax)-np.log(_xmin))/100
xmax = np.log(_xmax) + (np.log(_xmax)-np.log(_xmin))/100

# y軸の最大・最小
ymin = Weibull(0.001)
ymax = Weibull(0.999)

# 図の描画範囲
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# 図の表示書式
# ax.tick_params(labelbottom=False)
# ax.tick_params(labelleft=False)
# ax.tick_params(bottom=False)
ax.tick_params(direction="inout", axis="x", length=2, width=0)
ax.tick_params(direction="out", axis="y", length=2, width=0)
ax.tick_params(direction="inout", axis="x", which="minor", length=2, width=0)
ax.tick_params(which="minor", left=False)
ax.minorticks_on()


# y軸目盛用
_dy = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.999])
dy = Weibull(_dy)
_dy = _dy * 100

# 水平軸の描画用
_dy_tick = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.999])
dy_tick = Weibull(_dy_tick)

# 水平軸の描画
plt.hlines(dy_tick, xmin, xmax, color='mediumpurple', linewidth=0.1)
plt.hlines(0, xmin, xmax, color='black', linewidth=0.1)

# x軸の目盛ラベル
_dx = np.array([_xmin])
for i in range(1, 10):
    if(_xmin*10**i <= _xmax):
        _dx = np.append(_dx, _xmin*10**i)

dx = np.log(_dx)

# 鉛直軸
# 鉛直軸の準備
_dx_tick_pre = np.array([_xmin, _xmin*5])
# 実際の表示用配列
_dx_tick = np.array([_xmin, _xmin*5])
for i in range(1, 10):
    if(_xmin*10**i < _xmax):
        _dx_tick = np.append(_dx_tick, _dx_tick_pre*10**i)
    # xmaxのみ一つ追加
    if(_xmin*10**i == _xmax):
        _dx_tick = np.append(_dx_tick, _xmin*10**i)

# 副目盛
_dx_tick_pre = np.array([_xmin*2, _xmin*3, _xmin*4, _xmin*5])
# 実際の表示用配列
_dx_tick_sub = np.array([_xmin*2, _xmin*3, _xmin*4, _xmin*5])
for i in range(1, 10):
    if(_xmin*10**i < _xmax):
        _dx_tick_sub = np.append(_dx_tick_sub, _dx_tick_pre*10**i)

dx_tick = np.log(_dx_tick)
dx_tick_sub = np.log(_dx_tick_sub)

# 鉛直軸の描画
plt.vlines(dx_tick, ymin, ymax, color='mediumpurple', linewidth=0.1)
plt.vlines(0.0, ymin, ymax, color='black', linewidth=0.1)

# x軸目盛
# for i in range(7):
#     plt.text(_dx[i], ymin-0.1, str(round(_dx[i],2)), ha = 'center', va = 'top', fontsize=fs)
ax.get_xaxis().set_tick_params(pad=1)
ax.set_xticks(dx)
ax.set_xticklabels(np.round(_dx, 2), fontsize=5)
# 副目盛表示
ax.set_xticks(dx_tick_sub, minor=True)

# y軸目盛の値
# for i in range(9):
#     ax.text(xmin- (xmax-xmin)/100, dy[i], str(_dy[i]), ha = 'right', va = 'center', fontsize=fs)
ax.get_yaxis().set_tick_params(pad=1)
ax.set_yticks(dy)
ax.set_yticklabels(_dy, fontsize=5)

# 値のプロット
ax.scatter(qqq, ppp, s=2, alpha=0.8, linewidths=0.1, c="mediumslateblue", ec="navy", zorder=10)
ax.plot([xmin, xmax], [aa*xmin + bb, aa*xmax + bb], color='k', linestyle='-', linewidth=0.25, zorder=9)

# 文字のプロット
ax.text(xmin, ymax + (ymax-ymin)/50, "F(t) (%)   Median Ranks", ha='left', va='bottom', fontsize=5)

# 有効データ数
max = np.unique(Qin).size
print("有効データ数 = ", max)

# 統計量計算
m = aa
n = math.exp(-bb/aa)
print('形状パラメータ={m:10.6f}'.format(**locals()))
print('尺度パラメータ={n:10.6f}'.format(**locals()))

mean = n * math.gamma(1+1/m)
var = math.sqrt(n*n*(math.gamma(1+2/m) - (math.gamma(1+1/m))**2))
print('平均={mean:10.6f}'.format(**locals()))
print('標準偏差={var:10.6f}'.format(**locals()))

print('相関係数={rr[0][1]:10.6f}'.format(**locals()))
print('決定係数={r_squared:10.6f}'.format(**locals()))
plt.show()

# %%
pdf = PdfPages('./img/Weibull.pdf')
pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
pdf.close()
