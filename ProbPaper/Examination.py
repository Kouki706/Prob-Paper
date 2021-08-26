# %%
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 回帰式
def func(x, a, b):
    f = a*x + b
    return f

# Weibull逆関数
def Weibull(x):
    return np.log(np.log(1/(1-x)))

# 極値（最大）分布
def Gumbel(x):
    return -np.log(-np.log(x))

# データの読み込み
Qin = np.loadtxt('./Data.csv', delimiter=',', usecols=[0])
Qin = np.sort(Qin)

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
    Pin[i] = (i+0.7) / (max+0.4)

# 重複する値の判定
for i in range(max-2, 0, -1):
    if(Qin[i] == Qin[i+1]):
        Pin[i] = Pin[i+1]

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

# 正規分布について

# 正規分布の値を取得
ppp = norm.ppf(Pin, loc=0, scale=1)
qqq = Qin

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
tss = np.sum((ppp-np.mean(ppp))**2)
r_squared = 1 - (rss / tss)

# 図の書式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
fig = plt.figure(figsize=(8, 6), tight_layout=True)  # Figure
ax1 = fig.add_subplot(2, 2, 4)  # Axes
ax1.patch.set_facecolor('lavender')  # subplotの背景色
ax1.patch.set_alpha(0.2)  # subplotの背景透明度
ax1.spines['top'].set_linewidth(0.1)
ax1.spines['right'].set_linewidth(0.1)
ax1.spines['left'].set_linewidth(0.1)
ax1.spines['bottom'].set_linewidth(0.1)

# x軸の最大・最小
xmin = qqq[0] - (qqq[max-1]-qqq[0])/100
xmax = qqq[max-1] + (qqq[max-1]-qqq[0])/100

# y軸の最大・最小
if(Pin[0] > 0.001):
    ymin = norm.ppf(0.001, loc=0, scale=1)
elif(Pin[0] > 0.0001):
    ymin = norm.ppf(0.0001, loc=0, scale=1)
else:
    ymin = norm.ppf(0.00001, loc=0, scale=1)

ymax = norm.ppf(0.999, loc=0, scale=1)

# 図の描画範囲
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])

# 図の表示書式
ax1.tick_params(direction="inout", length=4, width=0.1)
ax1.tick_params(direction="inout", axis="x", which="minor", length=4, width=0.1)
ax1.tick_params(direction="inout", axis="y", which="minor", length=4, width=0.1)

# y軸目盛用
if(Pin[0] > 0.001):
    _dy = np.array([0.001, 0.01, 0.1, 0.5, 0.999])
elif(Pin[0] > 0.0001):
    _dy = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 0.999])
else:
    _dy = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.999])

dy = norm.ppf(_dy, loc=0, scale=1)
_dy = _dy * 100

# 水平軸の描画用
_dy_tick = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999])
dy_tick = norm.ppf(_dy_tick, loc=0, scale=1)
# 水平軸の描画
ax1.hlines(dy_tick, xmin, xmax, color='mediumslateblue', alpha=0.5, linewidth=0.1)
# 水平副目盛
if(Pin[0] > 0.001):
    _dy_tick_sub = np.array([0.005, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99])
elif(Pin[0] > 0.0001):
    _dy_tick_sub = np.array([0.0005, 0.005, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99])
else:
    _dy_tick_sub = np.array([0.00005, 0.0005, 0.005, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99])
dy_tick_sub = norm.ppf(_dy_tick_sub, loc=0, scale=1)

# x軸の目盛
_dx = np.empty(7)
_dx[0] = qqq[0]
_dx[6] = qqq[max-1]

# x軸の表示目盛の計算
ddx = (_dx[6]-_dx[0])/6
for i in range(1, 6, 1):
    _dx[i] = _dx[0] + ddx * i

# 鉛直軸の描画
ax1.vlines(_dx, ymin, ymax, color='mediumslateblue', alpha=0.5, linewidth=0.1)

# x軸目盛
ax1.get_xaxis().set_tick_params(pad=1)
ax1.set_xticks(_dx)
ax1.set_xticklabels(np.round(_dx, 1), fontsize=5)

# y軸目盛の値
ax1.get_yaxis().set_tick_params(pad=1)
ax1.set_yticks(dy)
ax1.set_yticklabels(_dy, fontsize=5)
# 副目盛表示
ax1.set_yticks(dy_tick_sub, minor=True)

# 値のプロット
ax1.scatter(qqq, ppp, s=8, marker=",", alpha=0.8, linewidths=0.1, c="fuchsia", ec="green", zorder=10)
ax1.plot([qqq[0], qqq[max - 1]], [aa * qqq[0] + bb, aa * qqq[max - 1] + bb], color='navy', linestyle='-', linewidth=0.5, zorder=9)

# 文字のプロット
ax1.text(xmin - (xmax-xmin)/35, ymax + (ymax-ymin)/50, "F(t)　　　正規確率紙", ha='left', va='bottom', font="IPAexGothic", fontsize=6)

# 統計量の計算
var = 1/aa
mean = -bb/aa
print("\n正規分布")
# 統計量の表示
print('平均 = {mean:10.6f}'.format(**locals()))
print('標準偏差 = {var:10.6f}'.format(**locals()))
print('相関係数 = {rr[0][1]:10.6f}'.format(**locals()))
print('決定係数 = {r_squared:10.6f}'.format(**locals()))

# 対数正規分布について

# 正規分布の値を取得
ppp = norm.ppf(Pin, loc=0, scale=1)
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

# 図の書式
ax2 = fig.add_subplot(2, 2, 3)  # Axes
ax2.patch.set_facecolor('lavender')  # subplotの背景色
ax2.patch.set_alpha(0.2)  # subplotの背景透明度
ax2.spines['top'].set_linewidth(0.1)
ax2.spines['right'].set_linewidth(0.1)
ax2.spines['left'].set_linewidth(0.1)
ax2.spines['bottom'].set_linewidth(0.1)

# x軸の最大・最小
for i in range(-10, 10):
    if(10**i < Qin[0] < 10**(i+1)):
        _xmin = (10**i)

    if(10**i < Qin[max-1] < 10**(i+1)):
        _xmax = (10**(i+1))

xmin = np.log(_xmin) - (np.log(_xmax)-np.log(_xmin))/100
xmax = np.log(_xmax) + (np.log(_xmax)-np.log(_xmin))/100

# y軸の最大・最小
# 正規分布と同様

# 図の描画範囲
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])

# 図の表示書式
ax2.tick_params(direction="inout", length=4, width=0.1)
ax2.tick_params(direction="inout", axis="x", which="minor", length=4, width=0.1)
ax2.tick_params(direction="inout", axis="y", which="minor", length=4, width=0.1)

# y軸目盛用
# 正規分布と同様

# 水平軸の描画用
# 正規分布と同様
# 水平軸の描画
plt.hlines(dy_tick, xmin, xmax, color='mediumslateblue', alpha=0.5, linewidth=0.1)
# y軸副目盛の設定
# 正規分布と同様

# x軸の目盛ラベル
_dx = np.array([_xmin])
for i in range(1, 20):
    if(_xmin*10**i <= _xmax):
        _dx = np.append(_dx, _xmin*10**i)

dx = np.log(_dx)

# 鉛直軸
# 鉛直軸の準備
_dx_tick_pre = np.array([_xmin, _xmin*5])
# 実際の表示用配列
_dx_tick = np.array([_xmin, _xmin*5])
for i in range(1, 20):
    if(_xmin*10**i < _xmax):
        _dx_tick = np.append(_dx_tick, _dx_tick_pre*10**i)
    # xmaxのみ一つ追加
    if(_xmin*10**i == _xmax):
        _dx_tick = np.append(_dx_tick, _xmin*10**i)

# 副目盛
_dx_tick_pre = np.array([_xmin*2, _xmin*3, _xmin*4, _xmin*5])
# 実際の表示用配列
_dx_tick_sub = np.array([_xmin*2, _xmin*3, _xmin*4, _xmin*5])
for i in range(1, 20):
    if(_xmin*10**i < _xmax):
        _dx_tick_sub = np.append(_dx_tick_sub, _dx_tick_pre*10**i)

dx_tick = np.log(_dx_tick)
dx_tick_sub = np.log(_dx_tick_sub)

# 鉛直軸の描画
plt.vlines(dx_tick, ymin, ymax, color='mediumslateblue', alpha=0.5, linewidth=0.1)

# x軸目盛
ax2.get_xaxis().set_tick_params(pad=1)
ax2.set_xticks(dx)
ax2.set_xticklabels(np.round(_dx, 2), fontsize=5)
# 副目盛表示
ax2.set_xticks(dx_tick_sub, minor=True)

# y軸目盛の値
ax2.get_yaxis().set_tick_params(pad=1)
ax2.set_yticks(dy)
ax2.set_yticklabels(_dy, fontsize=5)
# 副目盛表示
ax2.set_yticks(dy_tick_sub, minor=True)

# 値のプロット
ax2.scatter(qqq, ppp, s=8, marker="^", alpha=0.8, linewidths=0.1, c="lime", ec="navy", zorder=2)
ax2.plot([qqq[0], qqq[max - 1]], [aa * qqq[0] + bb, aa * qqq[max - 1] + bb], color='navy', linestyle='-', linewidth=0.5, zorder=9)

# 文字のプロット
ax2.text(xmin - (xmax-xmin)/35, ymax + (ymax-ymin)/50, "F(t)　　　対数正規確率紙", ha='left', va='bottom', font="IPAexGothic", fontsize=6)

# 統計量の計算
var_log = 1/aa
mean_log = -bb/aa
mean = math.exp(mean_log + var_log**2 / 2)
var = math.exp(2*mean_log + var_log**2)*(math.exp(var_log**2) - 1)
var = math.sqrt(var)
# 統計量の表示
print("\n対数正規分布")
print('対数平均 = {mean_log:10.6f}'.format(**locals()))
print('対数分散 = {var_log:10.6f}'.format(**locals()))
print('平均 = {mean:10.6f}'.format(**locals()))
print('標準偏差 = {var:10.6f}'.format(**locals()))
print('相関係数 = {rr[0][1]:10.6f}'.format(**locals()))
print('決定係数 = {r_squared:10.6f}'.format(**locals()))

# Weibull分布について

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
tss = np.sum((ppp-np.mean(ppp))**2)
r_squared = 1 - (rss / tss)

# 図の書式
ax3 = fig.add_subplot(2, 2, 1)  # Axes
ax3.patch.set_facecolor('lavender')  # subplotの背景色
ax3.patch.set_alpha(0.2)  # subplotの背景透明度
ax3.spines['top'].set_linewidth(0.1)
ax3.spines['right'].set_linewidth(0.1)
ax3.spines['left'].set_linewidth(0.1)
ax3.spines['bottom'].set_linewidth(0.1)

# x軸の最大・最小
for i in range(-10, 10):
    if(10**i < Qin[0] < 10**(i+1)):
        _xmin = (10**i)

    if(10**i < Qin[max-1] < 10**(i+1)):
        _xmax = (10**(i+1))

xmin = np.log(_xmin) - (np.log(_xmax)-np.log(_xmin))/100
xmax = np.log(_xmax) + (np.log(_xmax)-np.log(_xmin))/100

# y軸の最大・最小
if(Pin[0] > 0.001):
    ymin = Weibull(0.001)
elif(Pin[0] > 0.0001):
    ymin = Weibull(0.0001)
else:
    ymin = Weibull(0.00001)

ymax = Weibull(0.999)

# 図の描画範囲
ax3.set_xlim([xmin, xmax])
ax3.set_ylim([ymin, ymax])

# 図の表示書式
ax3.tick_params(direction="inout", length=4, width=0.1)
ax3.tick_params(direction="inout", axis="x", which="minor", length=4, width=0.1)
ax3.tick_params(direction="inout", axis="y", which="minor", length=4, width=0.1)

# y軸目盛用
if(Pin[0] > 0.001):
    _dy = np.array([0.001, 0.01, 0.1, 0.5, 0.999])
elif(Pin[0] > 0.0001):
    _dy = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 0.999])
else:
    _dy = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.999])
dy = Weibull(_dy)
_dy = _dy * 100

# 水平軸の描画用
_dy_tick = np.array([0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95])
dy_tick = Weibull(_dy_tick)
# 水平軸の描画
plt.hlines(dy_tick, xmin, xmax, color='mediumslateblue', alpha=0.5, linewidth=0.1)

# y軸副目盛用
if(Pin[0] > 0.001):
    _dy_tick_sub = np.array([0.002, 0.003, 0.004, 0.005, 0.02, 0.03, 0.04, 0.05, 0.2, 0.3, 0.4, 0.5, 0.8, 0.95])
elif(Pin[0] > 0.0001):
    _dy_tick_sub = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.002, 0.003, 0.004, 0.005, 0.02, 0.03, 0.04, 0.05, 0.2, 0.3, 0.4, 0.5, 0.8, 0.95])
else:
    _dy_tick_sub = np.array([0.00002, 0.00003, 0.00004, 0.00005, 0.0002, 0.0003, 0.0004, 0.0005, 0.002, 0.003, 0.004, 0.005, 0.02, 0.03, 0.04, 0.05, 0.2, 0.3, 0.4, 0.5, 0.8, 0.95])
dy_tick_sub = Weibull(_dy_tick_sub)

# x軸の目盛ラベル
_dx = np.array([_xmin])
for i in range(1, 20):
    if(_xmin*10**i <= _xmax):
        _dx = np.append(_dx, _xmin*10**i)

dx = np.log(_dx)

# 鉛直軸
# 鉛直軸の準備
_dx_tick_pre = np.array([_xmin, _xmin*5])
# 実際の表示用配列
_dx_tick = np.array([_xmin, _xmin*5])
for i in range(1, 20):
    if(_xmin*10**i < _xmax):
        _dx_tick = np.append(_dx_tick, _dx_tick_pre*10**i)
    # xmaxのみ一つ追加
    if(_xmin*10**i == _xmax):
        _dx_tick = np.append(_dx_tick, _xmin*10**i)

# 副目盛
_dx_tick_pre = np.array([_xmin*2, _xmin*3, _xmin*4, _xmin*5])
# 実際の表示用配列
_dx_tick_sub = np.array([_xmin*2, _xmin*3, _xmin*4, _xmin*5])
for i in range(1, 20):
    if(_xmin*10**i < _xmax):
        _dx_tick_sub = np.append(_dx_tick_sub, _dx_tick_pre*10**i)

dx_tick = np.log(_dx_tick)
dx_tick_sub = np.log(_dx_tick_sub)

# 鉛直軸の描画
plt.vlines(dx_tick, ymin, ymax, color='mediumslateblue', alpha=0.5, linewidth=0.1)

# x軸目盛
ax3.get_xaxis().set_tick_params(pad=1)
ax3.set_xticks(dx_tick)
ax3.set_xticklabels([])
# 副目盛表示
ax3.set_xticks(dx_tick_sub, minor=True)

# y軸目盛の値
ax3.get_yaxis().set_tick_params(pad=1)
ax3.set_yticks(dy)
ax3.set_yticklabels(_dy, fontsize=5)
# 副目盛表示
ax3.set_yticks(dy_tick_sub, minor=True)

# 値のプロット
ax3.scatter(qqq, ppp, s=8, alpha=0.8, linewidths=0.1, c="mediumslateblue", ec="navy", zorder=10)
ax3.plot([qqq[0], qqq[max - 1]], [aa * qqq[0] + bb, aa * qqq[max - 1] + bb], color='navy', linestyle='-', linewidth=0.5, zorder=9)

# 文字のプロット
ax3.text(xmin - (xmax-xmin)/35, ymax + (ymax-ymin)/50, "F(t)　　　ワイブル確率紙", ha='left', va='bottom', font="IPAexGothic", fontsize=6)

# メジアンランク法の記載
_gamma = -0.0
ax3.text(xmin, ymax + (ymax - ymin) / 18, "メジアンランク法", ha='left', va='bottom', font="IPAexGothic", fontsize=6)
ax3.text(xmin + (xmax - xmin) / 3.5, ymax + (ymax - ymin) / 20, "$\gamma$ ={_gamma:15.4f}".format(**locals()), ha='left', va='bottom', fontsize=6)

# 統計量計算
m = aa
n = math.exp(-bb/aa)
mean = n * math.gamma(1+1/m)
var = math.sqrt(n*n*(math.gamma(1+2/m) - (math.gamma(1+1/m))**2))
# 統計量表示
print("\nワイブル分布")
print('形状パラメータ={m:10.6f}'.format(**locals()))
print('尺度パラメータ={n:10.6f}'.format(**locals()))
print('平均={mean:10.6f}'.format(**locals()))
print('標準偏差={var:10.6f}'.format(**locals()))
print('相関係数={rr[0][1]:10.6f}'.format(**locals()))
print('決定係数={r_squared:10.6f}'.format(**locals()))

# 極値（最大分布について）

# 極値（最大）分布の値を取得
ppp = Gumbel(Pin)
qqq = Qin

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

# 図の書式
ax4 = fig.add_subplot(2, 2, 2)  # Axes
ax4.patch.set_facecolor('lavender')  # subplotの背景色
ax4.patch.set_alpha(0.2)  # subplotの背景透明度
ax4.spines['top'].set_linewidth(0.1)
ax4.spines['right'].set_linewidth(0.1)
ax4.spines['left'].set_linewidth(0.1)
ax4.spines['bottom'].set_linewidth(0.1)

# x軸の最大・最小
xmin = qqq[0] - (qqq[max-1]-qqq[0])/100
xmax = qqq[max-1] + (qqq[max-1]-qqq[0])/100

# y軸の最大・最小
if(Pin[0] > 0.001):
    ymin = Gumbel(0.001)
elif(Pin[0] > 0.0001):
    ymin = Gumbel(0.0001)
else:
    ymin = Gumbel(0.00001)
ymax = Gumbel(0.999)

# 図の描画範囲
ax4.set_xlim([xmin, xmax])
ax4.set_ylim([ymin, ymax])

# 図の表示書式
ax4.tick_params(direction="inout", length=4, width=0.1)
ax4.tick_params(direction="inout", axis="x", which="minor", length=4, width=0.1)
ax4.tick_params(direction="inout", axis="y", which="minor", length=4, width=0.1)

# y軸目盛用
if(Pin[0] > 0.001):
    _dy = np.array([0.001, 0.1, 0.9, 0.999])
elif(Pin[0] > 0.0001):
    _dy = np.array([0.0001, 0.001, 0.1, 0.9, 0.999])
else:
    _dy = np.array([0.00001, 0.001, 0.1, 0.9, 0.999])
dy = Gumbel(_dy)
_dy = _dy * 100

# 水平軸の描画用
_dy_tick = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.95, 0.99, 0.995])
dy_tick = Gumbel(_dy_tick)
# 水平軸の描画
ax4.hlines(dy_tick, xmin, xmax, color='mediumslateblue', alpha=0.5, linewidth=0.1)

# y軸副目盛用
if(Pin[0] > 0.001):
    _dy_tick_sub = np.array([0.01, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95, 0.99, 0.995])
elif(Pin[0] > 0.0001):
    _dy_tick_sub = np.array([0.01, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95, 0.99, 0.995])
else:
    _dy_tick_sub = np.array([0.0001, 0.01, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95, 0.99, 0.995])
dy_tick_sub = Gumbel(_dy_tick_sub)

# x軸の目盛
_dx = np.empty(7)
_dx[0] = qqq[0]
_dx[6] = qqq[max-1]

# x軸の表示目盛の計算
ddx = (_dx[6]-_dx[0])/6
for i in range(1, 6, 1):
    _dx[i] = _dx[0] + ddx * i

# 鉛直軸の描画
ax4.vlines(_dx, ymin, ymax, color='mediumslateblue', alpha=0.5, linewidth=0.1)

# x軸目盛
ax4.get_xaxis().set_tick_params(pad=1)
ax4.set_xticks(_dx)
ax4.set_xticklabels([])

# y軸目盛の値
ax4.get_yaxis().set_tick_params(pad=1)
ax4.set_yticks(dy)
ax4.set_yticklabels(_dy, fontsize=5)
# 副目盛表示
ax4.set_yticks(dy_tick_sub, minor=True)

# 値のプロット
ax4.scatter(qqq, ppp, s=8, marker="x", alpha=0.8, linewidths=0.1, c="red", zorder=10)
ax4.plot([qqq[0], qqq[max - 1]], [aa * qqq[0] + bb, aa * qqq[max - 1] + bb], color='navy', linestyle='-', linewidth=0.5, zorder=9)

# 文字のプロット
ax4.text(xmin - (xmax-xmin)/35, ymax + (ymax-ymin)/50, "F(t)　　　極値（最大）確率紙", ha='left', va='bottom', font="IPAexGothic", fontsize=6)

# 統計量の計算
mu = -bb/aa
n = 1/aa
mean = mu + np.euler_gamma*n
var = math.sqrt(np.pi**2 * n**2 / 6)
# 統計量の表示
print("\n極値（最大）分布")
print('位置パラメータ={mu:10.6f}'.format(**locals()))
print('尺度パラメータ={n:10.6f}'.format(**locals()))
print('平均={mean:10.6f}'.format(**locals()))
print('標準偏差={var:10.6f}'.format(**locals()))
print('相関係数={rr[0][1]:10.6f}'.format(**locals()))
print('決定係数={r_squared:10.6f}'.format(**locals()))

# 図の表示
plt.show()

# %%
pdf = PdfPages('./img/Examination.pdf')
pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
pdf.close()
