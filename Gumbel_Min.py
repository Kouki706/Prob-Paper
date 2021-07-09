#%%
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 極値（最大）分布
def Gumbel(x):
    return -np.log(-np.log(x))

# 回帰式
def func(x, a, b):
    f = a*x + b
    return f

# データの読み込み
Qin=np.loadtxt('./Data.csv', delimiter=',',usecols=[0])
# 0以下の削除
Qin = Qin[Qin > 0]
Qin = np.sort(Qin)

# データサイズの取得
max = Qin.size
print("データ数",max)

# メジアンランク法
Pin=np.empty(max)
for i in range(max):
    Pin[i] = (i+0.7) / (max+0.4)

# 重複する値を除く
for i in range(max-2,0,-1):
    if(Qin[i] == Qin[i+1]):
        Pin[i] = Pin[i+1]

# 最小値分布に変換
Pin = 1 - Pin

# データをファイル出力する
Data = [Qin,Pin]
Data = np.array(Data).T
np.savetxt("./tmp/Prob.dat",Data,delimiter="\t")

# 極値（最大）分布の値を取得
ppp=Gumbel(Pin)
qqq=Qin

# 回帰直線
popt, pcov = curve_fit(func,qqq,ppp)
rr=np.corrcoef(qqq,ppp)
aa = popt[0]
bb = popt[1]

# 決定係数
residuals =  ppp - func(qqq, popt[0],popt[1])
rss = np.sum(residuals**2)
tss = np.sum((ppp-np.mean(ppp))**2)
r_squared = 1 - (rss / tss)

# 図の書式
plt.rcParams['font.family'] = 'Times New Roman'
fig = plt.figure(figsize=(4,3)) # Figure
ax = fig.add_subplot()  # Axes
ax.patch.set_facecolor('lavender')  # subplotの背景色
ax.patch.set_alpha(0.2)  # subplotの背景透明度
ax.spines['top'].set_linewidth(0.1)
ax.spines['right'].set_linewidth(0.1)
ax.spines['left'].set_linewidth(0.1)
ax.spines['bottom'].set_linewidth(0.1)

# x軸の最大・最小
xmin=qqq[0] - (qqq[max-1]-qqq[0])/100
xmax=qqq[max-1] + (qqq[max-1]-qqq[0])/100

# y軸の最大・最小
ymin=Gumbel(0.001)
ymax=Gumbel(0.999)

# 図の描画範囲
ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])

# 図の表示書式
# ax.tick_params(labelbottom=False)
# ax.tick_params(labelleft=False)
# ax.tick_params(bottom=False)
ax.tick_params(direction = "inout", axis = "x", length=2, width=0)
ax.tick_params(direction = "out", axis = "y", length=2, width=0)

# y軸目盛用
_dy=np.array([0.001,0.01,0.1,0.5,0.9,0.95,0.99,0.995,0.999])
dy=Gumbel(_dy)
_dy=_dy * 100

# 水平軸の描画用
_dy_tick = np.array([0.001,0.01,0.1,0.5,0.9,0.95,0.99,0.995,0.999])
dy_tick=Gumbel(_dy_tick)
ax.hlines(dy_tick, xmin, xmax, color='mediumpurple',linewidth=0.1)
ax.hlines(0, xmin, xmax, color='black',linewidth=0.1)

# x軸の目盛
_dx=np.empty(7)
_dx[0] = qqq[0]
_dx[6] = qqq[max-1]

# x軸の表示目盛の計算
ddx = (_dx[6]-_dx[0])/6
for i in range(1,6,1):
    _dx[i] = _dx[0] + ddx * i

# 鉛直軸の描画
ax.vlines(_dx, ymin, ymax, color='mediumpurple',linewidth=0.1)

# x軸目盛
# for i in range(7):
#     ax.text(_dx[i], ymin-0.1, str(round(_dx[i],2)), ha = 'center', va = 'top', fontsize=fs)
ax.get_xaxis().set_tick_params(pad=1)
ax.set_xticks(_dx)
ax.set_xticklabels(np.round(_dx,2),fontsize=5)

# y軸目盛の値
# for i in range(9):
#     ax.text(xmin- (xmax-xmin)/100, dy[i], str(_dy[i]), ha = 'right', va = 'center', fontsize=fs)
ax.get_yaxis().set_tick_params(pad=1)
ax.set_yticks(dy)
ax.set_yticklabels(_dy, fontsize = 5)

# 値のプロット
ax.scatter(qqq,ppp,s = 2, alpha=0.8,linewidths=0.1,c="mediumslateblue",ec = "navy" ,zorder=10)
ax.plot([xmin, xmax], [aa*xmin + bb, aa*xmax + bb], color='k', linestyle='-', linewidth=0.25, zorder=9)

# 文字のプロット
ax.text(xmin, ymax + (ymax-ymin)/50, "F(t) (%)   Median Ranks", ha = 'left', va = 'bottom', fontsize=5)

# 有効データ数
max = np.unique(Qin).size
print("有効データ数 = ",max)

# 統計量
mu = -bb/aa
n = 1/aa
print('位置パラメータ={mu:10.6f}'.format(**locals()))
print('尺度パラメータ={n:10.6f}'.format(**locals()))

mean = mu + np.euler_gamma*n
var = math.sqrt( np.pi**2 * n**2 / 6 )
print('平均={mean:10.6f}'.format(**locals()))
print('標準偏差={var:10.6f}'.format(**locals()))

print('相関係数={rr[0][1]:10.6f}'.format(**locals()))
print('決定係数={r_squared:10.6f}'.format(**locals()))
plt.show()

#%%
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('./img/Gumbel_Min.pdf')
pdf.savefig(fig,bbox_inches="tight",pad_inches=0.02)
pdf.close()
