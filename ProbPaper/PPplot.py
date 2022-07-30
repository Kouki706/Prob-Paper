# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# scipyの関数
from scipy.stats import norm
from scipy.optimize import curve_fit

# 回帰式用の関数
def func(x, a, b):
    f = a * x + b
    return f


# weibull逆関数
def weibull(x):
    return np.log(np.log(1 / (1 - x)))


# 極値（最大）分布逆関数
def gumbel(x):
    return -np.log(-np.log(x))


class pyPPplot:
    def __init__(self, filename):
        """初期化"""
        plt.style.use(["science", "kulab"])

        # ファイル名の格納
        self.filename = filename
        # データの読み込み
        # ファイル名の分岐
        if filename.endswith(".csv"):
            self.data = np.loadtxt(filename, delimiter=",", usecols=[0])
        else:
            self.data = np.loadtxt(filename, delimiter="\t", usecols=[0])

        # データサイズ
        self.sample_num = self.data.size

        # 適合分布の比較と確率紙の判定
        self.single_mode = True

        # 決定係数
        self.r_squared = np.empty(0)
        # パラメータ
        self.param = np.empty((0, 2))
        # ラベル名
        self.name_of_prob_paper = ["norm", "lognorm", "weibull", "gumbel"]

    def _pre(self, more_than_zero=False):
        # 0以下データの削除とデータのソート
        if more_than_zero is True:
            self.data = self.data[self.data > 0]
        self.data = np.sort(self.data)

        # メジアンランク法
        self.prob = np.empty(self.data.size)
        for i in range(self.data.size):
            self.prob[i] = (i + 0.7) / (self.data.size + 0.4)

        # 重複する値の設定
        for i in range(self.data.size - 1, 1, -1):
            if self.data[i - 1] == self.data[i]:
                self.prob[i - 1] = self.prob[i]

        # 重複する値の削除
        self.prob = np.unique(self.prob)
        self.data = np.unique(self.data)

    def norm(self, figname):
        """正規分布のプロット"""
        # データの準備
        self._pre(False)
        self.single_mode = True

        # 決定係数とパラメータの初期化
        self.r_squared = np.empty(0)
        self.param = np.empty((0, 2))

        # プロットファイルの名称
        fig, ax = plt.subplots()

        # 各図の軸を空に
        self._empty_tick(ax)

        self._plot_by_prob_paper(ax, self.name_of_prob_paper[0])

        fig.savefig(figname, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    def lognorm(self, figname):
        """対数正規分布のプロット"""
        # データの準備
        self._pre(True)
        self.single_mode = True

        # 決定係数とパラメータの初期化
        self.r_squared = np.empty(0)
        self.param = np.empty((0, 2))

        # プロットファイルの名称
        fig, ax = plt.subplots()

        # 各図の軸を空に
        self._empty_tick(ax)

        self._plot_by_prob_paper(ax, self.name_of_prob_paper[1])

        fig.savefig(figname, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    def weibull(self, figname):
        """対数正規分布のプロット"""
        # データの準備
        self._pre(True)
        self.single_mode = True

        # 決定係数とパラメータの初期化
        self.r_squared = np.empty(0)
        self.param = np.empty((0, 2))

        # プロットファイルの名称
        fig, ax = plt.subplots()

        # 各図の軸を空に
        self._empty_tick(ax)

        self._plot_by_prob_paper(ax, self.name_of_prob_paper[2])

        fig.savefig(figname, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    def gumbel(self, figname):
        """対数正規分布のプロット"""
        # データの準備
        self._pre(False)
        self.single_mode = True

        # 決定係数とパラメータの初期化
        self.r_squared = np.empty(0)
        self.param = np.empty((0, 2))

        # プロットファイルの名称
        fig, ax = plt.subplots()

        # 各図の軸を空に
        self._empty_tick(ax)

        self._plot_by_prob_paper(ax, self.name_of_prob_paper[3])

        fig.savefig(figname, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    def examination(self, figname):
        """分布の検討のプロット"""
        print("サンプル数 : ", self.sample_num)

        # データの準備
        self._pre(True)
        self.single_mode = False

        # 決定係数とパラメータの初期化
        self.r_squared = np.empty(0)
        self.param = np.empty((0, 2))

        # 文字大きさの修正
        plt.rcParams["font.size"] = 5

        # プロットファイルの名称
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False)
        plt.subplots_adjust(wspace=0.2, hspace=0.1)

        # 各図の軸を空に
        self._empty_tick(axes[0, 0])
        self._empty_tick(axes[0, 1])
        self._empty_tick(axes[1, 0])
        self._empty_tick(axes[1, 1])

        # Artistの連番
        ax_number = [axes[1, 1], axes[1, 0], axes[0, 0], axes[0, 1]]

        for i in range(len(self.name_of_prob_paper)):
            self._plot_by_prob_paper(ax_number[i], self.name_of_prob_paper[i])

        fig.savefig(figname, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    def _plot_by_prob_paper(self, ax, name_of_prob_paper):
        if name_of_prob_paper == self.name_of_prob_paper[0]:
            self._plot_norm_paper(ax)
        elif name_of_prob_paper == self.name_of_prob_paper[1]:
            self._plot_logplot_norm_paper(ax)
        elif name_of_prob_paper == self.name_of_prob_paper[2]:
            self._plot_weibull_paper(ax)
        else:
            self._plot_gumbel_paper(ax)

    def _plot_norm_paper(self, ax):
        """正規分布"""
        ppp = norm.ppf(self.prob, loc=0, scale=1)
        qqq = self.data

        # 値のプロット
        if self.single_mode is True:
            mker = "o"
        else:
            mker = ","

        ax.scatter(
            qqq,
            ppp,
            marker=mker,
            s=2.5,
            alpha=1.0,
            linewidths=0.5,
            c="white",
            ec="black",
            zorder=2,
        )

        # 回帰直線
        popt, pcov = curve_fit(func, qqq, ppp)
        ax.plot(
            [qqq[0], qqq[qqq.size - 1]],
            [func(qqq[0], popt[0], popt[1]), func(qqq[qqq.size - 1], popt[0], popt[1])],
            color="black",
            linestyle="-",
            linewidth=0.5,
            zorder=1,
        )

        # x・y軸の設定
        self._set_y_tick(ax, self.name_of_prob_paper[0])
        self._set_x_tick(ax, self.name_of_prob_paper[0])

        # x・y軸の最大・最小
        self._set_max_min(ax, self.name_of_prob_paper[0])

        # 決定係数
        residuals = ppp - func(qqq, popt[0], popt[1])
        rss = np.sum(residuals**2)
        tss = np.sum((ppp - np.mean(ppp)) ** 2)
        r_squared = 1 - (rss / tss)
        print("正規分布\t\t: 決定係数 = {r_squared:10.6f}".format(**locals()))

        # 係数と決定係数
        self.r_squared = np.append(self.r_squared, r_squared)
        param1 = -popt[1] / popt[0]  # 平均
        param2 = 1 / popt[0]  # 標準偏差
        self.param = np.append(self.param, np.array([[param1, param2]]), axis=0)

        # 係数のプロット
        if self.single_mode is True:
            self._param_plot(ax, self.name_of_prob_paper[0], param1, param2)

    def _plot_logplot_norm_paper(self, ax):
        """対数正規分布"""
        ppp = norm.ppf(self.prob, loc=0, scale=1)
        qqq = np.log(self.data)

        # 値のプロット
        if self.single_mode is True:
            mker = "o"
        else:
            mker = "^"

        ax.scatter(
            qqq,
            ppp,
            marker=mker,
            s=2.5,
            alpha=1.0,
            linewidths=0.5,
            c="white",
            ec="black",
            zorder=2,
        )

        # 回帰直線
        popt, pcov = curve_fit(func, qqq, ppp)
        ax.plot(
            [qqq[0], qqq[qqq.size - 1]],
            [func(qqq[0], popt[0], popt[1]), func(qqq[qqq.size - 1], popt[0], popt[1])],
            color="black",
            linestyle="-",
            linewidth=0.5,
            zorder=1,
        )

        # x・y軸の設定
        self._set_y_tick(ax, self.name_of_prob_paper[1])
        self._set_x_tick(ax, self.name_of_prob_paper[1])

        # x・y軸の最大・最小
        self._set_max_min(ax, self.name_of_prob_paper[1])

        # 決定係数
        residuals = ppp - func(qqq, popt[0], popt[1])
        rss = np.sum(residuals**2)
        tss = np.sum((ppp - np.mean(ppp)) ** 2)
        r_squared = 1 - (rss / tss)
        print("対数正規分布\t\t: 決定係数 = {r_squared:10.6f}".format(**locals()))

        # 係数と決定係数
        self.r_squared = np.append(self.r_squared, r_squared)
        param1 = -popt[1] / popt[0]  # 平均
        param2 = 1 / popt[0]  # 標準偏差
        self.param = np.append(self.param, np.array([[param1, param2]]), axis=0)

        # 係数のプロット
        if self.single_mode is True:
            self._param_plot(ax, self.name_of_prob_paper[1], param1, param2)

    def _plot_weibull_paper(self, ax):
        """ワイブル分布"""
        # weibull分布について
        ppp = weibull(self.prob)
        qqq = np.log(self.data)

        # 値のプロット
        if self.single_mode is True:
            mker = "o"
        else:
            mker = "o"

        ax.scatter(
            qqq,
            ppp,
            marker=mker,
            s=2.5,
            alpha=1.0,
            linewidths=0.5,
            c="white",
            ec="black",
            zorder=2,
        )

        # 回帰直線
        popt, pcov = curve_fit(func, qqq, ppp)
        ax.plot(
            [qqq[0], qqq[qqq.size - 1]],
            [func(qqq[0], popt[0], popt[1]), func(qqq[qqq.size - 1], popt[0], popt[1])],
            color="black",
            linestyle="-",
            linewidth=0.5,
            zorder=1,
        )

        # x・y軸の設定
        self._set_y_tick(ax, self.name_of_prob_paper[2])
        self._set_x_tick(ax, self.name_of_prob_paper[2])

        # x・y軸の最大・最小
        self._set_max_min(ax, self.name_of_prob_paper[2])

        # 決定係数
        residuals = ppp - func(qqq, popt[0], popt[1])
        rss = np.sum(residuals**2)
        tss = np.sum((ppp - np.mean(ppp)) ** 2)
        r_squared = 1 - (rss / tss)
        print("ワイブル分布\t\t: 決定係数 = {r_squared:10.6f}".format(**locals()))

        # 係数と決定係数
        self.r_squared = np.append(self.r_squared, r_squared)
        param1 = popt[0]  # ワイブル係数（形状パラメータ）
        param2 = np.exp(-popt[1] / popt[0])  # 尺度パラメータ
        self.param = np.append(self.param, np.array([[param1, param2]]), axis=0)

        # 係数のプロット
        if self.single_mode is True:
            self._param_plot(ax, self.name_of_prob_paper[2], param1, param2)

    def _plot_gumbel_paper(self, ax):
        """グンベル分布"""
        # 極値（最大分布について）
        ppp = gumbel(self.prob)
        qqq = self.data

        # 値のプロット
        if self.single_mode is True:
            mker = "o"
        else:
            mker = "x"

        ax.scatter(
            qqq,
            ppp,
            marker=mker,
            s=2.5,
            alpha=1.0,
            linewidths=0.5,
            c="white",
            ec="black",
            zorder=2,
        )

        # 回帰直線
        popt, pcov = curve_fit(func, qqq, ppp)
        ax.plot(
            [qqq[0], qqq[qqq.size - 1]],
            [func(qqq[0], popt[0], popt[1]), func(qqq[qqq.size - 1], popt[0], popt[1])],
            color="black",
            linestyle="-",
            linewidth=0.5,
            zorder=1,
        )

        # x・y軸の設定
        self._set_y_tick(ax, self.name_of_prob_paper[3])
        self._set_x_tick(ax, self.name_of_prob_paper[3])

        # x・y軸の最大・最小
        self._set_max_min(ax, self.name_of_prob_paper[3])

        # 決定係数
        residuals = ppp - func(qqq, popt[0], popt[1])
        rss = np.sum(residuals**2)
        tss = np.sum((ppp - np.mean(ppp)) ** 2)
        r_squared = 1 - (rss / tss)
        print("極値（最大）分布\t: 決定係数 = {r_squared:10.6f}".format(**locals()))

        # 係数と決定係数
        self.r_squared = np.append(self.r_squared, r_squared)
        param1 = 1 / popt[0]  # 尺度パラメータ
        param2 = -popt[1] / popt[0]  # 位置パラメータ
        self.param = np.append(self.param, np.array([[param1, param2]]), axis=0)

        # 係数のプロット
        if self.single_mode is True:
            self._param_plot(ax, self.name_of_prob_paper[3], param1, param2)

    def _param_plot(self, ax, name_of_prob_paper, param1, param2):
        # 確率紙の判定
        if (
            name_of_prob_paper == self.name_of_prob_paper[0]
            or name_of_prob_paper == self.name_of_prob_paper[1]
        ):
            p1msg = "$\mu$"
            p2msg = "$\sigma$"
        elif name_of_prob_paper == self.name_of_prob_paper[2]:
            p1msg = "$m$"
            p2msg = "$\eta$ "
        else:
            p1msg = "$\\alpha$"
            p2msg = "$\\theta$"
        # 図中に値のプロット
        boxdic = {
            "facecolor": "white",
            "edgecolor": "black",
            "boxstyle": "square",
            "linewidth": 0.5,
        }
        msg = (
            p1msg
            + " ="
            + "{param1:8.4f}".format(**locals())
            + "\n"
            + p2msg
            + " ="
            + "{param2:8.4f}".format(**locals())
        )
        ax.text(
            0.025,
            0.965,
            msg,
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=boxdic,
        )

    def _empty_tick(self, ax):
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        # グリッドに関して
        ax.set_axisbelow(True)
        ax.grid(ls="--")

    def _set_x_tick(self, ax, name_of_prob_paper):
        if (
            name_of_prob_paper == self.name_of_prob_paper[0]
            or name_of_prob_paper == self.name_of_prob_paper[3]
        ):
            dx_tick_label = self._x_tick()
            ax.set_xticks(dx_tick_label)
            """確率紙と適合比較の判定"""
            if name_of_prob_paper == self.name_of_prob_paper[0]:
                ax.set_xticklabels(np.round(dx_tick_label, 1))
            if self.single_mode is True:
                ax.set_xticklabels(np.round(dx_tick_label, 5))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
        elif (
            name_of_prob_paper == self.name_of_prob_paper[1]
            or name_of_prob_paper == self.name_of_prob_paper[2]
        ):
            dx_tick_label, dx_tick_sub = self._log_x_tick()
            ax.set_xticks(np.log(dx_tick_label))
            """確率紙と適合比較の判定"""
            if name_of_prob_paper == self.name_of_prob_paper[1]:
                ax.set_xticklabels(np.round(dx_tick_label, 1))
            if self.single_mode is True:
                ax.set_xticklabels(np.round(dx_tick_label, 5))
            ax.set_xticks(np.log(dx_tick_sub), minor=True)

    def _x_tick(self):
        # 最大と最小
        xmin = np.min(self.data)
        xmax = np.max(self.data)

        # x軸のラベル
        dx_tick_label = np.empty(0)
        for i in range(5):
            dx_tick_label = np.append(dx_tick_label, xmin + (xmax - xmin) * i / 4)

        return dx_tick_label

    def _log_x_tick(self):
        # 最大と最小
        xmin = np.floor(np.log10(np.min(self.data)))
        xmax = np.ceil(np.log10(np.max(self.data)))

        # x軸のラベル
        dx_tick_label = np.empty(0)
        for i in range(int(xmin), int(xmax) + 1):
            dx_tick_label = np.append(dx_tick_label, 10**i)

        # x軸のサブラベル
        cal_dx_tick_sub = np.array([2, 3, 4, 5])
        dx_tick_sub = np.empty(0)
        for i in range(int(xmin), int(xmax)):
            dx_tick_sub = np.append(dx_tick_sub, cal_dx_tick_sub * 10**i)

        return dx_tick_label, dx_tick_sub

    def _set_y_tick(self, ax, name_of_prob_paper):
        """分布の適合と確率紙の確認"""
        if self.single_mode is True:
            if (
                name_of_prob_paper == self.name_of_prob_paper[0]
                or name_of_prob_paper == self.name_of_prob_paper[1]
            ):
                # y軸のラベル
                dy_tick_label = np.array(
                    [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999]
                )
                dy_tick = norm.ppf(dy_tick_label, loc=0, scale=1)
                ax.set_yticks(dy_tick)
                ax.set_yticklabels(dy_tick_label * 100)

                # y軸の副目盛表示
                dy_tick_sub = np.array([0.0005, 0.005, 0.05, 0.3, 0.4, 0.6, 0.7])
                dy_tick_sub = norm.ppf(dy_tick_sub, loc=0, scale=1)
                ax.set_yticks(dy_tick_sub, minor=True)
            elif name_of_prob_paper == self.name_of_prob_paper[2]:
                # y軸のラベル
                dy_tick_label = np.array(
                    [
                        0.0001,
                        0.0002,
                        0.0005,
                        0.001,
                        0.002,
                        0.005,
                        0.01,
                        0.02,
                        0.05,
                        0.1,
                        0.2,
                        0.5,
                        0.8,
                        0.95,
                        0.999,
                    ]
                )
                dy_tick = weibull(dy_tick_label)
                ax.set_yticks(dy_tick)
                ax.set_yticklabels(dy_tick_label * 100)

                # y軸 副目盛表示
                dy_tick_sub = np.array(
                    [
                        0.0003,
                        0.0004,
                        0.003,
                        0.004,
                        0.005,
                        0.03,
                        0.04,
                        0.05,
                        0.3,
                        0.4,
                    ]
                )
                dy_tick_sub = weibull(dy_tick_sub)
                ax.set_yticks(dy_tick_sub, minor=True)
            elif name_of_prob_paper == "gumbel":
                # y軸のラベル
                dy_tick_label = np.array(
                    [
                        0.001,
                        0.01,
                        0.1,
                        0.5,
                        0.9,
                        0.95,
                        0.99,
                        0.995,
                        0.999,
                    ]
                )
                dy_tick = gumbel(dy_tick_label)
                ax.set_yticks(dy_tick)
                ax.set_yticklabels(dy_tick_label * 100)

                # y軸 副目盛表示
                dy_tick_sub = np.array([0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
                dy_tick_sub = gumbel(dy_tick_sub)
                ax.set_yticks(dy_tick_sub, minor=True)
        else:
            if name_of_prob_paper == self.name_of_prob_paper[0]:
                # y軸のラベル
                dy_tick_label = np.array(
                    [
                        0.001,
                        0.01,
                        0.1,
                        0.5,
                        0.999,
                    ]
                )
                dy_tick = norm.ppf(dy_tick_label, loc=0, scale=1)
                ax.set_yticks(dy_tick)
                ax.set_yticklabels(dy_tick_label * 100)

                # y軸の副目盛表示
                dy_tick_sub = np.array(
                    [0.005, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]
                )
                dy_tick_sub = norm.ppf(dy_tick_sub, loc=0, scale=1)
                ax.set_yticks(dy_tick_sub, minor=True)
            elif name_of_prob_paper == self.name_of_prob_paper[1]:
                # y軸のラベル
                dy_tick_label = np.array(
                    [
                        0.001,
                        0.01,
                        0.1,
                        0.5,
                        0.999,
                    ]
                )
                dy_tick = norm.ppf(dy_tick_label, loc=0, scale=1)
                ax.set_yticks(dy_tick)
                ax.set_yticklabels(dy_tick_label * 100)

                # y軸 副目盛表示
                dy_tick_sub = np.array(
                    [0.005, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]
                )
                dy_tick_sub = norm.ppf(dy_tick_sub, loc=0, scale=1)
                ax.set_yticks(dy_tick_sub, minor=True)
            elif name_of_prob_paper == self.name_of_prob_paper[2]:
                # y軸のラベル
                dy_tick_label = np.array(
                    [
                        0.001,
                        0.01,
                        0.1,
                        0.5,
                        0.999,
                    ]
                )
                dy_tick = weibull(dy_tick_label)
                ax.set_yticks(dy_tick)
                ax.set_yticklabels(dy_tick_label * 100)

                # y軸 副目盛表示
                dy_tick_sub = np.array(
                    [
                        0.002,
                        0.003,
                        0.004,
                        0.005,
                        0.02,
                        0.03,
                        0.04,
                        0.05,
                        0.2,
                        0.3,
                        0.4,
                        0.8,
                        0.95,
                    ]
                )
                dy_tick_sub = weibull(dy_tick_sub)
                ax.set_yticks(dy_tick_sub, minor=True)
            elif name_of_prob_paper == "gumbel":
                # y軸のラベル
                dy_tick_label = np.array(
                    [
                        0.001,
                        0.1,
                        0.5,
                        0.9,
                        0.999,
                    ]
                )
                dy_tick = gumbel(dy_tick_label)
                ax.set_yticks(dy_tick)
                ax.set_yticklabels(dy_tick_label * 100)

                # y軸 副目盛表示
                dy_tick_sub = np.array(
                    [0.005, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.95, 0.99, 0.995]
                )
                dy_tick_sub = gumbel(dy_tick_sub)
                ax.set_yticks(dy_tick_sub, minor=True)

    def _set_max_min(self, ax, name_of_prob_paper):
        # y軸の最大・最小
        if np.min(self.prob) > 0.001:
            ymin = 0.001
        elif np.min(self.prob) > 0.0001:
            ymin = 0.0001
        else:
            ymin = 0.00001

        if np.max(self.prob) < 0.999:
            ymax = 0.999
        elif np.max(self.prob) < 0.9999:
            ymax = 0.9999
        else:
            ymax = 0.99999

        # x軸の最大・最小
        if name_of_prob_paper == self.name_of_prob_paper[0]:
            xmin = np.min(self.data) - (np.max(self.data) - np.min(self.data)) / 25
            xmax = np.max(self.data) + (np.max(self.data) - np.min(self.data)) / 25

            ax.set_xlim([xmin, xmax])
            ax.set_ylim(
                [norm.ppf(ymin, loc=0, scale=1), norm.ppf(ymax, loc=0, scale=1)]
            )
        elif name_of_prob_paper == self.name_of_prob_paper[1]:
            """10のべき乗を基準にx軸の最大・最小を判定"""
            xmin = 10 ** np.floor(np.log10(np.min(self.data)))
            xmax = 10 ** np.ceil(np.log10(np.max(self.data)))

            ax.set_xlim(
                [
                    np.log(xmin) - (np.log(xmax) - np.log(xmin)) / 25,
                    np.log(xmax) + (np.log(xmax) - np.log(xmin)) / 25,
                ]
            )
            """ 自然対数を基準にx軸の最大・最小を判定 """
            # xmin = np.floor(np.log(np.min(self.data)))
            # xmax = np.ceil(np.log(np.max(self.data)))
            # ax.set_xlim(
            #     [
            #         xmin - (xmax - xmin) / 25,
            #         xmax + (xmax - xmin) / 25,
            #     ]
            # )
            ax.set_ylim(
                [norm.ppf(ymin, loc=0, scale=1), norm.ppf(ymax, loc=0, scale=1)]
            )
        elif name_of_prob_paper == self.name_of_prob_paper[2]:
            xmin = 10 ** np.floor(np.log10(np.min(self.data)))
            xmax = 10 ** np.ceil(np.log10(np.max(self.data)))

            ax.set_xlim(
                [
                    np.log(xmin) - (np.log(xmax) - np.log(xmin)) / 25,
                    np.log(xmax) + (np.log(xmax) - np.log(xmin)) / 25,
                ]
            )
            ax.set_ylim([weibull(ymin), weibull(ymax)])
        else:
            xmin = np.min(self.data) - (np.max(self.data) - np.min(self.data)) / 25
            xmax = np.max(self.data) + (np.max(self.data) - np.min(self.data)) / 25

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([gumbel(ymin), gumbel(ymax)])
