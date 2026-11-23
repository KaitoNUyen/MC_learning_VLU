# -*- coding: utf-8 -*-
"""Script vẽ các biểu đồ trực quan phân tích cho Bài thực hành 8 (Phân cụm K-Means)."""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAITAP_DIR = os.path.join(BASE_DIR, "baitap08")
DATA_DIR = os.path.join(BASE_DIR, "data")


def ve_bieu_do_silhouette_va_3cot():
    """Hình 1: Bài 1 (Khách 113 có s nhỏ nhất trên K=5) và Bài 2 (Silhouette 2 cột vs 3 cột)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "khach_hang.csv"))

    # 2 cột (Mục 3 & Bài 1)
    X2 = df[["thu_nhap", "diem_chi_tieu"]].to_numpy()
    sc2 = StandardScaler().fit(X2)
    Xs2 = sc2.transform(X2)
    km5 = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs2)
    nhan = km5.labels_
    tam_goc = sc2.inverse_transform(km5.cluster_centers_)
    s_mau = silhouette_samples(Xs2, nhan)
    idx_min = int(np.argmin(s_mau))

    # Tính Silhouette K=2..10 cho 2 cột và 3 cột (Bài 2)
    Xs3 = StandardScaler().fit_transform(df[["tuoi", "thu_nhap", "diem_chi_tieu"]])
    cac_k = list(range(2, 11))
    sil_2cot, sil_3cot = [], []
    for k in cac_k:
        m2 = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs2)
        m3 = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs3)
        sil_2cot.append(silhouette_score(Xs2, m2.labels_))
        sil_3cot.append(silhouette_score(Xs3, m3.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # Subplot 1: 5 cụm khách hàng và vị trí khách 113 (s = 0.0286)
    mau_cum = {
        3: ("#1f77b4", "o", "Cum 3 (Thap-It)"),
        0: ("#d62728", "^", "Cum 0 (Thap-Nhieu)"),
        2: ("#17a589", "s", "Cum 2 (Vua-Vua)"),
        1: ("#e67e22", "D", "Cum 1 (Cao-It)"),
        4: ("#7d3c98", "v", "Cum 4 (Cao-Nhieu)"),
    }
    for c, (mau, mk, ten) in mau_cum.items():
        pts = X2[nhan == c]
        ax1.scatter(pts[:, 0], pts[:, 1], c=mau, marker=mk, s=36, alpha=0.82, label=ten, zorder=3)
        ax1.scatter(
            tam_goc[c, 0], tam_goc[c, 1],
            c=mau, marker="X", s=200, edgecolors="black", linewidths=1.1, zorder=5,
        )

    # Đánh dấu khách 113 có s nhỏ nhất và vẽ 2 đoạn nét đứt tới tâm cụm 2 & cụm 1
    p_min = X2[idx_min]
    ax1.plot(
        [p_min[0], tam_goc[2, 0]], [p_min[1], tam_goc[2, 1]],
        linestyle="--", color="#17a589", linewidth=1.5, alpha=0.85, zorder=4,
    )
    ax1.plot(
        [p_min[0], tam_goc[1, 0]], [p_min[1], tam_goc[1, 1]],
        linestyle="--", color="#e67e22", linewidth=1.5, alpha=0.85, zorder=4,
    )
    ax1.scatter(
        [p_min[0]], [p_min[1]],
        c="#ffeb3b", marker="*", s=290, edgecolors="#c0392b", linewidths=1.5,
        label=f"Khach {idx_min} (s min = {s_mau[idx_min]:.4f})", zorder=6,
    )
    ax1.annotate(
        f"Khach {idx_min} (25.4, 41)\na=0.918, b=0.945 -> s=0.029",
        xy=(p_min[0], p_min[1]),
        xytext=(15.5, 17.5),
        fontsize=8.6,
        fontweight="bold",
        color="#922b21",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff9e6", edgecolor="#c0392b", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3),
        zorder=7,
    )
    ax1.set_title("Bai 1: 5 cum khach hang & diem co s nho nhat", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Thu nhap (trieu dong / thang)")
    ax1.set_ylabel("Diem chi tieu (1 - 100)")
    ax1.set_xlim(1.5, 37.5)
    ax1.set_ylim(2, 112)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="upper center", ncol=2, fontsize=7.8, framealpha=0.92)

    # Subplot 2: So sánh Silhouette K = 2..10 giữa 2 cột và 3 cột (Bài 2)
    ax2.plot(cac_k, sil_2cot, marker="o", linewidth=2.2, color="#1f77b4", label="2 cot: thu_nhap, diem_chi_tieu (Muc 4)")
    ax2.plot(cac_k, sil_3cot, marker="s", linewidth=2.2, color="#d62728", label="3 cot: them tuoi (Bai 2)")

    # Khoanh tròn đỉnh K=5 cho cả 2 đường
    ax2.scatter([5], [sil_2cot[3]], s=180, facecolors="none", edgecolors="#1f77b4", linewidths=2.2, zorder=5)
    ax2.scatter([5], [sil_3cot[3]], s=180, facecolors="none", edgecolors="#d62728", linewidths=2.2, zorder=5)

    for k_val, v2, v3 in zip(cac_k, sil_2cot, sil_3cot):
        # Đặt nhãn ở hai phía trên/dưới tùy giá trị nào lớn hơn để tuyệt đối không chồng chữ
        if v2 >= v3:
            dy2, dy3 = 10, -15
            if k_val in (5, 6):
                dy3 = 10
        else:
            dy2, dy3 = -15, 10
        ax2.annotate(
            f"{v2:.3f}", (k_val, v2), textcoords="offset points", xytext=(0, dy2),
            ha="center", fontsize=8.0, color="#154360", fontweight="bold" if k_val == 5 else "normal",
        )
        ax2.annotate(
            f"{v3:.3f}", (k_val, v3), textcoords="offset points", xytext=(0, dy3),
            ha="center", fontsize=8.0, color="#922b21", fontweight="bold" if k_val == 5 else "normal",
        )

    ax2.set_title("Bai 2: Chi so Silhouette theo K (2 cot vs 3 cot)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("So cum K")
    ax2.set_ylabel("Chi so Silhouette trung binh")
    ax2.set_xticks(cac_k)
    ax2.set_ylim(0.30, 0.80)
    ax2.grid(True, linestyle=":", alpha=0.55)
    ax2.legend(loc="upper right", fontsize=8.6, framealpha=0.92)

    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_silhouette_va_3cot.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


def ve_bieu_do_dbscan_hai_vang_trang():
    """Hình 2: Bài 5 — Đối chiếu K-Means (K=2) và DBSCAN (eps = 0.2, 0.3, 0.4) trên hai vầng trăng."""
    df = pd.read_csv(os.path.join(DATA_DIR, "hai_vang_trang.csv"))
    X = df[["x1", "x2"]].to_numpy()
    y_that = df["nhan"].to_numpy()

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    nhan_km = km.labels_
    khop_km = max((nhan_km == y_that).mean(), (nhan_km != y_that).mean())

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4))

    # Panel 1: K-Means K=2 (vẽ thêm đường trung trực nối 2 tâm)
    ax = axes[0]
    for lop, mk in [(0, "o"), (1, "^")]:
        m0 = (y_that == lop) & (nhan_km == 0)
        m1 = (y_that == lop) & (nhan_km == 1)
        ax.scatter(X[m0, 0], X[m0, 1], c="#1f77b4", marker=mk, s=26, alpha=0.8)
        ax.scatter(X[m1, 0], X[m1, 1], c="#d62728", marker=mk, s=26, alpha=0.8)
    ax.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        c="#ffeb3b", marker="X", s=170, edgecolors="black", linewidths=1.1, label="Tam cum", zorder=5,
    )
    ax.set_title(f"K-Means (K = 2)\nKhop = {khop_km:.4f} (cat thang)", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_ylim(-1.60, 1.78)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    # Panel 2-4: DBSCAN với eps = 0.2, 0.3, 0.4
    for ax, e in zip(axes[1:], [0.2, 0.3, 0.4]):
        db = DBSCAN(eps=e, min_samples=5).fit(X)
        lbl = db.labels_
        so_cum = len(set(lbl)) - (1 if -1 in lbl else 0)
        so_nhieu = int((lbl == -1).sum())
        ari = adjusted_rand_score(y_that, lbl)

        # Vẽ điểm thuộc cụm 0 theo hình dạng của nhãn thật (o / ^) để thấy hai vầng trăng bị gộp
        for lop, mk, nhan_hinh in [(0, "o", "Lop that 0 (cum 0)"), (1, "^", "Lop that 1 (cum 0)")]:
            m = (y_that == lop) & (lbl == 0)
            ax.scatter(X[m, 0], X[m, 1], c="#2e86c1", marker=mk, s=26, alpha=0.72, label=nhan_hinh)

        # Vẽ điểm nhiễu (-1) màu đen nổi bật
        nhieu = lbl == -1
        if nhieu.any():
            ax.scatter(
                X[nhieu, 0], X[nhieu, 1],
                c="#111111", marker="x", s=45, linewidths=1.5,
                label=f"Nhieu -1 ({so_nhieu} diem)", zorder=5,
            )

        ax.set_title(
            f"DBSCAN (eps = {e}, min_samples = 5)\n{so_cum} cum, {so_nhieu} nhieu (ARI = {ari:+.3f})",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_ylim(-1.60, 1.78)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    fig.suptitle(
        "Bai 5: K-Means vs DBSCAN tren hai_vang_trang.csv (hinh dang o/^ la nhan that, mau sac la cum tim duoc)",
        fontsize=11.8,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_dbscan_hai_vang_trang.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


def ve_bieu_do_khoi_tao_va_digits():
    """Hình 3: Bài 3 (50 lần chạy random vs k-means++) và Bài 6 (Heatmap crosstab load_digits)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "khach_hang.csv"))
    Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

    in_rand = [
        KMeans(n_clusters=5, init="random", n_init=1, random_state=rs).fit(Xs).inertia_
        for rs in range(50)
    ]
    in_pp = [
        KMeans(n_clusters=5, init="k-means++", n_init=1, random_state=rs).fit(Xs).inertia_
        for rs in range(50)
    ]

    # Bài 6: crosstab load_digits
    Xd, yd = load_digits(return_X_y=True)
    kmd = KMeans(n_clusters=10, n_init=10, random_state=0).fit(Xd)
    ct = pd.crosstab(yd, kmd.labels_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.6), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Subplot 1: Bài 3 — 50 lần chạy inertia (random vs k-means++)
    cac_rs = np.arange(50)
    ax1.plot(
        cac_rs, in_rand, "o-", color="#d62728", markersize=4.8, linewidth=1.1, alpha=0.82,
        label='init="random" (30/50 dat 23.0, max 204.0)',
    )
    ax1.plot(
        cac_rs, in_pp, "s-", color="#1e8449", markersize=4.2, linewidth=1.8, alpha=0.92,
        label='init="k-means++" (50/50 dat 23.0)',
    )
    ax1.axhline(23.02, color="#1e8449", linestyle="--", linewidth=1.1, alpha=0.65)
    idx_max = int(np.argmax(in_rand))
    ax1.annotate(
        f"Te nhat: rs={idx_max}\ninertia={in_rand[idx_max]:.1f}",
        xy=(idx_max, in_rand[idx_max]),
        xytext=(max(2, idx_max - 19), in_rand[idx_max] - 32),
        fontsize=8.5,
        fontweight="bold",
        color="#922b21",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdebd0", edgecolor="#c0392b", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )
    ax1.set_title("Bai 3: Inertia qua 50 lan khoi tao (n_init = 1)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("random_state (0 -> 49)")
    ax1.set_ylabel("Inertia (nho hon la tot hon)")
    ax1.set_ylim(5, 252)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="upper right", fontsize=8.4, framealpha=0.92)

    # Subplot 2: Bài 6 — Heatmap pd.crosstab(nhan_that, cum)
    vals = ct.values
    im = ax2.imshow(vals, cmap="YlGnBu", aspect="auto")
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([str(c) for c in range(10)])
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([str(d) for d in range(10)])
    ax2.set_xlabel("Cum K-Means tim duoc (0 - 9)")
    ax2.set_ylabel("Chu so that (0 - 9)")
    ax2.set_title("Bai 6: Bang crosstab(nhan_that, cum) tren load_digits", fontsize=11.5, fontweight="bold")

    nguong_trang = vals.max() * 0.55
    for r in range(10):
        for c in range(10):
            v = vals[r, c]
            if v > 0:
                mau_chu = "white" if v >= nguong_trang else "#111111"
                ax2.text(
                    c, r, str(v), ha="center", va="center",
                    color=mau_chu, fontsize=8.8, fontweight="bold" if v >= 40 else "normal",
                )
            else:
                ax2.text(c, r, "0", ha="center", va="center", color="#bbbbbb", fontsize=7.5)

    # Khoanh viền đỏ cụm 1 (chứa 100 số 1 và 100 số 8) để nhấn mạnh cặp gom chung mạnh nhất
    rect = plt.Rectangle((1 - 0.5, -0.5), 1, 10, fill=False, edgecolor="#e74c3c", linewidth=2.2)
    ax2.add_patch(rect)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="So luong anh")

    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_khoi_tao_va_digits.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


if __name__ == "__main__":
    ve_bieu_do_silhouette_va_3cot()
    ve_bieu_do_dbscan_hai_vang_trang()
    ve_bieu_do_khoi_tao_va_digits()
