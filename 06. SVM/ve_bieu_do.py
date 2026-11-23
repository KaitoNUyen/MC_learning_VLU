# -*- coding: utf-8 -*-
"""Script vẽ các biểu đồ trực quan phân tích cho Bài thực hành 6 (SVM)."""

import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAITAP_DIR = os.path.join(BASE_DIR, "baitap06")
DATA_DIR = os.path.join(BASE_DIR, "data")


def ve_bieu_do_le_va_sv():
    """Hình 1: Minh họa Bài 1 & Bài 2 — Đường ranh giới, hai mép lề, vectơ hỗ trợ và thêm điểm."""
    X_goc, y_goc = make_blobs(
        n_samples=24, centers=[[2, 2], [6, 6]], cluster_std=0.9, random_state=0
    )
    y_goc = 1 - y_goc

    cac_truong_hop = [
        ("Goc (24 diem) & 2 diem moi", X_goc, y_goc, None),
        ("Them (-1, -1) lop 1\n(Ranh gioi KHONG doi)", np.vstack([X_goc, [[-1.0, -1.0]]]), np.append(y_goc, 1), (-1.0, -1.0)),
        ("Them (4.5, 4) lop 1\n(Ranh gioi xoay, le bi bop hep)", np.vstack([X_goc, [[4.5, 4.0]]]), np.append(y_goc, 1), (4.5, 4.0)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    xx, yy = np.meshgrid(np.linspace(-2.0, 9.0, 300), np.linspace(-2.0, 9.0, 300))
    luoi_diem = np.c_[xx.ravel(), yy.ravel()]

    for idx_ax, (ax, (tieu_de, X, y, diem_moi)) in enumerate(zip(axes, cac_truong_hop)):
        svm = SVC(kernel="linear", C=1000).fit(X, y)
        w = svm.coef_[0]
        le = 2.0 / np.linalg.norm(w)

        Z = svm.decision_function(luoi_diem).reshape(xx.shape)
        ax.contourf(xx, yy, Z > 0, levels=[-0.5, 0.5, 1.5], colors=["#f8d7da", "#d6eaf8"], alpha=0.55)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["#333333", "#111111", "#333333"],
                   linestyles=["--", "-", "--"], linewidths=[1.3, 2.0, 1.3])

        # Vẽ dữ liệu 2 lớp
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#d62728", marker="^", s=45, label="Lop 0", zorder=3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#1f77b4", marker="o", s=45, label="Lop 1", zorder=3)

        # Khoanh vàng các vectơ hỗ trợ
        sv = X[svm.support_]
        ax.scatter(sv[:, 0], sv[:, 1], s=180, facecolors="none", edgecolors="#f39c12",
                   linewidths=2.2, label=f"Vecto ho tro ({len(svm.support_)})", zorder=4)

        if idx_ax == 0:
            ax.scatter([3.0, 5.0], [3.0, 4.5], c="#27ae60", marker="*", s=180,
                       edgecolors="black", linewidths=0.8, label="Diem moi (3,3) & (5,4.5)", zorder=5)
            ax.annotate("(3, 3)\nd=1.40", (3.0, 3.0), xytext=(1.0, 4.8),
                        fontsize=8.5, fontweight="bold", color="#1e8449",
                        arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.0))
            ax.annotate("(5, 4.5)\nd=1.03", (5.0, 4.5), xytext=(5.6, 2.5),
                        fontsize=8.5, fontweight="bold", color="#1e8449",
                        arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.0))
        elif diem_moi is not None:
            ax.scatter([diem_moi[0]], [diem_moi[1]], c="#27ae60", marker="*", s=220,
                       edgecolors="black", linewidths=0.9, label="Diem them moi", zorder=5)

        ax.set_title(f"{tieu_de}\nDo rong le = {le:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-2.0, 9.0)
        ax.set_ylim(-2.0, 9.0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_le_va_sv.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


def ve_bieu_do_kernel_hai_vang_trang():
    """Hình 2: So sánh biên quyết định của Poly (degree 2, 3, 4) và RBF trên hai vầng trăng."""
    df = pd.read_csv(os.path.join(DATA_DIR, "hai_vang_trang.csv"))
    X = df[["x1", "x2"]].to_numpy()
    y = df["nhan"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    mo_hinh = [
        ("Poly bac 2 (C=1)", make_pipeline(StandardScaler(), SVC(kernel="poly", degree=2, C=1))),
        ("Poly bac 3 (C=1)", make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, C=1))),
        ("Poly bac 4 (C=100)", make_pipeline(StandardScaler(), SVC(kernel="poly", degree=4, C=100))),
        ("RBF (C=10, gamma=1)", SVC(kernel="rbf", C=10, gamma=1)),
    ]

    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    luoi = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, (ten, mh) in zip(axes, mo_hinh):
        mh.fit(X_tr, y_tr)
        acc = mh.score(X_te, y_te)
        Z = mh.predict(luoi).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#f8d7da", "#d6eaf8"], alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors="#222222", linewidths=1.4)
        ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], c="#d62728", s=22, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], c="#1f77b4", s=22, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.set_title(f"{ten}\nTest Acc = {acc:.4f}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_kernel_hai_vang_trang.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


def ve_bieu_do_luoi_min_va_thoi_gian():
    """Hình 3: Heatmap CV lưới mịn (Bài 4) và biểu đồ thời gian fit SVC vs LinearSVC (Bài 6)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "hai_vang_trang.csv"))
    X = df[["x1", "x2"]]
    y = df["nhan"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    luoi_min = {"C": [3, 10, 30], "gamma": [0.3, 1, 3]}
    tim = GridSearchCV(SVC(kernel="rbf"), luoi_min, cv=5).fit(X_tr, y_tr)
    bang = pd.DataFrame(tim.cv_results_["params"])
    bang["diem"] = tim.cv_results_["mean_test_score"]
    ma_tran = bang.pivot(index="C", columns="gamma", values="diem")

    # Đo thời gian Bài 6
    X_w, y_w = make_classification(n_samples=100, n_features=20, random_state=0)
    SVC().fit(X_w, y_w)
    LinearSVC(random_state=0).fit(X_w, y_w)

    cac_co = [2000, 5000, 10000, 20000]
    t_svc_list, t_lin_list = [], []
    for n in cac_co:
        Xc, yc = make_classification(n_samples=n, n_features=20, random_state=0)
        t0 = time.perf_counter()
        SVC().fit(Xc, yc)
        t_svc_list.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        LinearSVC(random_state=0).fit(Xc, yc)
        t_lin_list.append(time.perf_counter() - t0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Subplot 1: Heatmap lưới mịn
    im = ax1.imshow(ma_tran.values, cmap="Blues", aspect="auto", vmin=0.84, vmax=0.885)
    ax1.set_xticks(range(len(ma_tran.columns)))
    ax1.set_xticklabels([str(g) for g in ma_tran.columns])
    ax1.set_yticks(range(len(ma_tran.index)))
    ax1.set_yticklabels([str(c) for c in ma_tran.index])
    ax1.set_xlabel("gamma")
    ax1.set_ylabel("C")
    ax1.set_title("Bai 4: Diem CV (5-fold) cua luoi min", fontsize=11.5, fontweight="bold")
    for r in range(ma_tran.shape[0]):
        for c in range(ma_tran.shape[1]):
            val = ma_tran.values[r, c]
            mau_chu = "white" if val > 0.865 else "black"
            ax1.text(c, r, f"{val:.4f}", ha="center", va="center", color=mau_chu, fontweight="bold")
    # Viền vàng ô tốt nhất (C=30, gamma=1 -> row 2, col 1)
    best_r, best_c = np.unravel_index(np.argmax(ma_tran.values), ma_tran.shape)
    rect = plt.Rectangle((best_c - 0.5, best_r - 0.5), 1, 1, fill=False, edgecolor="#f39c12", linewidth=3)
    ax1.add_patch(rect)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Subplot 2: Biểu đồ thời gian
    ax2.plot(cac_co, t_svc_list, marker="o", linewidth=2.2, color="#d62728", label="SVC() (kernel RBF)")
    ax2.plot(cac_co, t_lin_list, marker="s", linewidth=2.2, color="#1f77b4", label="LinearSVC()")
    for x_val, y_s, y_l in zip(cac_co, t_svc_list, t_lin_list):
        ax2.annotate(f"{y_s:.2f}s", (x_val, y_s), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8.5, color="#c0392b", fontweight="bold")
        ax2.annotate(f"{y_l:.3f}s", (x_val, y_l), textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=8.5, color="#1f618d", fontweight="bold")
    ax2.set_title("Bai 6: Thoi gian .fit() theo so luong mau n", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("So mau (n)")
    ax2.set_ylabel("Thoi gian huan luyen (giay)")
    ax2.set_xticks(cac_co)
    ax2.set_ylim(-0.35, max(t_svc_list) * 1.14)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_luoi_min_va_thoi_gian.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


if __name__ == "__main__":
    ve_bieu_do_le_va_sv()
    ve_bieu_do_kernel_hai_vang_trang()
    ve_bieu_do_luoi_min_va_thoi_gian()
