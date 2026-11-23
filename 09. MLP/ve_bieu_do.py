# -*- coding: utf-8 -*-
"""Script vẽ các biểu đồ trực quan phân tích cho Bài thực hành 9 (MLP)."""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAITAP_DIR = os.path.join(BASE_DIR, "baitap09")
DATA_DIR = os.path.join(BASE_DIR, "data")


def ve_bieu_do_xor_bai1_bai2():
    """Hình 1: Bài 1 & Bài 2 — Ranh giới Logistic vs MLP 8 nơ-ron trên XOR và không gian ẩn Bài 2."""
    df = pd.read_csv(os.path.join(DATA_DIR, "xor.csv"))
    X = df[["x1", "x2"]].to_numpy()
    y = df["nhan"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    lr = LogisticRegression().fit(X_tr, y_tr)
    mlp = MLPClassifier(
        hidden_layer_sizes=(8,), solver="lbfgs", max_iter=5000, random_state=0
    ).fit(X_tr, y_tr)

    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 260), np.linspace(-1.1, 1.1, 260))
    luoi = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2))

    # Subplot 1: Hồi quy logistic trên XOR
    ax = axes[0]
    Z_lr = lr.predict(luoi).reshape(xx.shape)
    ax.contourf(xx, yy, Z_lr, levels=[-0.5, 0.5, 1.5], colors=["#f8d7da", "#d6eaf8"], alpha=0.62)
    ax.contour(xx, yy, Z_lr, levels=[0.5], colors="#222222", linewidths=1.5)
    ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], c="#d62728", s=26, alpha=0.8, edgecolors="white", linewidths=0.4, label="Lop 0")
    ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], c="#1f77b4", s=26, alpha=0.8, edgecolors="white", linewidths=0.4, label="Lop 1")
    ax.set_title(
        f"Hoi quy logistic (1 no-ron)\nTest Acc = {accuracy_score(y_te, lr.predict(X_te)):.4f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    # Subplot 2: MLP 8 nơ-ron ẩn + đánh dấu 5 điểm kiểm tra của Bài 1
    ax = axes[1]
    Z_mlp = mlp.predict(luoi).reshape(xx.shape)
    ax.contourf(xx, yy, Z_mlp, levels=[-0.5, 0.5, 1.5], colors=["#f8d7da", "#d6eaf8"], alpha=0.62)
    ax.contour(xx, yy, Z_mlp, levels=[0.5], colors="#222222", linewidths=1.5)
    ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], c="#d62728", s=26, alpha=0.72, edgecolors="white", linewidths=0.4, label="Lop 0")
    ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], c="#1f77b4", s=26, alpha=0.72, edgecolors="white", linewidths=0.4, label="Lop 1")

    X5 = X_te[:5]
    p5 = mlp.predict_proba(X5)[:, 1]
    ax.scatter(X5[:, 0], X5[:, 1], c="#f39c12", marker="*", s=200, edgecolors="black", linewidths=0.8, label="5 diem dau (Bai 1)", zorder=5)
    offsets = [(0.12, -0.25), (0.14, 0.14), (0.04, -0.28), (-0.16, 0.18), (0.14, 0.18)]
    for idx, (pt, prob, off) in enumerate(zip(X5, p5, offsets), start=1):
        ax.annotate(
            f"#{idx} (p={prob:.2f})",
            (pt[0], pt[1]),
            xytext=(pt[0] + off[0], pt[1] + off[1]),
            fontsize=8.0,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#f39c12", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9),
            zorder=6,
        )
    ax.set_title(
        f"Bai 1: MLP 8 no-ron an (XOR)\nTest Acc = {accuracy_score(y_te, mlp.predict(X_te)):.4f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    # Subplot 3: Không gian ẩn (h1, h2) của mạng XOR Bài 2
    ax = axes[2]
    h_grid = np.linspace(-0.2, 1.3, 200)
    hh1, hh2 = np.meshgrid(h_grid, h_grid)
    Z_h = (hh1 + hh2 >= 0.5).astype(int)
    ax.contourf(hh1, hh2, Z_h, levels=[-0.5, 0.5, 1.5], colors=["#f8d7da", "#d6eaf8"], alpha=0.62)
    t_line = np.linspace(-0.15, 0.65, 100)
    ax.plot(t_line, 0.5 - t_line, "k--", linewidth=1.8, label="Ranh gioi h1 + h2 = 0.5")

    ax.scatter([0.0], [0.0], c="#d62728", s=160, edgecolors="black", linewidths=1.1, zorder=5, label="Lop 0: (0,0) & (1,1)")
    ax.scatter([1.0, 0.0], [0.0, 1.0], c="#1f77b4", s=160, marker="s", edgecolors="black", linewidths=1.1, zorder=5, label="Lop 1: (1,0) & (0,1)")

    ax.annotate(
        "(0,0) & (1,1) -> (0, 0)\n   y = 0",
        (0.0, 0.0),
        xytext=(0.06, -0.20),
        fontsize=8.4,
        fontweight="bold",
        color="#922b21",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d62728", alpha=0.94),
        arrowprops=dict(arrowstyle="->", color="#922b21", lw=1.0),
    )
    ax.annotate(
        "(1,0) -> (1, 0)\n   y = 1",
        (1.0, 0.0),
        xytext=(0.64, 0.30),
        fontsize=8.4,
        fontweight="bold",
        color="#1b4f72",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#1f77b4", alpha=0.94),
        arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=1.0),
    )
    ax.annotate(
        "(0,1) -> (0, 1)\n   y = 1",
        (0.0, 1.0),
        xytext=(0.24, 0.94),
        fontsize=8.4,
        fontweight="bold",
        color="#1b4f72",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#1f77b4", alpha=0.94),
        arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=1.0),
    )
    ax.set_title("Bai 2: Khong gian an (h1, h2)\nReLU gap phang tach tuyen tinh XOR", fontsize=11, fontweight="bold")
    ax.set_xlabel("h1 = max(0, x1 - x2)")
    ax.set_ylabel("h2 = max(0, x2 - x1)")
    ax.set_xlim(-0.18, 1.25)
    ax.set_ylim(-0.26, 1.25)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_xor_bai1_bai2.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


def ve_bieu_do_kich_hoat_bai3():
    """Hình 2: Bài 3 — So sánh ranh giới của logistic, tanh và relu trên hai vầng trăng."""
    df = pd.read_csv(os.path.join(DATA_DIR, "hai_vang_trang.csv"))
    X = df[["x1", "x2"]]
    y = df["nhan"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    x_min, x_max = X["x1"].min() - 0.35, X["x1"].max() + 0.35
    y_min, y_max = X["x2"].min() - 0.35, X["x2"].max() + 0.35
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    luoi_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["x1", "x2"])

    X_tr_np, y_tr_np = X_tr.to_numpy(), y_tr.to_numpy()
    X_te_np, y_te_np = X_te.to_numpy(), y_te.to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))
    for ax, ham in zip(axes, ["logistic", "tanh", "relu"]):
        def tao():
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(100, 100),
                    alpha=1.0,
                    activation=ham,
                    solver="lbfgs",
                    max_iter=20000,
                    random_state=0,
                ),
            )

        mh = tao().fit(X_tr, y_tr)
        acc_tr = accuracy_score(y_tr, mh.predict(X_tr))
        acc_te = accuracy_score(y_te, mh.predict(X_te))
        cv = cross_val_score(tao(), X_tr, y_tr, cv=5).mean()

        Z = mh.predict(luoi_df).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#f8d7da", "#d6eaf8"], alpha=0.62)
        ax.contour(xx, yy, Z, levels=[0.5], colors="#222222", linewidths=1.6)
        ax.scatter(X_tr_np[y_tr_np == 0, 0], X_tr_np[y_tr_np == 0, 1], c="#d62728", s=22, alpha=0.65, edgecolors="white", linewidths=0.35, label="Lop 0 (hoc)")
        ax.scatter(X_tr_np[y_tr_np == 1, 0], X_tr_np[y_tr_np == 1, 1], c="#1f77b4", s=22, alpha=0.65, edgecolors="white", linewidths=0.35, label="Lop 1 (hoc)")
        ax.scatter(X_te_np[y_te_np == 0, 0], X_te_np[y_te_np == 0, 1], c="#d62728", marker="^", s=46, edgecolors="black", linewidths=0.6, label="Lop 0 (kiem tra)")
        ax.scatter(X_te_np[y_te_np == 1, 0], X_te_np[y_te_np == 1, 1], c="#1f77b4", marker="^", s=46, edgecolors="black", linewidths=0.6, label="Lop 1 (kiem tra)")

        danh_dau = " [TOT NHAT]" if ham == "relu" else ""
        ax.set_title(
            f"activation = '{ham}'{danh_dau}\nTrain={acc_tr:.4f} | Test={acc_te:.4f} | CV={cv:.4f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="lower left", fontsize=7.6, framealpha=0.9)

    fig.suptitle(
        "Bai 3: Anh huong cua ham kich hoat tren hai vang trang - mang (100, 100), alpha = 1.0",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_kich_hoat_hai_vang_trang.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


def ve_bieu_do_luoi_va_ngoai_suy():
    """Hình 3: Heatmap GridSearchCV (Bài 5) và so sánh ngoại suy giá nhà 250 m² (Bài 6)."""
    # --- Phần trái: Heatmap Bài 5 ---
    Xd, yd = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xd, yd, test_size=0.25, random_state=17, stratify=yd
    )
    pipe = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, random_state=0))
    luoi = {
        "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
        "mlpclassifier__alpha": [0.0001, 0.01, 1],
    }
    tim = GridSearchCV(pipe, luoi, cv=5, n_jobs=-1).fit(X_tr, y_tr)
    acc_mlp_best = accuracy_score(y_te, tim.predict(X_te))
    acc_svm = accuracy_score(y_te, SVC().fit(X_tr, y_tr).predict(X_te))

    bang = pd.DataFrame({
        "lop_an": [str(p["mlpclassifier__hidden_layer_sizes"]) for p in tim.cv_results_["params"]],
        "alpha": [p["mlpclassifier__alpha"] for p in tim.cv_results_["params"]],
        "cv": tim.cv_results_["mean_test_score"],
    })
    thu_tu = ["(10,)", "(50,)", "(100,)", "(50, 50)"]
    ma_tran = bang.pivot(index="lop_an", columns="alpha", values="cv").loc[thu_tu]

    # --- Phần phải: Ngoại suy Bài 6 ---
    dfg = pd.read_csv(os.path.join(DATA_DIR, "gia_nha.csv"))
    Xg = dfg[["dien_tich"]]
    yg = dfg["gia"]
    Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg, test_size=0.2, random_state=42)

    hqtt = LinearRegression().fit(Xg_tr, yg_tr)
    mlp_relu = make_pipeline(
        StandardScaler(), MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0)
    ).fit(Xg_tr, yg_tr)
    mlp_tanh = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), activation="tanh", max_iter=5000, random_state=0),
    ).fit(Xg_tr, yg_tr)
    mlp_sig = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), activation="logistic", max_iter=5000, random_state=0),
    ).fit(Xg_tr, yg_tr)

    mae_lin = mean_absolute_error(yg_te, hqtt.predict(Xg_te))
    mae_relu = mean_absolute_error(yg_te, mlp_relu.predict(Xg_te))

    x_plot = np.linspace(30, 260, 300)
    df_plot = pd.DataFrame({"dien_tich": x_plot})
    y_lin = hqtt.predict(df_plot)
    y_relu = mlp_relu.predict(df_plot)
    y_tanh = mlp_tanh.predict(df_plot)
    y_sig = mlp_sig.predict(df_plot)

    df_250 = pd.DataFrame({"dien_tich": [250.0]})
    p250_lin = hqtt.predict(df_250)[0]
    p250_relu = mlp_relu.predict(df_250)[0]
    p250_tanh = mlp_tanh.predict(df_250)[0]
    p250_sig = mlp_sig.predict(df_250)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.4))

    # Subplot 1: Heatmap Bài 5
    im = ax1.imshow(ma_tran.values, cmap="Blues", aspect="auto", vmin=0.945, vmax=0.982)
    ax1.set_xticks(range(len(ma_tran.columns)))
    ax1.set_xticklabels([str(a) for a in ma_tran.columns])
    ax1.set_yticks(range(len(ma_tran.index)))
    ax1.set_yticklabels(ma_tran.index)
    ax1.set_xlabel("alpha (dieu chuan L2)", fontsize=10.5)
    ax1.set_ylabel("hidden_layer_sizes", fontsize=10.5)
    ax1.set_title(
        f"Bai 5: GridSearchCV 5-fold tren bo chu so\n"
        f"Tot nhat: (50,), alpha=1 | MLP Test={acc_mlp_best:.4f} < SVM={acc_svm:.4f}",
        fontsize=10.8,
        fontweight="bold",
    )
    for r in range(ma_tran.shape[0]):
        for c in range(ma_tran.shape[1]):
            val = ma_tran.values[r, c]
            mau_chu = "white" if val > 0.968 else "black"
            ax1.text(c, r, f"{val:.4f}", ha="center", va="center", color=mau_chu, fontweight="bold", fontsize=10)
    best_r, best_c = np.unravel_index(np.argmax(ma_tran.values), ma_tran.shape)
    rect = plt.Rectangle((best_c - 0.5, best_r - 0.5), 1, 1, fill=False, edgecolor="#f39c12", linewidth=3.2)
    ax1.add_patch(rect)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="5-fold CV Accuracy")

    # Subplot 2: Ngoại suy Bài 6
    ax2.axvspan(35.5, 117.5, color="#e8f8f5", alpha=0.85, label="Mien du lieu da hoc [35.5, 117.5] m2")
    ax2.axvline(117.5, color="#7f8c8d", linestyle=":", linewidth=1.3)
    ax2.axvline(250.0, color="#95a5a6", linestyle="--", linewidth=1.0)

    ax2.scatter(Xg_tr["dien_tich"], yg_tr, c="#34495e", s=26, alpha=0.65, label="Tap hoc (48 can)", zorder=3)
    ax2.scatter(Xg_te["dien_tich"], yg_te, c="#27ae60", marker="^", s=50, edgecolors="black", linewidths=0.5, label="Tap kiem tra (12 can)", zorder=4)

    ax2.plot(x_plot, y_lin, color="#1f77b4", linewidth=2.4, label=f"Hoi quy tuyen tinh (MAE={mae_lin:.3f})")
    ax2.plot(x_plot, y_relu, color="#d62728", linewidth=2.4, linestyle="--", label=f"MLP (20,) ReLU (MAE={mae_relu:.3f})")
    ax2.plot(x_plot, y_tanh, color="#ff7f0e", linewidth=1.9, linestyle="-.", label="MLP (20,) tanh (bao hoa)")
    ax2.plot(x_plot, y_sig, color="#8e44ad", linewidth=1.8, linestyle=":", label="MLP (20,) logistic (bao hoa)")

    ax2.scatter([250, 250, 250, 250], [p250_lin, p250_relu, p250_tanh, p250_sig],
                c=["#1f77b4", "#d62728", "#ff7f0e", "#8e44ad"], s=65, edgecolors="black", zorder=5)

    ax2.annotate(f"ReLU: {p250_relu:.2f} ty\n(+8.26 ty)", (250, p250_relu), xytext=(155, 27.2),
                 fontsize=8.5, fontweight="bold", color="#c0392b",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))
    ax2.annotate(f"Tuyen tinh: {p250_lin:.2f} ty", (250, p250_lin), xytext=(164, 15.6),
                 fontsize=8.5, fontweight="bold", color="#1f618d",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color="#1f618d", lw=1.0))
    ax2.annotate(f"tanh: {p250_tanh:.2f} ty", (250, p250_tanh), xytext=(178, 12.6),
                 fontsize=8.3, fontweight="bold", color="#d35400",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.0))
    ax2.annotate(f"logistic: {p250_sig:.2f} ty", (250, p250_sig), xytext=(165, 6.2),
                 fontsize=8.3, fontweight="bold", color="#6c3483",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color="#6c3483", lw=1.0))

    ax2.set_title("Bai 6: Ngoai suy gia can 250 m2 (Tuyen tinh vs MLPRegressor)", fontsize=10.8, fontweight="bold")
    ax2.set_xlabel("Dien tich (m2)", fontsize=10.5)
    ax2.set_ylabel("Gia ban (ty dong)", fontsize=10.5)
    ax2.set_xlim(28, 265)
    ax2.set_ylim(1.5, 31.5)
    ax2.grid(True, linestyle=":", alpha=0.55)
    ax2.legend(loc="upper left", fontsize=7.9, framealpha=0.92)

    fig.tight_layout()
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_luoi_va_ngoai_suy.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


if __name__ == "__main__":
    ve_bieu_do_xor_bai1_bai2()
    ve_bieu_do_kich_hoat_bai3()
    ve_bieu_do_luoi_va_ngoai_suy()
