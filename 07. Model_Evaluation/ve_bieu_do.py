# -*- coding: utf-8 -*-
"""Script vẽ các biểu đồ trực quan phân tích cho Bài thực hành 7 (Đánh giá mô hình)."""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAITAP_DIR = os.path.join(BASE_DIR, "baitap07")
DATA_DIR = os.path.join(BASE_DIR, "data")


def doc_du_lieu_bo_hoc():
    df = pd.read_csv(os.path.join(DATA_DIR, "sinh_vien_bo_hoc.csv"))
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]
    return train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)


def ve_bieu_do_accuracy_va_nguong():
    """Hình 1: So sánh thước đo theo mô hình/ngưỡng (Bài 1 & 3) và đường CV chọn ngưỡng (Bài 3)."""
    X_tr, X_te, y_tr, y_te = doc_du_lieu_bo_hoc()

    luoi = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
    xs_te = lr.predict_proba(X_te)[:, 1]

    cac_cau_hinh = [
        ("Mo hinh luoi\n(mac dinh)", luoi.predict(X_te)),
        ("Logistic\n(t = 0.50)", (xs_te >= 0.50).astype(int)),
        ("Logistic\n(t = 0.25, max F1)", (xs_te >= 0.25).astype(int)),
        ("Logistic\n(t = 0.15, Bai 3)", (xs_te >= 0.15).astype(int)),
    ]

    ten_list = [c[0] for c in cac_cau_hinh]
    accs = [accuracy_score(y_te, d) for _, d in cac_cau_hinh]
    precs = [precision_score(y_te, d, zero_division=0) for _, d in cac_cau_hinh]
    recs = [recall_score(y_te, d) for _, d in cac_cau_hinh]
    f1s = [f1_score(y_te, d, zero_division=0) for _, d in cac_cau_hinh]

    # Đường Precision / Recall / F1 theo ngưỡng trên 5-fold CV tập học (Bài 3)
    xs_hoc = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression()),
        X_tr,
        y_tr,
        cv=5,
        method="predict_proba",
    )[:, 1]
    nguong_grid = np.round(np.arange(0.05, 0.91, 0.05), 2)
    p_cv = [precision_score(y_tr, (xs_hoc >= t).astype(int), zero_division=0) for t in nguong_grid]
    r_cv = [recall_score(y_tr, (xs_hoc >= t).astype(int)) for t in nguong_grid]
    f_cv = [f1_score(y_tr, (xs_hoc >= t).astype(int), zero_division=0) for t in nguong_grid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.4))

    # Subplot 1: Grouped bars Accuracy vs Precision vs Recall vs F1
    x = np.arange(len(ten_list))
    w = 0.19
    b1 = ax1.bar(x - 1.5 * w, accs, w, label="Accuracy", color="#4e79a7")
    b2 = ax1.bar(x - 0.5 * w, precs, w, label="Precision", color="#f28e2b")
    b3 = ax1.bar(x + 0.5 * w, recs, w, label="Recall", color="#e15759")
    b4 = ax1.bar(x + 1.5 * w, f1s, w, label="F1-score", color="#59a14f")

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    ax1.axhline(0.90, color="#666666", linestyle=":", linewidth=1.1, label="Muc san accuracy (0.90)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ten_list, fontsize=9)
    ax1.set_ylim(0, 1.24)
    ax1.set_ylabel("Gia tri thuoc do (tap kiem tra)", fontsize=10.5)
    ax1.set_title("Bai 1 & Bai 3: Accuracy gan nhau, nhung Recall khac han", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.55)
    ax1.legend(loc="upper center", ncol=3, fontsize=8.4, framealpha=0.94)

    # Subplot 2: Chọn ngưỡng trên 5-fold CV (tập học)
    ax2.plot(nguong_grid, p_cv, marker="o", markersize=4.5, linewidth=2.0, color="#1f4e79", label="Precision (CV)")
    ax2.plot(nguong_grid, r_cv, marker="s", markersize=4.5, linewidth=2.1, color="#c0392b", linestyle="--", label="Recall (CV)")
    ax2.plot(nguong_grid, f_cv, marker="^", markersize=4.5, linewidth=2.0, color="#2e7d32", linestyle=":", label="F1 (CV)")

    ax2.axhline(0.80, color="#c0392b", linestyle="-.", linewidth=1.2, alpha=0.75, label="Muc tieu Recall >= 0.80")
    ax2.axvline(0.15, color="#e67e22", linestyle="--", linewidth=1.6)
    ax2.axvline(0.25, color="#2e7d32", linestyle="--", linewidth=1.3, alpha=0.7)

    ax2.annotate(
        "Chon t = 0.15 (Bai 3)\nCV Recall = 0.8315\nTest Recall = 0.8333",
        xy=(0.15, 0.8315),
        xytext=(0.30, 0.87),
        fontsize=8.5,
        fontweight="bold",
        color="#b35900",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff8eb", ec="#e67e22", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2),
    )
    ax2.annotate(
        "t = 0.25\n(Max CV F1 = 0.6473)",
        xy=(0.25, 0.6473),
        xytext=(0.33, 0.30),
        fontsize=8.3,
        fontweight="bold",
        color="#1e6b24",
        bbox=dict(boxstyle="round,pad=0.28", fc="#eef9ef", ec="#2e7d32", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.1),
    )

    ax2.set_xlabel("Nguong xac suat (t)", fontsize=10.5)
    ax2.set_ylabel("Gia tri tren 5-fold CV tap hoc", fontsize=10.5)
    ax2.set_title("Bai 3: Chon nguong bang cross_val_predict tren tap hoc", fontsize=11.5, fontweight="bold")
    ax2.set_xlim(0.02, 0.93)
    ax2.set_ylim(0.0, 1.07)
    ax2.grid(True, linestyle=":", alpha=0.55)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    fig.tight_layout(pad=2.0)
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_ma_tran_va_nguong.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


def ve_bieu_do_roc_va_can_bang():
    """Hình 2: So sánh đường ROC của 6 mô hình (Mục 6 & Bài 4) và tác động cân bằng lớp (Bài 5)."""
    X_tr, X_te, y_tr, y_te = doc_du_lieu_bo_hoc()

    mo_hinh = {
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
        "Naive Bayes": GaussianNB(),
        "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "Rung 200 cay": RandomForestClassifier(200, random_state=0),
        "SVM RBF": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
        "Cay sau 4": DecisionTreeClassifier(max_depth=4, random_state=0),
    }
    mau_roc = ["#1f4e79", "#2e7d32", "#8e44ad", "#d95f02", "#c0392b", "#7f8c8d"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.5))

    # Subplot 1: Đường ROC của cả 6 mô hình
    for (ten, mh), mau in zip(mo_hinh.items(), mau_roc):
        mh.fit(X_tr, y_tr)
        p = mh.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        auc = roc_auc_score(y_te, p)
        lw = 2.4 if ten in ("Logistic", "Rung 200 cay") else 1.5
        ls = "-" if ten != "Rung 200 cay" else "-."
        ax1.plot(fpr, tpr, color=mau, linewidth=lw, linestyle=ls, label=f"{ten} (AUC = {auc:.4f})")

    ax1.plot([0, 1], [0, 1], color="#444444", linestyle="--", linewidth=1.1, label="Ngau nhien (AUC = 0.5000)")
    ax1.set_title("Muc 6 & Bai 4: Duong ROC cua 6 mo hinh tren tap kiem tra", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Ty le bao dong nham (FPR)", fontsize=10.5)
    ax1.set_ylabel("Recall (TPR)", fontsize=10.5)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.04)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="lower right", fontsize=8.6, framealpha=0.94)

    # Subplot 2: Trước vs Sau class_weight='balanced' (Bài 5 + đối chiếu Logistic)
    cac_mo_hinh_cb = [
        (
            "RandomForest",
            RandomForestClassifier(200, random_state=0),
            RandomForestClassifier(200, random_state=0, class_weight="balanced"),
        ),
        (
            "SVC (RBF)",
            make_pipeline(StandardScaler(), SVC(random_state=0)),
            make_pipeline(StandardScaler(), SVC(random_state=0, class_weight="balanced")),
        ),
        (
            "Logistic\n(doi chieu)",
            make_pipeline(StandardScaler(), LogisticRegression()),
            make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")),
        ),
    ]

    ten_cb = []
    p_goc, p_cb, r_goc, r_cb, f_goc, f_cb = [], [], [], [], [], []
    for ten, m0, m1 in cac_mo_hinh_cb:
        d0 = m0.fit(X_tr, y_tr).predict(X_te)
        d1 = m1.fit(X_tr, y_tr).predict(X_te)
        ten_cb.append(ten)
        p_goc.append(precision_score(y_te, d0, zero_division=0))
        p_cb.append(precision_score(y_te, d1, zero_division=0))
        r_goc.append(recall_score(y_te, d0))
        r_cb.append(recall_score(y_te, d1))
        f_goc.append(f1_score(y_te, d0))
        f_cb.append(f1_score(y_te, d1))

    idx = np.arange(len(ten_cb))
    w = 0.13
    bars_info = [
        (idx - 2.5 * w, p_goc, "#85c1e9", "Prec (goc)"),
        (idx - 1.5 * w, p_cb, "#1f618d", "Prec (balanced)"),
        (idx - 0.5 * w, r_goc, "#f5b7b1", "Recall (goc)"),
        (idx + 0.5 * w, r_cb, "#c0392b", "Recall (balanced)"),
        (idx + 1.5 * w, f_goc, "#a9dfbf", "F1 (goc)"),
        (idx + 2.5 * w, f_cb, "#1e8449", "F1 (balanced)"),
    ]
    for pos, vals, col, lbl in bars_info:
        rects = ax2.bar(pos, vals, w, color=col, label=lbl, edgecolor="white", linewidth=0.5)
        for rect in rects:
            h = rect.get_height()
            ax2.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + 0.014,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
                rotation=0,
            )

    ax2.set_xticks(idx)
    ax2.set_xticklabels(ten_cb, fontsize=10)
    ax2.set_ylim(0, 1.16)
    ax2.set_ylabel("Gia tri thuoc do (tap kiem tra)", fontsize=10.5)
    ax2.set_title("Bai 5: Truoc vs Sau class_weight='balanced' (SVC doi manh nhat)", fontsize=11.5, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.55)
    ax2.legend(loc="upper right", ncol=3, fontsize=8.0, framealpha=0.92)

    fig.tight_layout(pad=2.0)
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_roc_va_can_bang.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


def ve_bieu_do_chu_so_va_ngoai_lai():
    """Hình 3: Precision/Recall từng chữ số (Bài 2) và tác động +5 tỷ lên MAE, RMSE, R^2 (Bài 6)."""
    # Dữ liệu Bài 2 (load_digits + SVM RBF)
    Xd, yd = load_digits(return_X_y=True)
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
        Xd, yd, test_size=0.25, random_state=17, stratify=yd
    )
    svm = SVC(kernel="rbf").fit(Xd_tr, yd_tr)
    rep = classification_report(yd_te, svm.predict(Xd_te), output_dict=True)
    digits = [str(i) for i in range(10)]
    p_dig = [rep[d]["precision"] for d in digits]
    r_dig = [rep[d]["recall"] for d in digits]

    # Dữ liệu Bài 6 (gia_nha.csv + LinearRegression)
    dg = pd.read_csv(os.path.join(DATA_DIR, "gia_nha.csv"))
    Xg = dg[["dien_tich", "so_phong", "tuoi_nha"]]
    yg = dg["gia"]
    Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(Xg_tr, yg_tr)
    pred0 = reg.predict(Xg_te)
    pred1 = pred0.copy()
    pred1[0] += 5.0

    mae0, rmse0, r20 = (
        mean_absolute_error(yg_te, pred0),
        float(np.sqrt(mean_squared_error(yg_te, pred0))),
        r2_score(yg_te, pred0),
    )
    mae1, rmse1, r21 = (
        mean_absolute_error(yg_te, pred1),
        float(np.sqrt(mean_squared_error(yg_te, pred1))),
        r2_score(yg_te, pred1),
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.9), gridspec_kw={"width_ratios": [1.55, 1.0, 0.85]})

    # Subplot 1: Bài 2 — Precision & Recall từng chữ số
    x = np.arange(10)
    w = 0.36
    ax1.bar(x - w / 2, p_dig, w, color="#2980b9", label="Precision")
    ax1.bar(x + w / 2, r_dig, w, color="#e67e22", label="Recall")
    ax1.axhline(rep["macro avg"]["f1-score"], color="#27ae60", linestyle="--", linewidth=1.3,
                label=f"macro/weighted avg (~{rep['macro avg']['f1-score']:.4f})")
    ax1.annotate(
        "So 8 thap nhat\nP = R = 0.9767",
        xy=(8, 0.9767),
        xytext=(5.1, 0.961),
        fontsize=8.5,
        fontweight="bold",
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.28", fc="#fdecea", ec="#c0392b", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(digits)
    ax1.set_ylim(0.955, 1.008)
    ax1.set_xlabel("Chu so (0..9)", fontsize=10.5)
    ax1.set_ylabel("Gia tri (zoom 0.955 -> 1.005)", fontsize=10)
    ax1.set_title("Bai 2: Precision & Recall 10 chu so (SVM RBF)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.55)
    ax1.legend(loc="lower left", fontsize=8.3, framealpha=0.92)

    # Subplot 2: Bài 6 — MAE vs RMSE trước và sau khi cộng +5 tỷ
    x2 = np.arange(2)
    w2 = 0.34
    g0 = ax2.bar(x2 - w2 / 2, [mae0, rmse0], w2, color="#4e79a7", label="Truoc (+0 ty)")
    g1 = ax2.bar(x2 + w2 / 2, [mae1, rmse1], w2, color="#e15759", label="Sau (+5 ty, 1 can)")
    for bars in (g0, g1):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.04,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    ax2.annotate(f"x{mae1/mae0:.2f}", xy=(0 + w2 / 2, mae1), xytext=(0 + w2 / 2, mae1 + 0.22),
                 ha="center", fontsize=9, fontweight="bold", color="#b03a2e")
    ax2.annotate(f"x{rmse1/rmse0:.2f}", xy=(1 + w2 / 2, rmse1), xytext=(1 + w2 / 2, rmse1 + 0.22),
                 ha="center", fontsize=9.5, fontweight="bold", color="#b03a2e")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["MAE (ty)", "RMSE (ty)"], fontsize=10)
    ax2.set_ylim(0, 1.85)
    ax2.set_ylabel("Sai so (ty dong)", fontsize=10)
    ax2.set_title("Bai 6: RMSE tang gap 4.92x vs MAE 2.72x", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.55)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # Subplot 3: Bài 6 — R^2 trước và sau khi cộng +5 tỷ
    b_r2 = ax3.bar(["Truoc\n(+0 ty)", "Sau\n(+5 ty, 1 can)"], [r20, r21], width=0.5, color=["#2e7d32", "#c0392b"])
    for bar in b_r2:
        h = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.025,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
        )
    ax3.annotate(
        f"Giam -{r20 - r21:.4f}",
        xy=(1.0, r21 + 0.11),
        xytext=(0.45, 0.78),
        fontsize=8.8,
        fontweight="bold",
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fdecea", ec="#c0392b", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )
    ax3.set_ylim(0, 1.15)
    ax3.set_ylabel("He so xac dinh R^2", fontsize=10)
    ax3.set_title("Bai 6: R^2 tut manh do SSE", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle=":", alpha=0.55)

    fig.tight_layout(pad=2.0)
    save_path = os.path.join(BAITAP_DIR, "hinh_anh_chu_so_va_ngoai_lai.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Da tao bieu do: {save_path}")


if __name__ == "__main__":
    ve_bieu_do_accuracy_va_nguong()
    ve_bieu_do_roc_va_can_bang()
    ve_bieu_do_chu_so_va_ngoai_lai()
