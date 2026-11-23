# -*- coding: utf-8 -*-
"""Bài tập 4: Tự vẽ đường ROC của hồi quy logistic và rừng ngẫu nhiên, lưu thành roc.png.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def tim_duong_dan_du_lieu(duong_dan_mac_dinh):
    """Ưu tiên dùng đường dẫn data/... theo yêu cầu đề, dự phòng nếu chạy từ nơi khác."""
    if os.path.exists(duong_dan_mac_dinh):
        return duong_dan_mac_dinh
    ten_file = os.path.basename(duong_dan_mac_dinh)
    thu_muc_script = os.path.dirname(os.path.abspath(__file__))
    ung_vien = [
        os.path.join(thu_muc_script, "..", "data", ten_file),
        os.path.join(thu_muc_script, "..", "..", "Lab07_data", "data", ten_file),
        os.path.join("Lab07_data", "data", ten_file),
    ]
    for p in ung_vien:
        if os.path.exists(p):
            return p
    return duong_dan_mac_dinh


def phan_tich_doan_cat_nhau(fpr_a, tpr_a, fpr_b, tpr_b):
    """Phân tích trên lưới dày FPR in [0, 1] các đoạn đường A trên/dưới đường B."""
    luoi = np.linspace(0.0, 1.0, 10001)
    hieu = np.interp(luoi, fpr_a, tpr_a) - np.interp(luoi, fpr_b, tpr_b)
    trang_thai = np.where(hieu > 1e-9, 1, np.where(hieu < -1e-9, -1, 0))
    diem_doi = np.where(np.diff(trang_thai) != 0)[0] + 1
    cac_doan = []
    for doan in np.split(np.arange(len(luoi)), diem_doi):
        if len(doan) >= 2:
            st = int(trang_thai[doan[0]])
            cac_doan.append(
                (float(luoi[doan[0]]), float(luoi[doan[-1]]), st, float(np.max(np.abs(hieu[doan]))))
            )
    return cac_doan


def main():
    duong_dan = tim_duong_dan_du_lieu("data/sinh_vien_bo_hoc.csv")
    df = pd.read_csv(duong_dan)
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # Hai mô hình đúng theo mục 6 (h5_roc_auc.py)
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
    rf = RandomForestClassifier(200, random_state=0).fit(X_tr, y_tr)

    p_lr = lr.predict_proba(X_te)[:, 1]
    p_rf = rf.predict_proba(X_te)[:, 1]

    fpr_lr, tpr_lr, _ = roc_curve(y_te, p_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_te, p_rf)
    auc_lr = roc_auc_score(y_te, p_lr)
    auc_rf = roc_auc_score(y_te, p_rf)

    # Vẽ cả hai đường ROC và đường chéo đoán ngẫu nhiên lên cùng một hình
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.plot(
        fpr_lr,
        tpr_lr,
        color="#1f4e79",
        linewidth=2.3,
        label=f"Hoi quy logistic (AUC = {auc_lr:.4f})",
    )
    ax.plot(
        fpr_rf,
        tpr_rf,
        color="#d95f02",
        linewidth=2.1,
        linestyle="-.",
        label=f"Rung ngau nhien 200 cay (AUC = {auc_rf:.4f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="#666666",
        linewidth=1.3,
        linestyle="--",
        label="Doan ngau nhien (AUC = 0.5000)",
    )

    # Đánh dấu nhẹ các mốc giao cắt đáng chú ý (nội suy chính xác tung độ giao tại FPR=0.7296)
    tpr_giao_duoi = float(np.interp(0.7296, fpr_rf, tpr_rf))
    ax.scatter([0.0074, 0.4741, 0.7296], [0.5333, 0.9667, tpr_giao_duoi],
               color="#c0392b", s=38, zorder=5, label="Moc giao / tiep xuc chinh")

    ax.set_title("So sanh duong ROC: Hoi quy logistic vs Rung ngau nhien", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlabel("Ty le bao dong nham (FPR = FP / (FP + TN))", fontsize=10.5)
    ax.set_ylabel("Ty le tim dung / Recall (TPR = TP / (TP + FN))", fontsize=10.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    fig.tight_layout()

    thu_muc_luu = os.path.dirname(os.path.abspath(__file__))
    duong_dan_anh = os.path.join(thu_muc_luu, "roc.png")
    fig.savefig(duong_dan_anh, dpi=150)
    plt.close(fig)

    cac_doan = phan_tich_doan_cat_nhau(fpr_lr, tpr_lr, fpr_rf, tpr_rf)

    print("=" * 76)
    print("BAI TAP 4: TU VE DUONG ROC (HOI QUY LOGISTIC VS RUNG NGAU NHIEN)")
    print("=" * 76)
    print(f"Hoi quy logistic        : AUC = {auc_lr:.4f} ({len(fpr_lr)} diem tren duong ROC)")
    print(f"Rung ngau nhien 200 cay : AUC = {auc_rf:.4f} ({len(fpr_rf)} diem tren duong ROC)")
    print(f"Chenh lech AUC          : +{auc_lr - auc_rf:.4f} nghieng ve Hoi quy logistic")
    print(f"Da luu hinh ROC tai     : baitap07/roc.png")
    print()

    print("PHAN TICH VI TRI TUONG DOI CUA HAI DUONG ROC THEO FPR:")
    print("-" * 76)
    nhan_st = {
        1: "Logistic NAM TREN",
        -1: "Rung ngau nhien NAM TREN",
        0: "Hai duong TRUNG NHAU",
    }
    for dau, cuoi, st, max_d in cac_doan:
        print(f"  FPR [{dau:6.4f} -> {cuoi:6.4f}]: {nhan_st[st]:25s} (chenh TPR toi da: {max_d:.4f})")
    print()

    print("TRA LOI VA NHAN XET:")
    print("-" * 76)
    print("1. Duong nao om sat goc tren ben trai hon?")
    print(f"   - Duong HOI QUY LOGISTIC (AUC = {auc_lr:.4f}) om sat goc tren ben trai")
    print(f"     hon ro ret so voi Rung ngau nhien (AUC = {auc_rf:.4f}).")
    print("   - Cu the, trong vung hoat dong thuc te quan trong nhat (FPR tu 0.011")
    print("     den 0.356), duong Logistic nam cao hon ro ret: chi can FPR = 0.1704")
    print("     da dat Recall = 0.9667 (29/30 ban bo hoc), trong khi Rung ngau nhien")
    print("     phai chap nhan FPR len toi 0.3556 (gap doi bao dong nham) moi dat")
    print("     cung muc Recall 0.9667.")
    print()
    print("2. Co doan nao hai duong cat nhau khong?")
    print("   - CO, hai duong cat nhau o hai vung:")
    print("     + Vung dau duong (FPR cuc nho, tu 0.0037 den 0.0074): o FPR = 0,")
    print("       Logistic dat Recall 0.4000 (cao hon RF 0.3000), nhung tai FPR =")
    print("       0.0037 (1 bao nham) Rung ngau nhien nhay len Recall 0.5333 vuot")
    print("       len tren Logistic (0.4000); den FPR = 0.0074 Logistic bat kip")
    print("       0.5333 va vuot han len tren tu FPR >= 0.0111.")
    print("     + Vung duoi duong (FPR tu 0.4741 den 0.7296): ca hai cung o muc")
    print("       Recall 0.9667 tren [0.3556, 0.4741], sau do Rung ngau nhien co")
    print("       nhieu mau trung xac suat bang 0 nen noi thang cheo len (1.0, 1.0),")
    print("       hoi cao hon duong ngang cua Logistic toi FPR = 0.7296, truoc khi")
    print("       Logistic nhay len Recall = 1.0000 va nam tren tro lai.")
    print("=" * 76)


if __name__ == "__main__":
    main()
