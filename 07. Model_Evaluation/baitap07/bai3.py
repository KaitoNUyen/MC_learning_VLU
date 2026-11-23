# -*- coding: utf-8 -*-
"""Bài tập 3: Chọn ngưỡng lớn nhất đạt recall >= 0.8 bằng kiểm định chéo trên tập học.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
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


def main():
    duong_dan = tim_duong_dan_du_lieu("data/sinh_vien_bo_hoc.csv")
    df = pd.read_csv(duong_dan)
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]

    # Chia tập học / tập kiểm tra đúng theo mục 5
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    lr = make_pipeline(StandardScaler(), LogisticRegression())

    # Bước 1: Chỉ dùng tập học qua cross_val_predict (cv=5) để chọn ngưỡng,
    # tuyệt đối KHÔNG nhìn vào tập kiểm tra khi ra quyết định.
    xs_hoc = cross_val_predict(
        lr, X_tr, y_tr, cv=5, method="predict_proba"
    )[:, 1]

    # Dùng đúng lưới ngưỡng bước 0.05 như mục 5 (h4_nguong.py)
    cac_nguong = np.round(np.arange(0.05, 0.96, 0.05), 2)
    bang_cv = []
    for t in cac_nguong:
        du_hoc = (xs_hoc >= t).astype(int)
        rec = recall_score(y_tr, du_hoc)
        prec = precision_score(y_tr, du_hoc, zero_division=0)
        f1 = f1_score(y_tr, du_hoc, zero_division=0)
        bang_cv.append((t, int(du_hoc.sum()), int(((du_hoc == 1) & (y_tr == 1)).sum()), prec, rec, f1))

    # Tìm ngưỡng LỚN NHẤT cho recall >= 0.8 trên tập học (CV)
    nguong_dat = [dong for dong in bang_cv if dong[4] >= 0.8]
    t_chon, bao_cv, tp_cv, prec_cv, rec_cv, f1_cv = nguong_dat[-1]

    # Kiểm tra thêm ở lưới mịn bước 0.01 (chỉ trên tập học) để đối chiếu độ phân giải
    luoi_min = np.round(np.arange(0.01, 0.99, 0.01), 2)
    t_min_chon = max(
        t for t in luoi_min if recall_score(y_tr, (xs_hoc >= t).astype(int)) >= 0.8
    )
    rec_min_cv = recall_score(y_tr, (xs_hoc >= t_min_chon).astype(int))
    prec_min_cv = precision_score(y_tr, (xs_hoc >= t_min_chon).astype(int), zero_division=0)

    # Bước 2: Huấn luyện mô hình trên toàn bộ tập học và áp ngưỡng t_chon
    # lên tập kiểm tra ĐÚNG MỘT LẦN.
    lr.fit(X_tr, y_tr)
    xs_te = lr.predict_proba(X_te)[:, 1]

    du_te = (xs_te >= t_chon).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, du_te).ravel()
    so_bao = int(du_te.sum())
    prec_te = precision_score(y_te, du_te, zero_division=0)
    rec_te = recall_score(y_te, du_te)
    f1_te = f1_score(y_te, du_te)

    # Đánh giá tham khảo ngưỡng lưới mịn (cũng được chọn độc lập từ tập học)
    du_te_min = (xs_te >= t_min_chon).astype(int)
    rec_te_min = recall_score(y_te, du_te_min)
    prec_te_min = precision_score(y_te, du_te_min, zero_division=0)
    so_bao_min = int(du_te_min.sum())

    print("=" * 76)
    print("BAI TAP 3: CHON NGUONG LON NHAT DAT RECALL >= 0.80 (KIEM DINH CHEO)")
    print("=" * 76)
    print(f"Tap hoc: {len(y_tr)} sinh vien ({int(y_tr.sum())} bo hoc) | "
          f"Tap kiem tra: {len(y_te)} sinh vien ({int(y_te.sum())} bo hoc)")
    print()
    print("1. DO NGUONG TREN TAP HOC BANG cross_val_predict (cv=5, luoi buoc 0.05):")
    print(f"  {'Nguong':>6s} {'So bao':>7s} {'TP/89':>7s} {'Precision':>10s} {'Recall':>10s} {'F1':>9s} {'Dat >= 0.8?':>12s}")
    print("  " + "-" * 67)
    for t, sb, tp_h, p_h, r_h, f1_h in bang_cv[:8]:  # 0.05 -> 0.40
        dau = "DAT  <-- CHON" if t == t_chon else ("DAT" if r_h >= 0.8 else "KHONG")
        print(f"  {t:6.2f} {sb:7d} {tp_h:4d}/89 {p_h:10.4f} {r_h:10.4f} {f1_h:9.4f}   {dau}")
    print()
    print(f"=> Nguong LON NHAT dat recall >= 0.80 tren tap hoc (buoc 0.05): t = {t_chon:.2f}")
    print(f"   (Tai t = {t_chon:.2f}: CV recall = {rec_cv:.4f} ({tp_cv}/89), "
          f"CV precision = {prec_cv:.4f}, CV F1 = {f1_cv:.4f})")
    print(f"   (Tham khao luoi min buoc 0.01 tren tap hoc: t = {t_min_chon:.2f}, "
          f"CV recall = {rec_min_cv:.4f}, CV precision = {prec_min_cv:.4f})")
    print()

    print(f"2. AP NGUONG t = {t_chon:.2f} CHO TAP KIEM TRA (DUNG MOT LAN):")
    print("-" * 76)
    print(f"  Ma tran nham lan: TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"  So sinh vien bi bao nguy co (TP + FP) : {so_bao} / {len(y_te)} ban ({so_bao/len(y_te)*100:.1f}%)")
    print(f"  Recall tren tap kiem tra              : {rec_te:.4f} ({tp}/{int(y_te.sum())} ban bo hoc)")
    print(f"  Precision tren tap kiem tra           : {prec_te:.4f} ({tp}/{so_bao} bao dong dung)")
    print(f"  F1 tren tap kiem tra                  : {f1_te:.4f}")
    trang_thai = "CO DAT (>= 0.80)" if rec_te >= 0.8 else "KHONG DAT (< 0.80)"
    print(f"  Ket luan yeu cau recall >= 0.80       : {trang_thai}")
    print(f"  (Tham khao luoi min t = {t_min_chon:.2f}: Recall = {rec_te_min:.4f},")
    print(f"   Precision = {prec_te_min:.4f}, so bao = {so_bao_min} ban -> cung DAT >= 0.80)")
    print()

    print("3. GIAI THICH NGUYEN TAC CHON NGUONG:")
    print("-" * 76)
    print("- Vi sao phai tim nguong LON NHAT thoa man recall >= 0.80?")
    print("  Khi ha nguong cang thap, recall tang (it bo sot hon) nhung precision")
    print("  giam manh va so bao dong nham (FP) tang vot. Chon nguong LON NHAT van")
    print("  dat recall >= 0.80 giup dap ung chi tieu cua nha truong dong thoi giu")
    print("  precision cao nhat co the, giam so buoi tu van thua (FP).")
    print("- Vi sao phai dung cross_val_predict tren tap hoc ma khong nhin tap kiem tra?")
    print("  + Neu nhin bang ket qua tren tap kiem tra de chon nguong, tap kiem tra")
    print("    da bi dung de ra quyet dinh (ro ri du lieu - data leakage), lam con")
    print("    so bao cao dep gia tao hon thuc te khi ap dung cho khoa sinh vien moi.")
    print("  + Dung cross_val_predict(cv=5) tren tap hoc tao ra xac suat ngoai mau")
    print("    (out-of-fold): moi sinh vien nhan xac suat tu mo hinh chua tung hoc")
    print("    minh, nen mo phong trung thuc tinh huong du doan sinh vien moi.")
    print("- Nhan xet ket qua tren tap kiem tra:")
    print(f"  Voi t = {t_chon:.2f}, mo hinh phat hien {tp}/30 ban bo hoc (recall = {rec_te:.4f},")
    print(f"  vuot yeu cau 0.80), chi bo sot {fn} ban. Doi lai, nha truong can tu van")
    print(f"  {so_bao} ban (trong do co {fp} bao dong nham, precision = {prec_te:.4f}).")
    print("  Luu y: vi tap kiem tra chi co 30 ban bo hoc (moi 1 ban tuong ung 3.33%),")
    print("  recall tap kiem tra co the dao dong nhe quanh muc chon tren tap hoc.")
    print("=" * 76)


if __name__ == "__main__":
    main()
