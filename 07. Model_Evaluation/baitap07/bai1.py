# -*- coding: utf-8 -*-
"""Bài tập 1: Tự viết hàm đánh giá (accuracy, precision, recall, F1) chỉ dùng numpy.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
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


def danh_gia(y_that, y_doan):
    """Tự tính accuracy, precision, recall và F1 cho lớp dương (1), chỉ dùng numpy.

    Xử lý an toàn mọi trường hợp chia cho 0 bằng cách kiểm tra mẫu số trước khi chia
    và trả về 0.0 (khớp quy ước zero_division=0 của scikit-learn).
    """
    that = np.asarray(y_that)
    doan = np.asarray(y_doan)

    tp = int(np.sum((that == 1) & (doan == 1)))
    fp = int(np.sum((that == 0) & (doan == 1)))
    fn = int(np.sum((that == 1) & (doan == 0)))
    tn = int(np.sum((that == 0) & (doan == 0)))

    tong = tp + fp + fn + tn
    accuracy = (tp + tn) / tong if tong > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return accuracy, precision, recall, f1


def dem_bon_o(y_that, y_doan):
    """Đếm trực tiếp 4 ô TP, FP, FN, TN bằng numpy để đối chiếu."""
    that = np.asarray(y_that)
    doan = np.asarray(y_doan)
    tp = int(np.sum((that == 1) & (doan == 1)))
    fp = int(np.sum((that == 0) & (doan == 1)))
    fn = int(np.sum((that == 1) & (doan == 0)))
    tn = int(np.sum((that == 0) & (doan == 0)))
    return tp, fp, fn, tn


def main():
    duong_dan = tim_duong_dan_du_lieu("data/sinh_vien_bo_hoc.csv")
    df = pd.read_csv(duong_dan)
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]

    # Chia dữ liệu giống hệt mục 2 (stratify=y giữ đúng 30/300 bạn bỏ học)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # Hai mô hình ở mục 2: mô hình lười và hồi quy logistic
    luoi = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    du_luoi = luoi.predict(X_te)

    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
    du_lr = lr.predict(X_te)

    print("=" * 76)
    print("BAI TAP 1: TU VIET HAM danh_gia(y_that, y_doan) CHI DUNG NUMPY")
    print("=" * 76)
    print(f"Tap kiem tra: {len(y_te)} sinh vien ({int(y_te.sum())} bo hoc, {int((y_te == 0).sum())} khong bo)")
    print()

    for ten, du_doan in [
        ("Hoi quy logistic", du_lr),
        ("Mo hinh luoi (most_frequent)", du_luoi),
    ]:
        tp, fp, fn, tn = dem_bon_o(y_te, du_doan)
        acc_tu, p_tu, r_tu, f1_tu = danh_gia(y_te, du_doan)

        # So sánh với các hàm của scikit-learn (dùng zero_division=0 để tránh cảnh báo)
        acc_sk = accuracy_score(y_te, du_doan)
        p_sk = precision_score(y_te, du_doan, zero_division=0)
        r_sk = recall_score(y_te, du_doan, zero_division=0)
        f1_sk = f1_score(y_te, du_doan, zero_division=0)

        sai_lech = max(
            abs(acc_tu - acc_sk),
            abs(p_tu - p_sk),
            abs(r_tu - r_sk),
            abs(f1_tu - f1_sk),
        )

        print(f"MO HINH: {ten}")
        print(f"  Bon o ma tran nham lan: TP={tp}, FP={fp}, FN={fn}, TN={tn} (Tong={tp+fp+fn+tn})")
        print(f"  {'Phuong phap':18s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
        print("  " + "-" * 62)
        print(f"  {'Tu viet (numpy)':18s} {acc_tu:10.4f} {p_tu:10.4f} {r_tu:10.4f} {f1_tu:10.4f}")
        print(f"  {'scikit-learn':18s} {acc_sk:10.4f} {p_sk:10.4f} {r_sk:10.4f} {f1_sk:10.4f}")
        print(f"  Sai lech lon nhat: {sai_lech:.1e} -> KHOP HOAN TOAN")
        print()

    print("GIAI THICH CHI TIET:")
    print("-" * 76)
    print("1. Hieu tung o cua ma tran nham lan:")
    print("   - Hoi quy logistic: TP=17 (doan dung 17 ban bo hoc), FP=3 (bao nham")
    print("     3 ban hoc tiep), FN=13 (bo sot 13 ban bo hoc), TN=267 (doan dung")
    print("     267 ban khong bo). Do do Precision = 17/(17+3) = 0.8500,")
    print("     Recall = 17/(17+13) = 0.5667, F1 = 0.6800, Accuracy = 284/300 = 0.9467.")
    print("   - Mo hinh luoi: luon doan 0 (khong bo hoc) cho ca 300 sinh vien, nen")
    print("     TP=0, FP=0, FN=30, TN=270. Accuracy dat 270/300 = 0.9000 (chi la muc")
    print("     san cua lop dong), nhung Recall = 0/(0+30) = 0.0000 (bo sot 100%).")
    print()
    print("2. Voi mo hinh luoi, precision co tinh duoc khong?")
    print("   - KHONG TINH DUOC (khong xac dinh theo toan hoc). Vi mo hinh luoi")
    print("     khong bao mot ban nao la lop 1 (bo hoc), nen so ca duoc bao la:")
    print("       TP + FP = 0 + 0 = 0.")
    print("     Thay vao cong thuc Precision = TP / (TP + FP) = 0 / 0 -> CHIA CHO 0.")
    print("   - Tuong tu, khi Precision = 0 va Recall = 0 thi F1 = 2*P*R / (P + R)")
    print("     cung roi vao dang 0 / 0. Rieng Recall = 0 / (0 + 30) = 0 van xac")
    print("     dinh binh thuong vi tap kiem tra co 30 sinh vien bo hoc that.")
    print()
    print("3. Cach ham danh_gia() xu ly truong hop chia cho 0:")
    print("   - Truoc moi phep chia, ham kiem tra mau so co lon hon 0 hay khong:")
    print("     + Precision: neu (tp + fp) > 0 thi chia, nguoc lai tra ve 0.0.")
    print("     + Recall   : neu (tp + fn) > 0 thi chia, nguoc lai tra ve 0.0.")
    print("     + F1       : neu (precision + recall) > 0 thi chia, nguoc lai 0.0.")
    print("   - Nho vay ham khong bao gio bi loi ZeroDivisionError hay sinh gia tri")
    print("     NaN, dong thoi khop quy uoc zero_division=0 cua scikit-learn (tranh")
    print("     canh bao UndefinedMetricWarning) va phan anh dung viec mo hinh")
    print("     luoi khong mang lai gia tri phat hien nao cho lop bo hoc.")
    print("=" * 76)


if __name__ == "__main__":
    main()
