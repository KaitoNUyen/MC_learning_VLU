# -*- coding: utf-8 -*-
"""Bài tập 3: Kernel đa thức trên bộ dữ liệu hai vầng trăng.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
        os.path.join(thu_muc_script, "..", "..", "Lab06_data", "data", ten_file),
        os.path.join("Lab06_data", "data", ten_file),
    ]
    for p in ung_vien:
        if os.path.exists(p):
            return p
    return duong_dan_mac_dinh


def main():
    duong_dan = tim_duong_dan_du_lieu("data/hai_vang_trang.csv")
    df = pd.read_csv(duong_dan)
    X = df[["x1", "x2"]]
    y = df["nhan"]

    # Chia tập học và tập kiểm tra giống mục 4 và mục 5
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    luoi = {"svc__C": [0.1, 1, 10, 100]}
    ket_qua = []

    print("=" * 74)
    print("BAI TAP 3: KERNEL DA THUC TREN BO HAI VANG TRANG")
    print("=" * 74)
    print("Chi tiet diem kiem dinh cheo (5-fold CV) cho tung bac d va tham so C:")
    print()

    for bac in [2, 3, 4]:
        pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel="poly", degree=bac),
        )
        tim = GridSearchCV(pipeline, luoi, cv=5)
        tim.fit(X_tr, y_tr)

        # In điểm CV cho từng giá trị C của bậc hiện tại
        cac_c = [p["svc__C"] for p in tim.cv_results_["params"]]
        cac_diem = tim.cv_results_["mean_test_score"]
        chi_tiet = ", ".join(f"C={c}: {d:.4f}" for c, d in zip(cac_c, cac_diem))
        print(f"Bac d = {bac}: {chi_tiet}")

        acc_kt = tim.score(X_te, y_te)
        ket_qua.append({
            "bac": bac,
            "C_tot_nhat": tim.best_params_["svc__C"],
            "cv_tot_nhat": tim.best_score_,
            "acc_kiem_tra": acc_kt,
        })

    # Đối chiếu trực tiếp với kernel RBF ở mục 5
    luoi_rbf = {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]}
    tim_rbf = GridSearchCV(SVC(kernel="rbf"), luoi_rbf, cv=5)
    tim_rbf.fit(X_tr, y_tr)
    rbf_cv = tim_rbf.best_score_
    rbf_kt = tim_rbf.score(X_te, y_te)

    print()
    print("BANG TONG HOP KET QUA:")
    print("-" * 74)
    print(f"{'Mo hinh':26s} {'Tham so tot nhat':18s} {'CV (5-fold)':>12s} {'Tap kiem tra':>14s}")
    print("-" * 74)
    for kq in ket_qua:
        ten = f"Poly (degree={kq['bac']}, scaler)"
        tham_so = f"C = {kq['C_tot_nhat']}"
        print(f"{ten:26s} {tham_so:18s} {kq['cv_tot_nhat']:12.4f} {kq['acc_kiem_tra']:14.4f}")
    print("-" * 74)
    rbf_ts = f"C={tim_rbf.best_params_['C']}, g={tim_rbf.best_params_['gamma']}"
    print(f"{'RBF (muc 5, doi chieu)':26s} {rbf_ts:18s} {rbf_cv:12.4f} {rbf_kt:14.4f}")
    print()

    # Tìm bậc đa thức tốt nhất theo điểm CV (và tập kiểm tra)
    tot_nhat = max(ket_qua, key=lambda item: (item["cv_tot_nhat"], item["acc_kiem_tra"]))

    print("NHAN XET:")
    print("-" * 74)
    print(
        f"- Bac tot nhat trong ba bac (2, 3, 4) la bac d = {tot_nhat['bac']} "
        f"(voi C = {tot_nhat['C_tot_nhat']}),"
    )
    print(
        f"  dat diem kiem dinh cheo {tot_nhat['cv_tot_nhat']:.4f} "
        f"va accuracy tap kiem tra {tot_nhat['acc_kiem_tra']:.4f}."
    )
    vuot_cv = tot_nhat["cv_tot_nhat"] > rbf_cv
    vuot_kt = tot_nhat["acc_kiem_tra"] > rbf_kt
    if vuot_cv or vuot_kt:
        ket_luan = "Co vuot o mot so tieu chi."
    else:
        ket_luan = "KHONG vuot duoc kernel RBF o muc 5."
    print(
        f"- So voi kernel RBF o muc 5 (CV = {rbf_cv:.4f}, tap kiem tra = {rbf_kt:.4f}): "
        f"{ket_luan}"
    )
    print(
        "- Giai thich: Bien quyet dinh cua bo hai vang trang uon luon hinh chu S."
    )
    print(
        "  Da thuc bac 2 va bac 4 co tinh doi xung chan nen kho uon theo dang chu S;"
    )
    print(
        "  da thuc bac 3 co the tao duong cong chu S nen tot hon bac chan, nhung"
    )
    print(
        "  van kem linh hoat hon kernel RBF (von tao bien cong cuc bo theo khoang cach)."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
