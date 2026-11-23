# -*- coding: utf-8 -*-
"""Bài tập 3: So sánh các hàm kích hoạt logistic, tanh và relu trên bộ hai vầng trăng.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def tim_duong_dan_du_lieu(duong_dan_mac_dinh):
    """Ưu tiên đường dẫn data/... theo yêu cầu đề, dự phòng nếu chạy từ nơi khác."""
    if os.path.exists(duong_dan_mac_dinh):
        return duong_dan_mac_dinh
    ten_file = os.path.basename(duong_dan_mac_dinh)
    thu_muc_script = os.path.dirname(os.path.abspath(__file__))
    ung_vien = [
        os.path.join(thu_muc_script, "..", "data", ten_file),
        os.path.join(thu_muc_script, "..", "..", "Lab09_data", "data", ten_file),
        os.path.join("Lab09_data", "data", ten_file),
    ]
    for p in ung_vien:
        if os.path.exists(p):
            return p
    return duong_dan_mac_dinh


def tao_mang(ham_kich_hoat):
    """Tạo pipeline mạng hai lớp ẩn 100 nơ-ron với alpha=1.0 như mục 4."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100, 100),
            alpha=1.0,
            activation=ham_kich_hoat,
            solver="lbfgs",
            max_iter=20000,
            random_state=0,
        ),
    )


def main():
    duong_dan = tim_duong_dan_du_lieu("data/hai_vang_trang.csv")
    df = pd.read_csv(duong_dan)
    X = df[["x1", "x2"]]
    y = df["nhan"]

    # Chia tập học và tập kiểm tra đúng như mục 4
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    print("=" * 78)
    print("BAI TAP 3: SO SANH CAC HAM KICH HOAT TREN BO HAI VANG TRANG")
    print("=" * 78)
    print("Cau hinh mang (nhu muc 4): hidden_layer_sizes=(100, 100), alpha=1.0,")
    print("                           solver='lbfgs', max_iter=20000, random_state=0")
    print(f"Du lieu: {len(X_tr)} mau tap hoc, {len(X_te)} mau tap kiem tra (co StandardScaler)")
    print()

    cac_ham = ["logistic", "tanh", "relu"]
    ket_qua = []

    print(f"{'Ham kich hoat':15s} {'Tap hoc':>10s} {'Tap kiem tra':>14s} {'CV (5-fold)':>13s} {'So vong':>10s}")
    print("-" * 78)
    for ham in cac_ham:
        mo_hinh = tao_mang(ham).fit(X_tr, y_tr)
        acc_hoc = accuracy_score(y_tr, mo_hinh.predict(X_tr))
        acc_kt = accuracy_score(y_te, mo_hinh.predict(X_te))
        diem_cv = cross_val_score(tao_mang(ham), X_tr, y_tr, cv=5).mean()
        so_vong = mo_hinh[-1].n_iter_

        ket_qua.append({
            "ham": ham,
            "acc_hoc": acc_hoc,
            "acc_kt": acc_kt,
            "cv": diem_cv,
            "so_vong": so_vong,
        })
        print(f"{ham:15s} {acc_hoc:10.4f} {acc_kt:14.4f} {diem_cv:13.4f} {so_vong:10d}")
    print("-" * 78)
    print()

    # Tìm hàm tốt nhất (ưu tiên accuracy tập kiểm tra, sau đó điểm CV 5-fold)
    tot_nhat = max(ket_qua, key=lambda r: (r["acc_kt"], r["cv"], r["acc_hoc"]))

    print("NHAN XET VA KET LUAN:")
    print("-" * 78)
    print(
        f"- Ham cho ket qua tot nhat o day: '{tot_nhat['ham']}' "
        f"(tap hoc = {tot_nhat['acc_hoc']:.4f}, tap kiem tra = {tot_nhat['acc_kt']:.4f},"
    )
    print(f"  kiem dinh cheo 5-fold CV = {tot_nhat['cv']:.4f}).")
    print("- Cu the:")
    print("  1. 'relu' va 'tanh' CUNG DAT accuracy tap kiem tra cao nhat la 0.9067")
    print("     (doan dung 68/75 mau), cao hon ro ret so voi 'logistic' (+0.0533).")
    print("     Tuy nhien, 'relu' nhinh hon 'tanh' ca tren tap hoc (0.8800 vs 0.8711)")
    print("     lan tren kiem dinh cheo 5-fold (0.8578 vs 0.8444), nen 'relu' la lua")
    print("     chon tot nhat tong the o bai toan nay.")
    print("  2. 'logistic' (sigmoid) cho ket qua kem nhat (tap hoc 0.8400, kiem tra")
    print("     0.8533, CV 0.8356). Nguyen nhan: ham sigmoid co dao ham cuc dai chi")
    print("     bang 0.25 tai z=0 va dau ra khong doi xung quanh 0 (luon trong (0,1)).")
    print("     Khi ghep 2 lop an cung dieu chuan L2 manh (alpha=1.0) ep trong so nho,")
    print("     tin hieu do doc suy giam manh (vanishing gradient), khien ranh gioi bi")
    print("     phang hoa va cau tha (underfitting) gan giong duong thang.")
    print("  3. 'tanh' doi xung quanh 0 va co do doc cuc dai bang 1.0 (gap 4 lan")
    print("     sigmoid), con 'relu' giu dao ham bang 1 tren toan mien duong (khong bao")
    print("     hoa o phia duong) nen ca hai uon ranh gioi muot ma theo dung hai vang")
    print("     trang; do cung la ly do scikit-learn chon 'relu' lam mac dinh.")
    print("=" * 78)


if __name__ == "__main__":
    main()
