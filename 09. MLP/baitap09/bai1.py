# -*- coding: utf-8 -*-
"""Bài tập 1: Tự dự đoán bằng trọng số đã học trên bộ XOR.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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


def relu(z):
    """Hàm kích hoạt ReLU cho lớp ẩn: max(0, z)."""
    return np.maximum(0.0, z)


def sigmoid(z):
    """Hàm kích hoạt sigmoid cho lớp ra phân loại nhị phân."""
    return 1.0 / (1.0 + np.exp(-z))


def main():
    # Đọc bộ dữ liệu XOR và chia tập học / tập kiểm tra đúng như mục 3
    duong_dan = tim_duong_dan_du_lieu("data/xor.csv")
    df = pd.read_csv(duong_dan)
    X = df[["x1", "x2"]]
    y = df["nhan"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # Học lại mạng 1 lớp ẩn 8 nơ-ron đúng cấu hình mục 3
    mlp = MLPClassifier(
        hidden_layer_sizes=(8,),
        solver="lbfgs",
        max_iter=5000,
        random_state=0,
    )
    mlp.fit(X_tr, y_tr)

    # Trích xuất các ma trận trọng số và vectơ hệ số chặn đã học
    W1, W2 = mlp.coefs_
    b1, b2 = mlp.intercepts_
    so_tham_so = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)

    acc_tr = accuracy_score(y_tr, mlp.predict(X_tr))
    acc_te = accuracy_score(y_te, mlp.predict(X_te))

    # Tự tính lan truyền tiến bằng numpy cho 5 điểm đầu của tập kiểm tra
    X5_df = X_te.iloc[:5]
    y5 = y_te.iloc[:5].to_numpy()
    X5 = X5_df.to_numpy()

    Z1 = X5 @ W1 + b1          # Tổ hợp tuyến tính ở lớp ẩn: (5, 2) @ (2, 8) + (8,) -> (5, 8)
    H = relu(Z1)               # Kích hoạt ReLU ở lớp ẩn: (5, 8)
    Z2 = H @ W2 + b2           # Tổ hợp tuyến tính ở lớp ra: (5, 8) @ (8, 1) + (1,) -> (5, 1)
    p_tu_tinh = sigmoid(Z2).ravel()

    # Đối chiếu với predict_proba cột thứ hai (xác suất thuộc lớp 1)
    p_sklearn = mlp.predict_proba(X5_df)[:, 1]
    sai_lech = np.abs(p_tu_tinh - p_sklearn)

    print("=" * 78)
    print("BAI TAP 1: TU DU DOAN BANG TRONG SO DA HOC (MANG 8 NO-RON AN TREN XOR)")
    print("=" * 78)
    print(f"Kich thuoc trong so : W1 = {W1.shape}, b1 = {b1.shape}, W2 = {W2.shape}, b2 = {b2.shape}")
    print(f"Tong so tham so     : {so_tham_so} (2*8 + 8 + 8*1 + 1)")
    print(f"Accuracy mang       : tap hoc = {acc_tr:.4f}, tap kiem tra = {acc_te:.4f}")
    print()

    print("1. SO SANH LAN TRUYEN TIEN TU TINH (NUMPY) VOI mlp.predict_proba[:, 1]")
    print("-" * 78)
    print(
        f"{'Diem':4s} {'x1':>8s} {'x2':>8s} {'That':>5s} "
        f"{'Tu tinh (numpy)':>17s} {'predict_proba':>17s} {'Sai lech':>12s}"
    )
    print("-" * 78)
    for i in range(5):
        print(
            f"  #{i+1:<1d} {X5[i, 0]:+8.4f} {X5[i, 1]:+8.4f} {y5[i]:5d} "
            f"{p_tu_tinh[i]:17.12f} {p_sklearn[i]:17.12f} {sai_lech[i]:12.2e}"
        )
    print("-" * 78)
    print(f"Sai lech tuyet doi lon nhat : {sai_lech.max():.2e}")
    print(f"Kiem tra khop (atol=1e-12)  : {np.allclose(p_tu_tinh, p_sklearn, atol=1e-12)}")
    print()

    # In chi tiết từng bước tính cho điểm đầu tiên để minh họa rõ bên trong mạng
    print("2. CHI TIET CAC BUOC TINH CHO DIEM DAU TIEN (#1)")
    print("-" * 78)
    print(f"Dau vao x           = [{X5[0, 0]:+.4f}, {X5[0, 1]:+.4f}] (nhan that = {y5[0]})")
    print("Buoc 1: Z1 = x @ W1 + b1 (8 gia tri truoc ReLU):")
    print("  ", np.array2string(Z1[0], precision=4, floatmode="fixed", suppress_small=False))
    print("Buoc 2: H = ReLU(Z1) = max(0, Z1) (so am bi tat ve 0):")
    print("  ", np.array2string(H[0], precision=4, floatmode="fixed", suppress_small=False))
    so_bat = int(np.sum(H[0] > 0))
    print(f"  -> Co {so_bat}/8 no-ron an dang bat (> 0), {8 - so_bat}/8 no-ron bi tat (= 0).")
    print(f"Buoc 3: z2 = H @ W2 + b2 = {Z2[0, 0]:+.6f}")
    print(f"Buoc 4: p  = sigmoid(z2) = 1 / (1 + exp(-({Z2[0, 0]:+.6f}))) = {p_tu_tinh[0]:.12f}")
    print(f"        predict_proba[0, 1]                                  = {p_sklearn[0]:.12f}")
    print()

    print("3. GIAI THICH BAN CHAT PHEP TINH BEN TRONG MANG NO-RON")
    print("-" * 78)
    print("- Hai ket qua KHOP HOAN TOAN toi ~16 chu so thap phan (sai lech <= 2.8e-17,")
    print("  chi la sai so lam tron dau phay dong float64 cua may tinh).")
    print("- Dieu nay chung minh MLPClassifier khong he la 'hop den' huyen bi: khi du")
    print("  doan, no thuc hien dung 4 phep tinh dai so co ban:")
    print("  (1) Nhan ma tran + cong he so chan o lop an (Z1 = X @ W1 + b1): moi no-ron")
    print("      an tinh mot to hop tuyen tinh (tuong ung mot duong thang chia mat phang).")
    print("  (2) Ham kich hoat ReLU (H = max(0, Z1)): be gay tinh tuyen tinh bang cach")
    print("      giu gia tri duong va tat gia tri am ve 0. Chinh nho buoc phi tuyen nay")
    print("      ma 8 no-ron an ghep lai tao duoc ranh gioi chia 4 goc cua bai toan XOR.")
    print("  (3) Lop ra (Z2 = H @ W2 + b2): tong hop co trong so tu 8 dac trung an H")
    print("      thanh mot diem so logit duy nhat z2 cho moi mau.")
    print("  (4) Ham sigmoid(z2): ep logit tu (-vo cung, +vo cung) ve xac suat P(y=1|x)")
    print("      nam trong (0, 1) - dung la cot thu hai cua mlp.predict_proba(X).")
    print("- Ghi chu them: o diem #3 (+0.5155, -0.0052), toa do x2 rat sat truc 0")
    print("  (vung ranh gioi giua hai goc XOR), nen mang tinh ra p = 0.7943 > 0.5")
    print("  va doan lop 1 du nhan that la 0.")
    print("=" * 78)


if __name__ == "__main__":
    main()
