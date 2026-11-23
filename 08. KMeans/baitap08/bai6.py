# -*- coding: utf-8 -*-
"""Bài tập 6: Phân cụm bộ chữ số viết tay load_digits bằng K-Means (K = 10).

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    # Nạp bộ 1797 ảnh chữ số viết tay 8x8 (64 đặc trưng cùng thang đo 0..16)
    X, nhan_that = load_digits(return_X_y=True)

    # Phân cụm không dùng nhãn thật với K = 10
    km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
    cum = km.labels_

    # Sau khi phân cụm xong mới đối chiếu với nhãn thật bằng bảng crosstab
    bang = pd.crosstab(
        pd.Series(nhan_that, name="That"),
        pd.Series(cum, name="Cum"),
    )

    # Tính độ tinh khiết (purity) và chỉ số Adjusted Rand Index (ARI)
    do_tinh_khiet = bang.max(axis=0).sum() / len(nhan_that)
    ari = adjusted_rand_score(nhan_that, cum)

    print("=" * 76)
    print("BAI TAP 6: PHAN CUM CHU SO VIET TAY load_digits (K = 10)")
    print("=" * 76)
    print(
        f"Du lieu: {X.shape[0]} anh x {X.shape[1]} dac trung (0..16) | "
        f"Inertia = {km.inertia_:.1f}"
    )
    print(f"Do tinh khiet (purity, gan nhan da so moi cum): {do_tinh_khiet:.4f} ({do_tinh_khiet*100:.2f}%)")
    print(f"Adjusted Rand Index (ARI)                     : {ari:.4f}")
    print()

    print("1. BANG DOI CHIEU pd.crosstab(nhan_that, cum)")
    print("-" * 76)
    print(bang.to_string())
    print()

    print("2. MUC DO GOM GON CUA TUNG CHU SO (0..9)")
    print("-" * 76)
    print(f"{'Chu so':>6s} {'Tong anh':>9s} {'Cum chinh':>10s} {'So anh o cum chinh':>19s} {'Ty le gom gon':>14s}")
    thong_ke_so = []
    for so in range(10):
        hang = bang.loc[so]
        tong = int(hang.sum())
        cum_chinh = int(hang.idxmax())
        so_chinh = int(hang[cum_chinh])
        ty_le = so_chinh / tong * 100.0
        thong_ke_so.append((so, tong, cum_chinh, so_chinh, ty_le))
        print(f"{so:6d} {tong:9d} {cum_chinh:10d} {so_chinh:19d} {ty_le:13.1f}%")
    print()

    print("3. THANH PHAN CHINH TRONG TUNG CUM (0..9)")
    print("-" * 76)
    for c in range(10):
        cot = bang[c]
        tong_c = int(cot.sum())
        top = cot[cot > 0].sort_values(ascending=False)
        mo_ta = ", ".join(f"so {so}: {sl}" for so, sl in top.iloc[:3].items())
        print(f"  Cum {c} ({tong_c:3d} anh): {mo_ta}")
    print()

    print("4. TRA LOI VA NHAN XET")
    print("-" * 76)
    print("- Nhung chu so nao duoc gom gon nhat vao mot cum?")
    print("  + Chu so 0: gom gon nhat, 176/178 anh (98.9%) nam trong cum 0, va cum 0")
    print("    gan nhu tinh khiet tuyet doi (chi lan dung 2 anh cua so 2 va so 6).")
    print("  + Chu so 6: 177/181 anh (97.8%) nam gon trong cum 5 (chi lan 5 anh khac).")
    print("  + Chu so 7: 174/179 anh (97.2%) nam gon trong cum 2.")
    print("  + Chu so 4: 165/181 anh (91.2%) nam gon trong cum 6 (cum 6 rat sach).")
    print("  (Ly do: hinh dang cua 0, 6, 7, 4 co cau truc net rat dac trung va it")
    print("  trung lap theo khoang cach Euclidean giua cac mang pixel 8x8).")
    print("- Nhung chu so nao hay bi gom chung?")
    print("  + So 1 va so 8 bi gom chung nang nhat trong cum 1 (100 anh so 1 va 100")
    print("    anh so 8, ti le dung 50/50!). Ngoai ra so 1 con bi tach sang cum 3")
    print("    (54 anh, gom chung voi 20 anh so 9) va cum 9 (24 anh, chung voi so 2).")
    print("    Do do chu so 8 khong co cum rieng ma bi xe doi sang cum 1 (100 anh)")
    print("    va cum 8 (52 anh).")
    print("  + So 9, so 8 va so 5 hay bi gom chung trong cum 8 (139 so 9, 52 so 8,")
    print("    41 so 5 va 13 so 3) vi chung deu co nhieu net cong tron kin o phan")
    print("    than/dau nen khoang cach pixel 8x8 rat gan nhau.")
    print("  + So 2 va mot phan so 1 cung lan nhau o cum 9 (148 so 2, 24 so 1),")
    print("    va so 3 co 13 anh lan sang cum 8 (cung so 9, 8, 5).")
    print("- Ket luan: Du KHONG dung bat ky nhan nao, K-Means van tu gom dung khoang")
    print(f"  {do_tinh_khiet*100:.1f}% anh (purity = {do_tinh_khiet:.4f}, ARI = {ari:.4f}). Tuy nhien,")
    print("  khoang cach Euclidean tren pixel tho nhay voi do nghieng/dich chuyen")
    print("  net viet nen cac chu so cung nhieu net cong (8, 9, 5, 3) hay cung cot")
    print("  doc o giua (1, 8) de bi gom chung.")
    print("=" * 76)


if __name__ == "__main__":
    main()
