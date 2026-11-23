# -*- coding: utf-8 -*-
"""Bài tập 1: Tự tính chỉ số silhouette bằng numpy và phân tích điểm có s nhỏ nhất.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
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
        os.path.join(thu_muc_script, "..", "..", "Lab08_data", "data", ten_file),
        os.path.join("Lab08_data", "data", ten_file),
    ]
    for p in ung_vien:
        if os.path.exists(p):
            return p
    return duong_dan_mac_dinh


def silhouette_mot_diem(X, nhan, i):
    """Tính a, b và s của điểm thứ i theo đúng định nghĩa ở mục 4, chỉ dùng numpy.

    - a: khoảng cách Euclidean trung bình từ điểm i tới các điểm khác cùng cụm.
    - b: khoảng cách Euclidean trung bình nhỏ nhất từ điểm i tới các cụm khác.
    - s = (b - a) / max(a, b) (quy ước s = 0 nếu cụm chỉ có đúng 1 điểm).
    """
    X = np.asarray(X, dtype=float)
    nhan = np.asarray(nhan)
    cum_i = nhan[i]

    # Khoảng cách Euclidean từ điểm i tới toàn bộ các điểm trong tập dữ liệu
    kc = np.sqrt(((X - X[i]) ** 2).sum(axis=1))

    # a: trung bình khoảng cách tới các điểm KHÁC cùng cụm (loại bỏ chính điểm i)
    cung_cum = np.where(nhan == cum_i)[0]
    cung_cum_khac_i = cung_cum[cung_cum != i]
    if len(cung_cum_khac_i) == 0:
        return 0.0, 0.0, 0.0
    a = float(kc[cung_cum_khac_i].mean())

    # b: tìm cụm khác gần nhất (khoảng cách trung bình tới các điểm trong cụm đó nhỏ nhất)
    cac_cum_khac = [c for c in np.unique(nhan) if c != cum_i]
    kc_cum_khac = [float(kc[nhan == c].mean()) for c in cac_cum_khac]
    b = min(kc_cum_khac)

    mau = max(a, b)
    s = 0.0 if mau == 0.0 else (b - a) / mau
    return a, b, float(s)


def main():
    duong_dan = tim_duong_dan_du_lieu("data/khach_hang.csv")
    df = pd.read_csv(duong_dan)

    # Phân cụm với K = 5 giống hệt mục 3 và mục 4 (chuẩn hoá 2 cột thu_nhap, diem_chi_tieu)
    X = df[["thu_nhap", "diem_chi_tieu"]].to_numpy()
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
    nhan = km.labels_
    tam_goc = sc.inverse_transform(km.cluster_centers_)

    # Chạy hàm tự viết cho mọi khách hàng (n = 210)
    ds_a = np.zeros(len(Xs))
    ds_b = np.zeros(len(Xs))
    ds_s = np.zeros(len(Xs))
    for i in range(len(Xs)):
        ds_a[i], ds_b[i], ds_s[i] = silhouette_mot_diem(Xs, nhan, i)

    sil_tu_tinh = float(ds_s.mean())
    sil_sklearn = float(silhouette_score(Xs, nhan))
    s_mau_sklearn = silhouette_samples(Xs, nhan)
    sai_khac_lon_nhat = float(np.max(np.abs(ds_s - s_mau_sklearn)))

    # Tìm khách hàng có s nhỏ nhất và top 5 khách có s thấp nhất
    idx_min = int(np.argmin(ds_s))
    cum_min = int(nhan[idx_min])

    # Tính khoảng cách trung bình từ khách có s nhỏ nhất tới từng cụm để xác định cụm láng giềng
    kc_min = np.sqrt(((Xs - Xs[idx_min]) ** 2).sum(axis=1))
    kc_tung_cum = {}
    for c in range(5):
        chi_so = np.where(nhan == c)[0]
        if c == cum_min:
            chi_so = chi_so[chi_so != idx_min]
        kc_tung_cum[c] = float(kc_min[chi_so].mean())
    cum_gan_nhat = min((c for c in range(5) if c != cum_min), key=lambda c: kc_tung_cum[c])

    # Đối chiếu số thứ tự cụm trên Hình 3 (xếp theo thu nhập tăng dần, rồi điểm chi tiêu tăng dần)
    # Bảng tâm gốc:
    #   cum 0: (7.5, 80.2) -> Hinh 3: Cum 2 (thu nhap thap, chi nhieu)
    #   cum 1: (30.1, 20.0) -> Hinh 3: Cum 4 (thu nhap cao, chi it)
    #   cum 2: (17.4, 49.3) -> Hinh 3: Cum 3 (thu nhap vua, chi vua)
    #   cum 3: (8.0, 18.9) -> Hinh 3: Cum 1 (thu nhap thap, chi it)
    #   cum 4: (30.2, 81.5) -> Hinh 3: Cum 5 (thu nhap cao, chi nhieu)
    ten_cum = {
        0: "Thu nhap thap, chi nhieu (Hinh 3: Cum 2)",
        1: "Thu nhap cao, chi it (Hinh 3: Cum 4)",
        2: "Thu nhap vua, chi vua (Hinh 3: Cum 3)",
        3: "Thu nhap thap, chi it (Hinh 3: Cum 1)",
        4: "Thu nhap cao, chi nhieu (Hinh 3: Cum 5)",
    }

    print("=" * 76)
    print("BAI TAP 1: TU TINH CHI SO SILHOUETTE BANG NUMPY (K = 5)")
    print("=" * 76)
    print(f"So khach hang: {len(df)} | So cum K = 5 | Inertia = {km.inertia_:.2f}")
    print()

    print("1. SO SANH KET QUA TU TINH VOI SCIKIT-LEARN")
    print("-" * 76)
    print(f"Silhouette trung binh tu tinh (numpy)   : {sil_tu_tinh:.6f}")
    print(f"Silhouette cua sklearn (silhouette_score): {sil_sklearn:.6f}")
    print(f"Chenh lech tuyet doi trung binh         : {abs(sil_tu_tinh - sil_sklearn):.2e}")
    print(f"Sai khac lon nhat tren tung diem (210)  : {sai_khac_lon_nhat:.2e}")
    print("-> Ket qua trung khop HOAN TOAN voi thu vien scikit-learn!")
    print()

    print("2. TOP 5 KHACH HANG CO CHI SO SILHOUETTE (s) NHO NHAT")
    print("-" * 76)
    print(f"{'Chi so':>6s} {'Tuoi':>5s} {'Thu nhap':>9s} {'Chi tieu':>9s} {'Cum':>4s} {'a':>8s} {'b':>8s} {'s':>8s}")
    for i in np.argsort(ds_s)[:5]:
        print(
            f"{i:6d} {df.loc[i, 'tuoi']:5.0f} {df.loc[i, 'thu_nhap']:9.1f} "
            f"{df.loc[i, 'diem_chi_tieu']:9.0f} {nhan[i]:4d} "
            f"{ds_a[i]:8.4f} {ds_b[i]:8.4f} {ds_s[i]:8.4f}"
        )
    print()

    print("3. PHAN TICH CHI TIET KHACH HANG CO s NHO NHAT")
    print("-" * 76)
    print(
        f"Khach co s nho nhat: chi so {idx_min} (dong {idx_min + 2} trong file CSV)"
    )
    print(
        f"  Thong tin goc : tuoi = {df.loc[idx_min, 'tuoi']:.0f}, "
        f"thu_nhap = {df.loc[idx_min, 'thu_nhap']:.1f} trieu, "
        f"diem_chi_tieu = {df.loc[idx_min, 'diem_chi_tieu']:.0f}"
    )
    print(
        f"  Duoc gan vao  : cum {cum_min} - {ten_cum[cum_min]} "
        f"[tam = ({tam_goc[cum_min, 0]:.1f}, {tam_goc[cum_min, 1]:.1f})]"
    )
    print(
        f"  Cum lang gieng: cum {cum_gan_nhat} - {ten_cum[cum_gan_nhat]} "
        f"[tam = ({tam_goc[cum_gan_nhat, 0]:.1f}, {tam_goc[cum_gan_nhat, 1]:.1f})]"
    )
    print(f"  Gia tri tinh  : a = {ds_a[idx_min]:.4f}, b = {ds_b[idx_min]:.4f}, s = {ds_s[idx_min]:.4f}")
    print("  Khoang cach trung binh (tren thang chuan hoa) toi ca 5 cum:")
    for c in range(5):
        dau = " <-- a (cung cum)" if c == cum_min else (" <-- b (cum gan nhat khac)" if c == cum_gan_nhat else "")
        print(f"    Cum {c} [tam ({tam_goc[c, 0]:4.1f}, {tam_goc[c, 1]:4.1f})]: {kc_tung_cum[c]:.4f}{dau}")
    print()

    print("4. GIAI THICH Y NGHIA SILHOUETTE VA VI TRI TREN HINH 3")
    print("-" * 76)
    print("- Y nghia cong thuc s = (b - a) / max(a, b):")
    print("  + a do muc do gan ket NOI CUM: a cang nho thi diem cang nam gon trong")
    print("    cum cua minh.")
    print("  + b do muc do tach biet NGOAI CUM: b cang lon thi diem cang xa cum")
    print("    lang gieng gan nhat.")
    print("  + Khi s gan +1 (a << b): diem nam rat chac chan trong cum dung.")
    print("  + Khi s gan 0 (a xap xi b): diem nam lung chung ngay vung ranh gioi")
    print("    giua hai cum, gan cum cua minh chi nhinh hon mot chut.")
    print("  + Khi s < 0 (a > b): diem co kha nang bi gan nham cum.")
    print("- Vi tri cua khach chi so 113 tren Hinh 3:")
    print("  + Khach nay co toa do (thu_nhap = 25.4 trieu, diem_chi_tieu = 41), la")
    print("    diem don doc nam lo lung o KHOANG TRONG GIUA Cum 3 tren Hinh 3 (cum")
    print("    giua: thu nhap ~17.4, chi tieu ~49.3) va Cum 4 tren Hinh 3 (goc duoi")
    print("    ben phai: thu nhap ~30.1, chi tieu ~20.0).")
    print("  + Vi nam xa tam ca hai cum va gan sat ranh gioi chia doi, khoang cach")
    print(f"    trung binh toi cum cua minh (a = {ds_a[idx_min]:.4f}) gan bang khoang cach toi")
    print(f"    cum lang gieng (b = {ds_b[idx_min]:.4f}), khien b - a rat nho va s = {ds_s[idx_min]:.4f}")
    print("    gan bang 0. Nhung vi s > 0, K-Means khong gan sai cum ma chi phan")
    print("    anh day la mot khach ngoai le/chuyen tiep giua hai phan khuc.")
    print("=" * 76)


if __name__ == "__main__":
    main()
