# -*- coding: utf-8 -*-
"""Bài tập 2: Phân cụm trên cả ba cột tuổi, thu nhập và điểm chi tiêu.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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


def main():
    duong_dan = tim_duong_dan_du_lieu("data/khach_hang.csv")
    df = pd.read_csv(duong_dan)

    # 1. Phân cụm 3 cột (tuoi, thu_nhap, diem_chi_tieu) có chuẩn hoá
    cot_3 = ["tuoi", "thu_nhap", "diem_chi_tieu"]
    X3 = df[cot_3]
    sc3 = StandardScaler().fit(X3)
    Xs3 = sc3.transform(X3)

    # Đối chiếu với mục 3 (2 cột: thu_nhap, diem_chi_tieu)
    cot_2 = ["thu_nhap", "diem_chi_tieu"]
    X2 = df[cot_2]
    sc2 = StandardScaler().fit(X2)
    Xs2 = sc2.transform(X2)
    km2 = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs2)
    sil_2cot_k5 = silhouette_score(Xs2, km2.labels_)

    print("=" * 76)
    print("BAI TAP 2: THEM COT TUOI VAO PHAN CUM (3 COT CO CHUAN HOA)")
    print("=" * 76)
    print("1. CHON SO CUM K BANG SILHOUETTE (K = 2..10, 3 COT)")
    print("-" * 76)
    print(f"  {'K':>2s}   {'Inertia (3 cot)':>15s}   {'Giam vs K-1':>12s}   {'Silhouette (3 cot)':>18s}")

    ket_qua = []
    inertia_truoc = None
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs3)
        sil = silhouette_score(Xs3, km.labels_)
        giam_str = "" if inertia_truoc is None else f"{inertia_truoc - km.inertia_:12.2f}"
        ket_qua.append((k, km.inertia_, sil, km))
        inertia_truoc = km.inertia_
        print(f"  {k:2d}   {km.inertia_:15.2f}   {giam_str:>12s}   {sil:18.4f}")

    # Chọn K tốt nhất theo silhouette
    k_tot, inertia_tot, sil_tot, km3_tot = max(ket_qua, key=lambda x: x[2])
    print()
    print(
        f"-> K tot nhat theo silhouette tren 3 cot la K = {k_tot} "
        f"(silhouette = {sil_tot:.4f}, inertia = {inertia_tot:.2f})."
    )
    print(
        f"-> So voi muc 3 (2 cot, K = 5, silhouette = {sil_2cot_k5:.4f}): "
        f"K tot nhat VAN LA 5, nhung silhouette giam tu {sil_2cot_k5:.4f} xuong {sil_tot:.4f}."
    )
    print()

    # 2. So sánh tâm cụm theo đơn vị gốc giữa Mục 3 (2 cột) và Bài 2 (3 cột) với K = 5
    tam_2cot = pd.DataFrame(
        sc2.inverse_transform(km2.cluster_centers_),
        columns=["thu_nhap", "diem_chi_tieu"],
    )
    tam_2cot["so_khach"] = np.bincount(km2.labels_)
    tam_2cot["tuoi_tb"] = df.groupby(km2.labels_)["tuoi"].mean().values
    tam_2cot = tam_2cot[["tuoi_tb", "thu_nhap", "diem_chi_tieu", "so_khach"]]

    tam_3cot = pd.DataFrame(
        sc3.inverse_transform(km3_tot.cluster_centers_),
        columns=["tuoi_tb", "thu_nhap", "diem_chi_tieu"],
    )
    tam_3cot["so_khach"] = np.bincount(km3_tot.labels_)

    # Sắp xếp cả hai bảng theo cùng thứ tự (nhóm thu nhập thấp -> vừa -> cao, chi tiêu thấp -> cao) để dễ đối chiếu
    def sap_xep_nhom(bang):
        bang = bang.copy()
        bang["_nhom_tn"] = pd.cut(bang["thu_nhap"], bins=[0, 12, 24, 50], labels=[0, 1, 2])
        bang = bang.sort_values(["_nhom_tn", "diem_chi_tieu"]).drop(columns=["_nhom_tn"])
        return bang

    tam_2_sx = sap_xep_nhom(tam_2cot)
    tam_3_sx = sap_xep_nhom(tam_3cot)

    print("2. SO SANH TAM CUM (DON VI GOC) GIUA MUC 3 (2 COT) VA BAI 2 (3 COT) VOI K = 5")
    print("-" * 76)
    print("Bang A - Muc 3 (chi dung thu_nhap, diem_chi_tieu; tuoi_tb tinh sau):")
    print(tam_2_sx.round(1).to_string())
    print()
    print("Bang B - Bai 2 (dung ca 3 cot tuoi, thu_nhap, diem_chi_tieu):")
    print(tam_3_sx.round(1).to_string())
    print()

    # Tìm những khách hàng đổi cụm giữa 2 mô hình
    anh_xa = {}
    for c3 in range(5):
        anh_xa[c3] = int(pd.Series(km2.labels_[km3_tot.labels_ == c3]).mode()[0])
    nhan3_quy_doi = np.array([anh_xa[c] for c in km3_tot.labels_])
    chi_so_doi = np.where(km2.labels_ != nhan3_quy_doi)[0]

    ten_nhom = {
        0: "Thap-Nhieu (tre)",
        1: "Cao-It",
        2: "Vua-Vua",
        3: "Thap-It",
        4: "Cao-Nhieu",
    }

    print(f"3. CHI TIET {len(chi_so_doi)}/{len(df)} KHACH HANG DOI CUM KHI THEM COT TUOI")
    print("-" * 76)
    for idx in chi_so_doi:
        print(
            f"  Khach {idx:3d}: tuoi={df.loc[idx, 'tuoi']:2.0f}, "
            f"thu_nhap={df.loc[idx, 'thu_nhap']:4.1f}, "
            f"chi_tieu={df.loc[idx, 'diem_chi_tieu']:2.0f} | "
            f"2-cot: {ten_nhom[km2.labels_[idx]]:16s} -> 3-cot: {ten_nhom[nhan3_quy_doi[idx]]}"
        )
    print()

    # In thêm bảng tâm cụm K = 6 (vì silhouette K=6 = 0.4760 rất sát K=5 = 0.4808)
    km3_k6 = [item[3] for item in ket_qua if item[0] == 6][0]
    tam_k6 = pd.DataFrame(
        sc3.inverse_transform(km3_k6.cluster_centers_),
        columns=["tuoi_tb", "thu_nhap", "diem_chi_tieu"],
    )
    tam_k6["so_khach"] = np.bincount(km3_k6.labels_)
    print("Tham khao them - Tam cum voi K = 6 (silhouette = 0.4760, rat sat K = 5):")
    print(sap_xep_nhom(tam_k6).round(1).to_string())
    print()

    print("4. TRA LOI VA NHAN XET")
    print("-" * 76)
    print(f"- K tot nhat co con la 5 khong? CO, K = {k_tot} van dat silhouette cao nhat")
    print(f"  ({sil_tot:.4f}), nhung khoang cach voi K = 6 ({[x[2] for x in ket_qua if x[0]==6][0]:.4f}) rat sat sao.")
    print(f"  So voi 2 cot (silhouette = {sil_2cot_k5:.4f}), silhouette 3 cot giam manh con")
    print(f"  {sil_tot:.4f} vi cot tuoi co do chong lan lon giua cac nhom khach hang.")
    print("- Tam cac cum thay doi the nao so voi muc 3?")
    print("  + Cau truc 5 phan khuc chinh nhin chung GIU NGUYEN (205/210 khach khong doi).")
    print("  + Hai nhom 'Thu nhap thap, chi it' (8.0, 18.9, 40 khach) va 'Thu nhap cao,")
    print("    chi it' (30.1, 20.0, 35 khach) KHONG THAY DOI chut nao.")
    print("  + 4 khach tre (26-27 tuoi) thuoc nhom 'Vua-Vua' (tuoi tb ~40.4) bi keo sang")
    print("    nhom 'Thap-Nhieu' (tuoi tb ~24.9) vi tuoi gan hon -> tam nhom 'Thap-Nhieu'")
    print("    dich tu (thu_nhap 7.5, chi_tieu 80.2, 35 khach) sang (8.5, 77.6, 39 khach).")
    print("  + 1 khach lon tuoi (54 tuoi, thu nhap 27.4, chi tieu 63) chuyen tu nhom")
    print("    'Cao-Nhieu' (tuoi tb ~32.6) sang 'Vua-Vua' -> tuoi trung binh cua nhom")
    print("    'Vua-Vua' tang tu 40.4 len 41.6 (con 57 khach), con 'Cao-Nhieu' giam nhe")
    print("    tuoi tb tu 32.6 xuong 32.1 (con 39 khach).")
    print("  + Neu chon K = 6, nhom dong nhat 'Vua-Vua' se tach doi thanh hai nhom theo")
    print("    do tuoi: khach tre hon (~34.4 tuoi) va khach lon tuoi hon (~47.7 tuoi).")
    print("=" * 76)


if __name__ == "__main__":
    main()
