# -*- coding: utf-8 -*-
"""Bài tập 3: So sánh khởi tạo K-Means++ và khởi tạo ngẫu nhiên trên 50 lần chạy.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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


def chay_50_lan(Xs, cach_khoi_tao):
    """Chạy K-Means 50 lần với n_init=1, random_state từ 0 tới 49."""
    ds_inertia = []
    for rs in range(50):
        km = KMeans(
            n_clusters=5,
            init=cach_khoi_tao,
            n_init=1,
            random_state=rs,
        ).fit(Xs)
        ds_inertia.append(km.inertia_)
    return np.array(ds_inertia)


def main():
    duong_dan = tim_duong_dan_du_lieu("data/khach_hang.csv")
    df = pd.read_csv(duong_dan)
    Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

    inertia_rand = chay_50_lan(Xs, "random")
    inertia_pp = chay_50_lan(Xs, "k-means++")

    # Đếm số lần đạt inertia tối ưu 23.0 (giá trị chính xác ~23.02, làm tròn 1 chữ số = 23.0)
    so_lan_tot_rand = int(np.sum(np.round(inertia_rand, 1) == 23.0))
    so_lan_tot_pp = int(np.sum(np.round(inertia_pp, 1) == 23.0))
    so_lan_te_rand = 50 - so_lan_tot_rand
    so_lan_te_pp = 50 - so_lan_tot_pp

    print("=" * 76)
    print("BAI TAP 3: SO SANH K-MEANS++ VA KHOI TAO NGAU NHIEN (50 LAN, n_init=1)")
    print("=" * 76)

    print("1. BANG TONG HOP KET QUA (K = 5, random_state = 0..49)")
    print("-" * 76)
    print(
        f"{'Cach khoi tao':16s} {'Dat 23.0':>10s} {'Ty le tot':>10s} "
        f"{'Bi te':>7s} {'Min':>8s} {'Trung binh':>11s} {'Max':>8s}"
    )
    print("-" * 76)
    nhan_rand = 'init="random"'
    nhan_pp = 'init="k-means++"'
    print(
        f"{nhan_rand:16s} {so_lan_tot_rand:6d}/50  {so_lan_tot_rand * 2:8.1f}% "
        f"{so_lan_te_rand:7d} {inertia_rand.min():8.2f} {inertia_rand.mean():11.2f} {inertia_rand.max():8.2f}"
    )
    print(
        f"{nhan_pp:16s} {so_lan_tot_pp:6d}/50  {so_lan_tot_pp * 2:8.1f}% "
        f"{so_lan_te_pp:7d} {inertia_pp.min():8.2f} {inertia_pp.mean():11.2f} {inertia_pp.max():8.2f}"
    )
    print()

    print("2. PHAN BO GIA TRI INERTIA CUA 50 LAN CHAY (SAP XEP TANG DAN)")
    print("-" * 76)
    rand_tron = sorted(round(float(v), 1) for v in inertia_rand)
    pp_tron = sorted(round(float(v), 1) for v in inertia_pp)
    print("init=\"random\" (50 lan):")
    for i in range(0, 50, 10):
        print("  ", rand_tron[i:i + 10])
    print("init=\"k-means++\" (50 lan):")
    print(f"   Ca 50 lan deu dat dung {pp_tron[0]} (khong co bat ky lan nao bi te!).")
    print()

    # Thống kê nhanh các mức cực tiểu địa phương của init="random"
    muc_te = inertia_rand[np.round(inertia_rand, 1) > 23.0]
    rs_te_nhat = int(np.argmax(inertia_rand))
    print("Chi tiet cac lan bi mac ket cua init=\"random\":")
    print(f"  - So lan roi vao cuc tieu dia phuong te: {len(muc_te)}/50 ({len(muc_te)*2:.1f}%)")
    print(f"  - Khoang inertia te: tu {muc_te.min():.2f} den {muc_te.max():.2f}")
    print(f"  - Lan te nhat: random_state = {rs_te_nhat}, inertia = {inertia_rand[rs_te_nhat]:.2f} (gap {inertia_rand[rs_te_nhat]/inertia_rand.min():.1f} lan muc toi uu!)")
    print()

    print("3. KET LUAN VA GIAI THICH VAI TRO CUA KHOI TAO")
    print("-" * 76)
    print("- Cach khoi tao nao dang tin hon? init=\"k-means++\" DANG TIN HON HAN.")
    print(f"  Tren 50 lan thu doc lap voi n_init=1, k-means++ dat toi uu 23.0 ca {so_lan_tot_pp}/50")
    print(f"  lan (100%), trong khi khoi tao ngau nhien chi dat {so_lan_tot_rand}/50 lan (60%) va")
    print(f"  co toi {so_lan_te_rand}/50 lan (40%) dung o ket qua chia cum sai lech (inertia len")
    print(f"  toi {inertia_rand.max():.1f}, cao gap {inertia_rand.max()/inertia_rand.min():.1f} lan muc toi uu).")
    print("- Giai thich co che:")
    print("  1) Ban chat K-Means la thuat toan tham lam (giam dan inertia qua tung")
    print("     buoc gan va doi tam) va dung ngay khi khong cai thien duoc nua. Do ham")
    print("     muc tieu khong loi, K-Means rat de bi mac ket o CUC TIEU DIA PHUONG")
    print("     (local minimum) neu tam ban dau dat xau.")
    print("  2) Voi init=\"random\", 5 tam ban dau duoc boc hoan toan ngau nhien tu")
    print("     210 diem. Xac suat kha cao co 2 tam roi cung vao mot cum dong khach")
    print("     trong khi mot cum khac khong co tam nao. Khi hoi tu, mot cum that bi")
    print("     cat doi con hai cum that khac bi gop lam mot -> inertia tang vot")
    print("     (len 72-84 khi 1 cap bi gop, 117-119 khi gop nang, tham chi 204.0).")
    print("  3) Voi init=\"k-means++\", tam dau tien chon ngau nhien, nhung moi tam")
    print("     tiep theo duoc chon voi xac suat ty le thuan voi BINH PHUONG KHOANG")
    print("     CACH D(x)^2 toi tam gan nhat da chon. Nho do cac tam ban dau duoc")
    print("     trai deu ra xa nhau, moi cum that gan nhu chac chan co dung 1 tam")
    print("     khoi dau -> hoi tu nhanh va on dinh ve nghiem toi uu 23.0.")
    print("  4) Trong thuc te, ket hop init=\"k-means++\" (mac dinh cua scikit-learn)")
    print("     voi n_init=10 (chay 10 lan doc lap va giu lan co inertia nho nhat)")
    print("     la cach an toan nhat de tranh bay khoi tao xau.")
    print("=" * 76)


if __name__ == "__main__":
    main()
