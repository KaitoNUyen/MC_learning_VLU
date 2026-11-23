# -*- coding: utf-8 -*-
"""Bài tập 5: Chạy DBSCAN trên bộ hai vầng trăng và so sánh với K-Means.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

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
    duong_dan = tim_duong_dan_du_lieu("data/hai_vang_trang.csv")
    df = pd.read_csv(duong_dan)
    X = df[["x1", "x2"]].to_numpy()
    y_that = df["nhan"].to_numpy()

    # Đối chiếu với K-Means (K = 2) từ mục 5
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    nhan_km = km.fit_predict(X)
    khop_km = max((nhan_km == y_that).mean(), (nhan_km != y_that).mean())
    ari_km = adjusted_rand_score(y_that, nhan_km)

    print("=" * 76)
    print("BAI TAP 5: DBSCAN TREN BO HAI VANG TRANG (min_samples = 5)")
    print("=" * 76)
    print(f"Tap du lieu: {len(df)} diem, 2 lop that (moi lop {(y_that == 0).sum()} diem)")
    print(f"Doi chieu K-Means (K=2): ty le khop nhan that = {khop_km:.4f}, ARI = {ari_km:.4f}")
    print()

    print("1. KET QUA DBSCAN VOI eps = 0.2, 0.3, 0.4 (min_samples = 5)")
    print("-" * 76)
    print(
        f"{'eps':>5s} {'So cum':>8s} {'Diem nhieu (-1)':>17s} "
        f"{'Trong cum':>11s} {'ARI vs nhan that':>18s}"
    )
    print("-" * 76)

    for e in [0.2, 0.3, 0.4]:
        db = DBSCAN(eps=e, min_samples=5).fit(X)
        nhan = db.labels_
        # Không tính nhãn -1 (nhiễu) vào số cụm tìm được
        so_cum = len(set(nhan)) - (1 if -1 in nhan else 0)
        so_nhieu = int((nhan == -1).sum())
        trong_cum = len(nhan) - so_nhieu
        ari = adjusted_rand_score(y_that, nhan)
        print(f"{e:5.1f} {so_cum:8d} {so_nhieu:10d} ({so_nhieu/len(nhan)*100:4.1f}%) {trong_cum:11d} {ari:18.4f}")
    print()

    # Khảo sát bổ sung khi eps nhỏ hơn (0.15, 0.18) và khi tăng min_samples để thấy rõ cơ chế
    print("2. KHAO SAT BO SUNG DE HIEU RO CO CHE (eps nho hon / min_samples lon hon)")
    print("-" * 76)
    for e, ms in [(0.15, 5), (0.18, 5), (0.25, 10), (0.30, 12)]:
        db = DBSCAN(eps=e, min_samples=ms).fit(X)
        nhan = db.labels_
        so_cum = len(set(nhan)) - (1 if -1 in nhan else 0)
        so_nhieu = int((nhan == -1).sum())
        ari = adjusted_rand_score(y_that, nhan)
        bo_sung = ""
        if so_cum == 2:
            # Tính tỷ lệ khớp trên các điểm không phải nhiễu và toàn bộ
            k0 = ((nhan == 0) & (y_that == 0)).sum() + ((nhan == 1) & (y_that == 1)).sum()
            k1 = ((nhan == 0) & (y_that == 1)).sum() + ((nhan == 1) & (y_that == 0)).sum()
            khop_all = max(k0, k1) / len(y_that)
            bo_sung = f", khop={khop_all:.4f}"
        print(
            f"  eps={e:.2f}, min_samples={ms:2d} -> "
            f"so_cum={so_cum:2d}, nhieu={so_nhieu:3d}, ARI={ari:+.4f}{bo_sung}"
        )
    print()

    print("3. TRA LOI VA NHAN XET VAI TRO CUA eps")
    print("-" * 76)
    print("- Co gia tri nao trong {0.2, 0.3, 0.4} tach duoc hai vang trang tot hon")
    print("  K-Means khong? KHONG.")
    print("  Ca ba gia tri eps = 0.2, 0.3, 0.4 (voi min_samples = 5) deu chi tim")
    print("  duoc DUNG 1 CUM DUY NHAT (ARI xap xi 0.00), tuc la gop chung ca hai")
    print("  vang trang lam mot, kem hon K-Means (ty le khop 0.7500, ARI 0.2475).")
    print("- Vi sao lai xay ra hien tuong nay tren data/hai_vang_trang.csv?")
    print("  + Ve ly thuyet, DBSCAN co the tach cum hinh cong (khong tron) rat tot")
    print("    neu giua hai vang trang co khoang trong mat do thap ro rang.")
    print("  + Tuy nhien, bo data/hai_vang_trang.csv co DO NHIEU CAO, khien hai")
    print("    vang trang chong lan nhau o vung giua. Cac diem nhieu o vung tiep")
    print("    giap tao thanh 'cau noi mat do' (hieu ung bac cau / chaining), noi")
    print("    lien hai vang trang thanh mot cum lien thong duy nhat.")
    print("- Nhan xet vai tro cua tham so eps:")
    print("  + eps la ban kinh lan can dung de dem so hang xom quanh moi diem")
    print("    (diem co >= min_samples diem trong ban kinh eps la diem loi) va")
    print("    noi cac diem loi gan nhau vao cung mot cum.")
    print("  + Khi eps TANG (0.2 -> 0.3 -> 0.4): vung lan can rong hon, so diem")
    print("    nhieu giam manh (40 -> 10 -> 3), nhung cac cum o gan nhau rat de")
    print("    bi GOP DINH thanh 1 cum lon.")
    print("  + Khi eps QUA NHO (vi du 0.15 - 0.18): vung lan can qua hep, hai vang")
    print("    trang bi VO VUN thanh 5-16 cum nho va luong nhieu tang vot (50-99).")
    print("  + Ket luan: DBSCAN khac phuc duoc gioi han cum hinh tron cua K-Means")
    print("    khi du lieu sach/tach biet mat do, nhung rat nhay voi eps va de that")
    print("    bai khi hai cum co nhieu bac cau qua nhau.")
    print("=" * 76)


if __name__ == "__main__":
    main()
