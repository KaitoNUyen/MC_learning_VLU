# -*- coding: utf-8 -*-
"""Bài tập 6: MAE, RMSE và R^2 của hồi quy tuyến tính khi một căn bị đoán hỏng +5 tỷ.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
    duong_dan = tim_duong_dan_du_lieu("data/gia_nha.csv")
    df = pd.read_csv(duong_dan)

    # Dùng đầy đủ 3 đặc trưng của dữ liệu giá nhà bài 1, chia 80/20 (random_state=42) như bài 1
    X = df[["dien_tich", "so_phong", "tuoi_nha"]]
    y = df["gia"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mo_hinh = LinearRegression().fit(X_tr, y_tr)
    du_doan_goc = mo_hinh.predict(X_te)

    # Tính ba thước đo ban đầu trên tập kiểm tra
    mae_goc = mean_absolute_error(y_te, du_doan_goc)
    rmse_goc = float(np.sqrt(mean_squared_error(y_te, du_doan_goc)))
    r2_goc = r2_score(y_te, du_doan_goc)

    # Cộng thêm đúng 5 tỷ vào dự đoán của đúng một căn (căn đầu tiên trong tập kiểm tra)
    du_doan_hong = du_doan_goc.copy()
    du_doan_hong[0] += 5.0

    mae_sau = mean_absolute_error(y_te, du_doan_hong)
    rmse_sau = float(np.sqrt(mean_squared_error(y_te, du_doan_hong)))
    r2_sau = r2_score(y_te, du_doan_hong)

    # Phân tích tỷ trọng đóng góp của riêng căn bị đoán hỏng vào tổng sai số
    sai_so_sau = np.abs(y_te.to_numpy() - du_doan_hong)
    ty_trong_abs = sai_so_sau[0] / np.sum(sai_so_sau) * 100.0
    ty_trong_sq = (sai_so_sau[0] ** 2) / np.sum(sai_so_sau ** 2) * 100.0

    n_te = len(y_te)
    print("=" * 76)
    print("BAI TAP 6: MAE, RMSE VA R2 KHI CONG THEM 5 TY VAO DU DOAN DUNG MOT CAN")
    print("=" * 76)
    print(f"Du lieu gia_nha.csv: {len(df)} can (tap hoc: {len(y_tr)}, tap kiem tra: {n_te} can)")
    print(f"Can bi doan hong (can dau tien trong tap kiem tra):")
    print(f"  Gia that = {y_te.iloc[0]:.2f} ty | Du doan goc = {du_doan_goc[0]:.2f} ty "
          f"-> Du doan hong = {du_doan_hong[0]:.2f} ty (+5.00 ty)")
    print()

    print(f"{'Thuoc do':12s} {'Truoc (+0)':>12s} {'Sau (+5 ty)':>13s} {'Chenh lech':>13s} {'Ty le / Muc doi':>18s}")
    print("-" * 76)
    print(f"{'MAE (ty)':12s} {mae_goc:12.4f} {mae_sau:13.4f} {mae_sau - mae_goc:+13.4f} "
          f"{f'gap {mae_sau/mae_goc:.2f}x (+{(mae_sau/mae_goc-1)*100:.1f}%)':>18s}")
    print(f"{'RMSE (ty)':12s} {rmse_goc:12.4f} {rmse_sau:13.4f} {rmse_sau - rmse_goc:+13.4f} "
          f"{f'gap {rmse_sau/rmse_goc:.2f}x (+{(rmse_sau/rmse_goc-1)*100:.1f}%)':>18s}")
    print(f"{'R^2':12s} {r2_goc:12.4f} {r2_sau:13.4f} {r2_sau - r2_goc:+13.4f} "
          f"{f'giam {abs(r2_sau-r2_goc)/r2_goc*100:.1f}% gia tri':>18s}")
    print("-" * 76)
    print(f"Dong gop cua rieng 1 can hong vao tong sai so tuyet doi (MAE) : {ty_trong_abs:5.1f}%")
    print(f"Dong gop cua rieng 1 can hong vao tong binh phuong sai so     : {ty_trong_sq:5.1f}%")
    print()

    print("TRA LOI VA GIAI THICH:")
    print("-" * 76)
    print("1. Con so nao thay doi nhieu nhat?")
    print(f"   - Xet theo muc tang tuong doi giua hai thuoc do sai so: RMSE thay doi")
    print(f"     nhieu nhat, tang vot gap {rmse_sau/rmse_goc:.2f} lan (+{(rmse_sau/rmse_goc-1)*100:.1f}%, tu {rmse_goc:.4f} len {rmse_sau:.4f} ty),")
    print(f"     trong khi MAE chi tang gap {mae_sau/mae_goc:.2f} lan (+{(mae_sau/mae_goc-1)*100:.1f}%, tu {mae_goc:.4f} len {mae_sau:.4f} ty).")
    print(f"   - Dong thoi, R^2 cung bi danh guc rat nang (tut tu {r2_goc:.4f} xuong con")
    print(f"     {r2_sau:.4f}, giam {r2_goc - r2_sau:.4f}) vi cong thuc R^2 = 1 - SSE / SST cung dung")
    print("     tong binh phuong sai so SSE = n * RMSE^2.")
    print()
    print("2. Giai thich vi sao RMSE phat nang sai so lon hon MAE:")
    print("   - MAE = (1/n) * sum(|e_i|): moi sai so tham gia TUYEN TINH (bac 1).")
    print(f"     Khi 1 can lech them 5 ty trong n = {n_te} can, sai so do duoc chia")
    print(f"     deu cho ca {n_te} can nen MAE chi tang toi da 5 / {n_te} = {5/n_te:.4f} ty.")
    print("   - RMSE = sqrt((1/n) * sum(e_i^2)): moi sai so bi BINH PHUONG (bac 2)")
    print("     truoc khi lay trung binh roi moi khai can. Mot sai so ~5 ty khi")
    print(f"     binh phuong len toi ~25, chiem toi {ty_trong_sq:.1f}% tong binh phuong sai so")
    print(f"     cua ca {n_te} can, day RMSE len xap xi 5 / sqrt({n_te}) = {5/np.sqrt(n_te):.4f} ty")
    print(f"     (thuc te dat {rmse_sau:.4f} ty, lon hon muc tang MAE ~sqrt({n_te}) = {np.sqrt(n_te):.2f} lan).")
    print("   - Ket luan: RMSE (va R^2) rat nhay voi cac diem ngoai lai / ca doan")
    print("     hong nang; chon RMSE khi muon phat nang sai lech lon, chon MAE khi")
    print("     muon danh gia sai so dien hinh it bi meo boi vai ca ca biet.")
    print("=" * 76)


if __name__ == "__main__":
    main()
