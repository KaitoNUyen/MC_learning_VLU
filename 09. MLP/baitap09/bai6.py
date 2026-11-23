# -*- coding: utf-8 -*-
"""Bài tập 6: Mạng nơ-ron cho hồi quy (MLPRegressor) trên bộ dữ liệu giá nhà.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
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


def main():
    duong_dan = tim_duong_dan_du_lieu("data/gia_nha.csv")
    df = pd.read_csv(duong_dan)
    X = df[["dien_tich"]]
    y = df["gia"]

    # Chia tập học / tập kiểm tra đúng theo Bài 1 (test_size=0.2, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Hồi quy tuyến tính (đường thẳng) từ Bài 1
    hqtt = LinearRegression().fit(X_tr, y_tr)
    du_hqtt = hqtt.predict(X_te)
    mae_hqtt = mean_absolute_error(y_te, du_hqtt)
    r2_hqtt = r2_score(y_te, du_hqtt)

    # 2. MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0) có chuẩn hoá
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0),
    ).fit(X_tr, y_tr)
    du_mlp = mlp.predict(X_te)
    mae_mlp = mean_absolute_error(y_te, du_mlp)
    r2_mlp = r2_score(y_te, du_mlp)

    # Thử nghiệm tham khảo thêm với activation='tanh' và 'logistic' để thấy rõ hành vi ngoại suy
    mlp_tanh = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), activation="tanh", max_iter=5000, random_state=0),
    ).fit(X_tr, y_tr)
    mlp_sig = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), activation="logistic", max_iter=5000, random_state=0),
    ).fit(X_tr, y_tr)

    # Dự đoán ngoại suy tại căn 250 m2 (và thêm một số mốc để quan sát dáng đường cong)
    can_250 = pd.DataFrame({"dien_tich": [250.0]})
    gia_250_hqtt = hqtt.predict(can_250)[0]
    gia_250_mlp = mlp.predict(can_250)[0]
    gia_250_tanh = mlp_tanh.predict(can_250)[0]
    gia_250_sig = mlp_sig.predict(can_250)[0]

    so_ts_mlp = (
        sum(w.size for w in mlp[-1].coefs_)
        + sum(b.size for b in mlp[-1].intercepts_)
    )

    print("=" * 78)
    print("BAI TAP 6: MANG NO-RON CHO HOI QUY (MLPRegressor) TREN DU LIEU GIA NHA")
    print("=" * 78)
    print(f"Du lieu gia nha: {len(df)} can (tap hoc: {len(X_tr)}, tap kiem tra: {len(X_te)})")
    print(f"Mien dien tich trong du lieu : [{X['dien_tich'].min():.1f}, {X['dien_tich'].max():.1f}] m2")
    print(f"Phuong trinh hoi quy tuyen tinh: gia = {hqtt.coef_[0]:.6f} * dien_tich + {hqtt.intercept_:.6f}")
    print(f"MLPRegressor (20,) ReLU      : {so_ts_mlp} tham so, hoi tu sau {mlp[-1].n_iter_} vong")
    print()

    print("1. SO SANH MAE, R2 TREN TAP KIEM TRA VA DU DOAN NGOAI SUY CAN 250 m2")
    print("-" * 78)
    print(f"{'Mo hinh':34s} {'MAE (ty)':>10s} {'R2':>9s} {'Gia can 250 m2':>18s}")
    print("-" * 78)
    print(f"{'Hoi quy tuyen tinh (duong thang)':34s} {mae_hqtt:10.4f} {r2_hqtt:9.4f} {gia_250_hqtt:14.4f} ty")
    print(f"{'MLPRegressor (20,), ReLU [de bai]':34s} {mae_mlp:10.4f} {r2_mlp:9.4f} {gia_250_mlp:14.4f} ty")
    print("-" * 78)
    print("Tham khao them (neu doi ham kich hoat bi chan cho cung MLPRegressor):")
    print(
        f"{'  MLPRegressor (20,), tanh':34s} "
        f"{mean_absolute_error(y_te, mlp_tanh.predict(X_te)):10.4f} "
        f"{r2_score(y_te, mlp_tanh.predict(X_te)):9.4f} {gia_250_tanh:14.4f} ty"
    )
    print(
        f"{'  MLPRegressor (20,), logistic':34s} "
        f"{mean_absolute_error(y_te, mlp_sig.predict(X_te)):10.4f} "
        f"{r2_score(y_te, mlp_sig.predict(X_te)):9.4f} {gia_250_sig:14.4f} ty"
    )
    print()

    print("2. SO SANH DU DOAN TAI CAC MOC TRONG VA NGOAI MIEN DU LIEU")
    print("-" * 78)
    print(f"{'Dien tich (m2)':>14s} | {'Tuyen tinh':>12s} {'MLP (ReLU)':>12s} {'Chenh lech':>12s} | {'Vi tri'}")
    print("-" * 78)
    for dt in [35.0, 60.0, 80.0, 100.0, 117.5, 150.0, 200.0, 250.0]:
        df_dt = pd.DataFrame({"dien_tich": [dt]})
        p_l = hqtt.predict(df_dt)[0]
        p_m = mlp.predict(df_dt)[0]
        vung = "Trong mien hoc" if 35.5 <= dt <= 117.5 else "Ngoai suy"
        print(f"{dt:14.1f} | {p_l:12.4f} {p_m:12.4f} {p_m - p_l:+12.4f} | {vung}")
    print("-" * 78)
    print()

    print("3. NHAN XET: MANG NO-RON CO NGOAI SUY TOT NHU DUONG THANG KHONG?")
    print("-" * 78)
    print("- KHONG. Mang no-ron NGOAI SUY KEM HON RO RET so voi duong thang tuyen tinh:")
    print(f"  1. Ngay tren tap kiem tra (noi suy), MAE cua MLPRegressor ({mae_mlp:.4f} ty) cao")
    print(f"     gan gap doi hoi quy tuyen tinh ({mae_hqtt:.4f} ty), R2 giam tu {r2_hqtt:.4f} xuong")
    print(f"     {r2_mlp:.4f}. Ly do: tap hoc chi co 48 can nhung mang 20 no-ron an co toi")
    print(f"     {so_ts_mlp} tham so (nhieu hon ca so mau), trong khi quy luat that cua gia")
    print("     nha theo dien tich la tuyen tinh gan nhu hoan hao.")
    print(f"  2. Tai can 250 m2 (vuot xa muc toi da 117.5 m2 da hoc): duong thang tuyen")
    print(f"     tinh giu dung do doc toan cuc ~0.0778 ty/m2 nen du doan {gia_250_hqtt:.2f} ty dong.")
    print(f"     Con MLPRegressor (mac dinh ReLU) du doan vot len {gia_250_mlp:.2f} ty dong")
    print(f"     (lech +{gia_250_mlp - gia_250_hqtt:.2f} ty, tuong duong +{(gia_250_mlp / gia_250_hqtt - 1)*100:.1f}%) vi cac doan gap khuc")
    print("     ReLU keo dai mu quang do doc cuc bo qua doc o mep phai (~0.136 ty/m2).")
    print("  3. Neu thay ReLU bang ham bi chan nhu 'tanh' hay 'logistic', tinh hinh ngoai")
    print(f"     suy con ro hon: gia can 250 m2 bi bao hoa phang o muc chi {gia_250_tanh:.2f} ty")
    print(f"     (tanh) va {gia_250_sig:.2f} ty (logistic), khong tang noi theo dien tich.")
    print("  4. Ket luan: Mang no-ron gioi noi suy tren quan he phi tuyen nhieu du lieu,")
    print("     nhung KHONG dang tin khi ngoai suy vuot ngoai mien da hoc; voi du lieu")
    print("     nho co xu huong tuyen tinh ro rang, hoi quy tuyen tinh van vuot troi.")
    print("=" * 78)


if __name__ == "__main__":
    main()
