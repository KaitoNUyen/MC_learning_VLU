# -*- coding: utf-8 -*-
"""Bài tập 4: Dò lưới mịn quanh cặp tham số tốt nhất của mục 5.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
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

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # Đối chiếu kết quả lưới thưa ở mục 5
    luoi_thua = {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]}
    tim_thua = GridSearchCV(SVC(kernel="rbf"), luoi_thua, cv=5)
    tim_thua.fit(X_tr, y_tr)
    cv_thua = tim_thua.best_score_
    kt_thua = tim_thua.score(X_te, y_te)

    # Dò lưới mịn quanh cặp tốt nhất (C=10, gamma=1) của mục 5
    luoi_min = {"C": [3, 10, 30], "gamma": [0.3, 1, 3]}
    tim_min = GridSearchCV(SVC(kernel="rbf"), luoi_min, cv=5)
    tim_min.fit(X_tr, y_tr)
    cv_min = tim_min.best_score_
    kt_min = tim_min.score(X_te, y_te)

    bang = pd.DataFrame(tim_min.cv_results_["params"])
    bang["diem"] = tim_min.cv_results_["mean_test_score"]

    print("=" * 72)
    print("BAI TAP 4: DO LUOI MIN QUANH CAP TOT NHAT CUA MUC 5")
    print("=" * 72)
    print("Diem kiem dinh cheo (5-fold CV) cua luoi min, hang la C, cot la gamma:")
    print(bang.pivot(index="C", columns="gamma", values="diem").round(4).to_string())
    print()

    print("SO SANH LUOI THUA (MUC 5) VA LUOI MIN (BAI 4):")
    print("-" * 72)
    print(
        f"  Luoi thua (muc 5): cap = {tim_thua.best_params_}, "
        f"CV = {cv_thua:.4f}, tap kiem tra = {kt_thua:.4f}"
    )
    print(
        f"  Luoi min  (bai 4): cap = {tim_min.best_params_}, "
        f"CV = {cv_min:.4f}, tap kiem tra = {kt_min:.4f}"
    )
    print()

    print("NHAN XET:")
    print("-" * 72)
    chenh_cv = cv_min - cv_thua
    chenh_kt = kt_min - kt_thua
    if chenh_cv > 1e-9:
        print(
            f"1. Diem kiem dinh cheo tot nhat CO cao hon 0.8711 "
            f"(tang tu {cv_thua:.4f} len {cv_min:.4f}, chenh {chenh_cv:+.4f})."
        )
    else:
        print(
            f"1. Diem kiem dinh cheo tot nhat KHONG cao hon 0.8711 "
            f"(dat {cv_min:.4f} so voi {cv_thua:.4f})."
        )

    if abs(chenh_kt) > 1e-9:
        print(
            f"2. Accuracy tren tap kiem tra CO thay doi "
            f"(tu {kt_thua:.4f} thanh {kt_min:.4f}, chenh {chenh_kt:+.4f})."
        )
    else:
        print(
            f"2. Accuracy tren tap kiem tra KHONG doi (van giu nguyen {kt_min:.4f})."
        )

    print("3. Nhan xet ve cai loi cua viec do that min:")
    print("   - Quanh vung tham so tot, be mat diem kiem dinh cheo thuong kha phang,")
    print("     nen do min chi cai thien rat it (hoac khong cai thien) tren tap kiem tra.")
    print("   - Neu do qua min tren tap du lieu nho, chi phi tinh toan tang nhanh")
    print("     va mo hinh con co nguy co bam theo nhieu cua tap kiem dinh cheo.")
    print("   - Vi vay chien luoc hop ly la do luoi thua theo boi so 10 truoc de tim")
    print("     dung vung tot, roi chi do min vua phai quanh vung do.")
    print("=" * 72)


if __name__ == "__main__":
    main()
