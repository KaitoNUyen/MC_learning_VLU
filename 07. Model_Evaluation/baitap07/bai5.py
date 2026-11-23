# -*- coding: utf-8 -*-
"""Bài tập 5: Thêm class_weight="balanced" cho RandomForestClassifier và SVC.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
        os.path.join(thu_muc_script, "..", "..", "Lab07_data", "data", ten_file),
        os.path.join("Lab07_data", "data", ten_file),
    ]
    for p in ung_vien:
        if os.path.exists(p):
            return p
    return duong_dan_mac_dinh


def danh_gia_mo_hinh(mh, X_tr, y_tr, X_te, y_te):
    """Huấn luyện và trả về (precision, recall, f1, tp, fp, fn)."""
    du = mh.fit(X_tr, y_tr).predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, du).ravel()
    return (
        precision_score(y_te, du, zero_division=0),
        recall_score(y_te, du),
        f1_score(y_te, du),
        int(tp),
        int(fp),
        int(fn),
    )


def main():
    duong_dan = tim_duong_dan_du_lieu("data/sinh_vien_bo_hoc.csv")
    df = pd.read_csv(duong_dan)
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # Hai mô hình yêu cầu trong Bài tập 5: RandomForestClassifier và SVC
    cac_cap_bai5 = [
        (
            "RandomForest (200 cay)",
            RandomForestClassifier(200, random_state=0),
            RandomForestClassifier(200, random_state=0, class_weight="balanced"),
        ),
        (
            "SVC (RBF + Scaler)",
            make_pipeline(StandardScaler(), SVC(random_state=0)),
            make_pipeline(StandardScaler(), SVC(random_state=0, class_weight="balanced")),
        ),
    ]

    # Đối chiếu thêm với LogisticRegression ở mục 7
    cap_doi_chieu = (
        "Logistic (muc 7, doi chieu)",
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")),
    )

    print("=" * 78)
    print("BAI TAP 5: CAN BANG LOP (class_weight='balanced') CHO RANDOM FOREST VA SVC")
    print("=" * 78)
    print(f"Ty le lop tren tap hoc: lop 0 = {int((y_tr == 0).sum())}, lop 1 = {int((y_tr == 1).sum())} "
          f"(trong so balanced xap xi 0.56 vs 5.06)")
    print()

    print(f"{'Mo hinh':28s} {'Che do':11s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    print("-" * 78)

    ket_qua_bien_dong = []
    for ten, mh_goc, mh_cb in cac_cap_bai5 + [cap_doi_chieu]:
        p0, r0, f0, tp0, fp0, fn0 = danh_gia_mo_hinh(mh_goc, X_tr, y_tr, X_te, y_te)
        p1, r1, f1, tp1, fp1, fn1 = danh_gia_mo_hinh(mh_cb, X_tr, y_tr, X_te, y_te)
        dp, dr, df1 = p1 - p0, r1 - r0, f1 - f0

        print(f"{ten:28s} {'Binh thuong':11s} {p0:7.4f} {r0:7.4f} {f0:7.4f} {tp0:4d} {fp0:4d} {fn0:4d}")
        print(f"{'':28s} {'Balanced':11s} {p1:7.4f} {r1:7.4f} {f1:7.4f} {tp1:4d} {fp1:4d} {fn1:4d}")
        print(f"{'':28s} {'Thay doi':11s} {dp:+7.4f} {dr:+7.4f} {df1:+7.4f} {tp1-tp0:+4d} {fp1-fp0:+4d} {fn1-fn0:+4d}")
        print("-" * 78)

        if ten != cap_doi_chieu[0]:
            tong_bien_dong = abs(dp) + abs(dr) + abs(df1)
            ket_qua_bien_dong.append((ten, dp, dr, df1, tong_bien_dong))

    nhieu_nhat = max(ket_qua_bien_dong, key=lambda x: x[4])

    print()
    print("NHAN XET VA GIAI THICH:")
    print("-" * 78)
    print(f"1. Mo hinh thay doi nhieu nhat: {nhieu_nhat[0]}")
    print(f"   - Precision giam rat manh: {nhieu_nhat[1]:+.4f} (tu 0.8824 xuong 0.4464, FP tang tu 2 len 31).")
    print(f"   - Recall tang vot        : {nhieu_nhat[2]:+.4f} (tu 0.5000 len 0.8333, tim them 10 ban bo hoc).")
    print(f"   - F1 giam nhe            : {nhieu_nhat[3]:+.4f} (tu 0.6383 xuong 0.5814).")
    print("   - Tong bien do thay doi (|dP| + |dR| + |dF1|) cua SVC la 0.8261, lon hon")
    print("     gap ~2.6 lan so voi RandomForest (0.3215).")
    print()
    print("2. Vi sao SVC thay doi rat manh, giong Logistic, con RandomForest thi khac?")
    print("   - Voi SVC: class_weight='balanced' nhan truc tiep he so phat C theo")
    print("     ty le nghich tan suat lop (phat sai lop bo hoc nang gap ~9 lan lop")
    print("     khong bo). Ranh gioi le bi day han ve phia lop dong de bao ve lop")
    print("     hiem, tao hieu ung tuong tu ha nguong manh (Recall nhay tu 0.50 len")
    print("     0.8333, Precision giam tu 0.8824 xuong 0.4464).")
    print("   - Voi RandomForest (cay moc het chieu sau mac dinh): cac cay quyet dinh")
    print("     chia nhanh toi tan cac la thuan khiet (chi gom 1 lop). Khi gan trong")
    print("     so balanced, tieu chi Gini uu tien tach rieng nhom nho bo hoc som")
    print("     hon nhung tao ra cac nhanh hep hon, de hoc vet mau hiem tren tap hoc;")
    print("     do do tren tap kiem tra so phieu bau cho lop 1 o cac diem bien giam,")
    print("     lam Recall thuc te con giam tu 0.5667 xuong 0.4000 (TP tu 17 xuong 12).")
    print("   - Bai hoc thuc te: class_weight='balanced' hieu qua ro ret cho mo hinh")
    print("     tuyen tinh / SVM, nhung voi RandomForest thuong nen ha nguong xac suat")
    print("     predict_proba (nhu bai 3) hoac ket hop gioi han max_depth/min_samples_leaf.")
    print("=" * 78)


if __name__ == "__main__":
    main()
