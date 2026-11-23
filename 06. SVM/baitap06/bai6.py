# -*- coding: utf-8 -*-
"""Bài tập 6: Đo thời gian huấn luyện SVM trên dữ liệu lớn (SVC vs LinearSVC).

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import math
import sys
import time
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    # Chạy khởi động nhẹ để tránh nhiễu nạp thư viện ở lần đo đầu tiên
    X_w, y_w = make_classification(n_samples=100, n_features=20, random_state=0)
    SVC().fit(X_w, y_w)
    LinearSVC(random_state=0).fit(X_w, y_w)

    cac_co = [2000, 5000, 10000, 20000]
    ket_qua = []

    print("=" * 76)
    print("BAI TAP 6: DO THOI GIAN HUAN LUYEN SVM TREN DU LIEU LON")
    print("=" * 76)
    print(f"{'So mau (n)':>12s} {'SVC() (giay)':>16s} {'LinearSVC() (giay)':>20s} {'SVC / LinearSVC':>18s}")
    print("-" * 76)

    for n in cac_co:
        X, y = make_classification(
            n_samples=n,
            n_features=20,
            random_state=0,
        )

        # Đo thời gian .fit của SVC() (kernel RBF mặc định)
        t0 = time.perf_counter()
        SVC().fit(X, y)
        t_svc = time.perf_counter() - t0

        # Đo thời gian .fit của LinearSVC()
        t0 = time.perf_counter()
        LinearSVC(random_state=0).fit(X, y)
        t_linear = time.perf_counter() - t0

        ti_le = t_svc / t_linear if t_linear > 0 else float("inf")
        ket_qua.append((n, t_svc, t_linear))
        print(f"{n:12,d} {t_svc:16.4f} {t_linear:20.4f} {ti_le:17.1f}x")

    print()

    # Tính hệ số tăng thời gian khi dữ liệu tăng gấp đôi: 5,000 -> 10,000 và 10,000 -> 20,000
    tra_cuu = {n: (t_svc, t_lin) for n, t_svc, t_lin in ket_qua}
    svc_5_10 = tra_cuu[10000][0] / tra_cuu[5000][0]
    svc_10_20 = tra_cuu[20000][0] / tra_cuu[10000][0]
    svc_tb = math.sqrt(svc_5_10 * svc_10_20)

    lin_5_10 = tra_cuu[10000][1] / tra_cuu[5000][1]
    lin_10_20 = tra_cuu[20000][1] / tra_cuu[10000][1]
    lin_tb = math.sqrt(lin_5_10 * lin_10_20)

    print("HE SO TANG THOI GIAN KHI DU LIEU TANG GAP DOI (2x):")
    print("-" * 76)
    print(f"  Tu  5,000 -> 10,000 mau: SVC tang {svc_5_10:.2f} lan | LinearSVC tang {lin_5_10:.2f} lan")
    print(f"  Tu 10,000 -> 20,000 mau: SVC tang {svc_10_20:.2f} lan | LinearSVC tang {lin_10_20:.2f} lan")
    print(f"  Trung binh moi lan 2x  : SVC tang ~{svc_tb:.2f} lan | LinearSVC tang ~{lin_tb:.2f} lan")
    print()

    print("NHAN XET:")
    print("-" * 76)
    print(
        f"- Doi voi SVC() (kernel RBF): khi du lieu tang gap doi, thoi gian hoc"
    )
    print(
        f"  tang trung binh ~{svc_tb:.1f} lan (xap xi quy luat bac hai ~4 lan tro len)."
    )
    print(
        "  Nguyen nhan la SVC dung thuat toan SMO tren ma tran kernel giua cac cap"
    )
    print(
        "  diem du lieu, co do phuc tap tang khoang O(n^2) den O(n^3)."
    )
    print(
        f"- Doi voi LinearSVC(): khi du lieu tang gap doi, thoi gian hoc chi tang"
    )
    print(
        f"  trung binh ~{lin_tb:.1f} lan (xap xi tuyen tinh O(n)), va chay nhanh"
    )
    print(
        "  hon SVC() hang tram lan khi n lon (toi uu truc tiep theo w bang LIBLINEAR)."
    )
    print(
        "- Ket luan: Voi tap du lieu hang chuc nghin mau tro len, nen uu tien"
    )
    print(
        "  LinearSVC() (hoac thu tren mau nho truoc) thay vi dung SVC kernel RBF."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
