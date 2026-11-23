# -*- coding: utf-8 -*-
"""Bài tập 2: Thêm một điểm và quan sát sự thay đổi của đường ranh giới.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import sys
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def huan_luyen_va_thong_tin(X, y):
    """Huấn luyện SVM tuyến tính C=1000 và trả về các thông số chính."""
    svm = SVC(kernel="linear", C=1000)
    svm.fit(X, y)
    w1, w2 = svm.coef_[0]
    b = svm.intercept_[0]
    do_rong_le = 2.0 / np.sqrt(w1 ** 2 + w2 ** 2)
    return svm, w1, w2, b, do_rong_le


def main():
    # Bộ 24 điểm gốc của mục 2
    X_goc, y_goc = make_blobs(
        n_samples=24,
        centers=[[2, 2], [6, 6]],
        cluster_std=0.9,
        random_state=0,
    )
    y_goc = 1 - y_goc

    # Trường hợp 0: 24 điểm ban đầu
    svm0, w1_0, w2_0, b_0, le_0 = huan_luyen_va_thong_tin(X_goc, y_goc)

    # Trường hợp 1: Thêm điểm lớp 1 tại (-1, -1)
    diem_1 = np.array([[-1.0, -1.0]])
    X_them1 = np.vstack([X_goc, diem_1])
    y_them1 = np.append(y_goc, 1)
    svm1, w1_1, w2_1, b_1, le_1 = huan_luyen_va_thong_tin(X_them1, y_them1)
    f_diem1_cu = svm0.decision_function(diem_1)[0]

    # Trường hợp 2: Bỏ (-1, -1), thay bằng điểm lớp 1 tại (4.5, 4)
    diem_2 = np.array([[4.5, 4.0]])
    X_them2 = np.vstack([X_goc, diem_2])
    y_them2 = np.append(y_goc, 1)
    svm2, w1_2, w2_2, b_2, le_2 = huan_luyen_va_thong_tin(X_them2, y_them2)
    f_diem2_cu = svm0.decision_function(diem_2)[0]

    print("=" * 74)
    print("BAI TAP 2: THEM MOT DIEM VA TAC DONG CUA VECTO HO TRO")
    print("=" * 74)
    print(f"{'Truong hop':26s} {'w1':>8s} {'w2':>8s} {'b':>8s} {'Le':>8s} {'So SV':>6s}")
    print("-" * 74)
    print(
        f"{'0. Goc (24 diem)':26s} {w1_0:8.3f} {w2_0:8.3f} {b_0:8.3f} "
        f"{le_0:8.4f} {len(svm0.support_):6d}"
    )
    print(
        f"{'1. Them (-1, -1), lop 1':26s} {w1_1:8.3f} {w2_1:8.3f} {b_1:8.3f} "
        f"{le_1:8.4f} {len(svm1.support_):6d}"
    )
    print(
        f"{'2. Them (4.5, 4), lop 1':26s} {w1_2:8.3f} {w2_2:8.3f} {b_2:8.3f} "
        f"{le_2:8.4f} {len(svm2.support_):6d}"
    )
    print()

    print("CHI TIET CAC VECTO HO TRO:")
    print(f"  0. Goc              : chi so SV = {svm0.support_.tolist()}")
    print(
        f"  1. Them (-1, -1)    : chi so SV = {svm1.support_.tolist()} "
        f"(diem moi chi so 24 co f_cu = {f_diem1_cu:+.3f})"
    )
    print(
        f"  2. Them (4.5, 4)    : chi so SV = {svm2.support_.tolist()} "
        f"(diem moi chi so 24 co f_cu = {f_diem2_cu:+.3f})"
    )
    print()

    # In thông tin các vectơ hỗ trợ ở trường hợp 2
    print("Cac vecto ho tro sau khi them (4.5, 4):")
    for idx in svm2.support_:
        f_val = svm2.decision_function(X_them2[idx:idx + 1])[0]
        nhan_them = " <-- DIEM MOI" if idx == 24 else ""
        print(
            f"  diem {idx:2d}: x1 = {X_them2[idx, 0]:.2f}, "
            f"x2 = {X_them2[idx, 1]:.2f}, lop {y_them2[idx]}, "
            f"f(x) = {f_val:+.3f}{nhan_them}"
        )
    print()

    print("NHAN XET VA GIAI THICH BANG KHAI NIEM VECTO HO TRO:")
    print("-" * 74)
    print("1. Lan nao duong ranh gioi thay doi?")
    print("   - Lan 1 - Them (-1, -1) lop 1 : KHONG THAY DOI.")
    print("     He so w1, w2, b va do rong le giu nguyen hoan toan so voi ban dau.")
    print("   - Lan 2 - Them (4.5, 4) lop 1 : THAY DOI RO RET.")
    print(
        f"     w1 doi tu {w1_0:.3f} thanh {w1_2:.3f}, w2 tu {w2_0:.3f} thanh {w2_2:.3f}, "
        f"b tu {b_0:.3f} thanh {b_2:.3f},"
    )
    print(f"     va do rong le bi bop hep manh tu {le_0:.4f} xuong {le_2:.4f}.")
    print()
    print("2. Giai thich bang khai niem vecto ho tro:")
    print("   - Trong SVM, duong ranh gioi va hai mep le CHI duoc quyet dinh")
    print("     boi cac vecto ho tro (nhung diem nam sat ranh gioi nhat).")
    print("     Moi diem nam dung phia va o ngoai le khong gop phan nao vao w, b.")
    print(f"   - Diem (-1, -1) lop 1 co f_cu = {f_diem1_cu:+.3f} > +1, tuc la nam rat")
    print("     sau trong vung lop 1 va hoan toan ngoai le. No KHONG tro thanh")
    print("     vecto ho tro, tap 3 vecto ho tro cu [10, 13, 18] giu nguyen,")
    print("     nen duong ranh gioi khong he thay doi.")
    print(f"   - Diem (4.5, 4) lop 1 co f_cu = {f_diem2_cu:+.3f} < +1 (tham chi am,")
    print("     nam lan sang phia lop 0 cua ranh gioi cu). Diem nay vi pham le")
    print("     nen voi C = 1000, SVM buoc phai dich chuyen, xoay lai ranh gioi")
    print("     va thu hep le de tach dung no. Khi do (4.5, 4) tro thanh mot")
    print("     vecto ho tro moi quyet dinh mep le phia lop 1.")
    print("=" * 74)


if __name__ == "__main__":
    main()
