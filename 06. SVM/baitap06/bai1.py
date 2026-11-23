# -*- coding: utf-8 -*-
"""Bài tập 1: Tính tay khoảng cách tới ranh giới.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import sys
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def tinh_f_tay(w1, w2, b, x1, x2):
    """Tự tính giá trị f(x) = w1*x1 + w2*x2 + b bằng phép nhân và cộng."""
    return w1 * x1 + w2 * x2 + b


def tinh_khoang_cach(f_val, do_dai_w):
    """Tính khoảng cách từ điểm tới đường ranh giới: |f(x)| / ||w||."""
    return abs(f_val) / do_dai_w


def main():
    # Tạo lại đúng 24 điểm tách được ở mục 2
    X, y = make_blobs(
        n_samples=24,
        centers=[[2, 2], [6, 6]],
        cluster_std=0.9,
        random_state=0,
    )
    y = 1 - y  # Lớp 1 là cụm dưới bên trái

    # Huấn luyện SVM tuyến tính với C rất lớn (lề cứng)
    svm = SVC(kernel="linear", C=1000)
    svm.fit(X, y)

    w1, w2 = svm.coef_[0]
    b = svm.intercept_[0]
    do_dai_w = np.sqrt(w1 ** 2 + w2 ** 2)
    do_rong_le = 2.0 / do_dai_w
    nua_do_rong_le = do_rong_le / 2.0

    print("=" * 70)
    print("BAI TAP 1: TINH TAY KHOANG CACH TOI RANH GIOI")
    print("=" * 70)
    print(f"He so tu muc 2 : w1 = {w1:.6f}, w2 = {w2:.6f}, b = {b:.6f}")
    print(f"  (lam tron 3 so: w1 = {w1:.3f}, w2 = {w2:.3f}, b = {b:.3f})")
    print(f"Do dai ||w||   = sqrt(w1^2 + w2^2) = {do_dai_w:.4f}")
    print(f"Do rong le     = 2 / ||w||         = {do_rong_le:.4f}")
    print(f"Nua do rong le = 1 / ||w||         = {nua_do_rong_le:.4f}")
    print()

    # 1. Tự tính f(x) và khoảng cách cho hai điểm mới (3, 3) và (5, 4.5)
    print("1. TINH TAY f(x) VA KHOANG CACH CHO HAI DIEM MOI")
    print("-" * 70)
    cac_diem_moi = [(3.0, 3.0), (5.0, 4.5)]
    for x1, x2 in cac_diem_moi:
        f_val = tinh_f_tay(w1, w2, b, x1, x2)
        kc = tinh_khoang_cach(f_val, do_dai_w)
        lop = 1 if f_val > 0 else 0
        vi_tri = "ngoai le" if abs(f_val) > 1.0 else "ben trong le"
        print(f"Diem ({x1}, {x2}):")
        print(f"  f(x) = ({w1:.3f})*({x1}) + ({w2:.3f})*({x2}) + {b:.3f} = {f_val:+.3f}")
        print(f"  Khoang cach = |{f_val:+.3f}| / {do_dai_w:.4f} = {kc:.4f}")
        print(f"  Du doan lop {lop} ({vi_tri}, so voi nua le = {nua_do_rong_le:.4f})")
    print()

    # 2. Tự tính f(x) và khoảng cách cho ba vectơ hỗ trợ
    print("2. TINH TAY f(x) VA KHOANG CACH CUA BA VECTO HO TRO")
    print("-" * 70)
    for idx in svm.support_:
        x1, x2 = X[idx, 0], X[idx, 1]
        f_val = tinh_f_tay(w1, w2, b, x1, x2)
        kc = tinh_khoang_cach(f_val, do_dai_w)
        print(
            f"Diem {idx:2d} ({x1:.2f}, {x2:.2f}), lop {y[idx]}: "
            f"f(x) = {f_val:+.4f}, khoang cach = {kc:.4f}"
        )
    print(f"So sanh: nua do rong le = {do_rong_le:.4f} / 2 = {nua_do_rong_le:.4f}")
    print("  (Sai lech ~0.0001 la sai so hoi tu so hoc ben trong thu vien LIBSVM)")
    print()

    # 3. Giải thích vì sao khoảng cách của 3 vectơ hỗ trợ bằng đúng một nửa độ rộng lề
    print("3. GIAI THICH")
    print("-" * 70)
    print("- Duong ranh gioi quyet dinh la tap cac diem thoa man f(x) = 0.")
    print("- Hai mep le la hai duong song song nam doi xung hai ben duong")
    print("  ranh gioi: f(x) = +1 (phia lop 1) va f(x) = -1 (phia lop 0).")
    print("- Trong truong hop hai lop tach han nhau (le cung, C lon), moi")
    print("  vecto ho tro deu nam DUNG TREN mot trong hai mep le, tuc la")
    print("  chung luon thoa man |f(x)| = 1.")
    print("- Do do, khoang cach tu moi vecto ho tro toi duong ranh gioi la:")
    print("    d(x) = |f(x)| / sqrt(w1^2 + w2^2) = 1 / sqrt(w1^2 + w2^2).")
    print("- Trong khi do, do rong le la khoang cach giua hai mep le f(x) = +1")
    print("  va f(x) = -1, bang: le = 2 / sqrt(w1^2 + w2^2).")
    print("- Vi duong ranh gioi f(x) = 0 nam chinh giua hai mep le, nen")
    print("  khoang cach cua ca ba vecto ho tro bang dung 1/2 do rong le")
    print(f"  (cung bang {nua_do_rong_le:.4f}).")
    print("=" * 70)


if __name__ == "__main__":
    main()
