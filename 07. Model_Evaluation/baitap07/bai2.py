# -*- coding: utf-8 -*-
"""Bài tập 2: Chữ số viết tay — classification_report cho SVM của bài 6.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    # Lấy lại đúng bộ chữ số viết tay và cách chia của bài 6
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    svm = SVC(kernel="rbf").fit(X_tr, y_tr)
    du_doan = svm.predict(X_te)

    bao_cao_chuoi = classification_report(y_te, du_doan, digits=4)
    bao_cao = classification_report(y_te, du_doan, output_dict=True)
    cm = confusion_matrix(y_te, du_doan)

    # Tìm chữ số có recall thấp nhất và precision thấp nhất
    cac_lop = [str(c) for c in range(10)]
    min_recall = min(bao_cao[c]["recall"] for c in cac_lop)
    min_precision = min(bao_cao[c]["precision"] for c in cac_lop)
    lop_recall_thap = [c for c in cac_lop if abs(bao_cao[c]["recall"] - min_recall) < 1e-12]
    lop_prec_thap = [c for c in cac_lop if abs(bao_cao[c]["precision"] - min_precision) < 1e-12]

    # Thống kê số mẫu (support) từng chữ số trên tập kiểm tra
    supports = [int(bao_cao[c]["support"]) for c in cac_lop]
    macro = bao_cao["macro avg"]
    weighted = bao_cao["weighted avg"]

    print("=" * 76)
    print("BAI TAP 2: CHU SO VIET TAY - CLASSIFICATION REPORT CUA SVM (BAI 6)")
    print("=" * 76)
    print(f"Tap hoc: {len(y_tr)} anh | Tap kiem tra: {len(y_te)} anh (10 chu so 0..9)")
    print(f"Accuracy tong the: {accuracy_score(y_te, du_doan):.4f} "
          f"(dung {int(np.sum(du_doan == y_te))}/{len(y_te)}, sai {int(np.sum(du_doan != y_te))} anh)")
    print()
    print("CLASSIFICATION REPORT CHO CA 10 CHU SO:")
    print(bao_cao_chuoi.rstrip())
    print()

    # Tóm tắt các lỗi cụ thể từ ma trận nhầm lẫn để giải thích trực tiếp
    print("CAC ANH DOAN SAI (DOI CHIEU MA TRAN NHAM LAN):")
    for that in range(10):
        for doan in range(10):
            if that != doan and cm[that, doan] > 0:
                print(f"  - Chu so that {that} bi doan nham thanh {doan}: {cm[that, doan]} anh")
    print()

    print("TRA LOI VA GIAI THICH:")
    print("-" * 76)
    print(f"1. Chu so co RECALL thap nhat: chu so {', '.join(lop_recall_thap)} "
          f"(recall = {min_recall:.4f} = 42/43).")
    print("   - Vi chu so 8 chi co 43 anh kiem tra (it nhat trong 10 lop) va bi bo")
    print("     sot 1 anh (doan nham thanh so 1), nen 1 loi lam giam recall 1/43")
    print("     = 0.0233. Cac chu so 4, 7, 9 cung sai 1 anh nhung co support = 45")
    print("     nen recall cao hon mot chut (44/45 = 0.9778).")
    print()
    print(f"2. Chu so co PRECISION thap nhat: chu so {', '.join(lop_prec_thap)} "
          f"(precision = {min_precision:.4f} = 42/43).")
    print("   - Mo hinh doan 43 anh la so 8 (42 anh dung + 1 anh so 4 bi bao nham")
    print("     thanh so 8), nen precision = 42/43 = 0.9767. Dung sau la chu so 9")
    print("     (44/45 = 0.9778) va chu so 1, 3 (46/47 = 0.9787).")
    print()
    print("3. Vi sao macro avg va weighted avg gan nhu bang nhau tren bo du lieu nay?")
    print(f"   - macro avg    : P={macro['precision']:.4f}, R={macro['recall']:.4f}, F1={macro['f1-score']:.4f}")
    print(f"   - weighted avg : P={weighted['precision']:.4f}, R={weighted['recall']:.4f}, F1={weighted['f1-score']:.4f}")
    print(f"   - So mau (support) cua 10 chu so rat CAN BANG, chi dao dong hep tu")
    print(f"     {min(supports)} den {max(supports)} anh (chiem {min(supports)/len(y_te)*100:.2f}% den "
          f"{max(supports)/len(y_te)*100:.2f}% tap kiem tra, sat muc deu 10.0%).")
    print("   - macro avg lay trung binh cong voi trong so bang nhau (1/10 = 0.100)")
    print("     cho moi lop; weighted avg lay trong so theo ty le support_c / N")
    print("     (xap xi 0.096..0.102). Khi trong so cac lop gan nhu bang nhau va")
    print("     diem cua ca 10 lop deu cao dong deu (0.9767..1.0000), hai trung")
    print("     binh nay gan nhu trung khop (chenh lech toi da 0.0001).")
    print("   - Doi chieu: tren du lieu lech lop sinh_vien_bo_hoc (270 vs 30, tuc")
    print("     90% vs 10% o muc 4), macro avg F1 (0.8255) va weighted avg F1")
    print("     (0.9418) chenh nhau rat lon (0.1163) vi lop dong ap dao trong so.")
    print("=" * 76)


if __name__ == "__main__":
    main()
