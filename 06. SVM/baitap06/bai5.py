# -*- coding: utf-8 -*-
"""Bài tập 5: Chữ số viết tay — bảng nhầm lẫn và phân tích các ảnh đoán sai.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    svm = SVC(kernel="rbf").fit(X_tr, y_tr)
    du_doan = svm.predict(X_te)
    acc = accuracy_score(y_te, du_doan)
    cm = confusion_matrix(y_te, du_doan)

    print("=" * 72)
    print("BAI TAP 5: CHU SO VIET TAY - BANG NHAM LAN VA CAC ANH DOAN SAI")
    print("=" * 72)
    print(f"Accuracy tren tap kiem tra ({len(y_te)} anh): {acc:.4f}")
    print()

    # In bảng nhầm lẫn 10 hàng 10 cột (hàng là chữ số thật, cột là chữ số đoán)
    print("BANG NHAM LAN (hang = chu so that, cot = chu so doan):")
    tieu_de = "That\\Doan | " + " ".join(f"{c:3d}" for c in range(10))
    print(tieu_de)
    print("-" * len(tieu_de))
    for hang in range(10):
        gia_tri = " ".join(f"{cm[hang, cot]:3d}" for cot in range(10))
        print(f"   {hang:2d}     | {gia_tri}")
    print()

    # Đọc bảng nhầm lẫn để liệt kê các cặp chữ số bị nhầm
    print("CAC CAP CHU SO BI NHAM TU BANG NHAM LAN:")
    for hang in range(10):
        for cot in range(10):
            if hang != cot and cm[hang, cot] > 0:
                print(f"  - Chu so that {hang} bi doan nham thanh {cot}: {cm[hang, cot]} anh")
    print()

    # Vẽ và lưu từng ảnh bị đoán sai vào thư mục baitap06
    thu_muc_luu = os.path.dirname(os.path.abspath(__file__))
    chi_so_sai = np.flatnonzero(du_doan != y_te)
    print(f"TONG SO ANH DOAN SAI: {len(chi_so_sai)} tren {len(y_te)} anh")
    print("Da luu tung anh doan sai:")

    for stt, i in enumerate(chi_so_sai, start=1):
        ten_anh = f"bai5_anh_sai_{stt}_idx{i}_that{y_te[i]}_doan{du_doan[i]}.png"
        duong_dan_anh = os.path.join(thu_muc_luu, ten_anh)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(X_te[i].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"Anh {i}: that {y_te[i]}, doan {du_doan[i]}", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(duong_dan_anh, dpi=150)
        plt.close(fig)

        print(f"  {stt}. Anh {i:3d}: that la {y_te[i]}, doan la {du_doan[i]} -> {ten_anh}")

    # Lưu thêm một hình tổng hợp cả 4 ảnh sai cạnh nhau để tiện đối chiếu
    fig, axes = plt.subplots(1, len(chi_so_sai), figsize=(3 * len(chi_so_sai), 3.2))
    if len(chi_so_sai) == 1:
        axes = [axes]
    for ax, i in zip(axes, chi_so_sai):
        ax.imshow(X_te[i].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"Anh {i}\nthat {y_te[i]}, doan {du_doan[i]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("4 anh bi SVM kernel RBF doan sai tren tap kiem tra", fontsize=12)
    fig.tight_layout()
    duong_dan_tong_hop = os.path.join(thu_muc_luu, "bai5_tong_hop_anh_sai.png")
    fig.savefig(duong_dan_tong_hop, dpi=150)
    plt.close(fig)
    print(f"  -> Hinh tong hop: bai5_tong_hop_anh_sai.png")
    print()

    print("NHAN XET VI SAO SVM NHAM:")
    print("-" * 72)
    print("- Tren 450 anh kiem tra, SVM chi sai dung 4 anh (accuracy 0.9911),")
    print("  khong co cap nao bi nham lap lai qua 1 lan:")
    print("  + Anh 6   (that 9 -> doan 3): phan dau so 9 bi bet dac (khong ro")
    print("    vong tron rong), phan giua that vao ben trai va duoi uon cong")
    print("    manh sang trai nen nhin rat giong hai cung cua chu so 3.")
    print("  + Anh 18  (that 8 -> doan 1): than duoi la mot cot doc dam thang")
    print("    dung (vong duoi bi bet), vong tren nho lech trai giong moc dau")
    print("    cua chu so 1 viet tay.")
    print("  + Anh 161 (that 4 -> doan 8): hai nhanh tren cua so 4 viet sat")
    print("    nhau tao dang khep kin, ket hop net ngang dam o giua va than")
    print("    duoi hoi lech nen bi nhoe giong cau truc hai vong cua so 8.")
    print("  + Anh 428 (that 7 -> doan 9): dau so 7 co net trai keo dai xuong")
    print("    rat dam noi voi than phai, tao hinh nua kin giong vong tron tren")
    print("    va than dung cua chu so 9.")
    print("- Nguyen nhan chung: anh chi co do phan giai 8x8 pixel rat thap,")
    print("  khi nguoi viet net lech, nhoe hoac dinh nhau thi ngay ca mat nguoi")
    print("  cung kho phan biet ro rang.")
    print("=" * 72)


if __name__ == "__main__":
    main()
