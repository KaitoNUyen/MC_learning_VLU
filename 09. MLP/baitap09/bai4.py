# -*- coding: utf-8 -*-
"""Bài tập 4: Khảo sát ảnh hưởng của tốc độ học (learning_rate_init) trên bộ chữ số.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    cac_toc_do = [0.0001, 0.001, 0.01, 0.1]
    ket_qua = []

    print("=" * 78)
    print("BAI TAP 4: ANH HUONG CUA TOC DO HOC (learning_rate_init) TREN BO CHU SO")
    print("=" * 78)
    print("Mo hinh: StandardScaler() + MLPClassifier(hidden_layer_sizes=(100,),")
    print("         solver='adam', max_iter=1000, random_state=0)")
    print()
    print(
        f"{'learning_rate_init':>18s} {'So vong (n_iter_)':>18s} "
        f"{'Mat mat cuoi':>14s} {'Mat mat min':>13s} {'Test Acc':>11s}"
    )
    print("-" * 78)

    for lr in cac_toc_do:
        mo_hinh = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(100,),
                learning_rate_init=lr,
                max_iter=1000,
                random_state=0,
            ),
        )
        mo_hinh.fit(X_tr, y_tr)
        mlp = mo_hinh[-1]
        acc_kt = accuracy_score(y_te, mo_hinh.predict(X_te))
        mat_mat_cuoi = mlp.loss_curve_[-1]
        mat_mat_min = min(mlp.loss_curve_)

        ket_qua.append({
            "lr": lr,
            "n_iter": mlp.n_iter_,
            "loss_curve": mlp.loss_curve_,
            "loss_end": mat_mat_cuoi,
            "loss_min": mat_mat_min,
            "acc": acc_kt,
        })
        print(
            f"{lr:18g} {mlp.n_iter_:18d} {mat_mat_cuoi:14.6f} "
            f"{mat_mat_min:13.6f} {acc_kt:11.4f}"
        )
    print("-" * 78)

    # Vẽ bốn đường loss_curve_ lên cùng một hình và lưu thành toc_do_hoc.png
    mau_sac = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    kieu_net = ["-", "-", "--", "-."]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for kq, mau, net in zip(ket_qua, mau_sac, kieu_net):
        vong = range(1, len(kq["loss_curve"]) + 1)
        nhan = (
            f"lr = {kq['lr']:g} "
            f"(vong={kq['n_iter']}, loss={kq['loss_end']:.4f}, acc={kq['acc']:.4f})"
        )
        ax.plot(vong, kq["loss_curve"], color=mau, linestyle=net, linewidth=2.1, label=nhan)

    ax.set_title(
        "Bai 4: Duong mat mat (loss_curve_) theo toc do hoc tren bo chu so viet tay",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Vong hoc (epoch)", fontsize=11)
    ax.set_ylabel("Mat mat tren tap hoc (Cross-Entropy Loss)", fontsize=11)
    ax.set_xlim(0, max(kq["n_iter"] for kq in ket_qua) + 15)
    ax.set_ylim(-0.05, 2.90)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.2, framealpha=0.95)

    # Thêm khung phóng to 60 vòng đầu để nhìn rõ dao động của lr=0.1 và tốc độ rơi của lr=0.01
    ax_zoom = ax.inset_axes([0.50, 0.30, 0.46, 0.41])
    for kq, mau, net in zip(ket_qua, mau_sac, kieu_net):
        do_dai = min(60, len(kq["loss_curve"]))
        vong = range(1, do_dai + 1)
        ax_zoom.plot(vong, kq["loss_curve"][:do_dai], color=mau, linestyle=net, linewidth=1.8)
    ax_zoom.set_title("Phong to 60 vong dau (thay ro dao dong lr=0.1)", fontsize=8.8, fontweight="bold")
    ax_zoom.set_xlabel("Vong", fontsize=8)
    ax_zoom.set_ylabel("Loss", fontsize=8)
    ax_zoom.set_xlim(1, 60)
    ax_zoom.set_ylim(-0.02, 0.55)
    ax_zoom.tick_params(labelsize=8)
    ax_zoom.grid(True, linestyle=":", alpha=0.5)

    thu_muc_luu = os.path.dirname(os.path.abspath(__file__))
    duong_dan_anh = os.path.join(thu_muc_luu, "toc_do_hoc.png")
    fig.savefig(duong_dan_anh, dpi=150, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    print(f"Da luu bieu do 4 duong loss_curve_ tai: baitap09/toc_do_hoc.png")
    print()

    print("NHAN XET: TOC DO HOC QUA NHO VA QUA LON GAY RA CHUYEN GI?")
    print("-" * 78)
    print("1. Toc do hoc QUA NHO (lr = 0.0001):")
    print("   - Moi buoc cap nhat trong so rat nho nen mat mat giam cuc ky cham:")
    print("     mang phai chay toi 582 vong (gap ~4 lan lr=0.001 va ~15 lan lr=0.01)")
    print("     nhung mat mat cuoi van con 0.0189 (cao hon muc 0.0052 cua lr=0.001).")
    print("   - Ton nhieu thoi gian huan luyen; neu de max_iter mac dinh 200 thi mang")
    print("     se dung giua chung khi chua kip hoi tu (dinh ConvergenceWarning).")
    print("   - Tuy nhien, khi duoc cap du vong (max_iter=1000), buoc di nho va on dinh")
    print("     giup mang dat accuracy rat tot tren tap kiem tra (0.9889, sai chi 5/450).")
    print()
    print("2. Toc do hoc VUA PHAI (lr = 0.001 mac dinh va lr = 0.01):")
    print("   - lr = 0.001 hoi tu muot ma sau 144 vong (mat mat cuoi 0.0052, accuracy")
    print("     0.9844 - khop dung ket qua muc 6). Day la diem can bang tot nhat giua")
    print("     toc do va do on dinh.")
    print("   - lr = 0.01 hoi tu rat nhanh (38 vong, mat mat 0.0011), nhung do chinh xac")
    print("     tap kiem tra giam nhe xuong 0.9778.")
    print()
    print("3. Toc do hoc QUA LON (lr = 0.1):")
    print("   - Buoc nhay qua lon khien thuat toan NHAY VOT QUA day cuc tieu va DAO DONG")
    print("     that thuong: mat mat tung cham 0.0027 o vong 19 nhung ngay sau do bat")
    print("     nguoc len 0.0715, 0.1757 va ket thuc o 0.1204 (vong 30) khi dieu kien")
    print("     khong cai thien sau 10 vong kich hoat dung.")
    print("   - Hau qua: mang khong hoi tu ve nghiem tot, accuracy tap kiem tra roi manh")
    print("     xuong con 0.9511 (sai toi 22/450 anh, kem nhat trong ca 4 muc).")
    print("=" * 78)


if __name__ == "__main__":
    main()
