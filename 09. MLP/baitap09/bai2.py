# -*- coding: utf-8 -*-
"""Bài tập 2: Kiểm chứng và giải thích một mạng XOR khác bằng numpy.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import sys
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def relu(z):
    """Hàm kích hoạt ReLU: max(0, z)."""
    return np.maximum(0.0, z)


def main():
    # Mạng cần kiểm chứng:
    #   h1 = max(0, x1 - x2)
    #   h2 = max(0, x2 - x1)
    #   y  = h1 + h2
    # Viết dưới dạng ma trận trọng số và hệ số chặn:
    W1 = np.array([
        [ 1.0, -1.0],   # Trọng số của x1 vào h1 (+1) và h2 (-1)
        [-1.0,  1.0],   # Trọng số của x2 vào h1 (-1) và h2 (+1)
    ])
    b1 = np.array([0.0, 0.0])   # Không cần hệ số chặn ở lớp ẩn
    W2 = np.array([1.0, 1.0])   # Cộng hai nơ-ron ẩn
    b2 = 0.0                    # Không cần hệ số chặn ở lớp ra

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    xor_chuan = np.array([0.0, 1.0, 1.0, 0.0])

    print("=" * 76)
    print("BAI TAP 2: KIEM CHUNG VA GIAI THICH MOT MANG XOR KHAC")
    print("=" * 76)
    print("Cong thuc mang:")
    print("  h1 = max(0, x1 - x2)")
    print("  h2 = max(0, x2 - x1)")
    print("  y  = h1 + h2")
    print()
    print("Dang ma tran trong so (khong can he so chan, b1 = [0, 0], b2 = 0):")
    print("  W1 = [[ 1, -1],   (cot 1 tinh x1 - x2, cot 2 tinh x2 - x1)")
    print("        [-1,  1]]")
    print("  W2 = [1, 1],  b1 = [0, 0],  b2 = 0")
    print()

    # 1. Tính từng dòng một để quan sát rõ hoạt động của h1 và h2
    print("1. BANG KIEM CHUNG TUNG DAU VAO BANG NUMPY")
    print("-" * 76)
    print(f"{'x1':>4s} {'x2':>4s} | {'x1-x2':>6s} {'x2-x1':>6s} | {'h1':>4s} {'h2':>4s} | {'y=h1+h2':>7s} {'XOR':>5s} {'Khop?':>6s}")
    print("-" * 76)
    for x, y_chuan in zip(X, xor_chuan):
        z1 = x @ W1 + b1
        h = relu(z1)
        y = h @ W2 + b2
        khop = "DUNG" if np.isclose(y, y_chuan) else "SAI"
        print(
            f"{x[0]:4.0f} {x[1]:4.0f} | {z1[0]:+6.0f} {z1[1]:+6.0f} | "
            f"{h[0]:4.0f} {h[1]:4.0f} | {y:7.0f} {y_chuan:5.0f} {khop:>6s}"
        )
    print("-" * 76)

    # 2. Tính cả 4 đầu vào cùng lúc bằng phép nhân ma trận
    H = relu(X @ W1 + b1)
    Y = H @ W2 + b2
    print(f"Tinh ca bang mot lan (Y = relu(X @ W1 + b1) @ W2 + b2): {Y}")
    print(f"Ket qua XOR chuan                                     : {xor_chuan}")
    print(f"Kiem chung np.array_equal(Y, xor_chuan)               : {np.array_equal(Y, xor_chuan)}")
    print()

    # 3. Giải thích chi tiết bằng lời (từ đại số, vai trò từng nơ-ron đến hình học)
    print("2. GIAI THICH BANG LOI VI SAO MANG NAY TINH DUNG XOR")
    print("-" * 76)
    print("a) Vai tro chuyen biet cua tung no-ron an:")
    print("   - No-ron h1 = max(0, x1 - x2) chi BAT (= 1) khi x1 > x2, tuc la nhan dien")
    print("     duy nhat truong hop (x1, x2) = (1, 0). O 3 truong hop con lai,")
    print("     x1 - x2 <= 0 nen ham ReLU ep h1 ve 0.")
    print("   - No-ron h2 = max(0, x2 - x1) chi BAT (= 1) khi x2 > x1, tuc la nhan dien")
    print("     duy nhat truong hop (x1, x2) = (0, 1). O 3 truong hop con lai,")
    print("     x2 - x1 <= 0 nen ham ReLU ep h2 ve 0.")
    print("   - Khi hai dau vao GIONG NHAU, tuc (0, 0) hoac (1, 1), ca x1 - x2 va")
    print("     x2 - x1 deu bang 0 nen ca hai no-ron deu TAT (h1 = 0, h2 = 0) -> y = 0.")
    print("   - Khi hai dau vao KHAC NHAU, tuc (0, 1) hoac (1, 0), co DUNG MOT trong hai")
    print("     no-ron bat len bang 1, no-ron con lai bang 0 -> tong y = h1 + h2 = 1.")
    print()
    print("b) Ban chat dai so (tai sao cong thuc rat gon dep):")
    print("   - Voi moi so thuc a, ta luon co: max(0, a) + max(0, -a) = |a|.")
    print("   - Thay a = x1 - x2, dau ra cua mang chinh la:")
    print("       y = max(0, x1 - x2) + max(0, -(x1 - x2)) = |x1 - x2|.")
    print("   - Tren hai gia tri nhi phan {0, 1}, |x1 - x2| bang 0 khi x1 = x2 va bang 1")
    print("     khi x1 != x2 - day chinh la dinh nghia toan hoc cua phep XOR.")
    print()
    print("c) Goc nhin hinh hoc & so sanh voi mang XOR o muc 2:")
    print("   - Trong khong gian an (h1, h2), lop an da 'gap' mat phang doc theo duong")
    print("     cheo x1 = x2: ca hai diem lop 0 la (0, 0) va (1, 1) deu bi gom ve cung")
    print("     goc toa do (0, 0), con hai diem lop 1 la (1, 0) va (0, 1) duoc giu o")
    print("     (1, 0) va (0, 1). Luc nay mot duong thang (h1 + h2 = 0.5) tach gon hai lop.")
    print("   - Khac voi mang o muc 2 (h1 = max(0, x1+x2), h2 = max(0, x1+x2-1),")
    print("     y = h1 - 2*h2) phai dung he so chan b1 = [0, -1] va trong so am -2 de")
    print("     keo (1, 1) ve 0, mang nay KHONG CAN he so chan nao (b1 = [0, 0], b2 = 0).")
    print("   - Dieu nay cung cho thay mot bai toan co the co NHIEU BO TRONG SO KHAC NHAU")
    print("     cung giai dung trong mang no-ron.")
    print("=" * 76)


if __name__ == "__main__":
    main()
