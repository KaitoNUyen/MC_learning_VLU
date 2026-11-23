# -*- coding: utf-8 -*-
"""Bài tập 4: Lưu ảnh nén 16 màu bằng K-Means (china.jpg và ảnh cá nhân PNG 4 kênh RGBA).

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def chuan_hoa_anh_rgb(anh):
    """Đưa ảnh về mảng float32 trong đoạn [0, 1] và bỏ kênh alpha (kênh thứ 4) nếu có."""
    bo_alpha = False
    if anh.ndim == 3 and anh.shape[2] == 4:
        anh = anh[:, :, :3]
        bo_alpha = True
    elif anh.ndim == 2:
        anh = np.stack([anh, anh, anh], axis=-1)

    anh_f = anh.astype(np.float64)
    if anh_f.max() > 1.0:
        anh_f = anh_f / 255.0
    anh_f = np.clip(anh_f, 0.0, 1.0)
    return anh_f, bo_alpha


def nen_anh_kmeans(anh_rgb_01, so_mau=16, so_diem_hoc=5000, seed=0):
    """Nén màu ảnh RGB [0, 1] về còn so_mau màu bằng K-Means (học trên mẫu ngẫu nhiên)."""
    cao, rong, _ = anh_rgb_01.shape
    diem = anh_rgb_01.reshape(-1, 3)

    # Đếm số màu gốc trên thang 0..255
    anh_u8 = np.round(anh_rgb_01 * 255).astype(np.uint8)
    so_mau_goc = len(np.unique(anh_u8.reshape(-1, 3), axis=0))

    # Học trên tối đa 5000 điểm ảnh rút ngẫu nhiên cho nhanh, rồi gán cho toàn bộ ảnh
    rng = np.random.default_rng(seed)
    n_hoc = min(so_diem_hoc, len(diem))
    mau_hoc = diem[rng.choice(len(diem), n_hoc, replace=False)]

    km = KMeans(n_clusters=so_mau, n_init=4, random_state=seed).fit(mau_hoc)
    nhan = km.predict(diem)
    diem_moi = np.clip(km.cluster_centers_[nhan], 0.0, 1.0)
    anh_moi = diem_moi.reshape(cao, rong, 3)

    sai_lech = float(np.abs(diem_moi - diem).mean() * 255.0)
    return anh_moi, so_mau_goc, sai_lech


def tao_anh_ca_nhan_rgba(duong_dan_png):
    """Tạo sẵn tệp PNG cá nhân 4 kênh (RGBA) nếu chưa có để kiểm thử đọc bằng plt.imread."""
    if os.path.exists(duong_dan_png):
        return
    # Dùng ảnh hoa thật có sẵn trong scikit-learn và thêm kênh thứ 4 (alpha = 255)
    # để mô phỏng đúng định dạng tệp PNG 4 kênh RGBA phổ biến ngoài thực tế
    anh_hoa = load_sample_image("flower.jpg")
    cao, rong, _ = anh_hoa.shape
    kenh_alpha = np.full((cao, rong, 1), 255, dtype=np.uint8)
    anh_rgba = np.concatenate([anh_hoa, kenh_alpha], axis=2)
    Image.fromarray(anh_rgba, mode="RGBA").save(duong_dan_png)


def main():
    thu_muc_luu = os.path.dirname(os.path.abspath(__file__))

    print("=" * 76)
    print("BAI TAP 4: LUU ANH DA NEN 16 MAU VA SO SANH DUNG LUONG TEP")
    print("=" * 76)

    # ------------------------------------------------------------------
    # PHẦN 1: Ảnh mẫu china.jpg từ scikit-learn (theo k5_nen_anh.py)
    # ------------------------------------------------------------------
    anh_china = load_sample_image("china.jpg")
    anh_china_01, _ = chuan_hoa_anh_rgb(anh_china)
    anh_china_16, mau_goc_china, sai_china = nen_anh_kmeans(anh_china_01, so_mau=16)

    duong_dan_goc = os.path.join(thu_muc_luu, "anh_goc.png")
    duong_dan_16 = os.path.join(thu_muc_luu, "anh_16_mau.png")
    plt.imsave(duong_dan_goc, anh_china)
    plt.imsave(duong_dan_16, anh_china_16)

    dl_goc = os.path.getsize(duong_dan_goc)
    dl_16 = os.path.getsize(duong_dan_16)
    ty_le_china = dl_goc / dl_16
    giam_pt_china = (1.0 - dl_16 / dl_goc) * 100.0

    print("1. ANH MAU SCIKIT-LEARN (china.jpg)")
    print("-" * 76)
    print(f"Kich thuoc mang anh goc : {anh_china.shape} | anh_moi: {anh_china_16.shape}")
    print(f"So diem anh             : {anh_china.shape[0] * anh_china.shape[1]:,} | So mau goc: {mau_goc_china:,}")
    print(f"Sai lech trung binh     : {sai_china:.2f} muc sang (tren thang 255)")
    print(f"Dung luong anh_goc.png  : {dl_goc:>10,} byte ({dl_goc / 1024:6.1f} KB)")
    print(f"Dung luong anh_16_mau   : {dl_16:>10,} byte ({dl_16 / 1024:6.1f} KB)")
    print(f"Ket qua so sanh         : nho hon {ty_le_china:.2f} lan (giam {giam_pt_china:.1f}% dung luong)")
    print()

    # ------------------------------------------------------------------
    # PHẦN 2: Ảnh cá nhân định dạng PNG có kênh thứ 4 (RGBA), đọc bằng plt.imread
    # ------------------------------------------------------------------
    duong_dan_cn_goc = os.path.join(thu_muc_luu, "anh_ca_nhan.png")
    duong_dan_cn_16 = os.path.join(thu_muc_luu, "anh_ca_nhan_16_mau.png")
    tao_anh_ca_nhan_rgba(duong_dan_cn_goc)

    anh_cn_doc = plt.imread(duong_dan_cn_goc)
    hinh_dang_truoc = anh_cn_doc.shape
    anh_cn_01, da_bo_alpha = chuan_hoa_anh_rgb(anh_cn_doc)
    anh_cn_16, mau_goc_cn, sai_cn = nen_anh_kmeans(anh_cn_01, so_mau=16)

    plt.imsave(duong_dan_cn_16, anh_cn_16)
    dl_cn_goc = os.path.getsize(duong_dan_cn_goc)
    dl_cn_16 = os.path.getsize(duong_dan_cn_16)
    ty_le_cn = dl_cn_goc / dl_cn_16
    giam_pt_cn = (1.0 - dl_cn_16 / dl_cn_goc) * 100.0

    print("2. ANH CA NHAN (DOC TU TEP PNG 4 KENH RGBA BANG plt.imread)")
    print("-" * 76)
    print(f"Tep doc vao             : baitap08/anh_ca_nhan.png")
    print(f"Kich thuoc khi doc vao  : {hinh_dang_truoc} (dtype = {anh_cn_doc.dtype})")
    print(f"Bo kenh alpha thu 4     : {'CO (cat [:, :, :3] -> ' + str(anh_cn_01.shape) + ')' if da_bo_alpha else 'Khong can'}")
    print(f"So diem anh             : {anh_cn_01.shape[0] * anh_cn_01.shape[1]:,} | So mau goc: {mau_goc_cn:,}")
    print(f"Sai lech trung binh     : {sai_cn:.2f} muc sang (tren thang 255)")
    print(f"Dung luong anh_ca_nhan  : {dl_cn_goc:>10,} byte ({dl_cn_goc / 1024:6.1f} KB)")
    print(f"Dung luong ban 16 mau   : {dl_cn_16:>10,} byte ({dl_cn_16 / 1024:6.1f} KB)")
    print(f"Ket qua so sanh         : nho hon {ty_le_cn:.2f} lan (giam {giam_pt_cn:.1f}% dung luong)")
    print()

    # Vẽ hình đối chiếu 2x2 (ảnh gốc vs ảnh 16 màu cho cả 2 bức ảnh)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    axes[0, 0].imshow(anh_china)
    axes[0, 0].set_title(
        f"china.jpg goc ({mau_goc_china:,} mau)\n{dl_goc / 1024:.1f} KB",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0, 1].imshow(anh_china_16)
    axes[0, 1].set_title(
        f"china.jpg nen 16 mau (sai lech {sai_china:.2f})\n{dl_16 / 1024:.1f} KB (nho hon {ty_le_china:.2f}x)",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1, 0].imshow(anh_cn_01)
    axes[1, 0].set_title(
        f"anh_ca_nhan.png goc ({mau_goc_cn:,} mau, RGBA->RGB)\n{dl_cn_goc / 1024:.1f} KB",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1, 1].imshow(anh_cn_16)
    axes[1, 1].set_title(
        f"anh_ca_nhan.png nen 16 mau (sai lech {sai_cn:.2f})\n{dl_cn_16 / 1024:.1f} KB (nho hon {ty_le_cn:.2f}x)",
        fontsize=10.5,
        fontweight="bold",
    )
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Bai 4: So sanh anh goc va anh nen 16 mau bang K-Means", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    duong_dan_so_sanh = os.path.join(thu_muc_luu, "bai4_so_sanh_nen_anh.png")
    fig.savefig(duong_dan_so_sanh, dpi=150)
    plt.close(fig)
    print(f"Da luu hinh doi chieu   : baitap08/bai4_so_sanh_nen_anh.png")
    print()

    print("3. NHAN XET VE DUNG LUONG TEP VA XU LY ANH PNG")
    print("-" * 76)
    print(f"- Voi china.jpg, tep PNG 16 mau giam tu {dl_goc/1024:.1f} KB xuong {dl_16/1024:.1f} KB")
    print(f"  (nho hon {ty_le_china:.2f} lan, giam {giam_pt_china:.1f}%), trong khi mat thuong van thay")
    print(f"  bo cuc va mau sac rat gan anh goc (sai lech trung binh chi {sai_china:.2f}/255).")
    print(f"- Voi anh_ca_nhan.png, tep 16 mau giam tu {dl_cn_goc/1024:.1f} KB xuong {dl_cn_16/1024:.1f} KB")
    print(f"  (nho hon {ty_le_cn:.2f} lan, giam {giam_pt_cn:.1f}%), sai lech trung binh {sai_cn:.2f}/255.")
    print("- Giai thich ty le nen thuc te vs ly thuyet (6.0 lan):")
    print("  + Theo ly thuyet luu tru tho (24 bit -> 4 bit bang mau), dung luong giam")
    print("    6.0 lan. Con ham plt.imsave van luu tep PNG dang RGB/RGBA 8-bit moi kenh,")
    print("    nhung nho anh 16 mau tao ra nhieu mang mau dong nhat lap lai, thuat toan")
    print("    nen khong mat du lieu DEFLATE cua PNG nen hieu qua hon han anh goc.")
    print("- Luu y ky thuat khi doc anh PNG bang plt.imread:")
    print("  + Anh PNG thuong tra ve mang float32 trong [0, 1] (khong can chia 255)")
    print("    va co the co 4 kenh (R, G, B, Alpha). Phai cat bo kenh thu 4 bang")
    print("    anh[:, :, :3] truoc khi dua vao K-Means de chi phan cum khong gian RGB.")
    print("=" * 76)


if __name__ == "__main__":
    main()
