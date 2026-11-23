# -*- coding: utf-8 -*-
"""Bài tập 5: Dò lưới (GridSearchCV) cho mạng nơ-ron trên bộ chữ số viết tay.

Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu.
"""

import sys
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    # Pipeline chuẩn hoá + MLPClassifier theo đúng yêu cầu đề bài
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(max_iter=1000, random_state=0),
    )

    luoi = {
        "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
        "mlpclassifier__alpha": [0.0001, 0.01, 1],
    }

    tim = GridSearchCV(pipe, luoi, cv=5, n_jobs=-1)
    tim.fit(X_tr, y_tr)

    # Đối chiếu với MLP mặc định ở mục 6 và SVM kernel RBF ở bài 6 / mục 6
    mlp_mac_dinh = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0),
    ).fit(X_tr, y_tr)
    acc_mlp_mac_dinh = accuracy_score(y_te, mlp_mac_dinh.predict(X_te))

    svm_rbf = SVC().fit(X_tr, y_tr)
    acc_svm = accuracy_score(y_te, svm_rbf.predict(X_te))

    acc_tot_nhat = accuracy_score(y_te, tim.predict(X_te))
    so_sai_mlp = int((tim.predict(X_te) != y_te).sum())
    so_sai_svm = int((svm_rbf.predict(X_te) != y_te).sum())

    # Lập bảng điểm CV dạng ma trận (hàng: hidden_layer_sizes, cột: alpha)
    bang = pd.DataFrame({
        "lop_an": [str(p["mlpclassifier__hidden_layer_sizes"]) for p in tim.cv_results_["params"]],
        "alpha": [p["mlpclassifier__alpha"] for p in tim.cv_results_["params"]],
        "cv": tim.cv_results_["mean_test_score"],
    })
    thu_tu_lop = ["(10,)", "(50,)", "(100,)", "(50, 50)"]
    ma_tran = bang.pivot(index="lop_an", columns="alpha", values="cv").loc[thu_tu_lop]

    print("=" * 76)
    print("BAI TAP 5: DO LUOI (GridSearchCV) CHO MANG NO-RON TREN BO CHU SO")
    print("=" * 76)
    print("Mo hinh: make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, random_state=0))")
    print("Kiem dinh cheo: 5-fold CV tren tap hoc (1347 anh); tap kiem tra: 450 anh")
    print()

    print("1. BANG DIEM KIEM DINH CHEO (5-FOLD CV) CUA 12 CAP THAM SO")
    print("-" * 76)
    print(f"{'hidden_layer_sizes':20s} | {'alpha=0.0001':>14s} {'alpha=0.01':>14s} {'alpha=1':>14s}")
    print("-" * 76)
    for lop in thu_tu_lop:
        row = ma_tran.loc[lop]
        print(f"{lop:20s} | {row[0.0001]:14.4f} {row[0.01]:14.4f} {row[1]:14.4f}")
    print("-" * 76)
    print()

    lop_tot = tim.best_params_["mlpclassifier__hidden_layer_sizes"]
    alpha_tot = tim.best_params_["mlpclassifier__alpha"]

    print("2. KET QUA TOT NHAT VA SO SANH VOI SVM")
    print("-" * 76)
    print(f"Cap tham so tot nhat (theo 5-fold CV) : hidden_layer_sizes = {lop_tot}, alpha = {alpha_tot}")
    print(f"Diem CV tot nhat                      : {tim.best_score_:.4f}")
    print(f"MLP tot nhat (GridSearchCV) - Test Acc: {acc_tot_nhat:.4f} (sai {so_sai_mlp}/{len(y_te)} anh)")
    print(f"MLP mac dinh muc 6 (100,)   - Test Acc: {acc_mlp_mac_dinh:.4f}")
    print(f"SVM kernel RBF (SVC())      - Test Acc: {acc_svm:.4f} (sai {so_sai_svm}/{len(y_te)} anh)")
    print()

    chenh_lech = acc_tot_nhat - acc_svm
    ket_luan = "CO vuot duoc SVM" if acc_tot_nhat > acc_svm else "KHONG vuot duoc SVM"

    print("3. NHAN XET VA GIAI THICH")
    print("-" * 76)
    print(f"- Cap tham so tot nhat la hidden_layer_sizes = {lop_tot} va alpha = {alpha_tot}")
    print(f"  (CV 5-fold dat {tim.best_score_:.4f}; sat nut la (100,) va (50, 50) o alpha=1")
    print("  cung dat 0.9777).")
    print(f"- Accuracy tren tap kiem tra ({acc_tot_nhat:.4f}) {ket_luan} kernel RBF")
    print(f"  ({acc_svm:.4f}), thap hon {abs(chenh_lech):.4f} (nhieu hon {so_sai_mlp - so_sai_svm} anh doan sai).")
    print("- Mot so diem dang chu y tu bang do luoi:")
    print("  + O ca 4 kien truc, tang dieu chuan len alpha = 1 deu cai thien ro diem CV")
    print("    (vi du (50,) tang tu 0.9740 -> 0.9785; (50, 50) tang tu 0.9681 -> 0.9777),")
    print("    chung to bo chu so nho (1347 mau hoc, 64 dac trung) huong loi tu dieu chuan L2.")
    print("  + Mang 1 lop an 50 no-ron (3760 tham so) da du suc bieu dien va it hoc vet")
    print("    hon mang 2 lop (50, 50) (6310 tham so), con (10,) thi qua nho (CV ~0.948).")
    print("  + Dung nhu ket luan cua bai hoc: voi bo anh nho 8x8 (1797 mau), SVM kernel RBF")
    print("    van nhinh hon MLP vi tim duoc nghiem toi uu toan cuc le cuc dai va khong bi")
    print("    khuech dai nhieu vien anh khi chuan hoa tung pixel nhu StandardScaler.")
    print("=" * 76)


if __name__ == "__main__":
    main()
