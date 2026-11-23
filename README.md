# 🎓 MC_learning_VLU

> Bài tập thực hành môn **Học máy ứng dụng (Machine Learning)** — Trường Đại học Văn Lang (VLU)

## 📋 Mô tả

Repository này chứa các bài tập thực hành Machine Learning, được tổ chức theo từng chủ đề/thuật toán. Mỗi thư mục là một buổi học với nhiều bài tập từ cơ bản đến nâng cao, sử dụng Python (`numpy`, `pandas`, `scikit-learn`, `matplotlib`).

## 📁 Cấu trúc thư mục

| STT | Chủ đề                  | Thư mục bài tập                                                            | Nội dung                                                                                                                                                                                            |
| --- | ----------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | **Linear Regression**   | `01. Linear_Regression/`                                                   | Hồi quy tuyến tính — phân tích dữ liệu giá nhà, gradient descent, learning rate                                                                                                                     |
| 02  | **Logistic Regression** | `02. Logistic_Regression/`                                                 | Hồi quy logistic — phân loại nhị phân, hàm sigmoid, dự đoán qua môn                                                                                                                                 |
| 03  | **Naive Bayes**         | `03. Naive_Bayes/`                                                         | Phân loại Bayes đơn giản — phân loại văn bản, dự đoán spam, email                                                                                                                                   |
| 04  | **KNN**                 | [04. KNN/baitap04](04.%20KNN/baitap04/README.md)                           | K láng giềng gần nhất — khoảng cách Euclid & Manhattan, chuẩn hoá `StandardScaler`, chọn $k$ tối ưu, KNN phân loại & hồi quy                                                                        |
| 06  | **SVM**                 | [06. SVM/baitap06](06.%20SVM/baitap06/README.md)                           | Máy vectơ hỗ trợ — lề rộng nhất, vectơ hỗ trợ, lề mềm & tham số $C$, kernel đa thức & RBF, dò lưới `GridSearchCV`, nhận dạng chữ số viết tay                                                        |
| 07  | **Model Evaluation**    | [07. Model_Evaluation/baitap07](07.%20Model_Evaluation/baitap07/README.md) | Đánh giá mô hình — ma trận nhầm lẫn, Precision/Recall/F1, chọn ngưỡng bằng CV, đường cong ROC & AUC, `class_weight="balanced"`, MAE/RMSE/$R^2$ với điểm ngoại lai                                   |
| 08  | **K-Means**             | [08. KMeans/baitap08](08.%20KMeans/baitap08/README.md)                     | Phân cụm không giám sát — tự tính Silhouette, chọn số cụm $K$, khởi tạo `random` vs `k-means++`, nén ảnh 16 màu, so sánh DBSCAN, phân cụm chữ số viết tay                                           |
| 09  | **MLP**                 | [09. MLP/baitap09](09.%20MLP/baitap09/README.md)                           | Mạng nơ-ron nhiều lớp — lan truyền xuôi thủ công, bài toán XOR, so sánh hàm kích hoạt (`relu`/`tanh`/`logistic`), tốc độ học, `GridSearchCV` trên `load_digits`, hồi quy & ngoại suy `MLPRegressor` |
