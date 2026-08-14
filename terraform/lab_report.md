# Báo cáo ngắn Lab 16

1. Hạ tầng AWS được triển khai thành công bằng Terraform tại `us-east-1` và trạng thái cuối là `No changes`.
2. Do AWS Free Plan không cho phép `t3.medium`, Compute Node dùng `c7i-flex.large`, vẫn cung cấp đúng 2 vCPU và 4 GiB RAM.
3. Bộ dữ liệu Credit Card Fraud Detection gồm 284.807 giao dịch và 30 đặc trưng; thời gian tải là 0,988 giây.
4. LightGBM hoàn tất huấn luyện trong 1,544 giây và early stopping chọn iteration 1.
5. Mô hình đạt AUC-ROC 0,9471, Accuracy 0,9990 và F1-score 0,7500 trên tập test độc lập.
6. Precision đạt 0,6667 và Recall đạt 0,8571, cho thấy mô hình ưu tiên phát hiện phần lớn giao dịch gian lận.
7. Inference một dòng mất khoảng 0,656 ms; throughput batch 1.000 dòng đạt khoảng 1,23 triệu dòng/giây.
8. Kết quả cho thấy LightGBM phù hợp với CPU nhỏ: huấn luyện nhanh, dùng ít RAM và không cần GPU.
