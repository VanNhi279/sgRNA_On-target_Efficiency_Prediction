# Dự đoán hiệu quả chỉnh sửa gen của CRISPR-Cas9

Dự án này sử dụng Deep Learning để dự đoán hiệu quả (On-target efficiency) của sgRNA trong công nghệ chỉnh sửa gen CRISPR-Cas9, đi kèm với ứng dụng web minh họa sử dụng Streamlit.

## 📌 Các tính năng chính
- Huấn luyện mô hình Deep Learning (CNN/LSTM) trên dữ liệu chuỗi DNA.
- Giải thích mô hình bằng phương pháp XAI (Saliency Map) để xác định tầm quan trọng của từng nucleotide.
- Giao diện web tương tác để dự đoán nhanh hiệu quả chuỗi sgRNA.

## 🛠 Cài đặt

1. **Clone repository:**
   git clone https://github.com/VanNhi279/sgRNA_On-target_Efficiency_Prediction.git
   cd sgRNA_On-target_Efficiency_Prediction

2. **Cài đặt thư viện: Nên sử dụng môi trường ảo (venv hoặc conda):**
    pip install -r requirements.txt   

# 🚀 Hướng dẫn chạy Code

1. **Huấn luyện mô hình:**
    Mở file CRISPR-Cas9.ipynb bằng Jupyter Notebook hoặc Google Colab để thực hiện quá trình tiền xử lý dữ liệu và huấn luyện mô hình

2. **Chạy ứng dụng Web (Streamlit):**
    streamlit run app.py      

# 📁 Cấu trúc thư mục
    CRISPR-Cas9.ipynb: Notebook huấn luyện mô hình và phân tích XAI.

    app.py: Mã nguồn giao diện Streamlit.

    requirements.txt: Danh sách các thư viện cần cài đặt.

    best_model.keras: File mô hình đã huấn luyện (cần thiết để chạy app.py).