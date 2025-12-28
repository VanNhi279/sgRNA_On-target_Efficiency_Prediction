import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import altair as alt  # Thư viện vẽ biểu đồ

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="CRISPR Gene Scanner (XAI)",
    page_icon="🧬",
    layout="wide"
)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_prediction_model():
    try:
        # Lưu ý: Bạn cần thay đường dẫn 'best_model.keras' bằng file thật của bạn
        # Nếu chưa có file, XAI sẽ báo lỗi vì không có GradientTape để tính toán
        model = tf.keras.models.load_model('best_model.keras')
        return model
    except:
        return None

model = load_prediction_model() 

# --- 3. CÁC HÀM XỬ LÝ (GIỮ NGUYÊN VÀ THÊM XAI) ---

def one_hot_encode_single(seq):
    """Mã hóa One-hot cho 1 chuỗi để đưa vào XAI"""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    return np.array([mapping.get(base, [0,0,0,0]) for base in seq])

def get_saliency_map(model, seq):
    """Tính toán Saliency Map (XAI)"""
    x = one_hot_encode_single(seq)
    x = tf.convert_to_tensor(x[np.newaxis, ...], dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
    
    grads = tape.gradient(prediction, x)
    # Lấy giá trị tuyệt đối và tổng hợp theo chiều đặc trưng (One-hot)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
    # Chuẩn hóa về 0-1
    if saliency.max() != saliency.min():
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency

def scan_long_sequence(long_seq):
    """Giữ nguyên hàm gốc của bạn"""
    long_seq = long_seq.upper().replace("\n", "").replace(" ", "").strip()
    candidates = [] 
    positions = []  
    seq_len = len(long_seq)
    window_size = 23
    limit = seq_len - window_size + 1
    if limit <= 0: return [], []
    for i in range(limit):
        sub_seq = long_seq[i : i + window_size]
        if all(c in 'ACGTN' for c in sub_seq):
            candidates.append(sub_seq)
            positions.append(i)
    return candidates, positions

# --- 4. GIAO DIỆN NGƯỜI DÙNG (UI) ---

st.title("🧬 CRISPR-Cas9 XAI Scanner")
st.markdown("""
Công cụ quét chuỗi DNA và giải thích dự án bằng **XAI (Saliency Maps)**.
""")

# Kiểm tra model
if model is None:
    st.error("⚠️ Không tìm thấy file 'best_model.keras'. Vui lòng kiểm tra lại đường dẫn model để chạy XAI.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1️⃣ Nhập chuỗi Gen đích")
    sample_gene = "TTCCCTGGATTGGGTGGGGGCTGGGGAGGGAGAGTCGTTGCCGCCCATCAACAGAAACCCGACCGTAGCCCGGCGGGCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGCGGGGCTGGAGAGTGTTGGTCTGATAGTGACTTCATCTGGATCGCTTTAGACCTCTCGTTAAGTTCAACTGCAGCTCCCTGTATGTGATTTCATCGTGGCAGGTGCCTCAGAGCGAGAGGAGAGAGAGAGAGAGAGAGAGAGAGACAGACAGATACAGAGAGGAGACGGACAGACAGCGGACAGACAGCGAGAGAGACAGAGACAGCGAGACAGAGACAGAGCGACAGAGAC"
    long_input = st.text_area("Dán đoạn DNA dài vào đây:", value=sample_gene, height=150)
    
    if st.button("🚀 Quét và Phân tích (Scan)", type="primary"):
        clean_input = long_input.replace("\n", "").replace(" ", "").strip()
        if len(clean_input) < 23:
            st.warning("⚠️ Chuỗi quá ngắn!")
        else:
            with st.spinner("Đang xử lý..."):
                candidates, positions = scan_long_sequence(clean_input)
                if len(candidates) > 0:
                    # Mã hóa toàn bộ để dự đoán
                    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
                    X_input = np.array([[mapping.get(b, [0,0,0,0]) for b in s] for s in candidates])
                    
                    if model:
                        scores = model.predict(X_input, verbose=0).flatten()
                    else:
                        scores = np.random.uniform(0.1, 0.9, size=len(candidates))

                    df_results = pd.DataFrame({'Index': positions, 'Sequence': candidates, 'Score': scores})
                    
                    def get_rank(s):
                        if s > 0.8: return "🌟 Excellent"
                        elif s > 0.6: return "✅ Good"
                        else: return "❌ Poor"
                    
                    df_results['Rank'] = df_results['Score'].apply(get_rank)
                    st.session_state.results = df_results
                else:
                    st.error("❌ Không tách được chuỗi.")

# --- HIỂN THỊ KẾT QUẢ VÀ XAI ---
if 'results' in st.session_state:
    df = st.session_state.results
    
    # 1. Biểu đồ tổng quan (Giữ nguyên)
    st.markdown("---")
    chart_line = alt.Chart(df).mark_line(color='#2980b9').encode(x='Index', y='Score')
    chart_pts = alt.Chart(df).mark_circle().encode(x='Index', y='Score', color='Score', tooltip=['Index', 'Sequence', 'Score'])
    st.altair_chart((chart_line + chart_pts).properties(height=300), use_container_width=True)

    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.subheader("📋 Danh sách đoạn cắt")
        st.dataframe(df, height=400)

    with res_col2:
        st.subheader("🔍 Giải thích mô hình (XAI)")
        if model:
            # Cho phép người dùng chọn 1 chuỗi để giải thích
            target_idx = st.selectbox("Chọn vị trí Index để xem giải thích:", options=df['Index'].tolist())
            
            # Lấy chuỗi tương ứng
            selected_seq = df[df['Index'] == target_idx]['Sequence'].values[0]
            selected_score = df[df['Index'] == target_idx]['Score'].values[0]
            
            # Tính Saliency
            saliency_scores = get_saliency_map(model, selected_seq)
            
            # Tạo DF cho biểu đồ XAI
            df_xai = pd.DataFrame({
                'Position': list(range(1, 24)),
                'Nucleotide': list(selected_seq),
                'Importance': saliency_scores
            })

            # Vẽ biểu đồ Bar Chart XAI
            xai_chart = alt.Chart(df_xai).mark_bar().encode(
                x=alt.X('Position:O', title='Vị trí trên chuỗi'),
                y=alt.Y('Importance:Q', title='Độ quan trọng (Saliency)'),
                color=alt.condition(
                    alt.datum.Importance > 0.5,
                    alt.value('red'), alt.value('steelblue')
                ),
                tooltip=['Position', 'Nucleotide', 'Importance']
            ).properties(title=f"Phân tích chuỗi tại Index {target_idx} (Score: {selected_score:.4f})")
            
            st.altair_chart(xai_chart, use_container_width=True)
            st.info("💡 **Gợi ý:** Các cột màu đỏ là những vị trí Nucleotide ảnh hưởng mạnh nhất đến quyết định của mô hình.")
        else:
            st.warning("Vui lòng tải model thật để sử dụng tính năng XAI.")