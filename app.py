import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import altair as alt  # Thư viện vẽ biểu đồ

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="CRISPR Gene Scanner (Sliding Window)",
    page_icon="🧬",
    layout="wide"
)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_prediction_model():
    try:
        # Giả lập model load
        return "Loaded" 
    except:
        return None

model = load_prediction_model() 

# --- 3. CÁC HÀM XỬ LÝ (ĐÃ SỬA LOGIC CẮT CHUỖI) ---

def scan_long_sequence(long_seq):
    """
    SỬA ĐỔI: Thuật toán Sliding Window (Cửa sổ trượt).
    Di chuyển từng bước 1 (stride=1) để cắt toàn bộ các đoạn 23bp có thể có.
    Không còn lọc theo PAM 'GG' nữa để đảm bảo lấy đủ số lượng như yêu cầu.
    """
    # Làm sạch chuỗi
    long_seq = long_seq.upper().replace("\n", "").replace(" ", "").strip()
    
    candidates = [] 
    positions = []  
    
    seq_len = len(long_seq)
    window_size = 23
    
    # Logic: Nếu chuỗi dài 30, window 23 -> chạy từ 0 đến 30-23 = 7 (tức là 8 đoạn: 0,1,2,3,4,5,6,7)
    limit = seq_len - window_size + 1
    
    if limit <= 0:
        return [], []

    # Duyệt qua từng index một
    for i in range(limit):
        # Cắt đoạn 23 ký tự
        sub_seq = long_seq[i : i + window_size]
        
        # Kiểm tra tính hợp lệ (chỉ chứa A,C,G,T,N)
        # Nếu bạn muốn chấp nhận mọi ký tự thì bỏ dòng if này đi
        if all(c in 'ACGTN' for c in sub_seq):
            candidates.append(sub_seq)
            positions.append(i)
                    
    return candidates, positions

# --- 4. GIAO DIỆN NGƯỜI DÙNG (UI) ---

st.title("🧬 CRISPR-Cas9 Sliding Window Scanner")
st.markdown("""
Công cụ quét **toàn bộ** các đoạn con 23bp theo cơ chế cửa sổ trượt (Sliding Window).
- Ví dụ: Chuỗi 30 ký tự sẽ sinh ra 8 đoạn con liên tiếp.
""")

# --- PHẦN NHẬP LIỆU ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1️⃣ Nhập chuỗi Gen đích")
    
    # Chuỗi mẫu dài (để test)
    sample_gene = "TTCCCTGGATTGGGTGGGGGCTGGGGAGGGAGAGTCGTTGCCGCCCATCAACAGAAACCCGACCGTAGCCCGGCGGGCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGCGGGGCTGGAGAGTGTTGGTCTGATAGTGACTTCATCTGGATCGCTTTAGACCTCTCGTTAAGTTCAACTGCAGCTCCCTGTATGTGATTTCATCGTGGCAGGTGCCTCAGAGCGAGAGGAGAGAGAGAGAGAGAGAGAGAGACAGACAGATACAGAGAGGAGACGGACAGACAGCGGACAGACAGCGAGAGAGACAGAGACAGCGAGACAGAGACAGAGCGACAGAGAC"
    
    # Text Area
    long_input = st.text_area(
        "Dán đoạn DNA dài vào đây:", 
        value=sample_gene, 
        height=150
    )
    
    # Hiển thị độ dài hiện tại để user dễ kiểm tra logic
    st.caption(f"Độ dài chuỗi hiện tại: **{len(long_input.replace(' ', '').strip())}** ký tự.")

    if st.button("🚀 Quét toàn bộ (Scan)", type="primary"):
        clean_input = long_input.replace("\n", "").replace(" ", "").strip()
        if len(clean_input) < 23:
            st.warning(f"⚠️ Chuỗi quá ngắn ({len(clean_input)} < 23)!")
        else:
            with st.spinner("Đang cắt chuỗi và dự đoán..."):
                # 1. Quét tìm ứng viên (Sliding Window)
                candidates, positions = scan_long_sequence(clean_input)
                
                if len(candidates) > 0:
                    # 2. Giả lập điểm số (Random demo)
                    # Lưu ý: Model thực tế có thể yêu cầu PAM ở cuối, nhưng ở đây ta chấm điểm tất cả
                    scores = np.random.uniform(0.1, 0.99, size=len(candidates))
                    
                    # 3. Tạo bảng kết quả
                    df_results = pd.DataFrame({
                        'Index': positions,
                        'Sequence': candidates,
                        'Score': scores
                    })
                    
                    # Phân loại
                    def get_rank(s):
                        if s > 0.85: return "🌟 Excellent"
                        elif s > 0.7: return "✅ Good"
                        elif s > 0.5: return "⚠️ Average"
                        else: return "❌ Poor"
                    
                    df_results['Rank'] = df_results['Score'].apply(get_rank)
                    
                    # Lưu vào session
                    st.session_state.results = df_results
                    st.success(f"✅ Đã cắt thành công {len(candidates)} đoạn (từ vị trí {positions[0]} đến {positions[-1]}).")
                    
                else:
                    st.error("❌ Không tách được chuỗi nào hợp lệ.")

# --- PHẦN HIỂN THỊ KẾT QUẢ ---
if 'results' in st.session_state:
    df = st.session_state.results
    
    st.markdown("---")
    
    # Layout: Biểu đồ bên trên (cho rộng), Bảng bên dưới (hoặc chia cột tùy ý)
    # Ở đây tôi chia cột như cũ nhưng tập trung vào biểu đồ
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col2:
        st.subheader("📊 Biểu đồ toàn bộ các chuỗi")
        st.info(f"Biểu đồ hiển thị điểm số của {len(df)} đoạn cắt liên tiếp.")
        
        # --- TẠO BIỂU ĐỒ TẬP TRUNG ---
        # Tooltip rất quan trọng để hover vào thấy ngay sequence
        
        # 1. Đường Line nối các điểm (thể hiện sự biến thiên liên tục của Sliding Window)
        line = alt.Chart(df).mark_line(
            color='#2980b9', 
            opacity=0.5,
            strokeWidth=2
        ).encode(
            x=alt.X('Index', title='Vị trí bắt đầu (Index)'),
            y=alt.Y('Score', title='Điểm dự đoán', scale=alt.Scale(domain=[0, 1]))
        )
        
        # 2. Các điểm tròn (Scatter) để hover
        points = alt.Chart(df).mark_circle(size=80).encode(
            x='Index',
            y='Score',
            color=alt.Color('Score', scale=alt.Scale(scheme='turbo'), title="Mức độ"),
            tooltip=[
                alt.Tooltip('Index', title='Vị trí'),
                alt.Tooltip('Sequence', title='Chuỗi (23bp)'),
                alt.Tooltip('Score', format='.4f', title='Điểm số'),
                alt.Tooltip('Rank', title='Xếp hạng')
            ]
        ).interactive() # Cho phép zoom/pan

        # 3. Đường tham chiếu (ngưỡng 0.8)
        rule = alt.Chart(pd.DataFrame({'y': [0.8]})).mark_rule(color='red', strokeDash=[4, 4]).encode(y='y')

        chart_combined = (line + points + rule).properties(
            height=500,
            title="Biến thiên điểm số trên toàn bộ chuỗi Gen"
        )
        
        st.altair_chart(chart_combined, use_container_width=True)

    with res_col1:
        st.subheader("📋 Danh sách chi tiết")
        
        # Thêm filter nhỏ để xem nhanh
        filter_top = st.checkbox("Chỉ hiện điểm cao (>0.8)")
        
        if filter_top:
            df_display = df[df['Score'] > 0.8].sort_values(by='Score', ascending=False)
        else:
            df_display = df # Mặc định hiển thị theo Index tăng dần (Sliding window)

        st.dataframe(
            df_display,
            column_config={
                "Index": st.column_config.NumberColumn("Index", format="%d"),
                "Sequence": st.column_config.TextColumn("Sequence", width="medium"),
                "Score": st.column_config.ProgressColumn(
                    "Score", format="%.4f", min_value=0, max_value=1
                ),
            },
            hide_index=True,
            use_container_width=False,
            height=500
        )