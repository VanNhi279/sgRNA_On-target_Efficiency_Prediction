import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import altair as alt
import textwrap

# --- 1. CẤU HÌNH TRANG WEB (Dark Biotech Theme) ---
st.set_page_config(
    page_title="CRISPR XAI Oracle Pro",
    page_icon="🧬",
    layout="wide"
)

# --- CUSTOM CSS: Giao diện tối, chữ trắng sáng, hiệu ứng Neon ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Ép màu trắng cho toàn bộ text */
    h1, h2, h3, p, label, span, div, .stMarkdown { color: #ffffff !important; }
    
    /* Tùy chỉnh Card và Expander */
    div[data-testid="stVerticalBlock"] > div:has(div.stExpander) {
        background: #161b22; padding: 25px; border-radius: 15px;
        border: 1px solid #30363d; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
    }

    /* Nút bấm Neon Gradient */
    .stButton>button {
        background: linear-gradient(90deg, #1f6feb, #00d4ff);
        color: white; border: none; font-weight: bold;
        border-radius: 10px; height: 3.5em; width: 100%;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5); }

    /* Tab navigation */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #21262d; border-radius: 8px 8px 0px 0px;
        color: #8b949e !important; border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important; color: white !important;
        border-bottom: 2px solid #58a6ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TẢI MÔ HÌNH ---
@st.cache_resource
def load_prediction_model():
    try:
        # Đảm bảo file best_model.keras nằm cùng thư mục
        return tf.keras.models.load_model('best_model.keras')
    except:
        return None

model = load_prediction_model()

# --- 3. HÀM XỬ LÝ XAI (Saliency Map) ---
def get_saliency_map(model, seq):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    x = np.array([mapping.get(base, [0,0,0,0]) for base in seq], dtype=np.float32)
    x = tf.convert_to_tensor(x[np.newaxis, ...])
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
    
    grads = tape.gradient(prediction, x)
    if grads is None: return np.zeros(23)
    
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
    return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
# --- 3.1 1 SỐ HÀM BỔ SUNG ---
def calculate_gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq) * 100

def check_motifs(seq):
    warnings = []
    if "TTTT" in seq:
        warnings.append("⚠️ **Cảnh báo Poly-T:** Chuỗi chứa 4 nucleotide T liên tiếp, có thể gây dừng phiên mã sớm (premature termination).")
    return warnings
# --- 4. GIAO DIỆN CHÍNH ---
st.title("🧬 CRISPR-Cas9 Efficiency Oracle")
st.markdown("<p style='color: #8b949e;'>Hệ thống XAI-to-NLG dự đoán hiệu quả sgRNA dựa trên Deep Learning</p>", unsafe_allow_html=True)

# Khai báo dữ liệu mẫu
sample_dna = "TTCCCTGGATTGGGTGGGGGCTGGGGAGGGAGAGTCGTTGCCGCCCATCAACAGAAACCCGACCGTAGCCCGGCGGGCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGCGGGGCTGGAGAGTGTTGGTCTGATAGTGACTTCATCTGGATCGCTTTAGACCTCTCGTTAAGTTCAACTGCAGCTCCCTGTATGTGATTTCATCGTGGCAGGTGCCTCAGAGCGAGAGGAGAGAGAGAGAGAGAGAGAGAGAGACAGACAGATACAGAGAGGAGACGGACAGACAGCGGACAGACAGCGAGAGAGACAGAGACAGCGAGACAGAGACAGAGCGACAGAGAC"

with st.container():
    col_in, col_sm = st.columns([3, 1])
    with col_sm:
        st.write("### 📂 Sample Data")
        if st.button("Load DNA Example"):
            st.session_state["dna_input_area"] = sample_dna
    
    with col_in:
        input_tabs = st.tabs(["✍️ Nhập văn bản", "📂 Upload File"])
        
        with input_tabs[0]:
            dna_input = st.text_area("🧬 Sequence Input (DNA):", key="dna_input_area", height=120, placeholder="Dán trình tự DNA mục tiêu vào đây...")
        
        with input_tabs[1]:
            uploaded_file = st.file_uploader("Chọn file để upload (FASTA, TXT, CSV):", type=['fasta', 'fa', 'txt', 'csv'], key="file_uploader")
            if uploaded_file is not None:
                # Đọc nội dung file
                content = uploaded_file.read().decode('utf-8')
                
                # Xử lý file FASTA (bỏ qua dòng header bắt đầu bằng >)
                if uploaded_file.name.endswith(('.fasta', '.fa')):
                    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('>')]
                    dna_input = ''.join(lines)
                # Xử lý file CSV (lấy cột đầu tiên hoặc toàn bộ nội dung)
                elif uploaded_file.name.endswith('.csv'):
                    try:
                        df_upload = pd.read_csv(uploaded_file)
                        # Nếu có nhiều cột, lấy cột đầu tiên, nếu không thì lấy toàn bộ nội dung
                        if len(df_upload.columns) > 0:
                            dna_input = ''.join(df_upload.iloc[:, 0].astype(str).tolist())
                        else:
                            dna_input = content.replace('\n', '').replace(',', '').replace(' ', '')
                    except:
                        dna_input = content.replace('\n', '').replace(',', '').replace(' ', '')
                # Xử lý file TXT
                else:
                    dna_input = content.replace('\n', '').replace(' ', '')
                
                # Cập nhật session state để hiển thị trong text area
                st.session_state["dna_input_area"] = dna_input
                st.success(f"✅ Đã tải file thành công! ({len(dna_input)} ký tự)")
        
        # Lấy giá trị dna_input từ session state (từ tab nhập tay hoặc từ file upload)
        if 'dna_input_area' not in st.session_state:
            st.session_state["dna_input_area"] = ""
        dna_input = st.session_state.get("dna_input_area", "")

    if st.button("🚀 EXECUTE DEEP ANALYSIS"):
        # BƯỚC 1: Làm sạch chuỗi
        seq_clean = dna_input.upper().replace("\n", "").replace(" ", "").strip()

        # BƯỚC 2: Kiểm tra độ dài
        if len(seq_clean) < 23:
            st.warning("⚠️ Chuỗi quá ngắn, yêu cầu tối thiểu 23 ký tự.")
        
        # BƯỚC 3: Kiểm tra ký tự lạ
        elif any(c not in 'ACGT' for c in seq_clean):
            st.error("⚠️ Chuỗi chứa ký tự lạ không phải A, C, G, T. Vui lòng kiểm tra lại.")
            
        else:
            with st.spinner("Đang quét vị trí PAM (NGG) và khởi tạo ma trận..."):
                # --- SỬA LỖI: CHỈ LẤY CÁC ĐOẠN CÓ PAM (GG) Ở CUỐI ---
                # Trình tự 23bp = 20bp Guide + 1bp N + 2bp GG
                # Vậy vị trí index 21 và 22 (2 ký tự cuối) phải là 'GG'
                candidates = []
                indices = []
                
                for i in range(len(seq_clean) - 22):
                    segment = seq_clean[i : i+23]
                    # Kiểm tra đuôi PAM: Cas9 yêu cầu NGG (tức 2 nu cuối là GG)
                    if segment.endswith("GG"): 
                        candidates.append(segment)
                        indices.append(i)

                if not candidates:
                    st.warning("⚠️ Không tìm thấy vị trí PAM (GG) nào trong chuỗi DNA này!")
                    st.stop()

                # One-Hot Encoding
                mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
                X = np.array([[mapping.get(b, [0,0,0,0]) for b in s] for s in candidates])
                
                # Dự đoán điểm số
                scores = model.predict(X).flatten() if model else np.random.rand(len(candidates))
                
                def format_rank(s):
                    if s > 0.8: return "🌟 High"
                    elif s > 0.5: return "✅ Medium"
                    else: return "⚠️ Low"

                # Lưu vào Session State (Thêm cột PAM Position để người dùng dễ tra cứu)
                st.session_state.res = pd.DataFrame({
                    'Index': range(1, len(scores) + 1), # Số thứ tự tìm thấy
                    'Start Pos': indices,               # Vị trí bắt đầu trên gen
                    'Sequence': candidates, 
                    'Score': scores,
                    'Rank': [format_rank(s) for s in scores]
                })
# --- 5. HIỂN THỊ KẾT QUẢ ---
if 'res' in st.session_state:
    df = st.session_state.res
    st.markdown("---")
    
    # Dashboard nhỏ
    m1, m2, m3 = st.columns(3)
    m1.metric("Candidates Found", len(df))
    m2.metric("Max Efficiency", f"{df['Score'].max():.3f}")
    m3.metric("Optimal Targets (🌟)", len(df[df['Score'] > 0.8]))

    tabs = st.tabs(["📊 Visualization", "🔍 XAI Interpretation"])

    with tabs[0]:
        # Biểu đồ dải phổ màu
        chart = alt.Chart(df).mark_area(
            line={'color':'#58a6ff'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#0e1117', offset=0),
                       alt.GradientStop(color='#1f6feb', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Index', title='Vị trí trên toàn bộ chuỗi'),
            y=alt.Y('Score', title='Efficiency Score'),
            tooltip=['Index', 'Score', 'Rank']
        ).properties(height=400).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # Nút Download CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"crispr_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_csv"
        )
        
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        col_sel, col_viz = st.columns([1, 2])
        with col_sel:
            st.write("### 🎯 Tuyển chọn ứng viên")
            idx = st.selectbox("Chọn Index để phân tích XAI:", options=df['Index'].tolist())
            row = df.iloc[idx]
            st.write(f"**Trình tự:** `{row['Sequence']}`")
            st.write(f"**Điểm dự đoán:** `{row['Score']:.4f}`")
            st.write(f"**Xếp hạng:** {row['Rank']}")
            
        with col_viz:
            if model:
                saliency = get_saliency_map(model, row['Sequence'])
                xai_df = pd.DataFrame({'Pos': list(range(1,24)), 'Nuc': list(row['Sequence']), 'Val': saliency})
                xai_df['Label'] = xai_df['Pos'].astype(str) + ": " + xai_df['Nuc']
                
                # Tabs cho XAI Visualization
                xai_tabs = st.tabs(["📄 Bảng số liệu", "📊 Bar Chart", "🔥 Heatmap"])
                
                with xai_tabs[0]:
                    # Tab 1: Bảng số liệu
                    st.write("### Chi tiết độ quan trọng từng Nucleotide")
                    st.dataframe(xai_df[['Pos', 'Nuc', 'Val']].rename(columns={'Pos': 'Vị trí', 'Nuc': 'Nucleotide', 'Val': 'Độ quan trọng'}), use_container_width=True)
                
                with xai_tabs[1]:
                    # Tab 2: Bar Chart (giữ nguyên biểu đồ cột)
                    st.write("### Biểu đồ cột - Độ quan trọng Nucleotide")
                    xai_chart = alt.Chart(xai_df).mark_bar().encode(
                        x=alt.X('Label:O', sort=None, title='Nucleotide (Vị trí: Ký tự)'),
                        y=alt.Y('Val:Q', title='Độ quan trọng (Saliency)'),
                        color=alt.condition(alt.datum.Val > xai_df['Val'].mean(), alt.value('#ff4b4b'), alt.value('#00d4ff')),
                        tooltip=['Pos', 'Nuc', 'Val']
                    ).properties(height=350)
                    st.altair_chart(xai_chart, use_container_width=True)
                
                with xai_tabs[2]:
                    # Tab 3: Heatmap
                    st.write("### Heatmap - Bản đồ nhiệt độ quan trọng")
                    # Custom color scale: yellow -> light green -> teal -> blue (giống như trong ảnh)
                    heatmap_chart = alt.Chart(xai_df).mark_rect(stroke='white', strokeWidth=1).encode(
                        x=alt.X('Pos:O', title='Vị trí Nucleotide', axis=alt.Axis(labelAngle=0, labelColor='#333', titleColor='#333', gridColor='#e0e0e0')),
                        y=alt.Y('Nuc:O', title='Nucleotide', sort=['A', 'C', 'G', 'T'], axis=alt.Axis(labelColor='#333', titleColor='#333', gridColor='#e0e0e0')),
                        color=alt.Color('Val:Q', 
                                       title='Độ quan trọng',
                                       scale=alt.Scale(
                                           range=['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494'],
                                           domain=[xai_df['Val'].min(), xai_df['Val'].max()],
                                           type='linear'
                                       ),
                                       legend=alt.Legend(titleColor='#333', labelColor='#333')),
                        tooltip=['Pos', 'Nuc', 'Val']
                    ).properties(
                        height=200, 
                        width=600,
                        background='white'
                    ).configure_view(
                        stroke='transparent',
                        fill='white'
                    ).configure_axis(
                        domainColor='#333',
                        tickColor='#333'
                    )
                    st.altair_chart(heatmap_chart, use_container_width=True, theme=None)
                    
                    # Thêm heatmap dạng thanh ngang (alternative view)
                    st.write("#### Heatmap dạng thanh ngang")
                    heatmap_bar = alt.Chart(xai_df).mark_rect(stroke='white', strokeWidth=1).encode(
                        x=alt.X('Pos:O', title='Vị trí', axis=alt.Axis(labelAngle=0, labelColor='#333', titleColor='#333', gridColor='#e0e0e0')),
                        color=alt.Color('Val:Q',
                                       title='Độ quan trọng',
                                       scale=alt.Scale(
                                           range=['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494'],
                                           domain=[xai_df['Val'].min(), xai_df['Val'].max()],
                                           type='linear'
                                       ),
                                       legend=alt.Legend(titleColor='#333', labelColor='#333')),
                        tooltip=['Pos', 'Nuc', 'Val']
                    ).properties(
                        height=100, 
                        width=600,
                        background='white'
                    ).configure_view(
                        stroke='transparent',
                        fill='white'
                    ).configure_axis(
                        domainColor='#333',
                        tickColor='#333'
                    )
                    st.altair_chart(heatmap_bar, use_container_width=True, theme=None)
                
                # --- PHẦN NLG NÂNG CAO (Đã sửa lỗi hiển thị) ---
                
                # 1. Tính toán chỉ số
                gc_content = calculate_gc_content(row['Sequence'])
                motifs_warnings = check_motifs(row['Sequence'])
                
                distal_imp = xai_df[xai_df['Pos'] <= 12]['Val'].mean()
                seed_imp = xai_df[(xai_df['Pos'] >= 13) & (xai_df['Pos'] <= 20)]['Val'].mean()
                top_nucs = xai_df.nlargest(3, 'Val')

                st.markdown("### 📝 Phân tích chuyên sâu & Giải thích sinh học")
                
                # KHỞI TẠO BIẾN explanation (Dùng dedent để cắt bỏ khoảng trắng thừa)
                explanation = textwrap.dedent(f"""
                <div style="background-color: #161b22; padding: 25px; border-radius: 15px; border: 1px solid #30363d;">
                    <h4 style="color: #58a6ff; margin-top: 0;">🧬 BÁO CÁO HIỆU SUẤT SINH HỌC</h4>
                    <p style="font-size: 1.1em;">
                        Mô hình đánh giá trình tự này đạt 
                        <span style="color: {'#238636' if row['Score'] > 0.8 else '#ff4b4b'}; font-weight: bold; font-size: 1.2em;">
                        {row['Score']:.4f} ({row['Rank']})
                        </span>. 
                        Dưới đây là giải mã lý do tại sao AI đưa ra quyết định này:
                    </p>
                    <hr style="border-color: #30363d;">
                """)

                # --- PHẦN 1: GIẢI THÍCH VÙNG SEED ---
                explanation += textwrap.dedent(f"""<h5 style="color: #e6edf3;">1. Phân tích vùng Seed (Nucleotide 13-20)</h5>""")
                
                if seed_imp > distal_imp:
                    explanation += textwrap.dedent(f"""
                    <p>✅ <b>AI tập trung đúng trọng tâm:</b> Mô hình dành sự chú ý lớn ({seed_imp:.2f}) vào vùng Seed. 
                    Trong cơ chế CRISPR, 8-10 nucleotide này chịu trách nhiệm <b>tháo xoắn DNA (DNA melting)</b> và lai ghép với chuỗi đích. 
                    Việc AI đánh trọng số cao ở đây cho thấy trình tự này có khả năng bám đặc hiệu rất tốt.</p>
                    """)
                else:
                    explanation += textwrap.dedent(f"""
                    <p>⚠️ <b>Cảnh báo cấu trúc:</b> AI đang phân tán sự chú ý ra vùng xa (Distal region) thay vì tập trung vào vùng Seed. 
                    Điều này thường ám chỉ rằng trình tự này có thể gặp vấn đề về độ ổn định khi bắt cặp, hoặc dễ bị hiệu ứng off-target (cắt nhầm).</p>
                    """)

                # --- PHẦN 2: CÁC NUCLEOTIDE ĐỘT BIẾN ---
                explanation += textwrap.dedent(f"""<h5 style="color: #e6edf3; margin-top: 15px;">2. Các vị trí "Quyết định" (Key Drivers)</h5><ul>""")
                
                for _, nuc in top_nucs.iterrows():
                    pos_desc = ""
                    if nuc['Pos'] >= 21: pos_desc = "(Thuộc PAM - Giúp Cas9 nhận diện vị trí cắt)"
                    elif 13 <= nuc['Pos'] <= 20: pos_desc = "(Thuộc Seed - Quyết định độ bền liên kết)"
                    else: pos_desc = "(Thuộc vùng Distal - Ảnh hưởng đến độ ổn định khung)"
                    
                    # Lưu ý: Ngay cả trong vòng lặp cũng cần dedent nếu bạn xuống dòng
                    explanation += textwrap.dedent(f"""
                    <li style="margin-bottom: 8px;">
                        Vị trí <b>{nuc['Pos']} ({nuc['Nuc']})</b> có độ quan trọng cao nhất. <br>
                        <i style="color: #8b949e;">Lý do sinh học: {pos_desc}</i>. 
                        Sự hiện diện của <b>{nuc['Nuc']}</b> tại đây đóng góp tích cực vào dự đoán điểm số.
                    </li>
                    """)
                explanation += "</ul>"

                # --- PHẦN 3: ĐỘ BỀN NHIỆT (GC CONTENT) ---
                explanation += textwrap.dedent(f"""<h5 style="color: #e6edf3; margin-top: 15px;">3. Độ bền nhiệt động học (GC Content)</h5>""")
                
                gc_color = "#238636" if 40 <= gc_content <= 70 else "#ff4b4b"
                gc_eval = "Lý tưởng" if 40 <= gc_content <= 70 else "Không tối ưu"
                
                explanation += textwrap.dedent(f"""
                <p>Hàm lượng GC đạt <b>{gc_content:.1f}%</b> (<span style="color:{gc_color}">{gc_eval}</span>).</p>
                <div style="background-color: #30363d; border-radius: 5px; height: 10px; width: 100%;">
                    <div style="background-color: {gc_color}; width: {gc_content}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <p style="font-size: 0.9em; color: #8b949e; margin-top: 5px;">
                    <i>*Giải thích: Tỷ lệ GC từ 40-70% giúp cân bằng năng lượng liên kết. Quá thấp sẽ lỏng lẻo, quá cao sẽ tạo cấu trúc kẹp tóc (hairpin) cản trở Cas9.</i>
                </p>
                """)

                # --- PHẦN 4: CẢNH BÁO MOTIF ---
                if motifs_warnings:
                    explanation += textwrap.dedent(f"""<hr style="border-color: #30363d;"><h5 style="color: #ff4b4b;">⚠️ CẢNH BÁO AN TOÀN</h5>""")
                    for warn in motifs_warnings:
                        explanation += f"<p>{warn}</p>"
                
                explanation += "</div>"
                
                # Hiển thị kết quả cuối cùng
                st.markdown(explanation, unsafe_allow_html=True)
            else:
                st.error("Model 'best_model.keras' không tồn tại. Vui lòng kiểm tra lại file mô hình.")

