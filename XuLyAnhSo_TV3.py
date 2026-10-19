import json
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Import trực tiếp mã nguồn Backend của nhóm (preprocess.py)
try:
    from preprocess import preprocess_image
except ImportError:
    preprocess_image = None

# ==============================================================================
# 1. QUẢN LÝ CẤU HÌNH THÔNG SỐ (CONFIG.JSON)
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"

DEFAULT_CONFIG = {
    "reference_length_mm": 297.0,
    "reference_width_mm": 210.0,
    "fixed_threshold": 130,
    "white_min_value": 150,
    "white_max_saturation": 50,
    "background_hsv_lower": [0, 60, 0],
    "background_hsv_upper": [15, 255, 255],
    "min_area_fraction": 0.0015,
    "reference_min_area_fraction": 0.03
}

def load_system_config():
    """Tự động đọc và liên kết dữ liệu từ tệp config.json vào ứng dụng web."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                for key, val in DEFAULT_CONFIG.items():
                    if key not in cfg:
                        cfg[key] = val
                return cfg
        except Exception as err:
            st.sidebar.warning(f"Lỗi đọc config.json ({err}), dùng cấu hình mặc định.")
    return DEFAULT_CONFIG.copy()

# ==============================================================================
# 2. XỬ LÝ ĐO KÍCH THƯỚC VẬT THỂ (TÍNH TỶ LỆ PPM & QUY ĐỔI SỐ ĐO)
# ==============================================================================
def order_points(pts):
    """Sắp xếp 4 đỉnh hình chữ nhật: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-Left
    rect[2] = pts[np.argmax(s)]   # Bottom-Right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left
    return rect

def measure_and_annotate(image, target_mask, reference_mask, ref_length_mm, ref_width_mm, min_area_px=300):
    """
    Tính toán tỷ lệ Pixel/mm dựa trên vật chuẩn và quy ra kích thước thực tế,
    đồng thời đưa cả vật chuẩn và vật thể vào bảng thống kê với nhãn phân biệt rõ ràng.
    """
    annotated = image.copy()
    measurements = []
    ppm = None
    ref_box_coords = None

    # Step 1: Tính tỷ lệ PPM và thêm vật chuẩn vào bảng thống kê
    if reference_mask is not None and np.count_nonzero(reference_mask) > 0:
        cnts_ref, _ = cv2.findContours(reference_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_ref:
            c_ref = max(cnts_ref, key=cv2.contourArea)
            rect_ref = cv2.minAreaRect(c_ref)
            box_ref = order_points(cv2.boxPoints(rect_ref))
            ref_box_coords = box_ref
            
            tl, tr, br, bl = box_ref[0], box_ref[1], box_ref[2], box_ref[3]
            w_px = np.linalg.norm(tl - tr)
            h_px = np.linalg.norm(tl - bl)
            
            ref_l_px, ref_w_px = max(w_px, h_px), min(w_px, h_px)
            ppm_l = ref_l_px / max(ref_length_mm, ref_width_mm)
            ppm_w = ref_w_px / min(ref_length_mm, ref_width_mm)
            ppm = (ppm_l + ppm_w) / 2.0
            
            # Khoanh vùng hiển thị vật chuẩn trên ảnh với chữ to rõ
            cv2.drawContours(annotated, [box_ref.astype("int")], -1, (255, 0, 0), 3)
            cv2.putText(annotated, "VAT CHUAN", (int(tl[0]), max(30, int(tl[1]) - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            # Đưa thông số của vật chuẩn vào bảng thống kê đầu tiên
            ref_area_mm2 = ref_length_mm * ref_width_mm
            ref_area_cm2 = ref_area_mm2 / 100.0
            measurements.append({
                "STT": "Vật chuẩn (REF)",
                "Chiều dài (mm)": round(ref_length_mm, 2),
                "Chiều rộng (mm)": round(ref_width_mm, 2),
                "Diện tích (mm²)": round(ref_area_mm2, 2),
                "Diện tích (cm²)": round(ref_area_cm2, 2),
                "Tỷ lệ PPM (px/mm)": round(ppm, 2) if ppm else 0
            })

    # Step 2: Tìm contours và đo kích thước các vật thể thực tế còn lại
    cnts_target, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_target:
        cnts_target = sorted(cnts_target, key=lambda c: cv2.boundingRect(c)[0])
        
    obj_count = 0
    for c in cnts_target:
        if cv2.contourArea(c) < min_area_px:  # Lọc nhiễu theo diện tích động
            continue
            
        rect = cv2.minAreaRect(c)
        box = order_points(cv2.boxPoints(rect))
        tl, tr, br, bl = box[0], box[1], box[2], box[3]
        
        # Kiểm tra nếu trùng vị trí với vật chuẩn thì bỏ qua
        if ref_box_coords is not None:
            c_center = np.mean(box, axis=0)
            ref_center = np.mean(ref_box_coords, axis=0)
            if np.linalg.norm(c_center - ref_center) < 50:
                continue
        
        w_px = np.linalg.norm(tl - tr)
        h_px = np.linalg.norm(tl - bl)
        
        if ppm is None or ppm == 0:
            continue

        # Tính kích thước vật thể
        dim1_mm = w_px / ppm
        dim2_mm = h_px / ppm
        length_mm = max(dim1_mm, dim2_mm)
        width_mm = min(dim1_mm, dim2_mm)
        
        # Nếu vật thể có kích thước xấp xỉ tờ giấy chuẩn thì bỏ qua để tránh trùng lặp
        if abs(length_mm - max(ref_length_mm, ref_width_mm)) < 30 and abs(width_mm - min(ref_length_mm, ref_width_mm)) < 30:
            continue

        obj_count += 1
        area_mm2 = length_mm * width_mm
        area_cm2 = area_mm2 / 100.0
        
        # Vẽ bounding box & đánh số đối tượng vật thể
        cv2.drawContours(annotated, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.putText(annotated, f"#{obj_count}", (int(tl[0]), max(20, int(tl[1]) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(annotated, f"{length_mm:.1f}x{width_mm:.1f}mm",
                    (int(tl[0]), int(tl[1]) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        measurements.append({
            "STT": f"Vật thể #{obj_count}",
            "Chiều dài (mm)": round(length_mm, 2),
            "Chiều rộng (mm)": round(width_mm, 2),
            "Diện tích (mm²)": round(area_mm2, 2),
            "Diện tích (cm²)": round(area_cm2, 2),
            "Tỷ lệ PPM (px/mm)": round(ppm, 2)
        })

    return annotated, pd.DataFrame(measurements), ppm

# ==============================================================================
# 3. MÀN HÌNH DASHBOARD TRỰC QUAN (STREAMLIT APP)
# ==============================================================================
def main():
    st.set_page_config(
        page_title="Đồ Án XLAS - Đo Kích Thước Vật Thể",
        page_icon="📏",
        layout="wide"
    )

    # Custom CSS tùy chỉnh giao diện chuyên nghiệp
    st.markdown("""
        <style>
        img {
            border-radius: 10px;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    # Tiêu đề Dashboard
    st.title("📏 HỆ THỐNG ĐO KÍCH THƯỚC VẬT THỂ 2D SỬ DỤNG VẬT CHUẨN")
    st.caption("Đồ án Xử Lý Ảnh Số | Đề tài 16: Đo chiều dài, chiều rộng, diện tích vật thể từ ảnh 2D")
    st.markdown("---")

    # Load cấu hình mặc định từ config.json
    config_settings = load_system_config()

    # --------------------------------------------------------------------------
    # SIDEBAR: THANH ĐIỀU KHIỂN & CẤU HÌNH THAM SỐ
    # --------------------------------------------------------------------------
    st.sidebar.header("⚙️ THANH ĐIỀU KHIỂN")

    # 1. Nguồn dữ liệu
    st.sidebar.subheader("1. Nguồn Ảnh Đầu Vào")
    uploaded_file = st.sidebar.file_uploader("Tải ảnh lên từ máy tính", type=["jpg", "png", "jpeg", "bmp"])

    images_dir = ROOT_DIR / "data" / "images"
    sample_files = []
    if images_dir.exists():
        sample_files = [p.name for p in images_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']]

    selected_sample = None
    if sample_files:
        selected_sample = st.sidebar.selectbox("Hoặc chọn ảnh mẫu từ data/images:", ["-- Chọn ảnh mẫu --"] + sample_files)

    # 2. Phương pháp phân vùng
    st.sidebar.subheader("2. Phương Pháp Phân Vùng")
    method_display = st.sidebar.radio(
        "Thuật toán tách mặt nạ (Mask):",
        options=["Fixed Threshold (Cố định)", "Otsu Threshold (Tự động)", "HSV Color Space (Màu sắc)"],
        index=2
    )
    method_map = {
        "Fixed Threshold (Cố định)": "fixed",
        "Otsu Threshold (Tự động)": "otsu",
        "HSV Color Space (Màu sắc)": "hsv"
    }
    selected_method = method_map[method_display]

    # 3. Kích thước vật chuẩn
    st.sidebar.subheader("3. Kích Thước Vật Chuẩn (mm)")
    ref_length = st.sidebar.number_input(
        "Chiều dài vật chuẩn (mm):",
        value=float(config_settings["reference_length_mm"]),
        step=5.0
    )
    ref_width = st.sidebar.number_input(
        "Chiều rộng vật chuẩn (mm):",
        value=float(config_settings["reference_width_mm"]),
        step=5.0
    )

    # 4. Điều chỉnh tham số thực nghiệm
    st.sidebar.subheader("4. Tham Số Thực Nghiệm")
    fixed_thresh_val = st.sidebar.slider(
        "Ngưỡng xám (Fixed Threshold):",
        min_value=0, max_value=255, value=int(config_settings["fixed_threshold"])
    )
    
    # Mở rộng dải tối đa lên 0.05 để có thể lọc được các mảng hoa văn vân gỗ lớn trên bàn
    min_area_frac = st.sidebar.slider(
        "Tỷ lệ lọc diện tích nhiễu:",
        min_value=0.0005, max_value=0.0500,
        value=float(config_settings["min_area_fraction"]),
        step=0.0010, format="%.4f"
    )

    # Đóng gói settings
    current_settings = config_settings.copy()
    current_settings["fixed_threshold"] = fixed_thresh_val
    current_settings["min_area_fraction"] = min_area_frac
    current_settings["reference_length_mm"] = ref_length
    current_settings["reference_width_mm"] = ref_width

    # --------------------------------------------------------------------------
    # NẠP VÀ THỰC THI THUẬT TOÁN
    # --------------------------------------------------------------------------
    image = None
    img_name = ""

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_name = uploaded_file.name
    elif selected_sample and selected_sample != "-- Chọn ảnh mẫu --":
        image_path = images_dir / selected_sample
        image = cv2.imread(str(image_path))
        img_name = selected_sample

    if image is None:
        st.info("👈 **Vui lòng tải ảnh lên hoặc chọn ảnh mẫu ở Sidebar để hiển thị Dashboard.**")
        st.stop()

    # Gọi Backend preprocess.py
    with st.spinner("🔄 Đang xử lý tiền xử lý & tạo mặt nạ phân vùng..."):
        if preprocess_image is not None:
            prep_data = preprocess_image(image, current_settings)
            target_mask = prep_data["masks"][selected_method]["objects_mask"]
            reference_mask = prep_data["reference_mask"]
            otsu_val = prep_data.get("otsu_threshold", 0)
        else:
            # Fallback nếu chưa kết nối tệp preprocess.py
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, target_mask = cv2.threshold(gray, fixed_thresh_val, 255, cv2.THRESH_BINARY)
            reference_mask = np.zeros_like(gray)
            otsu_val = 0

    # Tính toán ngưỡng pixel tối thiểu dựa trên diện tích ảnh thực tế
    min_pixels_calc = int(image.shape[0] * image.shape[1] * min_area_frac)

    # Gọi Backend Đo kích thước
    with st.spinner("📏 Đang quy đổi tỷ lệ PPM và đo kích thước vật thể..."):
        annotated_img, df_stats, ppm = measure_and_annotate(
            image=image,
            target_mask=target_mask,
            reference_mask=reference_mask,
            ref_length_mm=ref_length,
            ref_width_mm=ref_width,
            min_area_px=min_pixels_calc
        )

    # --------------------------------------------------------------------------
    # GIAO DIỆN CHÍNH DẠNG TABS 
    # --------------------------------------------------------------------------
    tab_main, tab_mask, tab_report = st.tabs([
        "📊 Kết Quả & Trực Quan Hóa", 
        "🔍 Kiểm Tra Tiền Xử Lý & Mặt Nạ", 
        "📋 Bảng Thống Kê & Dữ Liệu"
    ])

    # TAB 1: KẾT QUẢ TRỰC QUAN & CHỈ SỐ KPI
    with tab_main:
        st.subheader(f"🖼️ Khung Kết Quả Nhận Diện: `{img_name}`")
        col_img, col_metrics = st.columns([3, 2], gap="large")
        
        with col_img:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            if ppm:
                st.info(f"📌 Tỷ lệ chuyển đổi chuẩn (PPM): **{ppm:.2f} px/mm**")

        with col_metrics:
            st.markdown("### 📈 Chỉ Số Nổi Bật (KPIs)")
            target_objs_count = len(df_stats[df_stats["STT"].str.contains("Vật thể", na=False)])
            st.metric("Tổng số vật thể phát hiện", f"{target_objs_count} vật")
            st.metric("Tỷ lệ PPM", f"{ppm:.2f} px/mm" if ppm else "N/A")
            
            if not df_stats.empty:
                obj_df = df_stats[df_stats["STT"].str.contains("Vật thể", na=False)]
                if not obj_df.empty:
                    st.metric("DT vật thể lớn nhất", f"{obj_df['Diện tích (cm²)'].max():.2f} cm²")
                    st.metric("DT vật thể nhỏ nhất", f"{obj_df['Diện tích (cm²)'].min():.2f} cm²")
                else:
                    st.metric("DT lớn nhất", f"{df_stats['Diện tích (cm²)'].max():.2f} cm²")
                    st.metric("DT nhỏ nhất", f"{df_stats['Diện tích (cm²)'].min():.2f} cm²")

    # TAB 2: KIỂM TRA MẶT NẠ & PHÂN VÙNG
    with tab_mask:
        st.subheader("👁️ So Sánh Quá Trình Phân Vùng Ảnh")
        col_orig, col_m, col_ref = st.columns(3)
        
        with col_orig:
            st.markdown("**1. Ảnh Gốc (Original)**")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_m:
            st.markdown(f"**2. Mặt Nạ Vật Thể ({method_display})**")
            st.image(target_mask, use_container_width=True, channels="GRAY")
            if selected_method == "otsu" and otsu_val > 0:
                st.caption(f"Ngưỡng Otsu tự động: `{otsu_val:.1f}`")

        with col_ref:
            st.markdown("**3. Mặt Nạ Vật Chuẩn (REF)**")
            st.image(reference_mask, use_container_width=True, channels="GRAY")

    # TAB 3: BẢNG THỐNG KÊ CHI TIẾT & XUẤT CSV
    with tab_report:
        st.subheader("📊 Bảng Thống Kê Số Liệu Chi Tiết")
        if not df_stats.empty:
            st.dataframe(df_stats, use_container_width=True)
            csv_output = df_stats.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 Tải Bảng Số Đo Chuẩn (CSV)",
                data=csv_output,
                file_name=f"SoDo_KichThuoc_{Path(img_name).stem}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Không phát hiện đối tượng nào trong mặt nạ hiện tại. Thử chọn phương pháp khác hoặc chỉnh ngưỡng trên Sidebar.")

if __name__ == "__main__":
    main()