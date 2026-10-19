from pathlib import Path
import json
import cv2
import numpy as np

# Import hàm tiền xử lý từ preprocess.py của Thành viên 1
try:
    from preprocess import preprocess_image
except ImportError:
    preprocess_image = None

ROOT = Path(__file__).resolve().parent


def order_points(pts):
  """Sắp xếp 4 đỉnh của hình chữ nhật theo thứ tự: top-left, top-right, bottom-right, bottom-left."""
  rect = np.zeros((4, 2), dtype="float32")
  s = pts.sum(axis=1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  return rect


def process_and_measure(
    image_path, ref_width_mm=210.0, output_path="output_measured.png", method="hsv"
):
  """Thực hiện đo lường kích thước các vật thể dựa trên vật chuẩn sử dụng mặt nạ đã tiền xử lý."""
  # 1. Đọc ảnh gốc bằng imdecode để hỗ trợ đường dẫn có dấu trên Windows
  image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
  if image is None:
    print(f"Không thể đọc được ảnh tại: {image_path}")
    return

  # Đọc cấu hình settings từ config.json nếu có, hoặc dùng mặc định
  config_path = ROOT / "config.json"
  settings = {
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
  if config_path.exists():
      try:
          with open(config_path, "r", encoding="utf-8") as f:
              settings.update(json.load(f))
      except Exception:
          pass

  # Sử dụng kết quả tiền xử lý và tách mặt nạ từ preprocess.py thay vì dùng Canny trên ảnh gốc
  if preprocess_image is not None:
      result = preprocess_image(image, settings)
      reference_mask = result['reference_mask']
      if method not in result['masks']:
          method = 'hsv'
      target_mask = result['masks'][method]['objects_mask']
  else:
      # Fallback nếu chưa kết nối được preprocess.py
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      blurred = cv2.GaussianBlur(gray, (7, 7), 0)
      target_mask = cv2.Canny(blurred, 50, 100)
      reference_mask = np.zeros_like(gray)

  orig = image.copy()
  pixels_per_metric = None
  obj_count = 0
  ref_box = None

  # 2. Xử lý vật chuẩn (Reference) từ reference_mask để tính tỷ lệ PPM chính xác
  ref_cnts, _ = cv2.findContours(reference_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if ref_cnts:
      c_ref = max(ref_cnts, key=cv2.contourArea)
      ref_rect_cv = cv2.minAreaRect(c_ref)
      ref_box_pts = cv2.boxPoints(ref_rect_cv)
      ref_box = np.array(ref_box_pts, dtype="int")
      ref_ordered = order_points(ref_box.astype("float32"))
      
      tl, tr, br, bl = ref_ordered
      ref_w_px = np.linalg.norm(tl - tr)
      ref_h_px = np.linalg.norm(tl - bl)
      
      actual_ref_l = settings.get("reference_length_mm", 297.0)
      actual_ref_w = settings.get("reference_width_mm", 210.0)
      
      ref_l_px = max(ref_w_px, ref_h_px)
      pixels_per_metric = ref_l_px / max(actual_ref_l, actual_ref_w)
      print(f"   [Vật chuẩn] Tỷ lệ quy đổi: {pixels_per_metric:.2f} px/mm")

      # Vẽ khung vật chuẩn
      cv2.drawContours(orig, [ref_box], -1, (255, 0, 0), 2)
      cv2.putText(orig, "VAT CHUAN", (int(tl[0]), max(30, int(tl[1]) - 10)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

  # 3. Tìm contours (đường viền) của tất cả vật thể từ target_mask
  cnts, _ = cv2.findContours(
      target_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  if not cnts:
    print(f"Không tìm thấy vật thể nào trong ảnh: {Path(image_path).name}")
    cv2.imwrite(output_path, orig)
    return

  # Sắp xếp các contours từ trái sang phải
  cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

  # 4. Lặp qua từng đối tượng để tính toán kích thước
  for c in cnts:
    # Lọc bỏ các contour quá nhỏ (nhiễu hạt hoặc vân gỗ sót lại)
    if cv2.contourArea(c) < 300:
      continue

    # Bỏ qua nếu contour nằm trùng hoặc giao nhau với vùng vật chuẩn
    if ref_box is not None:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if cv2.pointPolygonTest(ref_box.astype(np.float32), (cx, cy), False) >= 0:
                continue

    obj_count += 1

    # Xác định minAreaRect (hình chữ nhật xoay ôm sát vật thể)
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    rect = order_points(box.astype("float32"))

    (tl, tr, br, bl) = rect
    width_px = np.linalg.norm(tl - tr)
    height_px = np.linalg.norm(tl - bl)

    if pixels_per_metric is None or pixels_per_metric == 0:
      print("   [Lỗi] Không xác định được tỷ lệ PPM từ vật chuẩn!")
      return

    # Tính chiều rộng, chiều dài và diện tích thực tế
    dim_w_mm = width_px / pixels_per_metric
    dim_h_mm = height_px / pixels_per_metric
    dim_l_mm = max(dim_w_mm, dim_h_mm)
    dim_s_mm = min(dim_w_mm, dim_h_mm)
    area_mm2 = dim_l_mm * dim_s_mm

    # 5. Vẽ khung, đánh số thứ tự và hiển thị kích thước lên ảnh
    cv2.drawContours(orig, [box], -1, (0, 255, 0), 2)

    # Đánh số thứ tự đối tượng (#1, #2, #3...)
    cv2.putText(
        orig,
        f"#{obj_count}",
        (int(tl[0]), int(tl[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    # Hiển thị kích thước (Dài x Rộng)
    cv2.putText(
        orig,
        f"{dim_l_mm:.1f}x{dim_s_mm:.1f}mm",
        (int(tl[0]), int(tl[1]) + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
    )

    print(
        f"   - Vật thể #{obj_count}: Rộng = {dim_s_mm:.2f}mm, Dài ="
        f" {dim_l_mm:.2f}mm, Diện tích = {area_mm2:.2f}mm²"
    )

  # Lưu kết quả ảnh đo lường
  cv2.imwrite(output_path, orig)
  print(f"-> Đã lưu kết quả đo tại: {output_path}\n")


if __name__ == "__main__":
  # Sử dụng đường dẫn tương đối qua ROOT để tránh lỗi trên Streamlit Cloud
  images_folder = ROOT / "data" / "images"
  output_folder = ROOT / "outputs_measured_tv2"
  output_folder.mkdir(exist_ok=True)

  EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
  image_paths = sorted(
      p for p in images_folder.iterdir() if p.suffix.lower() in EXTENSIONS
  )

  if not image_paths:
    print("Không tìm thấy ảnh nào trong thư mục data/images!")
  else:
    print(
        f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu thực hiện đo lường cho Thành"
        " viên 2...\n"
    )
    # Lựa chọn phương pháp phân vùng: 'hsv', 'otsu', hoặc 'fixed'
    chosen_method = "hsv"
    for idx, img_path in enumerate(image_paths, 1):
      print(f"[{idx}/{len(image_paths)}] Đang xử lý: {img_path.name}")
      output_file_path = output_folder / f"measured_{img_path.name}"

      # ref_width_mm: điều chỉnh kích thước thực tế của vật chuẩn
      process_and_measure(
          image_path=str(img_path),
          ref_width_mm=210.0,
          output_path=str(output_file_path),
          method=chosen_method,
      )

    print("=== Hoàn tất toàn bộ nhiệm vụ của Thành viên 2! ===")