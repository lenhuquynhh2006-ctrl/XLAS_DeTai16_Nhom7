"""Tiền xử lý và phân vùng. Chạy: python preprocess.py

Giả định: nền tối; bìa chuẩn trắng là vùng trắng lớn nhất; vật nằm tách rời.
Không đo kích thước và không tự tạo số đo Ground Truth.
"""
from pathlib import Path
import argparse
import csv
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def read_image(path):
    # Đọc bằng imdecode để hỗ trợ đường dẫn có dấu trên Windows.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Không đọc được ảnh: {path.name}')
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'Không lưu được ảnh: {path.name}')
    data.tofile(str(path))


def clean_mask(mask, min_pixels):
    # Tăng kích thước kernel lên (5,5) để xóa các vân gỗ/hoa văn bề mặt bàn hiệu quả hơn
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Chỉ lấp lỗ bên trong, không thay biên ngoài bằng hình chữ nhật.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    mask = cv2.bitwise_or(padded, cv2.bitwise_not(flooded))[1:-1, 1:-1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_pixels:
            result[labels == index] = 255
    return result


def preprocess_image(image, settings):
    """Hàm để tvien 2 import. Mọi mask giữ nguyên kích thước ảnh gốc."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Lọc nhiễu trước khi tăng tương phản.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    min_pixels = max(20, int(gray.size * settings['min_area_fraction']))

    # A: threshold xám cố định. B: Otsu tự chọn ngưỡng.
    _, fixed_raw = cv2.threshold(enhanced, settings['fixed_threshold'], 255, cv2.THRESH_BINARY)
    otsu_value, otsu_raw = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # C: tách theo màu nền HSV. Với nền nâu, loại vùng H/S/V của nền.
    background = cv2.inRange(hsv, np.array(settings['background_hsv_lower'], np.uint8),
                            np.array(settings['background_hsv_upper'], np.uint8))
    color_raw = cv2.bitwise_not(background)
    
    # Thêm bước lọc phụ để loại bỏ các đường vân mỏng trên bàn cho kênh màu HSV
    kernel_texture = np.ones((7, 7), np.uint8)
    color_raw = cv2.morphologyEx(color_raw, cv2.MORPH_OPEN, kernel_texture)

    # Tách bìa trắng: độ bão hòa thấp, độ sáng cao.
    white_raw = cv2.inRange(hsv, np.array([0, 0, settings['white_min_value']], np.uint8),
                           np.array([179, settings['white_max_saturation'], 255], np.uint8))
    white = clean_mask(white_raw, min_pixels)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    reference = np.zeros_like(gray)
    warnings = []
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        reference[labels == largest] = 255
        if np.count_nonzero(reference) / gray.size < settings['reference_min_area_fraction']:
            warnings.append('Vung chuan qua nho: kiem tra reference_mask')
    else:
        warnings.append('Khong tim thay bia trang: kiem tra anh/nguong HSV')

    masks = {}
    for name, raw in [('fixed', fixed_raw), ('otsu', otsu_raw), ('hsv', color_raw)]:
        foreground = clean_mask(raw, min_pixels)
        # Bỏ vùng chuẩn đã nhận diện khỏi mask vật cần đo.
        targets = cv2.bitwise_and(foreground, cv2.bitwise_not(reference))
        targets = clean_mask(targets, min_pixels)
        masks[name] = {'all_mask': cv2.bitwise_or(targets, reference),
                       'objects_mask': targets}
        components = cv2.connectedComponentsWithStats(targets)[0] - 1
        if components == 0:
            warnings.append(f'{name}: khong tach duoc vat can do')
    return {'gray': gray, 'hsv': hsv, 'blurred': blurred, 'enhanced': enhanced,
            'reference_mask': reference, 'masks': masks,
            'otsu_threshold': float(otsu_value), 'warnings': warnings}


def preview(image, result):
    # Chỉ thu nhỏ ảnh xem trước. Mask bàn giao luôn ở độ phân giải gốc.
    panels = [(image, 'Original'), (result['gray'], 'Grayscale'),
              (result['enhanced'], 'CLAHE'), (result['reference_mask'], 'Reference'),
              (result['masks']['fixed']['objects_mask'], 'Fixed: targets'),
              (result['masks']['otsu']['objects_mask'], 'Otsu: targets'),
              (result['masks']['hsv']['objects_mask'], 'HSV: targets')]
    overlay = image.copy()
    target = result['masks']['hsv']['objects_mask'] > 0
    ref = result['reference_mask'] > 0
    overlay[target] = (0.6 * overlay[target] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[ref] = (0.6 * overlay[ref] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    panels.append((overlay, 'HSV overlay: green target / blue ref'))
    tiles = []
    for panel, label in panels:
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        h, w = panel.shape[:2]
        scale = min(360 / w, 430 / h)
        resized = cv2.resize(panel, (int(w * scale), int(h * scale)))
        tile = np.full((470, 380, 3), 245, np.uint8)
        rh, rw = resized.shape[:2]
        tile[35:35+rh, (380-rw)//2:(380-rw)//2+rw] = resized
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 25, 25), 1)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])


def init_ground_truth(images, settings):
    # Đây chỉ là mẫu để nhập số đo bằng thước, không phải kết quả đo từ ảnh.
    path = ROOT / 'data' / 'ground_truth.csv'
    if path.exists():
        print('Đã có ground_truth.csv; giữ nguyên để không mất số em nhập.')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_name', 'object_id', 'role', 'length_mm', 'width_mm',
                         'area_mm2', 'area_definition', 'position_note'])
        for image in images:
            length, width = settings['reference_length_mm'], settings['reference_width_mm']
            writer.writerow([image.name, 'REF01', 'reference', length, width,
                             length * width, 'rectangle', 'white board'])
            writer.writerow([image.name, '', 'target', '', '', '', '', ''])
    print('Đã tạo data/ground_truth.csv. Điền mã vật, kích thước thực tế và thêm dòng cho ảnh nhiều vật.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Tên một ảnh trong data/images, bỏ qua để chạy toàn bộ')
    parser.add_argument('--init-gt', action='store_true', help='Tạo mẫu Ground Truth nếu chưa tồn tại')
    args = parser.parse_args()
    settings = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    folder = ROOT / 'data' / 'images'
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if args.image:
        images = [p for p in images if p.name == args.image]
    if not images:
        print('Chưa có ảnh phù hợp. Chép ảnh vào data/images và kiểm tra tên/đuôi file.')
        return 1
    if args.init_gt:
        init_ground_truth(images, settings)
    (ROOT / 'outputs').mkdir(parents=True, exist_ok=True)
    report = []
    for path in images:
        try:
            image = read_image(path)
            result = preprocess_image(image, settings)
            # Bao gồm đuôi ảnh để không ghi đè nhau.
            destination = ROOT / 'outputs' / path.name
            for name in ['gray', 'blurred', 'enhanced', 'reference_mask']:
                save_image(destination / f'{name}.png', result[name])
            # HSV gốc lưu .npy; chuyển thẳng HSV sang PNG sẽ hiển thị sai màu.
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / 'hsv.npy', result['hsv'])
            for name, masks in result['masks'].items():
                for kind, mask in masks.items():
                    save_image(destination / name / f'{kind}.png', mask)
            save_image(destination / 'preview.png', preview(image, result))
            entry = {'image': path.name, 'otsu_threshold': result['otsu_threshold'],
                     'warnings': result['warnings'], 'status': 'needs_visual_review'}
            (destination / 'processing_info.json').write_text(json.dumps(entry, indent=2), encoding='utf-8')
            report.append(entry)
            print(f'Đã xử lý {path.name}. Mở outputs/{path.name}/preview.png để kiểm tra.')
            for warning in result['warnings']:
                print('  Lưu ý:', warning)
        except Exception as error:
            report.append({'image': path.name, 'status': 'error', 'error': str(error)})
            print(f'Lỗi {path.name}: {error}')
    (ROOT / 'outputs' / 'processing_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Hoàn tất xử lý ảnh.')
    return int(any(row['status'] == 'error' for row in report))


if __name__ == '__main__':
    raise SystemExit(main())