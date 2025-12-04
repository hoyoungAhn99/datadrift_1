import yaml
from scipy import io
from pathlib import Path
import glob
from PIL import Image

with open("config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

annot_file = io.loadmat(Path(config['preprocess']['annot_path']))

in_data_path = Path(config['preprocess']['in_data_path'])
out_data_path = Path(config['preprocess']['out_data_path'])

# 출력 디렉토리가 없으면 생성합니다.
out_data_path.mkdir(parents=True, exist_ok=True)

annotations = annot_file['annotations'][0]
num_images = len(annotations)

print(f"총 {num_images}개의 이미지 크롭을 시작합니다...")

for i, annotation in enumerate(annotations):
    # .mat 파일에서 바운딩 박스 좌표와 파일명을 추출합니다.
    x1 = annotation['bbox_x1'][0][0]
    y1 = annotation['bbox_y1'][0][0]
    x2 = annotation['bbox_x2'][0][0]
    y2 = annotation['bbox_y2'][0][0]
    filename = annotation['fname'][0]

    # 이미지 열기, 자르기, 저장
    image_path = in_data_path / filename
    with Image.open(image_path) as img:
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img.save(out_data_path / filename)

    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{num_images} 처리 완료...")

print("모든 이미지 크롭이 완료되었습니다.")
