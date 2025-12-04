import os
import shutil
import re

def parse_label_file(file_path):
    """
    stepN.txt 파일에서 라벨 정보를 파싱합니다.
    예: "- ('SUV', '기아자동차', '모하비', '2020-'): 500개" -> ('SUV', '기아자동차', '모하비', '2020-')
    """
    labels = []
    # 정규 표현식을 사용하여 튜플 부분만 정확하게 추출합니다.
    # 그룹 1: 차종, 그룹 2: 제조사, 그룹 3: 모델, 그룹 4: 연식
    pattern = re.compile(r"\('([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'\)")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    # 각 그룹을 튜플로 만들어 리스트에 추가합니다.
                    labels.append(match.groups())
    except FileNotFoundError:
        print(f"경고: {file_path} 파일을 찾을 수 없습니다. 이 단계는 건너뜁니다.")
    return labels

def copy_files_for_step(step_name, labels, base_src_path, base_dst_path):
    """
    주어진 라벨에 해당하는 파일들을 원본 경로에서 대상 경로로 복사합니다.
    """
    print(f"--- {step_name} 처리 시작 ---")
    
    # 대상 폴더 생성 (images, labels)
    dst_images_path = os.path.join(base_dst_path, step_name, 'images')
    dst_labels_path = os.path.join(base_dst_path, step_name, 'labels')
    os.makedirs(dst_images_path, exist_ok=True)
    os.makedirs(dst_labels_path, exist_ok=True)

    for label_parts in labels:
        # label_parts 예: ('SUV', '기아자동차', '모하비', '2020-')
        relative_path = os.path.join(*label_parts)
        
        # 원본 경로 설정
        src_image_dir = os.path.join(base_src_path, 'all', 'images', relative_path)
        src_label_dir = os.path.join(base_src_path, 'all', 'labels', relative_path)

        # 대상 경로 설정
        dst_image_dir = os.path.join(dst_images_path, relative_path)
        dst_label_dir = os.path.join(dst_labels_path, relative_path)

        # shutil.copytree를 사용하여 디렉토리 전체 복사
        if os.path.exists(src_image_dir):
            if not os.path.exists(dst_image_dir):
                shutil.copytree(src_image_dir, dst_image_dir)
                print(f"복사 완료 (images): {src_image_dir} -> {dst_image_dir}")
            else:
                print(f"이미 존재하여 건너뜁니다 (images): {dst_image_dir}")
        if os.path.exists(src_label_dir):
            if not os.path.exists(dst_label_dir):
                shutil.copytree(src_label_dir, dst_label_dir)
                print(f"복사 완료 (labels): {src_label_dir} -> {dst_label_dir}")
            else:
                print(f"이미 존재하여 건너뜁니다 (labels): {dst_label_dir}")

if __name__ == "__main__":
    BASE_PATH = r"F:\dataset_final"
    # stepN.txt 파일들이 위치한 경로 (스크립트 실행 위치 기준)
    LABEL_FILE_PATH = r"c:\Users\DMLab\GITHUB\vericar" 

    for i in range(4):
        step = f"step{i}"
        label_file = os.path.join(LABEL_FILE_PATH, f"{step}.txt")
        
        labels_to_copy = parse_label_file(label_file)
        if labels_to_copy:
            copy_files_for_step(step, labels_to_copy, BASE_PATH, BASE_PATH)
