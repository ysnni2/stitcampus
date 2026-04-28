"""
image_stitching.py
==================
자동 이미지 스티칭 프로그램
- SIFT 특징점 추출 및 FLANN 매칭
- Lowe's ratio test로 좋은 매칭 선별
- RANSAC 기반 호모그래피 추정
- warpPerspective로 이미지 워핑
- ★ 추가 기능: Linear Feathering Blending (자연스러운 경계 합성)
"""

import cv2
import numpy as np
import os

# ────────────────────────────────────────────────────
# 설정: 사용할 이미지 경로 (왼쪽 → 오른쪽 순서)
# ────────────────────────────────────────────────────
IMAGE_PATHS = [
    "KakaoTalk_20260428_135350625.jpg",
    "KakaoTalk_20260428_140646008.jpg",
    "IMG_2108.jpeg",
]
OUTPUT_DIR  = "output"
MAX_WIDTH   = 1200
RATIO_TEST  = 0.80
RANSAC_THR  = 5.0
MIN_MATCHES = 10


def load_images(paths):
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"[ERROR] 이미지를 찾을 수 없습니다: {p}")
        h, w = img.shape[:2]
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        images.append(img)
        print(f"  [LOAD] {os.path.basename(p):40s} {img.shape[1]}x{img.shape[0]}")
    return images


def detect_and_match(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"  특징점 수: {len(kp1)} / {len(kp2)}")
    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
    print(f"  좋은 매칭 수 (ratio test): {len(good)}")
    if len(good) < MIN_MATCHES:
        raise RuntimeError(f"매칭 수 부족: {len(good)} < {MIN_MATCHES}")
    return kp1, kp2, good


def find_homography(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THR)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  RANSAC 인라이어: {inliers}/{len(matches)}")
    if H is None:
        raise RuntimeError("호모그래피 추정 실패")
    return H, mask


def warp_and_composite(img_src, img_dst, H):
    h_s, w_s = img_src.shape[:2]
    h_d, w_d = img_dst.shape[:2]
    corners_src = np.float32([[0,0],[w_s,0],[w_s,h_s],[0,h_s]]).reshape(-1,1,2)
    corners_dst = np.float32([[0,0],[w_d,0],[w_d,h_d],[0,h_d]]).reshape(-1,1,2)
    corners_warped = cv2.perspectiveTransform(corners_src, H)
    all_corners = np.concatenate([corners_dst, corners_warped], axis=0)
    x_min = int(np.floor(all_corners[:,:,0].min()))
    y_min = int(np.floor(all_corners[:,:,1].min()))
    x_max = int(np.ceil(all_corners[:,:,0].max()))
    y_max = int(np.ceil(all_corners[:,:,1].max()))
    tx, ty = -x_min, -y_min
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    warped_src = cv2.warpPerspective(img_src, T @ H, (canvas_w, canvas_h))
    canvas_dst = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_dst[ty:ty+h_d, tx:tx+w_d] = img_dst
    result = feathering_blend(canvas_dst, warped_src)
    return result


def feathering_blend(img_a, img_b):
    """★ 추가 기능: Linear Feathering Blending
    거리 변환 기반 alpha 맵으로 이미지 경계를 자연스럽게 합성"""
    mask_a = (img_a.sum(axis=2) > 0).astype(np.uint8)
    mask_b = (img_b.sum(axis=2) > 0).astype(np.uint8)
    dist_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 5)
    dist_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 5)
    total = dist_a + dist_b
    total[total == 0] = 1
    alpha = (dist_a / total)[..., np.newaxis]
    overlap = (mask_a & mask_b).astype(bool)
    only_a  = (mask_a & ~mask_b).astype(bool)
    only_b  = (~mask_a & mask_b).astype(bool)
    result = np.zeros_like(img_a, dtype=np.float32)
    img_a_f = img_a.astype(np.float32)
    img_b_f = img_b.astype(np.float32)
    result[overlap] = (alpha[overlap] * img_a_f[overlap] +
                       (1 - alpha[overlap]) * img_b_f[overlap])
    result[only_a]  = img_a_f[only_a]
    result[only_b]  = img_b_f[only_b]
    return result.clip(0, 255).astype(np.uint8)


def crop_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


def save_match_visualization(img1, img2, kp1, kp2, matches, mask, save_path):
    inlier_matches = [m for m, flag in zip(matches, mask.ravel()) if flag]
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2,
        inlier_matches[:60], None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(save_path, vis)
    print(f"  [SAVE] 매칭 시각화 -> {save_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  Image Stitching with Feathering Blend")
    print("=" * 55)

    print("\n[1] 이미지 로드")
    images = load_images(IMAGE_PATHS)

    result = images[0]

    for i in range(1, len(images)):
        print(f"\n[{i+1}] 이미지 {i} + 이미지 {i+1} 정합")
        try:
            kp1, kp2, good = detect_and_match(result, images[i])
            H, mask = find_homography(kp1, kp2, good)
            save_match_visualization(
                result, images[i], kp1, kp2, good, mask,
                os.path.join(OUTPUT_DIR, f"match_{i}.jpg")
            )
            result = warp_and_composite(result, images[i], H)
            result = crop_black_border(result)
            print(f"  현재 파노라마 크기: {result.shape[1]} x {result.shape[0]}")
            mid_path = os.path.join(OUTPUT_DIR, f"step_{i}.jpg")
            cv2.imwrite(mid_path, result)
            print(f"  [SAVE] 중간 결과 -> {mid_path}")
        except RuntimeError as e:
            print(f"  [SKIP] {e}")
            continue

    final_path = os.path.join(OUTPUT_DIR, "panorama_final.jpg")
    cv2.imwrite(final_path, result)
    print(f"\n{'='*55}")
    print(f"  [완료] 최종 파노라마 -> {final_path}")
    print(f"         크기: {result.shape[1]} x {result.shape[0]}")
    print(f"{'='*55}")