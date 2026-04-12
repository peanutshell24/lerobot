import cv2
import numpy as np
import glob

# ========== 1. 配置 ==========
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()

squares_x = 9
squares_y = 6
square_length = 0.02878   # 米
marker_length = 0.02120   # 米

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    aruco_dict
)

# ========== 2. 收集角点 ==========
all_charuco_corners = []
all_charuco_ids = []
image_size = None

image_files = glob.glob("calib_images/*.jpg")

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    # 检测 ArUco marker
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

    if ids is None or len(ids) == 0:
        continue

    # 基于 marker 插值出 ChArUco 角点
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )

    # 至少要有足够多的角点
    if ret is not None and ret >= 4:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        vis = img.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
        cv2.imshow("charuco", vis)
        cv2.waitKey(100)

cv2.destroyAllWindows()

if len(all_charuco_corners) < 5:
    raise RuntimeError("可用标定图太少，建议至少多拍一些。")

# ========== 3. 相机标定 ==========
retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("重投影误差:", retval)
print("camera_matrix =\n", camera_matrix)
print("dist_coeffs =\n", dist_coeffs)

# ========== 4. 保存 ==========
fs = cv2.FileStorage("camera_charuco.yml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("dist_coeffs", dist_coeffs)
fs.release()

print("已保存到 camera_charuco.yml")