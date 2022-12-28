import os
import cv2
import random
import glob
import time
import mediapipe as mp
import matplotlib.pyplot as plt

# Main Variable
CaptureDeviceID = 0
CaptureResolution = (640, 480, 15)  # Width, Height, fps
CroppingSize = 720
OutputResolution = 96

# Override Rectangle
AreaStartAt = ((CaptureResolution[0] - CaptureResolution[1]) / 2, 0) # (80, 0)
AreaEndAt = (
    ((CaptureResolution[0] - CaptureResolution[1]) / 2) + CaptureResolution[1],
    CaptureResolution[1],
) # (560, 480)

# Save Point
ImageOriginalFolder = "./original/"
ImageGenerateFolder = "./generate/"
ImageProcessFolder = "./process/"
ImageResizeFolder = "./resize/"

# Definition
def ImageGenerateFromFile(src: str, out: str = 'file', mode: str = 'blank', savePath: str = ImageGenerateFolder + "pose.jpg"):
    basename = os.path.basename(src)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)
    blank = cv2.flip(cv2.imread('./blank.png'), 1)
    image = cv2.flip(cv2.imread(src), 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmark_list = ['WRIST', 'THUMP_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(21):
            print(landmark_list[i])
            print(hand_landmarks.landmark[i])
    if mode == 'image':
        annotated_image = image.copy()
    elif mode == 'blank':
        annotated_image = blank.copy()
    for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
    output = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    if out == 'file':
        cv2.imwrite(savePath, output)
    elif out == 'display':
        plt.imshow(output)

def ImageCropFromFile(
    path: str,
    cropSize: int = CroppingSize,
    savePath: str = ImageProcessFolder + "crop.jpg",
) -> bool:
    basename = os.path.splitext(os.path.basename(path))[0]
    original = cv2.imread(path)
    cropAreaStartLimitH, cropAreaStartLimitW = (
        original.shape[0] - cropSize,
        original.shape[1] - cropSize,
    )
    RandomStartAtW, RandomStartAtH = random.randint(
        0, cropAreaStartLimitW
    ), random.randint(0, cropAreaStartLimitH)
    cropped = original[
        RandomStartAtH : RandomStartAtH + cropSize,
        RandomStartAtW : RandomStartAtW + cropSize,
    ]
    cv2.imwrite(savePath, cropped)
    return (
        True,
        [cropAreaStartLimitW, cropAreaStartLimitH],
        [RandomStartAtW, RandomStartAtH],
        [RandomStartAtW + cropSize, RandomStartAtH + cropSize],
    )

def ImageSquareFromFile(
    path: str, savePath: str = ImageProcessFolder + "sq.jpg"
):
    basename = os.path.splitext(os.path.basename(path))[0]
    original = cv2.imread(path)
    squared = original[
        # int(AreaStartAt[1]) : int(AreaEndAt[1]), int(AreaStartAt[0]) : int(AreaEndAt[0])
        0:1080, 420:1500 
    ]
    print(AreaStartAt[1], AreaEndAt[1], AreaStartAt[0], AreaEndAt[0])
    cv2.imwrite(savePath, squared)
    return True

def ImageResizerFromFile(
    path: str, savePath: str = ImageResizeFolder + "out.jpg"
) -> bool:
    basename = os.path.splitext(os.path.basename(path))[0]
    original = cv2.imread(path)
    resized = cv2.resize(original, dsize=[OutputResolution, OutputResolution])
    cv2.imwrite(savePath, resized)
    return True

if __name__ == '__main__':
    rsp = 'scissor'

    # Original Directory
    originals = glob.glob(ImageOriginalFolder + rsp + "/" + "*.png")
    print(originals)
    time.sleep(1)
    n = 0

    # Posing Output
    for original in originals: 
        basename = os.path.basename(original)
        print(
            basename,
            ImageGenerateFromFile(original, savePath=ImageGenerateFolder + rsp + "/" + basename + ".jpg")
        )

    # Posing Directory
    posings = glob.glob(ImageGenerateFolder + rsp + "/" + "*.jpg")
    n = 0

    print("posings:", posings)
    time.sleep(1)

    # Crop
    # for posing in posings:
    #     for i in range(10):
    #         n = n + 1
    #         print(
    #             ImageCropFromFile(
    #                 posing, savePath=ImageProcessFolder + str(n) + ".jpg"
    #             )
    #         )

    # Square
    for posing in posings:
        n = n + 1
        print(
            ImageSquareFromFile(
                posing, savePath=ImageProcessFolder + rsp + "/" + str(n) + ".jpg"
            )
        )

    # Process Directory
    processes = glob.glob(ImageProcessFolder + rsp + "/" + "*.jpg")

    print(processes)
    time.sleep(1)

    for process in processes:
        basename = os.path.basename(process)
        print(
            ImageResizerFromFile(process, ImageResizeFolder + rsp + "/" + basename)
        )

