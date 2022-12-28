import hands

import glob
import os
import csv

# Variable
rsp = "paper"

originals = glob.glob("./original/*.png")

for original in originals:
    print(original)

# Cut
n = 250
for original in originals:
    for i in range(10):
        n = n + 1
        print(
            hands.ImageCropFromFile(
                original, savePath=hands.ImagePreprocessingFolder + "/" + str(n) + ".png"
            )
        )

# Square
for original in originals:
    print(
        hands.ImageResizerFromFile(
            original, savePath=hands.ImagePreprocessingFolder + "/" + str(n) + ".png"
        )
    )

# Resize
n = 500
resizeDatas = glob.glob("./preprocessing/*.png")
for resizeData in resizeDatas:
    n = n + 1
    print(
        hands.ImageResizerFromFile(
            resizeData, savePath=hands.ImageResizeFolder + "/" + str(n) + ".png"
        )
    )

# Train Data TSV
with open("./processing/train_master.tsv", "a", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    for appendPath in glob.glob("./processing/*.png"):
        writer.writerow([str(appendPath), rsp])


# Delete
for f in resizeDatas:
    os.remove(f)
