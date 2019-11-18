import cv2
import os

tmp = "train_temp/"

def rebuild(dir):
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    for root, dirs, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                image = cv2.imread(filepath)
                dim = (227, 227)
                resized = cv2.resize(image, dim)
                path = tmp + file
                cv2.imwrite(path, resized)
            except:
                print(filepath)
                os.remove(filepath)


if __name__ == "__main__":
    rebuild("train/")
