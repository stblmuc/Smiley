import sys
import os
import shutil

image_size = "28"
fixed_cats = ["sad", "smile", "neutral", "upside-down"]

if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <import_directory> <output_directory>")
    print("Example: python C:\\Users\\iot\\Smiley\\collectImages.py C:\\Smiley C:\\Users\\iot\\Smiley\\smiley\\data\\categories\\28")
    exit(1)

importFolder = sys.argv[1] #path to folder with team1, team2, ...
outputFolder = sys.argv[2] #path to folder where subfolders with categories should be created
outputFolderUserCats = os.path.normpath(outputFolder) + "-user"

# if previous output folder should be deleted, uncomment this code
#if os.path.exists(outputFolder) and os.path.isdir(outputFolder):
#    shutil.rmtree(outputFolder)


def copy_image(imagePath, category):
    ouputDir = outputFolder if category in fixed_cats else outputFolderUserCats
    # create folder for category if it doesn't exist:
    path = os.path.join(ouputDir, category)
    if not os.path.exists(path):
        os.makedirs(path)

    image_name = max([0] + [int(p.split(".")[0]) for p in os.listdir(path)]) + 1

    shutil.copy2(imagePath, path + "/" + str(image_name) + ".png")


subfolder = os.path.join("categories", image_size)
for z in os.walk(importFolder):
    if len(str(z[0].split("/")[-1])) > 0:
        for x in os.walk(os.path.join(str(z[0].split("/")[-1]), subfolder)):
            if len(x[-1]) > 0:
                for y in os.walk(os.path.join(str(x[0].split("/")[-1]))):
                    for img in y[-1]:
                        if img.split(".")[-1] == "png":
                            copy_image(y[0] + "/" + img, y[0].split("\\")[-1])
