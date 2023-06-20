
import cv2, random
from tqdm import tqdm
print("Starting")
image = cv2.imread('000008.jpg')

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

ss.switchToSelectiveSearchQuality()
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))
print("Beginnign viz loop")
no_of_rects = 8
for i in tqdm(range(0, len(rects), no_of_rects)):
    # clone the original image so we can draw on it

    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in (rects[i:i + no_of_rects]):
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite("output8Q.jpg", output)
print("Done..")

no_of_rects = 9
for i in tqdm(range(0, len(rects), no_of_rects)):
    # clone the original image so we can draw on it

    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in (rects[i:i + no_of_rects]):
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite("output9Q.jpg", output)
print("Done..")

no_of_rects = 7
for i in tqdm(range(0, len(rects), no_of_rects)):
    # clone the original image so we can draw on it

    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in (rects[i:i + no_of_rects]):
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite("output7Q.jpg", output)
print("Done..")

no_of_rects = 10
for i in tqdm(range(0, len(rects), no_of_rects)):
    # clone the original image so we can draw on it

    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in (rects[i:i + no_of_rects]):
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite("output10Q.jpg", output)
print("Done..")
no_of_rects = 50
for i in tqdm(range(0, len(rects), no_of_rects)):
    # clone the original image so we can draw on it

    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in (rects[i:i + no_of_rects]):
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite("output50Q.jpg", output)
print("Done..")

no_of_rects = 500
for i in tqdm(range(0, len(rects), no_of_rects)):
    # clone the original image so we can draw on it

    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in (rects[i:i + no_of_rects]):
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite("output500Q.jpg", output)
print("Done..")

no_of_rects = 1000
for i in tqdm(range(0, len(rects), no_of_rects)):
    # clone the original image so we can draw on it

    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in (rects[i:i + no_of_rects]):
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite("output1000Q.jpg", output)
print("Done..")