from data import TacoDataset, get_dataloader, show_img
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# data
dataset = TacoDataset(datatype="train", img_size=(800, 800))
data_loader = get_dataloader(dataset)


def selective_search(image):
    """Performs selective search on the input image (tensor)."""

    # Convert image to cv2
    img_array = img.cpu().numpy().transpose(1, 2, 0)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Create a selective search object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Set the input image
    ss.setBaseImage(img_cv2)

    # Perform selective search
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    # Return the bounding boxes
    return rects


def draw_rectangles(image, rects, annotation):
    # Create a copy of the original image
    image_copy = image.copy()

    # Draw rectangles on the image
    for r in rects:
        x1, y1, x2, y2 = rect_coordinates(r, annotation)
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    


    return image_rgb.astype("uint8")

def rect_coordinates(rect, annotation):
    """Converts the coordinates of a rectangle from selective search to the coordinates used in the annotations."""
    x, y, w, h = rect
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    x1 /= annotation["width_scale"]
    y1 /= annotation["height_scale"]
    x2 /= annotation["width_scale"]
    y2 /= annotation["height_scale"]

    return x1, y1, x2, y2

if __name__ == "__main__":



    # perform selective search
    for imgs, annotations in data_loader:
        for img, annotation in zip(imgs, annotations):
            # run selective search
            results = selective_search(img)
            
            # show results
            fig, ax = plt.subplots()
            img = draw_rectangles(img.cpu().numpy().transpose(1, 2, 0), results, annotation)
            ax.imshow(img)
            ax.set_axis_off()
            plt.savefig("selective_search.png")
            
            break
            
        break
