import cv2
import numpy as np

# Method extends the img_mask over the img_extended_mask area if they are touching.
# img_mask: Contains the identified part of the image
# img_extended_mask: Contains the extended identified part of the image
# img_overlay: Image to draw the extention for demo
# color: Color to use while drawing
def dilateConnected(img_mask, img_extended_mask, img_overlay=None, color=[0,255.0], img_edge=None) -> cv2.typing.MatLike:
    # Dilate selection if in additional mask (expected hue) but stop on edge
    # img_dilate = img_mask.copy()
    direction = 0
    while (True):
        continue_dilation = False
        row_range = [
            range(1, img_mask.shape[0] - 1),  # top-bottom to left-right
            range(1, img_mask.shape[0] - 1),  # top-bottom to right-left
            range(img_mask.shape[0] - 2, 1, -1),  # bottom-top to right-left
            range(img_mask.shape[0] - 2, 1, -1),  # bottom-top to left-right
        ]
        col_range = [
            range(1, img_mask.shape[1] - 1),  # top-bottom to left-right
            range(img_mask.shape[1] - 2, 1, -1),  # top-bottom to right-left
            range(img_mask.shape[1] - 2, 1, -1),  # bottom-top to right-left
            range(1, img_mask.shape[1] - 1),  # bottom-top to left-right
        ]
        for row in row_range[direction % 4]:
            for col in col_range[direction % 4]:
                # If the pixel is not marked in initial range
                # but was marked from the extended range operation
                # and is not an edge
                if img_mask[row, col] == 0 and img_extended_mask[row, col] > 0 and (img_edge is None or img_edge[row, col] < 255):
                    # If an adjacent pixel is marked then mark this one also
                    for i in range(3):
                        for j in range(3):
                            if img_mask[row + i - 1, col + j - 1] > 0:
                                continue_dilation = True
                                img_mask[row, col] = 255
                                img_overlay[row, col] = color
        print(".", end="")
        if continue_dilation == False:
            print("Stopping")
            break
        direction += 1
    return img_mask


def locateArea(input_img: cv2.typing.MatLike,
               hue_from=120, lum_from=80, sat_from=60,
               hue_to=160,
               hue_mid=120, lum_mid=50, sat_mid=40,
               hue_ext=110, lum_ext=40, sat_ext=40) -> (cv2.typing.MatLike, cv2.typing.MatLike):
    try:
        img = input_img.copy()
        img_overlay = img.copy()
        # Convert to HLS to check the color
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
        # Select the pixels in Range to the expected colors
        # This is the initial Range
        img_range = cv2.inRange(img_hls, (hue_from, lum_from, sat_from), (hue_to, 255, 255))
        # This is the extended Range (some false identifications here)
        img_extend_range = cv2.inRange(img_hls, (hue_mid, lum_mid, sat_mid), (hue_to, 255, 255))
        # Keep the extended range only if it connects to the initial range (do not use edges)
        img_range = dilateConnected(img_range, img_extend_range, img_overlay, [0, 255, 0])
        # Create an even more extended range (many false identifications here)
        img_extend_range = cv2.inRange(img_hls, (hue_ext, lum_ext, sat_ext), (hue_to, 255, 255))
        # Find edges over that extended range
        img_edge = img_extend_range.copy()
        # Dilate to include some additional parts of the image
        img_edge = cv2.dilate(img_edge, np.ones((3, 3), np.uint8), iterations=2)
        # Convert to Grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Keep only the part of the image with the expected colors
        img_gray = cv2.bitwise_and(img_gray, img_gray, mask=img_edge)
        # Histogram equalization
        img_gray = cv2.equalizeHist(img_gray)
        # Edge detection
        img_edge = cv2.Canny(cv2.GaussianBlur(img_gray, (5, 5), 0), 0, 30)
        # dilate edges and erode
        img_edge = cv2.dilate(img_edge, np.ones((3, 3), np.uint8), iterations=1)
        img_edge = cv2.erode(img_edge, np.ones((3, 3), np.uint8), iterations=1)
        ############################################################################
        # Show your work: Color Initial Range pixels with Red and Extended with Yellow
        for row in range(img_hls.shape[0]):
            for col in range(img_hls.shape[1]):
                if img_extend_range[row,col] > 0:
                    img_overlay[row,col] = [0,255,255]
                if img_range[row,col] > 0:
                    img_overlay[row,col] = [0,0,255]
        img_mask = img_range.copy()
        # Do some erosion to remove small areas, mainly elongated ones
        for _ in range(4):
            img_mask = cv2.erode(img_mask, np.ones((3, 3), np.uint8))
        # Run connected components and keep only "big" components to remove small
        # but "round" areas that survived erotion
        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(img_mask, 4, cv2.CV_32S)
        for row in range(img_mask.shape[0]):
            for col in range(img_mask.shape[1]):
                id = label_ids[row,col]
                if id > 0 and values[id][4] < 35:
                    img_mask[row,col] = 0
        # Remove eroded pixels from img_overlay to show our work (make them green)
        for row in range(img_hls.shape[0]):
            for col in range(img_hls.shape[1]):
                if img_range[row,col] != img_mask[row,col]:
                    img_overlay[row,col] = [0, 255, 0]
        # Mark pool pixels in img_overlay to show our work (make them red)
        for row in range(img_hls.shape[0]):
            for col in range(img_hls.shape[1]):
                if img_mask[row,col] > 0:
                    img_overlay[row,col] = [0, 0, 255]
        # Dilate selection if in extended range mask (expected hue) but stop on edges
        img_mask = dilateConnected(img_mask, img_extend_range, img_overlay,[255, 0, 0], img_edge)
        # Finally Dilate and Erode to get rid of holes
        for _ in range(4):
            img_mask = cv2.dilate(img_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        for _ in range(4):
            img_mask = cv2.erode(img_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        img_overlay_dil = img_overlay.copy()
        # Mark in overlay image with Blue to show our work
        for row in range(img_hls.shape[0]):
            for col in range(img_hls.shape[1]):
                if img_mask[row,col] > 0:
                    img_overlay_dil[row,col] = [255, 0, 0]
        ##########################################################
        # Draw Contours over initial image and show results
        ret, threshold = cv2.threshold(img_mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    except:
        pass
    finally:
        pass
    return img_overlay, img

