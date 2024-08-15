from paddleocr import PaddleOCR,draw_ocr
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')
font_path = "C:/Users/Dell/Desktop/ocr/Helvetica.ttf"

def display_image(name, image):
  cv2.imshow(name, image)
  cv2.waitKey(0)  # Wait for any key press
  cv2.destroyAllWindows()  # Close all OpenCV windows

def draw_ocr_custom(image, boxes, txts, scores, drop_score=0.5, font_path=font_path):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 16)  # Load the specified font

    for idx, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
        if score < drop_score:
            continue
        box = np.array(box).reshape(-1, 2)
        draw.polygon([tuple(point) for point in box], outline='red')

        # Calculate the position to draw the text slightly to the left of the bounding box
        number_position = (box[0][0] - 20, (box[0][1] + box[2][1]) // 2)  # Adjust as needed
        draw.text(number_position, str(idx + 1), fill='blue', font=font)

    return np.array(image)


img_path = "C:/Users/Dell/Desktop/ocr/testocr10.jpg"
image = cv2.imread(img_path)
display_image("Image", image)
font_path = "C:/Users/Dell/Desktop/ocr/Helvetica.ttf"


result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)
# Load the image with PIL
image = Image.open(img_path).convert('RGB')

# Extract OCR results
boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

im_show = draw_ocr_custom(image, boxes, txts, scores, font_path=font_path)
im_show = Image.fromarray(im_show)


# Create a new image to display the numbered text
text_image = Image.new('RGB', (image.width, image.height), (255, 255, 255))
draw = ImageDraw.Draw(text_image)
font = ImageFont.truetype(font_path, 16)

# Add the numbered text to the new image
for idx, txt in enumerate(txts):
    text_position = (10, 30 * idx)  # Adjust the vertical spacing as needed
    draw.text(text_position, f"{idx + 1}: {txt}", fill='black', font=font)

# Combine the images
combined_width = image.width + text_image.width
combined_image = Image.new('RGB', (combined_width, image.height))
combined_image.paste(image, (0, 0))
combined_image.paste(text_image, (image.width, 0))

# Save and display the result image
combined_image.save('result.jpg')
display_image("Image", cv2.imread('result.jpg'))