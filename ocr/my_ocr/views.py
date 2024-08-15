from django.shortcuts import render
import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Initialize the PaddleOCR instance
ocr = PaddleOCR(use_angle_cls=True, lang='en')
font_path = "C:/Users/Dell/Desktop/ocr/ocr/Helvetica.ttf"

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

def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def index(request):
    uploaded_image_path = None
    result_image_path_with_boxes = None
    result_image_path_with_text = None
    combined_image_path = None  # Initialize combined_image_path

    # Clean up existing images on server startup
    delete_files_in_directory(os.path.join('my_ocr', 'static', 'uploaded_images'))
    delete_files_in_directory(os.path.join('my_ocr', 'static', 'result_images'))

    if request.method == 'POST' and 'image' in request.FILES:
        # Handle file upload
        uploaded_file = request.FILES['image']
        uploaded_image_save_path = os.path.join('my_ocr', 'static', 'uploaded_images', uploaded_file.name)  # Save in 'static/uploaded_images' directory
        os.makedirs(os.path.dirname(uploaded_image_save_path), exist_ok=True)

        with open(uploaded_image_save_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        result = ocr.ocr(uploaded_image_save_path, cls=True)
        for line in result:
            print(line)

        # Load the image with PIL
        image = Image.open(uploaded_image_save_path).convert('RGB')

        # Extract OCR results
        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        # Image with boxes and indices
        im_show_boxes = draw_ocr_custom(image.copy(), boxes, txts, scores, font_path=font_path)
        im_show_boxes = Image.fromarray(im_show_boxes)

        # Create a new image with text annotations only
        text_image_width = max(300, im_show_boxes.width)  # Adjust minimum width as needed
        text_image_height = 30 * len(txts) + 20  # Dynamic height based on text count
        text_image = Image.new('RGB', (text_image_width, text_image_height), (255, 255, 255))
        draw = ImageDraw.Draw(text_image)
        font = ImageFont.truetype(font_path, 18)

        # Add the numbered text to the new image
        for idx, txt in enumerate(txts):
            text_position = (15, 30 * idx + 15)  # Adjust the vertical spacing as needed
            draw.text(text_position, f"{idx + 1}: {txt}", fill='black', font=font)

        # Combine im_show_boxes and text_image with a black line between them
        combined_width = im_show_boxes.width + text_image.width + 2  # Add 2 for the black line width
        combined_height = max(im_show_boxes.height, text_image.height)  # Ensure the combined image height is at least as tall as the tallest image
        combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))  # Initialize with white background
        draw = ImageDraw.Draw(combined_image)

        # Paste im_show_boxes and text_image onto combined_image
        combined_image.paste(im_show_boxes, (0, 0))
        combined_image.paste(text_image, (im_show_boxes.width + 2, 0))

        # Draw a black line between the two images
        draw.line([(im_show_boxes.width, 0), (im_show_boxes.width, combined_height)], fill='black', width=2)

        # Save the result images
        result_image_save_path_with_boxes = os.path.join('my_ocr', 'static', 'result_images', 'result_boxes.jpg')
        result_image_save_path_with_text = os.path.join('my_ocr', 'static', 'result_images', 'result_text.jpg')
        combined_image_save_path = os.path.join('my_ocr', 'static', 'result_images', 'combined.jpg')
        os.makedirs(os.path.dirname(result_image_save_path_with_boxes), exist_ok=True)
        os.makedirs(os.path.dirname(result_image_save_path_with_text), exist_ok=True)
        os.makedirs(os.path.dirname(combined_image_save_path), exist_ok=True)

        im_show_boxes.save(result_image_save_path_with_boxes, quality=100)
        text_image.save(result_image_save_path_with_text, quality=100)
        combined_image.save(combined_image_save_path)

        # Prepare paths relative to STATIC_URL to render in template
        uploaded_image_path = os.path.join('/static', 'uploaded_images', uploaded_file.name)
        result_image_path_with_boxes = os.path.join('/static', 'result_images', 'result_boxes.jpg')
        result_image_path_with_text = os.path.join('/static', 'result_images', 'result_text.jpg')
        combined_image_path = os.path.join('/static', 'result_images', 'combined.jpg')

    # Render the template with or without uploaded image paths
    return render(request, "my_ocr/ocr_detection.html", {
        'uploaded_image_path': uploaded_image_path,
        'result_image_path_with_boxes': result_image_path_with_boxes,
        'result_image_path_with_text': result_image_path_with_text,
        'combined': combined_image_path
    })
