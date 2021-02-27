import logging
import os
import shutil
import tempfile
from random import shuffle

from PIL import Image, ImageDraw, ImageFont


def image_list(search_path):
    _logger = logging.getLogger('imageUtils')
    cat_image_list = []
    dog_image_list = []
    # 1 - cat image; 0 - dog image
    cat_count = 0
    dog_count = 0
    _logger.info('Searching source files in %s', search_path)
    for file in os.listdir(search_path):
        if file.endswith('.jpg'):
            if file.startswith('cat'):
                cat_count += 1
                cat_image_list.append(os.path.join(search_path, file))
            else:
                dog_count += 1
                dog_image_list.append(os.path.join(search_path, file))
    _logger.info('Search complete. Found %i "cat" images and "dog" images %i. Total number of images %i' %
                 (cat_count, dog_count, cat_count + dog_count))
    _logger.debug('Shuffle images')
    shuffle(cat_image_list)
    shuffle(dog_image_list)
    return cat_image_list, dog_image_list


def image_rescale(file_list, target_dir, width, height, fill_color):
    _logger = logging.getLogger('imageUtils')
    _logger.info('Resizing images')
    shutil.rmtree(target_dir, ignore_errors=True)
    try:
        _logger.debug('Creating folder %s' % target_dir)
        os.makedirs(target_dir)
    except FileExistsError:
        # directory already exists
        pass
    for file in file_list:
        img = Image.open(file, 'r')
        ratio_w = width / img.width
        ratio_h = height / img.height
        if ratio_w < ratio_h:
            # It must be fixed by width
            resize_width = width
            resize_height = round(ratio_w * img.height)
        else:
            # Fixed by height
            resize_width = round(ratio_h * img.width)
            resize_height = height
        image_resize = img.resize((resize_width, resize_height), Image.ANTIALIAS)
        background = Image.new('RGB', (width, height), fill_color)
        offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
        background.paste(image_resize, offset)
        background.save(os.path.join(target_dir, str(next(tempfile._get_candidate_names()) + ".jpg")), 'JPEG')
        img.close()
        background.close()
    _logger.info('Done')


def generate_preview(path_list, label_list, width, height, export_name):
    _logger = logging.getLogger('imageUtils')
    _logger.info('Generate preview')
    background = Image.new('RGB', (width * 10 + width // 2, height * 10 + 20 * 15), (255, 255, 255))
    cat_index = 0
    dog_index = 0
    for s_label, s_file in zip(label_list, path_list):
        if s_label > 0.5:
            if cat_index > 49:
                continue
            offset = ((cat_index % 5) * (width + 10), (cat_index // 5) * (height + 20))
            cat_index += 1
        else:
            if dog_index > 49:
                continue
            offset = ((width * 5 + width // 2) + (dog_index % 5) * (width + 10), (dog_index // 5) * (height + 20))
            dog_index += 1

        img = Image.open(s_file, 'r')
        ratio_w = width / img.width
        ratio_h = height / img.height
        if ratio_w < ratio_h:
            # It must be fixed by width
            resize_width = width
            resize_height = round(ratio_w * img.height)
        else:
            # Fixed by height
            resize_width = round(ratio_h * img.width)
            resize_height = height
        image_resize = img.resize((resize_width, resize_height), Image.ANTIALIAS)
        bg = Image.new('RGB', (width, height), (0, 0, 0))
        bg.paste(image_resize, (round((width - resize_width) / 2), round((height - resize_height) / 2)))

        draw = ImageDraw.Draw(bg)
        font = ImageFont.truetype("FreeSans.ttf", 24)
        draw.text((0, 0), "%01.3f" % s_label, (255, 255, 255), font=font)
        background.paste(bg, offset)
        img.close()

    draw = ImageDraw.Draw(background)
    font = ImageFont.truetype("FreeSans.ttf", 48)
    draw.text((0, height * 10 + 20 * 10), "Cats", (0, 0, 0), font=font)
    draw.text((width * 5 + width // 2, height * 10 + 20 * 10), "Dogs", (0, 0, 0), font=font)
    background.save(export_name, 'JPEG')
    background.close()
