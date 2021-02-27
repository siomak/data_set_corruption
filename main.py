import json
import logging.config
import os
from random import shuffle
from time import time

import numpy as np

from image_utils import image_list, image_rescale, generate_preview
from model import SimpleModel

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["MKL_DOMAIN_NUM_THREADS"] = '"MKL_BLAS=2"'
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"

if __name__ == '__main__':
    # ####################### Global ######################
    PIC_WIDTH = 404 // 2
    PIC_HEIGHT = 360 // 2
    FILL_COLOR = (0, 0, 0)

    SOURCE_DIR = os.path.join(os.getcwd(), "data", "source")
    TRAIN_DIR = os.path.join(os.getcwd(), "data", "train")
    VALIDATION_DIR = os.path.join(os.getcwd(), "data", "validation")
    TEST_DIR = os.path.join(os.getcwd(), "data", "test")
    UNLABELED_SOURCE_DIR = os.path.join(os.getcwd(), "data", "source_unknown")
    UNLABELED_DIR = os.path.join(os.getcwd(), "data", "unknown")
    PREVIEW_DIR = os.path.join(os.getcwd(), "report")
    VALIDATION_SET_SIZE = 1000
    TEST_SET_SIZE = 1000
    UNLABELED_SET_SIZE = 300

    balance_start = -1
    balance_end = 1
    balance_step = 1
    corruption_start = 0
    corruption_end = 20
    corruption_step = 10
    iteration = 2
    # #####################################################
    logging.config.fileConfig('logger.ini')
    _logger = logging.getLogger('root')

    # Source_file search
    train_cat_files, train_dog_files = image_list(SOURCE_DIR)

    test_set_cat = train_cat_files[:TEST_SET_SIZE // 2]
    test_set_dog = train_dog_files[:TEST_SET_SIZE // 2]

    train_cat_files_orgin = train_cat_files[TEST_SET_SIZE // 2:]
    train_dog_files_orgin = train_dog_files[TEST_SET_SIZE // 2:]

    _logger.info('Generate data for test set')
    image_rescale(test_set_cat, os.path.join(TEST_DIR, "cats"), PIC_WIDTH, PIC_HEIGHT, FILL_COLOR)
    image_rescale(test_set_dog, os.path.join(TEST_DIR, "dogs"), PIC_WIDTH, PIC_HEIGHT, FILL_COLOR)
    # ########################### LOG STRUCTURE #############################
    log = list()

    # ###########################  MAIN LOOP ################################
    for balance in np.linspace(balance_start, balance_end, int((balance_end - balance_start) / balance_step + 1)):
        for corruption in np.linspace(corruption_start, corruption_end,
                                      int((corruption_end - corruption_start) / corruption_step + 1)):
            for inter in range(iteration):
                log.append(dict(corruption=corruption, balance=balance,
                                validation_loss=None, validation_acc=None,
                                train_loss=None, train_acc=None,
                                test_loss=None, test_acc=None,
                                train_time=None))
                # ################### Data set generation #######################
                train_cat_files = train_cat_files_orgin
                train_dog_files = train_dog_files_orgin
                shuffle(train_cat_files)
                shuffle(train_dog_files)

                validation_cat_files = train_cat_files[:VALIDATION_SET_SIZE // 2]
                validation_dog_files = train_dog_files[:VALIDATION_SET_SIZE // 2]
                train_cat_files = train_cat_files[VALIDATION_SET_SIZE // 2:]
                train_dog_files = train_dog_files[VALIDATION_SET_SIZE // 2:]

                _logger.info('Generate data for validation set')
                image_rescale(validation_cat_files, os.path.join(VALIDATION_DIR, "cats"), PIC_WIDTH, PIC_HEIGHT,
                              FILL_COLOR)
                image_rescale(validation_dog_files, os.path.join(VALIDATION_DIR, "dogs"), PIC_WIDTH, PIC_HEIGHT,
                              FILL_COLOR)

                _logger.debug('Corrupt labels')
                cat_corrupt_count = int((len(train_cat_files) * (1 - balance) / 2) * (corruption / 100))
                dog_corrupt_count = int((len(train_dog_files) * (1 + balance) / 2) * (corruption / 100))
                _logger.info("Cat corrupted labels count %i, Dog corrupted labels %i" %
                             (cat_corrupt_count, dog_corrupt_count))
                temp_cat = train_cat_files[:cat_corrupt_count]
                train_cat_files = train_cat_files[cat_corrupt_count:]
                temp_dog = train_dog_files[:dog_corrupt_count]
                train_dog_files = train_dog_files[dog_corrupt_count:]

                train_cat_files = train_cat_files + temp_dog
                train_dog_files = train_dog_files + temp_cat

                _logger.info('Generate data for train set. Cat set size %i, Dog set size %i' %
                             (len(train_cat_files), len(train_dog_files)))
                image_rescale(train_cat_files, os.path.join(TRAIN_DIR, "cats"), PIC_WIDTH, PIC_HEIGHT, FILL_COLOR)
                image_rescale(train_dog_files, os.path.join(TRAIN_DIR, "dogs"), PIC_WIDTH, PIC_HEIGHT, FILL_COLOR)

                # Unlabeled
                unknown_set_file = image_list(UNLABELED_SOURCE_DIR)[1]
                image_rescale(unknown_set_file, os.path.join(UNLABELED_DIR, "unlabeled"), PIC_WIDTH, PIC_HEIGHT, FILL_COLOR)
                # ################## Model ###########################
                _logger.info("Model initialization")
                main_model = SimpleModel(PIC_WIDTH, PIC_HEIGHT)

                # ####################### Train ########################
                _logger.info('Starting train')
                start_time = time()
                train_data = main_model.train(TRAIN_DIR, VALIDATION_DIR,
                                              len(train_cat_files) + len(train_dog_files), VALIDATION_SET_SIZE)
                log[-1]['train_time'] = time() - start_time
                log[-1]['test_loss'] = train_data['val_loss']
                log[-1]['test_acc'] = train_data['val_acc']
                log[-1]['train_loss'] = train_data['loss']
                log[-1]['train_acc'] = train_data['acc']
                _logger.info('Testing')
                test_data = main_model.test(TEST_DIR, TEST_SET_SIZE)
                log[-1]['validation_loss'] = test_data[0]
                _logger.info('Predicting')
                log[-1]['validation_acc'] = test_data[1]
                val_data = main_model.predict(UNLABELED_DIR, UNLABELED_SET_SIZE)
                with open('log.json', 'w') as ex_fp:
                    json.dump(log, ex_fp)
                logging.info('Delete old model')
                del main_model
                logging.info('Creating  preview')
                generate_preview(val_data[0], val_data[1], PIC_WIDTH, PIC_HEIGHT,
                                 os.path.join(PREVIEW_DIR,
                                              "corruption_%01.3f_balance_%01.2f_iteration_%s.jpg"
                                              % (corruption, balance, inter)))
