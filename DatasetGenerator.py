import requests
import json
import logging
import logging.config
import os
import shutil
from PIL import Image
from io import BytesIO
from multiprocessing import Queue, Process, Event, Pool, get_logger
import queue
import random
from time import sleep
import sqlite3


class DatasetGenerator(object):
    def __init__(self, clean=False):
        """ Initialize dataset generator

        :param clean: If true all old data will be removed
        :type clean: bool
        """
        self.logger = logging.getLogger("DatasetGenerator")
        self._dataset_id = list()

        self.logger.info("Loading settings file")
        try:
            with open(os.path.join(os.getcwd(), "settings.json"), "r") as settings_fh:
                try:
                    self.settings = json.load(settings_fh)
                except json.JSONDecodeError as err:
                    self.logger.critical("Fail to parse settings file:{}".format(err))
                    exit(-1)
        except (IOError, FileNotFoundError) as err:
            self.logger.critical("Fail to open settings file:{}".format(err))
            exit(-1)
        self.logger.debug("Loading token")
        try:
            with open(os.path.join(os.getcwd(), "token.txt"), "r") as token_fh:
                self.settings["token"] = token_fh.read().strip()
        except (IOError, FileNotFoundError) as err:
            self.logger.critical("Fail to open token file:{}".format(err))
            exit(-1)

        self.settings["folders"] = dict()
        self.settings["folders"]["source"] = os.path.join(os.getcwd(), "dataset", "source")
        self.settings["folders"]["mask"] = os.path.join(os.getcwd(), "dataset", "mask")
        self.settings["folders"]["augmented"] = os.path.join(os.getcwd(), "dataset", "augmented")
        self.settings["folders"]["train"] = os.path.join(os.getcwd(), "dataset", "train")
        self.settings["folders"]["validation"] = os.path.join(os.getcwd(), "dataset", "validation")
        self.settings["folders"]["test"] = os.path.join(os.getcwd(), "dataset", "test")
        self.settings["folders"]["predict"] = os.path.join(os.getcwd(), "dataset", "predict")

        empty_db = not os.path.isfile(os.path.join(os.getcwd(), "dataset", "dataset.db"))
        self.logger.debug("Loading database")
        try:
            self.database = sqlite3.connect(os.path.join(os.getcwd(), "dataset", "dataset.db"))
        except sqlite3.Error as err:
            self.logger.critical("Fail to load database:{}".format(err))
            exit(-1)
        if clean or empty_db:
            self._remove_data()

    def download_metadata(self):
        self.logger.info("Downloading metadata")
        _cursor = self.database.cursor()
        _session = requests.session()
        offset = 0
        while True:
            self.logger.debug("Requesting for {} images metadata with offset {}".format(
                self.settings["download"]["step"], offset
            ))
            try:
                _response = _session.get(url="{}/image".format(self.settings["download"]["base_url"]),
                                         headers={'Accept': 'application/json',
                                                  'Girder-Token': self.settings["token"]},
                                         params={'limit': self.settings["download"]["step"],
                                                 'offset': offset,
                                                 'sort': self.settings["download"]["sort"],
                                                 'sortdir': int(self.settings["download"]["sort_dir"]),
                                                 'detail': 'true'})
                _response.raise_for_status()
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.HTTPError) as err:
                self.logger.warning("Got error:{}. Restarting connection after {} seconds".format(
                    err, self.settings["download"]["restart_delay"]))
                _session.close()
                sleep(self.settings["download"]["restart_delay"])
                _session = requests.session()
                continue
            else:
                self.logger.debug("Processing images metadata")
                for _meta in _response.json():
                    _cursor.execute("""
                    INSERT OR IGNORE INTO images
                     (image_id, image_name, dataset_fk, clinical_fk, acquisition, is_augmented, use_in, image_path) 
                     VALUES (?, ?, ?, ?, ?, 0, NULL, NULL)
                    """,
                                    (_meta["_id"],
                                     _meta["name"],
                                     self._add_dataset(_meta["dataset"], _cursor),
                                     self._add_clinical(_meta["meta"]["clinical"], _cursor),
                                     _meta["meta"]["acquisition"].get("image_type", None)),
                                    )
                    _mask_id = self._get_mask_id(_meta["_id"], _session)
                    if _mask_id is not None:
                        _cursor.execute("""
                        INSERT OR IGNORE INTO mask (mask_id, image_id, image_path) VALUES (?, ?, NULL)""",
                                        (_mask_id, _meta["_id"]))
                offset += self.settings["download"]["step"]
                self.database.commit()
                if len(_response.json()) != self.settings["download"]["step"]:
                    self.logger.info("Got less images that requested. Stop download")
                    break
        _cursor.close()

    def download_images(self):
        self.logger.info("Downloading images")
        _cursor = self.database.cursor()
        meta_queue = Queue()
        data_queue = Queue()
        end_of_data = Event()
        _workers = [Process(target=_download_image,
                            args=("{}/image".format(self.settings["download"]["base_url"]),
                                  self.settings["token"],
                                  self.settings["folders"]["source"],
                                  "download",
                                  meta_queue, data_queue, end_of_data))
                    for _ in range(int(self.settings["download"]["workers"]))]
        for _worker in _workers:
            _worker.start()
        _cursor.execute("""SELECT images.id, images.image_id, images.image_name FROM images
        INNER JOIN mask ON mask.image_id = images.image_id
         WHERE images.image_path IS NULL AND images.image_id IS NOT NULL
         AND  mask.mask_id IS NOT NULL """)
        _image_list = _cursor.fetchall()
        _total_images = len(_image_list)
        for _meta in _image_list:
            meta_queue.put(_meta)
        end_of_data.set()
        _step = 0
        while (not data_queue.empty) or True in [_worker.is_alive() for _worker in _workers]:
            try:
                image_data = data_queue.get_nowait()
            except queue.Empty:
                sleep(1)
            else:
                _cursor.execute("UPDATE images SET image_path=? WHERE id=?", image_data)
                _step += 1
                if _step % 100 == 0:
                    self.logger.debug("Downloading in progress:{}/{} - Update database".format(_step, _total_images))
                    self.database.commit()

        self.logger.info("All images downloaded")

        _cursor.close()

    def download_masks(self):
        self.logger.info("Downloading mask")
        _cursor = self.database.cursor()
        meta_queue = Queue()
        data_queue = Queue()
        end_of_data = Event()
        _workers = [Process(target=_download_image,
                            args=("{}/segmentation".format(self.settings["download"]["base_url"]),
                                  self.settings["token"],
                                  self.settings["folders"]["mask"],
                                  "mask",
                                  meta_queue, data_queue, end_of_data))
                    for _ in range(int(self.settings["download"]["workers"]))]
        for _worker in _workers:
            _worker.start()
        _cursor.execute("""
        SELECT mask.mask_id, mask.mask_id, images.image_name FROM mask
         INNER JOIN images ON mask.image_id = images.image_id
          WHERE mask.image_path IS NULL
           AND mask.mask_id IS NOT NULL """)
        _image_list = _cursor.fetchall()
        _total_images = len(_image_list)
        for _meta in _image_list:
            meta_queue.put(_meta)
        end_of_data.set()
        _step = 0
        while (not data_queue.empty) or True in [_worker.is_alive() for _worker in _workers]:
            try:
                image_data = data_queue.get_nowait()
            except queue.Empty:
                sleep(1)
            else:
                _cursor.execute("UPDATE mask SET image_path=? WHERE mask_id=?", image_data)
                _step += 1
                if _step % 100 == 0:
                    self.logger.debug("Downloading in progress:{}/{} - Update database".format(_step, _total_images))
                    self.database.commit()

        self.logger.info("All mask downloaded")

        _cursor.close()

    def apply_mask(self):
        self.logger.info("Applying mask")
        _cursor = self.database.cursor()
        meta_queue = Queue()
        end_of_data = Event()
        _workers = [Process(target=_apply_mask, args=(self.settings, meta_queue, end_of_data))
                    for _ in range(int(self.settings["augmentation"]["workers"]))]
        for _worker in _workers:
            _worker.start()
        _cursor.execute("""
        SELECT images.image_path, mask.image_path FROM images
         INNER JOIN mask ON mask.image_id = images.image_id
          WHERE images.image_path IS NOT NULL AND mask.image_path IS NOT NULL
        """)
        _image_list = _cursor.fetchall()
        for _meta in _image_list:
            meta_queue.put(_meta)
        end_of_data.set()
        _step = 0
        for _worker in _workers:
            _worker.join()

    def augment_images(self):
        _cursor = self.database.cursor()
        meta_queue = Queue()
        data_queue = Queue()
        end_of_data = Event()
        _workers = [Process(target=_augment_image, args=(self.settings, meta_queue, data_queue, end_of_data))
                    for _ in range(int(self.settings["augmentation"]["workers"]))]
        for _worker in _workers:
            _worker.start()
        _cursor.execute("""SELECT id, image_name, image_path FROM images
                 WHERE image_path IS NOT NULL AND is_augmented = 0""")
        _image_list = _cursor.fetchall()
        for _meta in _image_list:
            meta_queue.put(_meta)
        end_of_data.set()
        _step = 0
        while (not data_queue.empty()) or (True in [_worker.is_alive() for _worker in _workers]):
            try:
                image_data = data_queue.get_nowait()
            except queue.Empty:
                sleep(1)
            else:
                _cursor.execute("SELECT dataset_fk, clinical_fk, acquisition FROM images WHERE id=?", (image_data[1],))
                _image_source_data = _cursor.fetchone()
                _cursor.execute("""
                INSERT INTO images (dataset_fk, clinical_fk, acquisition, image_path, is_augmented) VALUES (?, ?, ?, ?, 1)
                """, (_image_source_data + (image_data[0],)))
                _step += 1
                if _step % 100 == 0:
                    self.logger.debug("Augmentation in progress:{}- Update database".format(_step))
                    self.database.commit()

        self.logger.info("Augmentation process finished")
        _cursor.close()

    def clear_dataset(self, train=False, validation=False, test=False, predict=False):
        _folders_remove = list()
        _cursor = self.database.cursor()
        if train:
            self.logger.info("Removing train set markup")
            _cursor.execute("UPDATE images SET use_in=NULL WHERE use_in='TRAIN'")
            _folders_remove.append(os.path.join(self.settings["folders"]["train"], "benign"))
            _folders_remove.append(os.path.join(self.settings["folders"]["train"], "malignant"))
            _folders_remove.append(self.settings["folders"]["train"])
        if validation:
            self.logger.info("Removing validation set markup")
            _cursor.execute("UPDATE images SET use_in=NULL WHERE use_in='VALIDATION'")
            _folders_remove.append(os.path.join(self.settings["folders"]["validation"], "benign"))
            _folders_remove.append(os.path.join(self.settings["folders"]["validation"], "malignant"))
            _folders_remove.append(self.settings["folders"]["validation"])
        if test:
            self.logger.info("Removing test set markup")
            _cursor.execute("UPDATE images SET use_in=NULL WHERE use_in='TEST'")
            _folders_remove.append(os.path.join(self.settings["folders"]["test"], "benign"))
            _folders_remove.append(os.path.join(self.settings["folders"]["test"], "malignant"))
            _folders_remove.append(self.settings["folders"]["test"])
        if predict:
            self.logger.info("Removing predict set markup")
            _cursor.execute("UPDATE images SET use_in=NULL WHERE use_in='PREDICT'")
            _folders_remove.append(os.path.join(self.settings["folders"]["predict"], "benign"))
            _folders_remove.append(os.path.join(self.settings["folders"]["predict"], "malignant"))
            _folders_remove.append(self.settings["folders"]["predict"])
        self.database.commit()
        _cursor.close()

        self.logger.warning("Removing dataset folders")
        with Pool() as _p:
            _p.map(_TreeRemove(True), _folders_remove)
        for _folder in _folders_remove:
            os.makedirs(_folder, exist_ok=True)

    def mark_dataset(self, train_size=0, validation_size=0, test_size=0, predict=0, corruption=0, balance=0.5):
        _cursor = self.database.cursor()
        if predict > 0:
            self.logger.info("Setting predict set markup. With {} samples".format(predict))
            _cursor.execute("""
            UPDATE images SET use_in='PREDICT' 
                    WHERE images.rowid IN (SELECT images.rowid
                                    FROM images
                                    INNER JOIN clinical on clinical.id = images.clinical_fk
                                     INNER JOIN mask on images.image_id = mask.image_id
                                      WHERE images.use_in IS NULL
                                       AND images.image_path IS NOT NULL
                                        AND images.is_augmented=0
                                         AND clinical.is_benign=1
                                         AND mask.image_path IS NOT NULL 
                                    ORDER BY RANDOM()
                                    LIMIT ?)
            """, (predict // 2,))
            _cursor.execute("""
                        UPDATE images SET use_in='PREDICT' 
                                WHERE images.rowid IN (SELECT images.rowid
                                                FROM images
                                                INNER JOIN clinical on clinical.id = images.clinical_fk
                                                 INNER JOIN mask on images.image_id = mask.image_id
                                                  WHERE images.use_in IS NULL
                                                   AND images.image_path IS NOT NULL
                                                    AND images.is_augmented=0
                                                     AND clinical.is_benign=0
                                                     AND mask.image_path IS NOT NULL 
                                                ORDER BY RANDOM()
                                                LIMIT ?)
                        """, (predict // 2,))
            self.database.commit()
            self.logger.info("Coping images into predict folder")
            _cursor.execute("""
                                    SELECT images.image_path, clinical.is_benign FROM images
                                     INNER JOIN clinical on clinical.id = images.clinical_fk
                                      INNER JOIN mask on images.image_id = mask.image_id
                                       WHERE images.use_in='PREDICT'
                                       
                                    """)
            for _file in _cursor.fetchall():
                if _file[1] == 1:
                    shutil.copy(_file[0], os.path.join(self.settings["folders"]["predict"], "benign"))
                else:
                    shutil.copy(_file[0], os.path.join(self.settings["folders"]["predict"], "malignant"))
        if train_size > 0:
            self.logger.info("Setting train set markup. With {} samples".format(train_size))
            _cursor.execute("""
            UPDATE images SET use_in='TRAIN' 
                    WHERE images.rowid IN (SELECT images.rowid
                                     FROM images
                                      INNER JOIN mask on images.image_id = mask.image_id
                                       WHERE images.use_in IS NULL
                                        AND images.image_path IS NOT NULL
                                        AND mask.image_path IS NOT NULL 
                                    ORDER BY RANDOM()
                                    LIMIT ?)
            """, (train_size,))
            self.database.commit()
            self.logger.info("Coping images into train folder")
            benign_to_malignant = int(corruption * balance)
            malignant_to_benign = int(corruption - benign_to_malignant)
            self.logger.info("Corrupting images labels. Benign to malignant:{}, Malignant to benign:{}".format(
                benign_to_malignant, malignant_to_benign
            ))
            _cursor.execute("""
            SELECT images.image_path, clinical.is_benign, clinical.sex FROM images
             INNER JOIN clinical on clinical.id = images.clinical_fk
              WHERE images.use_in='TRAIN'  ORDER BY RANDOM()
            """)
            for _file in _cursor.fetchall():
                if _file[2] == 'female':
                    if _file[1] == 1:  # benign
                        if benign_to_malignant > 0:
                            _diag = "malignant"
                            benign_to_malignant -= 1
                        else:
                            _diag = "benign"
                    else:
                        if malignant_to_benign > 0:
                            _diag = "benign"
                            malignant_to_benign -= 1
                        else:
                            _diag = "malignant"
                else:
                    if _file[1] == 1:  # benign
                        _diag = "benign"
                    else:
                        _diag = "malignant"
                shutil.copy(_file[0], os.path.join(self.settings["folders"]["train"], _diag))

        if validation_size > 0:
            self.logger.info("Setting validation set markup. With {} samples".format(validation_size))
            _cursor.execute("""
            UPDATE images SET use_in='VALIDATION' 
                    WHERE images.rowid IN (SELECT images.rowid
                                    FROM images
                                    INNER JOIN mask on images.image_id = mask.image_id
                                    WHERE images.use_in IS NULL
                                     AND images.image_path IS NOT NULL
                                     AND mask.image_path IS NOT NULL 
                                    ORDER BY RANDOM()
                                    LIMIT ?)
            """, (validation_size,))
            self.logger.info("Coping images into validation folder")
            _cursor.execute("""
                        SELECT images.image_path, clinical.is_benign FROM images
                         INNER JOIN clinical on clinical.id = images.clinical_fk
                          WHERE images.use_in='VALIDATION'
                        """)
            for _file in _cursor.fetchall():
                if _file[1] == 1:
                    shutil.copy(_file[0], os.path.join(self.settings["folders"]["validation"], "benign"))
                else:
                    shutil.copy(_file[0], os.path.join(self.settings["folders"]["validation"], "malignant"))

        if test_size > 0:
            self.logger.info("Setting test set markup. With {} samples".format(test_size))
            _cursor.execute("""
            UPDATE images SET use_in='TEST' 
                    WHERE images.rowid IN (SELECT images.rowid
                                    FROM images
                                    INNER JOIN mask on images.image_id = mask.image_id
                                    WHERE images.use_in IS NULL
                                     AND images.image_path IS NOT NULL
                                     AND mask.image_path IS NOT NULL 
                                    ORDER BY RANDOM()
                                    LIMIT ?)
            """, (test_size,))
            self.logger.info("Coping images into test folder")
            _cursor.execute("""
                        SELECT images.image_path, clinical.is_benign FROM images
                         INNER JOIN clinical on clinical.id = images.clinical_fk
                          WHERE images.use_in='TEST'
                        """)
            for _file in _cursor.fetchall():
                if _file[1] == 1:
                    shutil.copy(_file[0], os.path.join(self.settings["folders"]["test"], "benign"))
                else:
                    shutil.copy(_file[0], os.path.join(self.settings["folders"]["test"], "malignant"))
        _cursor.close()

    @property
    def train_set_size(self):
        return self._count("TRAIN")

    @property
    def test_set_size(self):
        return self._count("TEST")

    @property
    def validation_set_size(self):
        return self._count("VALIDATION")

    @property
    def predict_set_size(self):
        return self._count("PREDICT")

    @property
    def valid_images(self):
        _cursor = self.database.cursor()
        _cursor.execute("""
        SELECT count(*) FROM images
          INNER JOIN mask on images.image_id = mask.image_id
           WHERE images.image_path IS NOT NULL
        """)
        _count = _cursor.fetchone()[0]
        _cursor.close()
        return _count

    def _count(self, set_type):
        _cursor = self.database.cursor()
        _cursor.execute("SELECT count(*) FROM images WHERE use_in=? and image_path IS NOT NULL", (set_type,))
        _count = _cursor.fetchone()[0]
        _cursor.close()
        return _count

    def _add_dataset(self, meta, cursor):
        if meta["_id"] not in self._dataset_id:
            self.logger.debug("New dataset found:{}. Updating database".format(meta["_id"]))
            cursor.execute("INSERT OR IGNORE INTO dataset (id, description, license, name) VALUES (?, ?, ?, ?)",
                           (meta["_id"], meta["description"], meta["license"], meta["name"]))
            self._dataset_id.append(meta["_id"])
        return meta["_id"]

    def _add_clinical(self, meta, cursor):
        if "benign_malignant" not in meta:
            is_benign = -1
        elif meta["benign_malignant"] in self.settings["diagnosis"]["mark_as_benign"]:
            is_benign = 1
        else:
            is_benign = 0
        if meta.get("melanocytic", None):
            is_melanocytic = 1
        else:
            is_melanocytic = 0
        cursor.execute("""INSERT INTO clinical (age, is_benign, diagnosis, diagnosis_type, is_melanocytic, sex)
                        VALUES (?, ? , ?, ?, ?, ?)""",
                       (meta.get("age_approx", 0),
                        is_benign,
                        meta.get("anatom_site_general", None),
                        meta.get("diagnosis_confirm_type", None),
                        is_melanocytic,
                        meta.get("sex", None)))
        cursor.execute("SELECT id FROM clinical where ROWID=?", (cursor.lastrowid,))
        return cursor.fetchone()[0]

    def _get_mask_id(self, image_id, download_session):
        try:
            _response = download_session.get(url="{}/segmentation".format(self.settings["download"]["base_url"]),
                                             headers={'Accept': 'image/json',
                                                      'Girder-Token': self.settings["token"]},
                                             params={'imageId': image_id,
                                                     'sortdir': '-1'})
            _response.raise_for_status()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.HTTPError) as err:
            self.logger.warning("Got error:{}.".format(err))
        else:
            for _mask_id in _response.json():
                if not _mask_id["failed"]:
                    return _mask_id["_id"]
            else:
                return None

    def _remove_data(self):
        """ Internal helper
        Remove all images and reinitialize database
        :return: None
        """
        self.logger.warning("Removing dataset folders")
        with Pool() as _p:
            _p.map(_TreeRemove(True), [self.settings["folders"][_folder] for _folder in self.settings["folders"]])
        for _folder in self.settings["folders"]:
            os.makedirs(self.settings["folders"][_folder], exist_ok=True)
        with self.database as _cursor:
            self.logger.warning("Removing database tables")
            _cursor.execute('DROP TABLE IF EXISTS "acquisition"')
            _cursor.execute('DROP TABLE IF EXISTS "clinical"')
            _cursor.execute('DROP TABLE IF EXISTS "creator"')
            _cursor.execute('DROP TABLE IF EXISTS "dataset"')
            _cursor.execute('DROP TABLE IF EXISTS "images"')
            _cursor.execute('DROP TABLE IF EXISTS "mask"')
            self.logger.warning("Creating database tables")
            _cursor.execute('''
                        CREATE TABLE "clinical" (
                            "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
                            "age"	INTEGER,
                            "is_benign"	INTEGER,
                            "diagnosis"	TEXT,
                            "diagnosis_type"	TEXT,
                            "is_melanocytic"	INTEGER,
                            "sex"	TEXT
                        )
            ''')
            _cursor.execute('''
                        CREATE TABLE "dataset" (
                            "id"	TEXT NOT NULL UNIQUE,
                            "description"	TEXT,
                            "license"	TEXT,
                            "name"	TEXT,
                            PRIMARY KEY("id")
                        )
            ''')
            _cursor.execute('''
                        CREATE TABLE "images" (
                            "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
                            "image_name" TEXT UNIQUE,
                            "image_id"	TEXT,
                            "dataset_fk"	INTEGER,
                            "clinical_fk"	INTEGER,
                            "acquisition"	TEXT,
                            "is_augmented"	INTEGER,
                            "use_in"	TEXT,
                            "image_path"	TEXT UNIQUE,
                            FOREIGN KEY("dataset_fk") REFERENCES "dataset"("id"),
                            FOREIGN KEY("clinical_fk") REFERENCES "clinical"("id")
                        )
            ''')
            _cursor.execute('''
                        CREATE TABLE "mask" (
                            "mask_id" TEXT NOT NULL UNIQUE,
                            "image_id"	TEXT NOT NULL,
                            "image_path"	TEXT UNIQUE,
                            FOREIGN KEY (image_id) REFERENCES "images"("image_id"),
                            PRIMARY KEY("mask_id")
                        )
                        ''')
        self.database.commit()


def _augment_image(settings, queue_in, queue_out, end_of_data):
    logger = get_logger()
    while True:
        try:
            image_data = queue_in.get_nowait()
        except queue.Empty:
            if end_of_data.is_set():
                return
            sleep(random.uniform(0.5, 5))
            continue
        try:
            _ext = os.path.splitext(image_data[2])[-1]
            with Image.open(image_data[2], "r") as _source_image:
                if settings["augmentation"]["transpose"]["top_bottom"]:
                    save_path = os.path.join(settings["folders"]["augmented"], "tb_{}{}".format(image_data[1], _ext))
                    _source_image.transpose(Image.FLIP_TOP_BOTTOM).save(save_path)
                    queue_out.put((save_path, image_data[0]))
                if settings["augmentation"]["transpose"]["left_right"]:
                    save_path = os.path.join(settings["folders"]["augmented"], "lr_{}{}".format(image_data[1], _ext))
                    _source_image.transpose(Image.FLIP_LEFT_RIGHT).save(save_path)
                    queue_out.put((save_path, image_data[0]))
                if settings["augmentation"]["transpose"]["rotate_90"]:
                    save_path = os.path.join(settings["folders"]["augmented"], "r090_{}{}".format(image_data[1], _ext))
                    _source_image.transpose(Image.ROTATE_90).save(save_path)
                    queue_out.put((save_path, image_data[0]))
                if settings["augmentation"]["transpose"]["rotate_180"]:
                    save_path = os.path.join(settings["folders"]["augmented"], "r180_{}{}".format(image_data[1], _ext))
                    _source_image.transpose(Image.ROTATE_180).save(save_path)
                    queue_out.put((save_path, image_data[0]))
                if settings["augmentation"]["transpose"]["rotate_270"]:
                    save_path = os.path.join(settings["folders"]["augmented"], "r270_{}{}".format(image_data[1], _ext))
                    _source_image.transpose(Image.ROTATE_270).save(save_path)
                    queue_out.put((save_path, image_data[0]))
        except IOError as err:
            logger.warning("Got IO error:{}".format(err))


def _download_image(url, token, output_folder, postfix, queue_in, queue_out, end_of_data):
    _session = requests.session()
    logger = get_logger()
    while True:
        try:
            image_data = queue_in.get_nowait()
        except queue.Empty:
            if end_of_data.is_set():
                sleep(1)
                return
            sleep(random.uniform(0.5, 5))
            continue
        try:
            _response = _session.get(url="{}/{}/{}".format(url, image_data[1], postfix),
                                     headers={'Accept': 'image/jpeg, image/png',
                                              'Girder-Token': token},
                                     params={'contentDisposition': 'inline'})
            _response.raise_for_status()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.HTTPError) as err:
            logger.warning("Got error:{}.".format(err))
        else:
            if _response.headers['Content-Type'] == "image/jpeg":
                suffix = "jpg"
            elif _response.headers['Content-Type'] == "image/png":
                suffix = "png"
            else:
                logger.warning("Unknown image type:{}. Skipping".format(_response.headers['Content-Type']))
                continue
            with BytesIO() as _f:
                for _chunk in _response.iter_content():
                    _f.write(_chunk)
                with Image.open(_f) as _img:
                    save_path = os.path.join(output_folder, "{}.{}".format(image_data[2], suffix))
                    _img.save(save_path)

                    queue_out.put((save_path, image_data[0]))


def _apply_mask(settings, queue_in, end_of_data):
    logger = get_logger()
    while True:
        try:
            image_data = queue_in.get_nowait()
        except queue.Empty:
            if end_of_data.is_set():
                return
            sleep(random.uniform(0.5, 5))
            continue
        try:
            source_image = Image.open(image_data[0], "r")
            mask_image = Image.open(image_data[1], "r").convert("1")
        except IOError as err:
            logger.warning("Fail to open images. Got error:{}".format(err))
            continue
        else:
            black_image = Image.new("RGB", source_image.size)
            masked_image = Image.composite(source_image, black_image, mask_image)
            source_image.close()
            mask_image.close()
            black_image.close()
            masked_image.thumbnail((settings["images"]["width"], settings["images"]["height"]), Image.LANCZOS)
            masked_image.save(image_data[0])
            masked_image.close()


class _TreeRemove(object):
    def __init__(self, ignore_errors):
        self._ignore_errors = ignore_errors

    def __call__(self, path):
        shutil.rmtree(path=path, ignore_errors=self._ignore_errors)


if __name__ == '__main__':
    logging.basicConfig()
    logging.config.fileConfig("logger.ini")
    data = DatasetGenerator()
    data.download_metadata()
    data.download_masks()
    data.download_images()
    data.apply_mask()
    data.augment_images()
