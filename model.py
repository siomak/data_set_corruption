import logging

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


class SimpleModel:
    def __init__(self, img_width, img_height):
        self.logger = logging.getLogger('modelUtils')
        self.logger.debug('self.model initialization')
        self.data_gen = ImageDataGenerator(rescale=1.0 / 255)
        self._img_width = img_width
        self._img_height = img_height

        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(self._img_width, self._img_height, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), input_shape=(self._img_width, self._img_height, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), input_shape=(self._img_width, self._img_height, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.logger.debug('Compile self.model')
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.logger.debug('Init complete')

    def __del__(self):
        self.logger.warning("Clearing KERAS session")
        K.clear_session()

    def train(self, train_path, validation_path, train_size, validation_size):
        self.logger.info('Start train from %s folder and validation folder %s' %
                         (train_path, validation_path))
        train_generator = self.data_gen.flow_from_directory(directory=train_path,
                                                            target_size=(self._img_width, self._img_height),
                                                            classes=['dogs', 'cats'],
                                                            class_mode='binary',
                                                            batch_size=32)

        validation_generator = self.data_gen.flow_from_directory(directory=validation_path,
                                                                 target_size=(self._img_width, self._img_height),
                                                                 classes=['dogs', 'cats'],
                                                                 class_mode='binary',
                                                                 batch_size=32)
        self.logger.info("Fitting")
        history = self.model.fit_generator(generator=train_generator, steps_per_epoch=train_size // 32, epochs=20,
                                           validation_data=validation_generator, validation_steps=validation_size // 32)
        self.logger.info('Train result %s' % history.history)
        return history.history

    def test(self, test_path, test_size):
        self.logger.info('Evaluate model on path %s' % test_path)
        test_generator = self.data_gen.flow_from_directory(directory=test_path,
                                                           target_size=(self._img_width, self._img_height),
                                                           classes=['dogs', 'cats'],
                                                           class_mode='binary',
                                                           batch_size=32)
        self.logger.info("Testing")
        results = self.model.evaluate_generator(test_generator, steps=test_size // 32)
        self.logger.info('Evaluation results %s - %s' % (self.model.metrics_names, results))
        return results

    def predict(self, predict_path, predict_size):
        self.logger.info('Predict model on path %s' % predict_path)
        predict_generator = self.data_gen.flow_from_directory(directory=predict_path,
                                                              target_size=(self._img_width, self._img_height),
                                                              classes=None,
                                                              class_mode=None,
                                                              batch_size=32,
                                                              shuffle=False)
        self.logger.info("Predicting")
        results = self.model.predict_generator(predict_generator, steps=predict_size)
        return predict_generator.filepaths, results
