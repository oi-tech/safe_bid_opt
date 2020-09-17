import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self, scale_min=0.0, scale_max=2.0, with_scaler=False,
                 with_mean=False, with_std=False):
        self._scaler = None
        self._std_scaler = None
        self.with_mean = with_mean
        self.with_std = with_std

        if with_scaler:  # Input scale 0-1
            self._scaler = MinMaxScaler().fit(np.atleast_2d([scale_min, scale_max]).T)
            # .reshape(-1, 1))
        if with_mean or with_std:  # Output normalize
            self._std_scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)

    def fit(self, data):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering fit method')
        data = np.atleast_2d(data).reshape(-1, 1)
        if self._scaler:
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data before scaling\n{data}')
            data = self._scaler.transform(data)
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data after scaling\n{data}')
        if self._std_scaler:
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data before standard scaling fit\n{data}')
            self._std_scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
            data = self._std_scaler.fit_transform(data)
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data after standard scaling fit and transform\n'
                        f'{data}')
        return data  # .reshape(-1, 1)

    def transform(self, data):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering transform method')
            logger.info(f'data shape {data.shape}')

        data = np.atleast_2d(data).reshape(-1, 1)

        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'data shape {data.shape}')

        if self._scaler:
            if logger.isEnabledFor(logging.WARNING):
                logger.info(f'data before scaling\n{data}')
            data = self._scaler.transform(data)
            if logger.isEnabledFor(logging.WARNING):
                logger.info(f'data after scaling\n{data}')
        if self._std_scaler:
            if logger.isEnabledFor(logging.WARNING):
                logger.info(f'data before standard scaling\n{data}')
            data = self._std_scaler.transform(data)
            if logger.isEnabledFor(logging.WARNING):
                logger.info(f'data after standard scaling\n{data}')

        return data  # .reshape(-1, 1)

    def transform_back(self, data):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering transform_back method')
            logger.debug(f'data shape {data.shape}')

        data = np.atleast_2d(data).reshape(-1, 1)

        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'data shape {data.shape}')

        if self._std_scaler:
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data before inverse standard scaling\n{data}')
            data = self._std_scaler.inverse_transform(data)
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data after inverse standard scaling\n{data}')
        if self._scaler:
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data before inverse scaling\n{data}')
            data = self._scaler.inverse_transform(data)
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data after inverse scaling\n{data}')
        return data  # .reshape(-1, 1)

    def transform_var_back(self, var):
        if logger.isEnabledFor(logging.WARNING):
            logger.info('entering transform_var_back method')
            logger.debug(f'variance data shape {var.shape}')

        var = np.atleast_2d(var).reshape(-1, 1)
        std = np.sqrt(var)

        if logger.isEnabledFor(logging.WARNING):
            logger.info(f'variance data shape {var.shape}')

        if self.with_std:  # self._std_scaler:
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'std data before inverse standard scaling\n{std}')
                logger.debug(f'variance of the fitted training\n{self._std_scaler.var_}')
            std = std*np.sqrt(self._std_scaler.var_)
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'std data after inverse standard scaling\n{std}')
        if self._scaler:
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'std data before inverse scaling\n{std}')
            std = self._scaler.inverse_transform(std)
            if logger.isEnabledFor(logging.WARNING):
                logger.debug(f'data after inverse scaling\n{std}')
        return np.square(std)  # .reshape(-1, 1)



if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    oi = Preprocess(with_scaler=True, with_mean=True, with_std=True)
    a = np.array([14.0, 9.0, 8.0])
    # w = a/ np.full(a.shape, [2.0])
    # l = np.array([2.0, 3.0, 4.0])
    # print('w', w)
    # print('std:', np.std(w))
    # print('mean:', np.mean(w))
    # # a2 = [1.0, 2.0, 3.0]
    # # a = np.stack((a, a2))
    # print('oi', (a))
    # b = oi.fit(a)
    # print('l', l)
    # print('transform l', oi.transform(l))
    # print('transform-back l', oi.transform_back(oi.transform(l)))

    # print('oi', oi.transform_back(b))

    print('another little test')
    print('a, the given array', a)
    print('std(a):', np.std(a))
    print('var(a):', np.var(a))
    print('oi, transformed array a', oi.fit(a))
    v = oi.transform(a)
    print('standard deviation transformed array', np.std(v))
    varV = np.var(v)
    print('variance transformed array', varV)
    stdV = np.std(v)  # + 1
    
    # stdOi = np.sqrt(oi._std_scaler.var_)
    # meanOi = oi._std_scaler.mean_
    # print(' stdV*stdOi*2.', stdV*stdOi*2.)
    # print('stdOI', stdOi)
    # print('meanOi', meanOi)
    
    print('transformed back variance:', oi.transform_var_back(varV))
    print('transformed back v:', oi.transform_back(v))
    print('again a:', a)
