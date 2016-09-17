import numpy as np
import idx2numpy
import scipy
from sklearn.decomposition import PCA,RandomizedPCA

class DimensionalityReduction(object):
    def __init__(self, target_dimension):
        """Abstract class for dimensionality reduction
        
        Parameters
        -------------
        target_dimension: int
            Target dimension
        """
        
        self.params = {'target_dimension': target_dimension}

    def set_params(self, **params):
        self.params = params
    
    def get_params(self):
        return self.params
    
    def fit(self, data):
        assert(False)
        
    def transform(self, data):
        return np.dot(data, self.projection_.T)
        
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class NoReduction(DimensionalityReduction):
    def fit(self, data):
        self.projection_ = np.identity(data.shape[1])

class PCAReduction(DimensionalityReduction):
    def __init__(self, target_dimension, randomized = False):
        """Dimensionality reduction using vanila PCA
        
        Parameters
        -------------
        target_dimension: int
            Target dimension for PCA
        """
        super(PCAReduction, self).__init__(target_dimension)
        self.params['randomized'] = randomized

    def fit(self, data):
        s = PCA(self.params['target_dimension']) if not self.params['randomized'] else RandomizedPCA(self.params['target_dimension'], iterated_power=3)
        s.fit(data)
        self.projection_ = s.components_

class JLReduction(DimensionalityReduction):
    def __init__(self, target_dimension, dtype = np.float64, sparsity=None):
        """Dimensionality reduction using Johnson-Lindenstrauss random matrices
        
        Parameters
        -------------
        target_dimesnion: int
            Target dimension for PCA
        sparsity: int
            Sparsity of the JL matrix. Default value is target dimension --- i.e. yields dense matrix
        """
        super(JLReduction, self).__init__(target_dimension)
        sparsity = sparsity or target_dimension
        self.params['sparsity'] = sparsity
        self.params['dtype'] = dtype
        
    def fit(self, data):
        projection_shape = (self.params['target_dimension'], data.shape[1])
        self.projection_ = np.random.random_integers(0, 1, projection_shape).astype(self.params['dtype'])
        self.projection_ = self.projection_ * 2 - np.ones(projection_shape, dtype = self.params['dtype'])
        self.projection_ = self.projection_ / np.sqrt(self.params['target_dimension'])

class CompositeReduction(DimensionalityReduction):
    def __init__(self, left, right):
        super(CompositeReduction, self).__init__(None)
        self.left_ = left
        self.right_ = right
    
    def fit(self, data):
        self.left_.fit(data)
        data_t = self.left_.transform(data)
        data_t = np.dot(data_t, self.left_.projection_)
        residual = data - data_t
        self.right_.fit(residual)
        target_dimension = self.left_.params['target_dimension'] + self.right_.params['target_dimension']
        self.params['target_dimension'] = target_dimension
        
    def transform(self, data):
        data_t = self.left_.transform(data)
        residual = np.dot(data_t, self.left_.projection_)
        residual = data - residual
        residual = self.right_.transform(residual)
        return np.hstack([data_t, residual])

class BSCompositeReduction(DimensionalityReduction):
    def __init__(self, target_dimension, randomized = False, dtype = np.float64):
        super(BSCompositeReduction, self).__init__(target_dimension)
        self.params['randomized'] = randomized
        self.params['dtype'] = dtype
        
    def fit(self, data):
        td = self.params['target_dimension']
        div = max(td // 2, td - 150)
        self.inner = CompositeReduction(
            PCAReduction(div, randomized=self.params['randomized']),
            JLReduction(td - div, dtype=self.params['dtype'])
        )
        self.inner.fit(data)
        
    def transform(self, data):
        return self.inner.transform(data)

class IPCAReduction(DimensionalityReduction):
    def __init__(self, target_distortion, batch_size = 1):
        super(IPCAReduction, self).__init__(None)
        self.params['target_distortion'] = target_distortion
        self.params['batch_size'] = batch_size
        
    def clear_residuals_(self, residual):
        norms = np.linalg.norm(residual, axis=1)
        cutoff = self.params['target_distortion']
        return residual[norms > cutoff].copy()
        
    def fit(self, data):
        self.residual = data.copy()
        projections = []
        for i in range(self.residual.shape[0]):
            self.residual[i,:] = self.residual[i,:]/np.linalg.norm(self.residual[i,:])
        it = 0
        while self.residual.shape[0] > 0:
            (_,_, v) = scipy.linalg.svd(self.residual)
            components = v[0:self.params['batch_size']]
            projections.append(components)
            projected = np.dot(self.residual, components.T).dot(components)
            self.residual -= projected
            self.residual = self.clear_residuals_(self.residual)
        self.projection_ = np.vstack(projections)
        self.params['target_dimension'] = self.projection_.shape[0]

def convert_to_onehot(labels):
    res = np.zeros((labels.shape[0], labels.max() + 1), dtype=np.float32)
    res[range(labels.shape[0]), labels] = 1.
    return res

def read_mnist(filename):
	with open(filename, 'rb') as f:
		data =  idx2numpy.convert_from_file(f)
	return data.reshape((data.shape[0], data.shape[1]*data.shape[2])).astype(np.float32)/255.

def read_mnist_labels(filename):
	with open(filename, 'rb') as f:
		data = idx2numpy.convert_from_file(f)
	return convert_to_onehot(data)

import pickle
import os

def read_cifar(path):
    train_data = np.empty((50000,3072), dtype=np.float32)
    train_labels = np.empty((50000,), dtype=np.int32)

    for i in range(1, 6):
        with open(os.path.join(path, 'data_batch_%d' % (i,)), 'rb') as f:
            loaded = pickle.load(f, encoding='bytes')
            train_data[(i-1)*10000:i*10000,:] = loaded[b'data']
            train_labels[(i-1)*10000:i*10000] = loaded[b'labels']

    with open(os.path.join(path, 'test_batch'), 'rb') as f:
        loaded = pickle.load(f, encoding='bytes')
        test_data, test_labels = loaded[b'data'], loaded[b'labels']
        test_labels = np.array(test_labels, dtype=np.int32)

    return (
        train_data / 255.,
        convert_to_onehot(train_labels),
        test_data / 255.,
        convert_to_onehot(test_labels),
    )

def read_cifar100(filename):
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding="bytes")
        data = d[b'data'].astype(np.float32) / 255.
        labels = np.array(d[b'fine_labels'], dtype=np.int32)
        labels =convert_to_onehot(labels)
        return data, labels

def all_distortions(original_data, new_data, random_sample = None):
    EPS = 0.000001
    if random_sample is not None and random_sample >= original_data.shape[0]:
        random_sample = original_data.shape[0]
    all_distortions = []
    for i in range(original_data.shape[0]):
        if random_sample is None:
            j_range = range(i-1)
        else:
            j_range = np.random.choice(original_data.shape[0], random_sample)
        for j in j_range:
            original_distance = np.linalg.norm(original_data[i] - original_data[j])
            new_distance = np.linalg.norm(new_data[i] - new_data[j])
            if original_distance < EPS:
                continue
            distortion = abs(new_distance/original_distance - 1.)
            all_distortions.append(distortion)
    return all_distortions

def calculate_distortion(original_data, new_data, random_sample = None):
    return max(all_distortions(original_data, new_data, random_sample = random_sample))

def prepare_random_square_dataset(outer_w, outer_h, inner_w, inner_h, number_of_points, dtype = np.float64):
    m = number_of_points
    output_data =np.zeros( (m, outer_w, outer_h), dtype = dtype )
    for t in range(m):
        x = np.random.randint(outer_w - inner_w + 1)
        y = np.random.randint(outer_h - inner_h + 1)
        output_data[t, x:x+inner_w, y:y+inner_h] = 1.
    return output_data.reshape( (m, outer_w*outer_h))

def prepare_full_square_dataset(outer_size, inner_size, dtype = np.float64):
    m = (outer_size - inner_size + 1)**2
    output_data =np.zeros( (m, outer_size, outer_size), dtype = dtype )
    for x in range(outer_size - inner_size + 1):
        for y in range(outer_size - inner_size + 1):
            t = x*(outer_size - inner_size + 1) + y
            output_data[t, x:x+inner_size, y:y+inner_size] = 1.
    return output_data.reshape( (m, outer_size*outer_size))
