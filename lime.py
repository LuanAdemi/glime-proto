from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Tuple, List
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from PIL import Image


CUDA_AVAILABLE = torch.cuda.is_available()


class SimilarityKernel(ABC):
    """
    This is an abstract class for similarity kernel functions.

    A similarity kernel function is a function that takes two inputs and returns 
    a scalar value representing the degree of similarity between the two inputs. 
    It is employed in the LIME algorithm to assess the similarity between the original 
    input and the perturbed input. This similarity value can be calculated in either 
    the interpretable space or the original space.
    """
    @abstractmethod
    def __call__(self, x1, x2) -> float:
        pass

class Sampler(ABC):
    def __init__(self, segments: np.ndarray, n_jobs: int = 1, alpha: int = 0.3):
        self.segments = segments
        self.n_jobs = n_jobs
        self.alpha = alpha

    def sample(self, instance: np.ndarray, num_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        def generate_sample_parallel(instance, bitmap):
            return self.generate_sample(instance, bitmap), bitmap

        samples = Parallel(n_jobs=self.n_jobs)(
            delayed(generate_sample_parallel)(instance, np.random.choice([0, 1], size=len(self.segments), p=[self.alpha, 1-self.alpha]))
            for _ in range(num_samples)
        )
        return samples

    @abstractmethod
    def generate_sample(self, instance: np.ndarray, segment_bitmap: np.ndarray) -> np.ndarray:
        pass


class ExponentialKernel(SimilarityKernel):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x1, x2) -> float:
        return np.exp(-np.linalg.norm(x1 - x2) / self.sigma)

class BinarySampler(Sampler):
    """
    This is a concrete class for binary sampler.

    It fills each superpixel with a fill value or the original pixel values based on the bitmap.
    """
    def __init__(self, segments: np.ndarray, n_jobs: int = 1, alpha: int = 0.3):
        self.fill_value = 128
        super().__init__(segments, n_jobs, alpha)

    def generate_sample(self, instance: np.ndarray, segment_bitmap: np.ndarray) -> np.ndarray:
        sample = np.copy(instance)
        for i, bitmap in enumerate(segment_bitmap):
            if bitmap == 0:
                sample[self.segments == i] = self.fill_value
        return sample
    
class FlowSampler(Sampler):
    def __init__(self, flow, segments: np.ndarray, alpha: int = 0.3):
        self.flow = flow
        super().__init__(segments, alpha=alpha)

    def _latent_shuffling(self, x, beta=0.6):
        x = torch.tensor(x, dtype=torch.float32).to(self.flow.device)
        z = self.flow.to_latent(x.unsqueeze(0))
        # TODO: find manipulators for the latent space
        z[0] = z[0] + beta * torch.randn_like(z[0])
        return self.flow.to_image(z)[0].squeeze().detach().cpu().numpy()

    def generate_sample(self, instance: np.ndarray, segment_bitmap: np.ndarray) -> np.ndarray:
        sample = np.copy(instance)
        perturbed = self._latent_shuffling(instance.transpose(2, 0, 1)).transpose(1, 2, 0)
        print(sample.shape, perturbed.shape)
        
        for i, bitmap in enumerate(segment_bitmap):
            if bitmap == 0:
                sample[self.segments == i] = perturbed[self.segments == i]
        return sample

class LIMEExplainer:
    def __init__(self, 
                 similarity_kernel: SimilarityKernel,
                 sampler: Sampler,
                 model: callable, 
                 preprocess: callable = lambda x: x):
        
        self.similarity_kernel = similarity_kernel
        self.sampler = sampler

        self.device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        self.model = model.to(self.device)

        self.preprocess = preprocess

        # freeze the model
        self.model.eval()

    def get_class_probs(self, batch, cls_to_explain):
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            return probs[:, cls_to_explain]
    
    def build_batch(self, samples):
        batch = torch.stack([self.preprocess(Image.fromarray(sample)) for sample, _ in samples])
        return batch.to(self.device)

    def build_pertubations(self, instance: np.ndarray, num_samples: int):
        return self.sampler.sample(instance, num_samples)
    
    def build_sample_weights(self, instance: np.ndarray, samples: List[Tuple[np.ndarray, np.ndarray]]):
        sample_weights = np.zeros(len(samples))
        for i, (sample, bitmap) in enumerate(samples):
            sample_weights[i] = self.similarity_kernel(instance, sample)
        return sample_weights
    
    def train_model(self, samples, probs, sample_weights):
        model = LinearRegression()
        X = np.array([bitmap for _, bitmap in samples])
        y = probs
        model.fit(X, y, sample_weight=sample_weights)

        return model

    def get_importance(self, instance: np.ndarray, class_to_explain: int, n_pertubations: int = 500):
        samples = self.build_pertubations(instance, n_pertubations)
        batch = self.build_batch(samples)
        probs = self.get_class_probs(batch, class_to_explain).cpu().numpy()
        sample_weights = self.build_sample_weights(instance, samples)
        model = self.train_model(samples, probs, sample_weights)

        return model.coef_
    
    def explain(self, instance: np.ndarray, class_to_explain: int, n_pertubations: int = 1000):
        image = np.copy(instance)
        importance = self.get_importance(instance, class_to_explain, n_pertubations)
        
        segmentation = np.zeros_like(self.sampler.segments, dtype=np.float32)
        for i in range(len(importance)):
            segmentation[self.sampler.segments == i] = importance[i]

        # create subplots
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(image)
        ax1.imshow(segmentation, cmap='coolwarm', alpha=0.5)
        im2 = ax2.imshow(segmentation, cmap='coolwarm')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        plt.show()

        return importance