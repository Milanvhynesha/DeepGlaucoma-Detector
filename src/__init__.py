"""
DeepGlaucoma-Detector source package.
"""

from src.model import GlaucomaDetectionModel, create_model
from src.data_loader import GlaucomaDataset, create_dataloaders, split_data
from src.train import train_model, train_epoch, validate_epoch
from src.predict import GlaucomaPredictor, predict_glaucoma
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.utils import set_seed, get_device, count_parameters

__version__ = '1.0.0'
