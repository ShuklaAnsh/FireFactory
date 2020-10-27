from typing import List
import numpy as np


class Audio:
    pass


class Model:
    pass


def pre_process(lofi: List[str], non_lofi: List[str]) -> List[Audio]:
    """
    Proposed architecture:

    Pre-processes the data by analyzing the audio files
    Audio Fingerprinting:
    MFCC Vector Creation:
    Feature Extraction:
    Segmentation:

    Params:
        lofi: List of paths to lofi audio files
        non_lofi: List of paths to non-lofi audio files
    """
    pass


def initialize_model() -> Model:
    """
    Initializes the classifier by loading the data, pre-processing it, and then training the classifier
    Returns the Classifier to be used for prediction.
    """
    model = Model()

    # Load Data
    # dataset 1
    # dataset 2

    # Prepare data
    # audio = pre_process()
    print("Data prepared.")

    # model.train(data)
    print("Genetic Training complete")
    return model
