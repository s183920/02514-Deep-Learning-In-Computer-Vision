import wandb
from classifier import HotdogClassifier


classifier = HotdogClassifier(show_test_images=False)
classifier.train()