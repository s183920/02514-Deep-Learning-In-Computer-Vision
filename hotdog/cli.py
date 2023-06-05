import argparse
from classifier import HotdogClassifier

class AddToDict(argparse.Action):
    """An argparser action that adds arguments to a dictionary with the argument name as key"""
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            return 
        
        if not hasattr(namespace, self.dest):
            setattr(namespace, self.dest, {})
        elif getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        
        key = option_string.replace("-", "") # argument name
        getattr(namespace, self.dest).update({key: values})


parser = argparse.ArgumentParser(description='Hotdog classifier')

# general args
parser.add_argument("-mode", default = "train", choices = ["train", "test"], help = "Mode to run the classifier in")

# classifier args
parser.add_argument("--name", default = None, type = str, help = "Name of the model")
parser.add_argument('--show-test-images', action='store_true', default=False)
parser.add_argument("--model", type = str, default = "SimpleCNN", help = "Model to use for classification")
parser.add_argument("--use_wandb", action='store_true', default=True)

# config args
parser.add_argument("--optimizer", type = str, action=AddToDict, dest = "config", help = "Optimizer")
parser.add_argument("--loss_fun", type = str, action=AddToDict, dest = "config", help = "Loss function")
parser.add_argument("--lr", type = float, action=AddToDict, dest = "config", help = "Learning rate")
parser.add_argument("--num_epochs", type = int, action=AddToDict, dest = "config", help = "Batch size")

# parse args
args = parser.parse_args()

# create classifier
classifier = HotdogClassifier(name = args.name, show_test_images=args.show_test_images, use_wandb=args.use_wandb, model=args.model, **args.config)

if args.mode == "train":
    classifier.train()
elif args.mode == "test":
    classifier.test()

# print(args)