wandb_defaults = {
	# "sync_tensorboard":True, 
	"reinit":True,
	"entity" : "deepcomputer",
	# "name" : self.run_name,
	"project" : "Hotdog", # wandb project name, each project correpsonds to an experiment
	# "dir" : "logs/" + "GetStarted", # dir to store the run in
	# "group" : self.agent_name, # uses the name of the agent class
	"save_code" : True,
	"mode" : "online",
}

default_config = {
    # "model"	: "SimpleCNN",
	"optimizer" : "Adam",
	"loss_fun" : "CrossEntropyLoss",
	"num_epochs" : 10,
	"optimizer_kwargs" : {},
	"train_dataset_kwargs" : {},
	"test_dataset_kwargs" : {},
	"train_dataloader_kwargs" : {},
	"test_dataloader_kwargs" : {},
}

sweep_defaults = {
    'program': "hotdog/cli.py",
    'method': 'bayes',
    # 'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'Validation metrics/test_acc'
        },
    'parameters': {
        # 'batch_size': {'values': [16, 32, 64]},
        'num_epochs': {'max': 20, "min": 1, "distribution" : "int_uniform"},
        # 'lr': {'max': 0.1, 'min': 0.0001}
     },
    # "early_terminate": {
	# 	"type": "hyperband",
	# 	"min_iter": 1,
	# 	"max_iter": 3,
	# }
}