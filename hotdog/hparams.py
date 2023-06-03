wandb_defaults = {
	# "sync_tensorboard":True, 
	"reinit":True,
	# "entity" : "deepcomputer",
	# "name" : self.run_name,
	"project" : "Hotdog", # wandb project name, each project correpsonds to an experiment
	# "dir" : "logs/" + "GetStarted", # dir to store the run in
	# "group" : self.agent_name, # uses the name of the agent class
	"save_code" : True,
	"mode" : "online",
}

default_config = {
    "model"	: "SimpleCNN",
	"optimizer" : "Adam",
	"loss_fun" : "CrossEntropyLoss",
	"num_epochs" : 10,
	"optimizer_kwargs" : {},
	"train_dataset_kwargs" : {},
	"test_dataset_kwargs" : {},
	"train_dataloader_kwargs" : {},
	"test_dataloader_kwargs" : {},
}