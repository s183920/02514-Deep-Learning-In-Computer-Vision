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

default_classifier_config = {
    "model"	: "Resnet50",
    "seed":0,
	"optimizer" : "Adam",
	"loss_fun" : "CrossEntropyLoss",
	"num_epochs" : 25,
	### model params
	"finetune": False,
}

