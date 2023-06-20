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
    "model"	: "Resnet",
    "seed":0,
	"optimizer" : "Adam",
	"loss_fun" : "CrossEntropyLoss",
	"num_epochs" : 25,
	### model params
	"finetune": False,
    "classification_size" : (256, 256),
    "ss_size" : (500, 500),
    "k1" : 0.7,
    "k2" : 0.3,
    "train_size" : None,
    "val_size" : None,
    "test_size" : None,
}

