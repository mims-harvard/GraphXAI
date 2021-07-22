import tasks
import configs

prog_args = configs.arg_parse()

if prog_args.dataset is not None:
    if prog_args.dataset == "bitcoinalpha":
        print("Train bitcoinalpha dataset")
        training_function = "tasks.task_bitcoinalpha"
        eval(training_function)(prog_args)
    
    elif prog_args.dataset == "bitcoinotc":
        print("Train bitcoinotc dataset")
        training_function = "tasks.task_bitcoinotc"
        eval(training_function)(prog_args)
        
    elif prog_args.dataset == "amazon":
        print("Train amazon dataset")
                
    else:
        print("Training synthetic dataset")
        training_function = "tasks.task_syn"
        eval(training_function)(prog_args)
        
