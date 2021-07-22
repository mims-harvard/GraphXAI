import configs
import explain_tasks as tasks

prog_args = configs.arg_parse()

if prog_args.dataset is not None:
    if prog_args.dataset == "bitcoinalpha":
        print("Explain bitcoinalpha dataset")
        explaining_task = "tasks.bitcoin"
        eval(explaining_task)(prog_args)
    
    elif prog_args.dataset == "bitcoinotc":
        print("Explain bitcoinotc dataset")
        explaining_task = "tasks.bitcoin"
        eval(explaining_task)(prog_args)
        
    elif prog_args.dataset == "amazon":
        print("Explain amazon dataset")
                
    else:
        print("Explain synthetic dataset")
        explaining_task = "tasks.task_syn"
        eval(explaining_task)(prog_args)
