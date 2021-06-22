import configs
import utils

prog_args = configs.arg_parse()
ckpt = utils.load_ckpt(prog_args)

save_data = ckpt["save_data"] # get save data

print(ckpt["epoch"])
