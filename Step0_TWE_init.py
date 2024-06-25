import logging
import logging.config
import torch
import numpy as np
def log_init(logfile,debug_flag):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.NOTSET)

    ch = logging.StreamHandler()

    if debug_flag:  # self.opt.debug:
        ch.setLevel(logging.NOTSET)
    else:
        ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s: %(message)s")

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)


    return logger
def TWE_init():
    '''
    Function: initialization of TWE

    # Perset params:
        :param logfile: Path of log file
        :param debug_flag: Whether to print the debug information (logger.debug)
    # Derived parameters
        :param device: device choise
        :param logger: logger instance
    '''
    #%% 1. opt definition
    opt={}
    opt['logfile']='log.txt'
    opt['debug_flag']=False
    #%% 2. log initialization
    opt['logger']=log_init(opt['logfile'], opt['debug_flag'])
    #%% 3. numpy cuda initialization
    np.random.seed(0)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        opt['device'] = torch.device('cuda')
    else:
        opt['device'] = torch.device('cpu')
    return opt


