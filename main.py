import importlib
from config import *
from dataflow.DataController import DataController


if __name__ == '__main__':
    # ARGS = parse_args()
    ARGS.modeltypes = ARGS.modeltypes.split(',')
    if ARGS.datatsetstype != '':
        ARGS.datatsetstype = ARGS.datatsetstype.split(',')
    else:
        ARGS.datatsetstype = None
    if ARGS.device == 'cuda' or ARGS.device == 'gpu':
        ARGS.device = 'cuda'
    ARGS.Set_datactr_func = None
    if ARGS.set_datactr_func != '':
        tmp = ARGS.set_datactr_func.split(',')
        ARGS.Set_datactr_func=getattr(importlib.import_module(tmp[0]),tmp[1])
    
    ARGS.Set_modelctr_func = None
    if ARGS.set_modelctr_func != '':
        tmp = ARGS.set_modelctr_func.split(',')
        ARGS.Set_modelctr_func=getattr(importlib.import_module(tmp[0]),tmp[1])
        
    datactr = DataController(ARGS)
    # datactr = DataController(datatsetstype=datatsetstype,modeltypes=modeltypes,device=ARGS.device,
    #                          set_datactr_func=Set_datactr_func,
    #                          set_modelctr_func=Set_modelctr_func,)
    datactr.exec()