import time
import os

def mkfiledir(filedirlist):
    path = '.'
    for filesubdir in filedirlist:
        path = os.path.join(path,filesubdir)
        if not os.path.exists(path):
            os.mkdir(path)
    return path


class logger:
    def __init__(self,modeltype,datasettype,dataarch,trainratio):
        now = time.time()
        self.modeltype = modeltype
        self.datasettype = datasettype
        self.dataarch = dataarch
        
        self.savedir = mkfiledir(["datasets","datasetp",datasettype,trainratio,"checkpoints",modeltype,dataarch])
        self.logdir = mkfiledir(["datasets","datasetp",datasettype,trainratio,'logs',modeltype,dataarch])
        self.log = open(os.path.join(self.logdir,str(int(now))+'_log.txt'),'w')
        pass
    
    def log1(self):
        print("The log file is {}".format(self.logdir))
        self.log.write('This task is training on the ' + self.datasettype + ' dataset\n')
        print('This task is training on the ' + self.datasettype + ' dataset\n')
        if self.modeltype is not None:
            self.log.write('We now use ' + self.modeltype + ' model to do this task\n')
            print('We now use ' + self.modeltype + ' model to do this task\n')
        
        self.log.write('Init Start\n')
        print('Init Start\n')
        self.log.flush()

    def log2(self,epoch):
        print("Train init done")
        print("Epoch {}:".format(epoch))
        pass
    
    def log3(self,epoch,train_acc,loss):
        print("Epoch {}:".format(epoch))
        print('{} Train Done.\n train_acc={} train_loss={}\n Start Val:\n'.format(epoch+1,train_acc,loss))
        self.log.write('{} Train Done.\n train_acc={} train_loss={}\n Start Val:\n'.format(epoch+1,train_acc,loss))
        self.log.flush()
    
    def log4(self,epoch,train_args_val):
        print('||epoch:%d, Val_Accurate = %f, Val_F1_Score=%f, Val_ACER= %f\n' % (epoch + 1, train_args_val['Accurate']['all'], train_args_val['F1_score']['all'], train_args_val['ACER']['all'],))
        self.log.write('||epoch:%d, Val_Accurate = %f, Val_F1_Score=%f, Val_ACER= %f\n' % (epoch + 1, train_args_val['Accurate']['all'], train_args_val['F1_score']['all'], train_args_val['ACER']['all'],))
        self.log.flush()
    
    def log5(self):
        print('Finished Training')
        self.log.close()

    def log_best(self,epoch,val_args,best_args):
        print('epoch: {} The accurate is {} last best accurate is {}'.format(epoch,val_args['Accurate']['all'],best_args['Accurate']['all']))
        self.log.write('epoch: {} The best is {} last best is {}'.format(epoch,val_args['Accurate']['all'],best_args['Accurate']['all']))
        self.log.flush()
        pass