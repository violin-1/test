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
    
    def log3(self,epoch,loss):
        print("Epoch {}:".format(epoch))
        print('{} Train Done.\n train_loss={}\n Start Val:\n'.format(epoch+1,loss))
        self.log.write('{} Train Done.\n train_loss={}\n Start Val:\n'.format(epoch+1,loss))
        self.log.flush()
    
    def log4(self,epoch,train_args_val):
        print('||epoch:%d, Val_loss = %f\n' % (epoch + 1, train_args_val))
        self.log.write('||epoch:%d, Val_loss = %f\n' % (epoch + 1, train_args_val))
        self.log.flush()
    
    def log5(self):
        print('Finished Training')
        self.log.close()

    def log_best(self,epoch,val_args,best_args):
        print('epoch: {} The val loss is {} last minimum loss is {}'.format(epoch,val_args,best_args))
        self.log.write('epoch: {} The best minimum loss is {} last minimum loss is {}'.format(epoch,val_args,best_args))
        self.log.flush()
        pass