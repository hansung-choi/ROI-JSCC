from data_maker import *

def BaseSchedulerMaker(optimizer,initial_lr,mile_stone):
    #code from https://github.com/hansung-choi/TLA-linear-ascent
    initial_lr = initial_lr
    warm_up_step = -1
    mile_stone = [170,190] #[160,220]
    gamma = 0.1 #0.01
    
    #mile_stone = [250,400,450,475] #[160,220] from CAMixerSR paper
    mile_stone = mile_stone
    gamma = 0.5 #0.01
    

    def get_lr(epoch):
        #if epoch <= warm_up_step:
            #return initial_lr * epoch / warm_up_step

        decrease_rate = 1
        for th in mile_stone:
            if epoch >= th and th >0:
                decrease_rate *= gamma
        return initial_lr*decrease_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer,get_lr)

def get_base_ut_schedular(optimizer,initial_lr):
    # code from https://github.com/hansung-choi/TLA-linear-ascent
    initial_lr = initial_lr
    warm_up_step = 5
    mile_stone = [200,320]
    gamma = 0.01

    def get_lr(epoch):
        if epoch <= warm_up_step:
            return initial_lr * epoch / warm_up_step

        decrease_rate = 1
        for th in mile_stone:
            if epoch >= th:
                decrease_rate *= gamma
        return initial_lr*decrease_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer,get_lr)


def OptimizerMaker(model,cfg: DictConfig):
    optimizer = None
    if cfg.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),cfg.learning_rate)
    else:
        raise ValueError(f'{cfg.optimizer_name} is not implemented yet')
    return optimizer

