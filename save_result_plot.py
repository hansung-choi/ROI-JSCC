from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
from model_eval import *
from matplotlib.pyplot import cm
import csv
import random
import os
import gc

def get_model_save_name(cfg,model_name,rcpp,SNR):
    data = cfg.data_info.data_name
    cfg.model_name = model_name
    get_loss_info(cfg)
    get_task_info(cfg)
    task = cfg.task_name
    cfg.SNR_info = SNR
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    cfg.rcpp = rcpp
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    random_seed_num = cfg.random_seed
    random_num = str(random_seed_num).zfill(3)
      
    save_name = f"{task}_{data}_{chan_type}_SNR{SNR}_rcpp{rcpp}_{metric}_{model_name}.pt"

    return save_name    

    

def save_SNR_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,option=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    if option:
        metric = option + "_" + metric
    plot_save_name = f"{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']
    
    plt.rcParams["figure.figsize"] = (14,5) #(14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    
    for i, model_name in enumerate(model_name_list):
        valid_SNR_list = []
        valid_performance_list = []
        for SNR in SNR_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]            
            
            if eval_dict:
                valid_SNR_list.append(SNR)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        line = ax1.plot(valid_SNR_list, valid_performance_list,color_list[i],label=f'{cfg.model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('SNR (dB)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')

def save_rcpp_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp_list,SNR,option=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    if option:
        metric = option + "_" + metric
    plot_save_name = f"{chan_type}_rcpp_{metric}_at_SNR{str(SNR).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']
    
    plt.rcParams["figure.figsize"] = (10,8) #(14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        valid_rcpp_list = []
        valid_performance_list = []
        for rcpp in rcpp_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]
            if eval_dict:
                valid_rcpp_list.append(rcpp)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        valid_cpp_list = 1/np.array(valid_rcpp_list)        
        line = ax1.plot(valid_cpp_list, valid_performance_list,color_list[i],label=f'{cfg.model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('cpp', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')



def save_SNR_performance_opt_Avg_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,option=None,prefix=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    if option:
        metric = option + "_" + metric
    plot_save_name = f"opt_Avg_{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name
    
    if prefix:
        plot_save_name = prefix + "_" + plot_save_name


    color_list = ['b-','r-','g-','c-','m-','y-','k-']
    
    plt.rcParams["figure.figsize"] = (14,4) #(14,8), (14,5), (14,4)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    
    for i, model_name in enumerate(model_name_list):
        valid_SNR_list = []
        valid_performance_list = []
        for SNR in SNR_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]            
            
            if eval_dict:
                valid_SNR_list.append(SNR)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        line = ax1.plot(valid_SNR_list, valid_performance_list,color_list[i],label=f'{cfg.model_name}, {metric}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)
    
    metric = cfg.performance_metric    
    color_list = ['b--','r--','g--','c--','m--','y--','k--']
    for i, model_name in enumerate(model_name_list):
        valid_SNR_list = []
        valid_performance_list = []
        for SNR in SNR_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]            
            
            if eval_dict:
                valid_SNR_list.append(SNR)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        line = ax1.plot(valid_SNR_list, valid_performance_list,color_list[i],label=f'{cfg.model_name}, {metric}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('SNR (dB)', fontsize=18) #  fontsize=20, 10
    ax1.set_ylabel(cfg.performance_metric, fontsize=18)
    plt.xticks( fontsize=10) # fontsize=14
    plt.yticks(fontsize=10)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=12)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}', fontdict = {'fontsize' : 18})  # 'fontsize' : 20, 10, 12

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_SNR_performance_Ab_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,option=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    if option:
        metric = option + "_" + metric
    plot_save_name = f"Ab_{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


    color_list = ['b-','r-','g-','c-','m-','y-','k-']
    
    plt.rcParams["figure.figsize"] = (14,4) #(14,8), (14,5), (14,4)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    
    for i, model_name in enumerate(model_name_list):
        valid_SNR_list = []
        valid_performance_list = []
        for SNR in SNR_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]            
            
            if eval_dict:
                valid_SNR_list.append(SNR)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        line = ax1.plot(valid_SNR_list, valid_performance_list,color_list[i],label=f'{cfg.model_name} {metric}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line

    ax1.set_xlabel('SNR (dB)', fontsize=18) #  fontsize=20, 10
    ax1.set_ylabel(cfg.performance_metric, fontsize=18)
    plt.xticks( fontsize=10) # fontsize=14
    plt.yticks(fontsize=10)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=12)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}', fontdict = {'fontsize' : 18})  # 'fontsize' : 20, 10


    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')
    
    
def save_SNR_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,option=None,prefix=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    if option:
        metric = option + "_" + metric
    table_save_name = f"{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        table_save_name += "_" + model_name

    if prefix:
        table_save_name = prefix + "_" + table_save_name


    save_name = save_folder + table_save_name + ".csv"
    
    
    first_line = ["rcpp",rcpp,"metric",metric]
    second_line = ["SNR"]
    second_line.extend(SNR_list)
    second_line.append('GFlops')
    second_line.append('Mmemory')
    
    
    with open(save_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
    
        for i, model_name in enumerate(model_name_list):
            valid_performance_list = [model_name]
            for SNR in SNR_list:
                model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
                eval_dict = total_eval_dict[model_save_name]            
            
                if eval_dict:
                    performance = eval_dict[metric]
                    GFlops = eval_dict['GFlops']
                    Mmemory = eval_dict['Mmemory']
                    valid_performance_list.append(performance)
                else:
                    valid_performance_list.append("None")
            valid_performance_list.append(GFlops)
            valid_performance_list.append(Mmemory)
            writer.writerow(valid_performance_list)
                    
    f.close()  
        
    logger.info(f'{table_save_name} is saved')

def save_rcpp_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp_list,SNR,option=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    if option:
        metric = option + "_" + metric
    table_save_name = f"{chan_type}_rcpp_{metric}_at_SNR{str(SNR).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        table_save_name += "_" + model_name

    save_name = save_folder + table_save_name + ".csv"
    
    
    first_line = ["SNR",SNR,"metric",metric]
    second_line = ["rcpp"]
    second_line.extend(rcpp_list)
    second_line.append('GFlops')
    second_line.append('Mmemory')
    
    
    with open(save_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
    
        for i, model_name in enumerate(model_name_list):
            valid_performance_list = [model_name]
            for rcpp in rcpp_list:
                model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
                eval_dict = total_eval_dict[model_save_name]            
            
                if eval_dict:
                    performance = eval_dict[metric]
                    GFlops = eval_dict['GFlops']
                    Mmemory = eval_dict['Mmemory']
                    valid_performance_list.append(performance)
                else:
                    valid_performance_list.append("None")
            valid_performance_list.append(GFlops)
            valid_performance_list.append(Mmemory)
            writer.writerow(valid_performance_list)
                    
    f.close()  
        
    logger.info(f'{table_save_name} is saved')




def save_plot_legend_type1(cfg,logger,model_name_list,plot_name=None,ncol=5):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"plot_legend"
    if len(model_name_list)==0:
        return None
    
    if not plot_name:
        for model_name in model_name_list:
            plot_save_name += "_" + model_name
    else:
        plot_save_name +=plot_name



    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']

    plt.rcParams["figure.figsize"] = (20,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    for i, model_name in enumerate(model_name_list):
        line = ax1.plot([10], [10],color_list[i],label=f'{model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('...', fontsize=18)
    ax1.set_ylabel(metric, fontsize=15)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(loc='lower right', ncol=ncol, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0,0,0.6,0.8])

    save_name = save_folder + plot_save_name + f'ncol{ncol}' + "_type1.pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_plot_legend_type2(cfg,logger,model_name_list,plot_name=None,ncol=5):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"plot_legend"
    if len(model_name_list)==0:
        return None
    
    if not plot_name:
        for model_name in model_name_list:
            plot_save_name += "_" + model_name
    else:
        plot_save_name +=plot_name



    color_list = ['b-','r-','g-','c-','m-','y-','k-']

    plt.rcParams["figure.figsize"] = (20,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        line = ax1.plot([10], [10],color_list[i],label=f'{model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('...', fontsize=18)
    ax1.set_ylabel(metric, fontsize=15)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(loc='lower right', ncol=ncol, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0,0,0.6,0.8])

    save_name = save_folder + plot_save_name + f'ncol{ncol}' + "_type2.pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


