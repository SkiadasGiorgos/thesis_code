from matplotlib import pyplot as plt
import numpy as np
import json

#methods LwF and EwC results for initial classes 250 increment 250 resnet18
path_lwf_250_res = "/home/skiadasg/res/vggface2_lwf_lwf_250_res/results/acc_tag-2023-09-23-18-36.txt"
path_ewc_250_res = "/home/skiadasg/res/vggface2_ewc_ewc_250_res/results/acc_tag-2023-09-23-23-59.txt"

#methods LwF and EwC results for initial classes 250 increment 250 googlenet
path_lwf_250 = "none"
path_ewc_250 = "/home/skiadasg/res/vggface2_ewc_ewc_250/results/acc_tag-2023-09-24-17-03.txt"

#methods LwF and EwC results for initial classes 100 increment 134 googlenet
path_lwf_100 = "/home/skiadasg/res/googlenet/vggface2_lwf_lwf_googlenet/results/acc_tag-2023-09-06-01-40.txt" 
path_ewc_100 = "/home/skiadasg/res/googlenet/vggface2_ewc_lwf_googlenet/results/acc_tag-2023-09-04-23-29.txt"

#methods LwF and EwC results for initial classes 50 increment 50 googlenet
path_lwf_50 = "/home/skiadasg/res/googlenet/vggface2_lwf_lwf_googlenet/results/acc_tag-2023-09-05-10-53.txt" 
path_ewc_50 = "/home/skiadasg/res/googlenet/vggface2_ewc_lwf_googlenet/results/acc_tag-2023-09-05-18-42.txt"

#methods LwF and EwC results for initial classes 100 increment 134 resnet18
path_lwf_100_res = '/home/skiadasg/res/resnet/vggface2_lwf_lwf_init100/results/acc_tag-2023-09-06-11-03.txt'
path_ewc_100_res = '/home/skiadasg/res/resnet/vggface2_ewc_ewc_init100/results/acc_tag-2023-09-06-17-12.txt'

#methods LwF and EwC results for initial classes 50 increment 50 resnet18
path_lwf_50_res = '/home/skiadasg/res/resnet/vggface2_lwf_lwf_init50/results/acc_tag-2023-09-06-22-11.txt'
path_ewc_50_res = '/home/skiadasg/res/resnet/vggface2_ewc_ewc_init50/results/acc_tag-2023-09-07-04-48.txt'

#finetuning results for initial 250 increment 250 googlenet and resnet
path_ft_250 = "/home/skiadasg/res/vggface2_finetuning_finetuning_250/results/acc_tag-2023-09-24-22-52.txt"
path_ft_250_res = "/home/skiadasg/res/vggface2_finetuning_finetuning_250_res/results/acc_tag-2023-09-24-04-42.txt"

#finetuning results for initial 100 increment 134 googlenet and resnet
path_ft_100 = '/home/skiadasg/res/resnet/vggface2_finetuning_ft_cnn_100/results/acc_tag-2023-09-07-11-09.txt'
path_ft_100_res = '/home/skiadasg/res/resnet/vggface2_finetuning_ft_res_100/results/acc_tag-2023-09-07-16-32.txt'

#finetuning results for initial 50 increment 50 googlenet and resnet
path_ft_50 = '/home/skiadasg/res/resnet/vggface2_finetuning_ft_cnn_50/results/acc_tag-2023-09-07-21-01.txt'
path_ft_50_res = '/home/skiadasg/res/resnet/vggface2_finetuning_ft_res_100/results/acc_tag-2023-09-08-02-39.txt'

#iCarl results for initial 100 increment 134 resnet 
path_icarl_100_40 = '/home/skiadasg/res/replay_methods/googlenet/vggface2_icarl_icarl_100_num40/results/acc_tag-2023-09-13-01-04.txt'
path_icarl_100_20 = '/home/skiadasg/res/replay_methods/googlenet/vggface2_icarl_icarl_100_num20/results/acc_tag-2023-09-11-22-11.txt'
path_icarl_100_10 = '/home/skiadasg/res/replay_methods/googlenet/vggface2_icarl_icarl_100_num10/results/acc_tag-2023-09-11-13-40.txt'
path_icarl_100_10_res = '/home/skiadasg/res/replay_methods/vggface2_icarl_icarl_100_num10/results/acc_tag-2023-09-09-16-07.txt'
path_icarl_100_20_res = '/home/skiadasg/res/replay_methods/vggface2_icarl_icarl_100_20/results/acc_tag-2023-09-09-01-44.txt'
path_icarl_100_40_res = '/home/skiadasg/res/replay_methods/vggface2_icarl_icarl_100_num40/results/acc_tag-2023-09-10-14-13.txt'

#BiC results for initial 100 increment 134 resnet
path_bic_100_40 = '/home/skiadasg/res/replay_methods/googlenet/vggface2_bic_bic_100_num40/results/acc_tag-2023-09-13-10-46.txt'
path_bic_100_20 = '/home/skiadasg/res/replay_methods/googlenet/vggface2_bic_bic_100_num20/results/acc_tag-2023-09-12-14-17.txt'
path_bic_100_10 = '/home/skiadasg/res/replay_methods/googlenet/vggface2_bic_bic_100_num10/results/acc_tag-2023-09-12-06-44.txt'
path_bic_100_10_res = '/home/skiadasg/res/replay_methods/vggface2_bic_bic_100_num10/results/acc_tag-2023-09-09-09-30.txt'
path_bic_100_20_res = '/home/skiadasg/res/replay_methods/vggface2_bic_bic_100_num20/results/acc_tag-2023-09-08-15-31.txt'
path_bic_100_40_res = '/home/skiadasg/res/replay_methods/vggface2_bic_bic_100_num40/results/acc_tag-2023-09-10-22-45.txt'

#BiC results for initial 50 incrment 50 resnet
path_bic_50_40 ='/home/skiadasg/res/replay_methods/googlenet/vggface2_bic_bic_50_num40/results/acc_tag-2023-09-17-16-20.txt'
path_bic_50_20 ='/home/skiadasg/res/replay_methods/googlenet/vggface2_bic_bic_50_num20/results/acc_tag-2023-09-16-11-49.txt'
path_bic_50_10 ='/home/skiadasg/res/replay_methods/googlenet/vggface2_bic_bic_50_num10/results/acc_tag-2023-09-16-02-28.txt'
path_bic_50_40_res = '/home/skiadasg/res/replay_methods/vggface2_bic_bic_50_num40/results/acc_tag-2023-09-14-03-44.txt'
path_bic_50_20_res = '/home/skiadasg/res/replay_methods/vggface2_bic_bic_50_num20/results/acc_tag-2023-09-09-23-17.txt'
path_bic_50_10_res = '/home/skiadasg/res/replay_methods/vggface2_bic_bic_50_num10/results/acc_tag-2023-09-13-19-51.txt'

#icarl results for initial 50 incrment 50 resnet
path_icarl_50_40 ='/home/skiadasg/res/replay_methods/googlenet/vggface2_icarl_icarl_50_num40/results/acc_tag-2023-09-19-02-53.txt'
path_icarl_50_20 ='/home/skiadasg/res/replay_methods/googlenet/vggface2_icarl_icarl_50_num20/results/acc_tag-2023-09-18-15-24.txt'
path_icarl_50_10 ='/home/skiadasg/res/replay_methods/googlenet/vggface2_icarl_icarl_50_num10/results/acc_tag-2023-09-18-05-22.txt'
path_icarl_50_40_res = '/home/skiadasg/res/replay_methods/vggface2_icarl_icarl_50_num40/results/acc_tag-2023-09-15-09-32.txt'
path_icarl_50_20_res = '/home/skiadasg/res/replay_methods/vggface2_icarl_icarl_50_num20/results/acc_tag-2023-09-14-23-48.txt'
path_icarl_50_10_res = '/home/skiadasg/res/replay_methods/vggface2_icarl_icarl_50_num10/results/acc_tag-2023-09-14-14-54.txt'



save_path = "/home/skiadasg/thesis_code/thesis_code/plots"

# d = {}
# with open(config_path) as f:
#     data = f.read()

# js = json.loads(data)

# title = js["approach"]+"_"<+str(js["datasets"])+"_"+js['network']+"_"+str(js['nc_first_task'])+"_"+str(js['num_tasks'])+"_tag"



def create_results(path):
    avg_results = 0
    results = np.loadtxt(path)
    results[results==0] = np.nan
    avg_results = np.nanmean(results,axis=1)
    return avg_results


# title = 'Accuracy per sample increment BiC (10 Tasks)'


def plots_replay_samples_incr():
    samples_10 =create_results(path_bic_50_10)
    samples_20 = create_results(path_bic_50_20)
    samples_40 = create_results(path_bic_50_40)

    num_tasks = range(len(samples_10))

    plt.figure()
    plt.plot(samples_10,label='10 samples')
    plt.plot(samples_20,label='20 samples')
    plt.plot(samples_40,label='40 samples')
    plt.scatter(num_tasks,samples_10)
    plt.scatter(num_tasks,samples_20)
    plt.scatter(num_tasks,samples_40)
    plt.xlabel('Tasks')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)

def plot_last_acc():
    title = 'Accuracy per task after the end of training (10 Tasks)'
    save_path = "/home/skiadasg/thesis_code/thesis_code/plots/"+title+".png"
    last_lwf = np.loadtxt(path_lwf)[:][-1]
    last_ewc = np.loadtxt(path_ewc)[:][-1]
    last_lwf_res = np.loadtxt(path_lwf_res)[:][-1]
    last_ewc_res = np.loadtxt(path_ewc_res)[:][-1]

    num_tasks = range(len(last_ewc))
    plt.figure()
    plt.plot(last_lwf)
    plt.plot(last_ewc)
    plt.plot(last_ewc_res)
    plt.plot(last_lwf_res)
    plt.scatter(num_tasks,last_lwf,label='lwf')
    plt.scatter(num_tasks,last_ewc,label='ewc')
    plt.scatter(num_tasks,last_ewc_res,label='ewc_res')
    plt.scatter(num_tasks,last_lwf_res,label='lwf_res')
    plt.xticks([0,1,2,3,4,5,6,7,8,9],
    ['Task 0','Task 1','Task 2','Task 3','Task 4','Task 5','Task 6','Task 7','Task 8','Task 9'])
    plt.legend()
    plt.savefig(save_path)

# plot_last_acc()
# plots_replay_samples_incr()

title = 'regularization_methods_init(250)_incr_(250)'
save_path = "/home/skiadasg/thesis_code/thesis_code/plots/"+title+".png"

# results_bic = create_results(path_bic_100_10_res)
# results_icarl = create_results(path_icarl_100_10_res)
results_ft_res = create_results(path_ft_250)
# results_lwf = create_results(path_lwf_250_res)
results_ewc = create_results(path_ewc_250)

tasks = []
for i in range(len(results_ewc)):
    tasks.append(i)

plt.figure()
# plt.plot(results_bic,label='bic')
# plt.plot(results_icarl,label='icarl')
# plt.plot(results_lwf,label='lwf')
plt.plot(results_ewc,label='ewc')
# plt.scatter(tasks,results_bic)
# plt.scatter(tasks,results_icarl)
# plt.scatter(tasks,results_lwf)
plt.scatter(tasks,results_ewc)
plt.plot(results_ft_res,'k--',linewidth=0.5,label='ft')
plt.scatter(tasks,results_ft_res,facecolors='black')
plt.xlabel('Tasks')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig(save_path)
print('ok')