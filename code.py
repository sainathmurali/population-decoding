#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import matplotlib.ticker as ticker

# Set the parameters for your analysis
brain_area = "S1naive"          # S1/S1naive
type_of_analysis1 = "GO"        # GO/NOGO
type_of_analysis2 = "LICK"      # LICK/NOLICK
window_type = "cum"             # type of data chunk (cum - cumulative / mov - moving) 
time_window = 1                 # window width 1 for 100ms, 2 for 200ms, etc,. 
cv = 5                          # number of cross_validation
scoring = "balanced_accuracy"   # scoring method for example 'accuracy','balanced accuracy','f1_score'
model_name = 'svm'              # svm - support vector machines / lda - linear discriminant analysis

# Set the classifier based on the model name
if model_name =='svm':
    clf = svm.SVC(cache_size=1000, kernel="linear")
elif model_name == 'lda':
    clf = lda()

# Set the folder path to save the results
folder = 'final/'+ brain_area + '_'+ model_name + '/'
if not os.path.exists(folder):
    os.makedirs(folder)


def fetch_data(brain_area, type_of_analysis):
    # Load the data
    all_trials=np.load("processed_data/"+brain_area + "_all_trials.npy",allow_pickle=True)
    
    # Initialize arrays for storing trial type (GO/NOGO), fluorescence data (df/f) and behavioral data (LICK/NOLICK)
    trial_list=np.empty((all_trials.shape[0],4))
    trial_dff=np.empty((all_trials.shape[0],41)) 
    trial_licks=np.empty((all_trials.shape[0],41))
    
    for trial in range(all_trials.shape[0]):
        # Store trial information
        trial_list[trial,0]=all_trials[trial].neuron_num

        if type_of_analysis=="GO":
            # Label trials as GO/NOGO
            if (all_trials[trial].trial_type)=="go":
                trial_list[trial,1]=1
            elif (all_trials[trial].trial_type)=="nogo":
                trial_list[trial,1]=0
            else:
                print("wrong trial type on neuron ",all_trials[trial].neuron_num )
        elif type_of_analysis=="LICK":    
            # Label trials as LICK/NOLICK
            if (all_trials[trial].trial_outcome)=="FA":
                trial_list[trial,1]=1
            elif (all_trials[trial].trial_outcome)=="Hit":
                trial_list[trial,1]=1
            elif (all_trials[trial].trial_outcome)=="Miss":
                trial_list[trial,1]=0
            elif (all_trials[trial].trial_outcome)=="CR":
                trial_list[trial,1]=0
            else:
                print("wrong trial outcome on neuron ",all_trials[trial].neuron_num )
        else:
            print("Type of analysis can only be 'GO' or 'LICK'")

        trial_list[trial,2]=all_trials[trial].mouse_id
        trial_list[trial,3]=str(all_trials[trial].mouse_id)+str(all_trials[trial].date)   
	
	#moving the original timeseires left/right to make a new timeseries that is relative to lick
        if (all_trials[trial].trial_outcome)=="FA" or (all_trials[trial].trial_outcome)=="Hit":
            trial_dff[trial,0:41]=all_trials[trial].dff 
            trial_licks[trial,:]=all_trials[trial].licks
            lick_start=np.argmax(trial_licks[trial,:]==1,axis=0)
            trial_dff_rel_licks=np.zeros(trial_licks.shape[1])
            new_start=lick_start-4
            if new_start>=0:                    
                trial_dff_rel_licks[0:trial_licks.shape[1]-new_start]=all_trials[trial].dff[new_start: ]
                trial_dff_rel_licks[trial_licks.shape[1]-new_start :]=all_trials[trial].dff[-1]
            else: 
                trial_dff_rel_licks[0-new_start :]=all_trials[trial].dff[0:all_trials[trial].dff.shape[0]+new_start]
                trial_dff_rel_licks[0:-new_start]=all_trials[trial].dff[ 0 ]
        else:
            trial_dff[trial,0:41]=all_trials[trial].dff 
    return trial_list, trial_dff


def group_data(trial_list, trial_dff, mouse_id):
    '''Function to organize data based on trial type and behaviour type'''
    # Filter data based on mouse ID
    neuron_list=trial_list[:,0][trial_list[:,3]==mouse_id]
    neuron_list=np.unique(neuron_list)
    print("Num of neurons after filter=",neuron_list.shape[0])
    dgo = []
    dngo = []

    for neuron in neuron_list:
        # Split data into go/lick trials and nogo/nolick trials
        mask = (trial_list[:,0] == neuron)
        mask1 = mask & (trial_list[:,1] == 1)
        mask2 = mask & (trial_list[:,1] == 0)
        dff_go_lick = trial_dff[mask1]
        dff_nogo_nolick = trial_dff[mask2]
        dngo.append(dff_nogo_nolick)
        dgo.append(dff_go_lick)

    X_go_lick = np.rollaxis(np.asarray(dgo),1,0)
    X_nogo_nolick = np.rollaxis(np.asarray(dngo),1,0)
    X = np.concatenate((X_nogo_nolick,X_go_lick),axis = 0)

    y_go_lick = np.repeat(1,int(X_go_lick.shape[0]))
    y_nogo_nolick = np.repeat(0,int(X_nogo_nolick.shape[0]))
    y = np.concatenate((y_nogo_nolick,y_go_lick),axis = 0)

    return X,y, neuron_list


def split_windows(X, time_window, window_type):
    '''Function to split the data into cumulative/moving windows'''
    split_area_windows = []

    if window_type == 'cum':
        if time_window == 1:
            for i in range(time_window, X.shape[2]+1, time_window):
                split_area_windows.append(X[:, :, :i])
        else:
            for i in range(time_window+1, X.shape[2]+1, time_window):
                split_area_windows.append(X[:, :, :i])
    elif window_type == 'mov':
        split_area_windows = np.array_split(X, X.shape[2] / time_window, axis=2)
    
    print('split_area_windows info')
    print(type(split_area_windows))
    print('length ' + str(len(split_area_windows)))
    
    for i, window in enumerate(split_area_windows):
        print('element #' + str(i))
        print(window.shape)

    time = np.linspace(0, 4, num=len(split_area_windows))
    return split_area_windows, time


def cross_validate_clf(split_area_windows, y, cv, scoring, clf, labels, y_go_lick, mouse_id):
    '''Function to perform n-fold cross validation'''
    golick_accuracy = []

    if y_go_lick.shape[0] == 2:
        n_splits = np.min((cv, np.min(y_go_lick)))
    else:
        n_splits = np.min((cv, np.min((y_go_lick, 0))))

    j = 0
    for X_window in split_area_windows:
        j = j + 1
        X_window = np.reshape((X_window), (X_window.shape[0], X_window.shape[1] * X_window.shape[2]))

        if n_splits > 1:
            cv_results = cross_validate(clf, X_window, y, scoring=scoring,
                                        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10),
                                        return_train_score=False, n_jobs=-1)
            res = np.array(cv_results['test_score'])
            go_lick_acc_random = np.array_split(res, res.shape[0] / 5)
            acccc = np.asarray(go_lick_acc_random)
            mean_CV = np.mean(np.mean(acccc, axis=1))
            golick_accuracy.append(mean_CV)
        else:
            go_lick_acc_random1 = np.NaN
            golick_accuracy.append(go_lick_acc_random1)

    golick_accuracy = np.asarray([golick_accuracy])
    golick_accuracy = golick_accuracy[0]

    return golick_accuracy


def plot_decoding_results(golick1_accuracy, golick2_accuracy, time, mouse_id):
    '''Function to plot the decoding results'''
    plt.clf()
    plt.plot(time, golick1_accuracy, label=type_of_analysis1, marker='.', color='magenta')
    plt.plot(time, golick2_accuracy, label=type_of_analysis2, marker='*', color='blue')

    y_for_chance = np.repeat(0.50, time.shape[0])
    plt.plot(time, y_for_chance, '--', c='black')
    plt.ylim([0.4, 1])
    plt.grid(which='both')
    plt.minorticks_on()
    plt.xlabel('Time (s)', c='black')
    plt.legend(loc='upper right')
    plt.ylabel('Decoder accuracy', c='black')

    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    [i.set_color('black') for i in plt.gca().get_xticklabels()]
    [i.set_color('black') for i in plt.gca().get_yticklabels()]

    plt.title(brain_area + '_' + window_type + '_balanced_accuracy_' + str(time_window * 100) + 'ms (' +
              str(int(mouse_id)) + ') pca', c='red')
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    fig.savefig(folder + brain_area + '_' + window_type + '_' + str(int(mouse_id)) + '_balanced_accuracy_' +
                str(time_window * 100) + 'ms_pca.png', format='png')
    return


# Fetch data
go_trial_list, go_trial_dff = fetch_data(brain_area, type_of_analysis1)
lick_trial_list, lick_trial_dff = fetch_data(brain_area, type_of_analysis2)
mask = np.unique(go_trial_list[:, 3])

# Initialize empty lists
n_neurons = []
n_trials = []
n_go = []
n_lick = []
peak_lick = []
peak_go = []

# Main for-loop across each mouse
for mouse_id in mask:
    print(mouse_id)
    
    # Group data for go/nogo trials
    X_go, y_go, neuron_list = group_data(go_trial_list, go_trial_dff, mouse_id)
    n_trials.append(y_go.shape[0])
    n_go.append(np.nan_to_num(np.bincount(y_go)))
    
    # Group data for lick/nolick trials
    X_lick, y_lick, neuron_list = group_data(lick_trial_list, lick_trial_dff, mouse_id)
    n_lick.append(np.nan_to_num(np.bincount(y_lick)))
    n_neurons.append(neuron_list.shape[0])
    
    # Split windows for go/nogo trials
    split_area_windows_go, time = split_windows(X_go, time_window, window_type)
    
    # Split windows for lick/nolick trials
    split_area_windows_lick, time = split_windows(X_lick, time_window, window_type)
    
    # Perform cross-validation for go/nogo trials
    go_acc = cross_validate_clf(split_area_windows_go, y_go, cv, scoring, clf, np.unique(y_go), np.bincount(y_go), mouse_id)
    np.savetxt(folder + brain_area + '_' + str(int(mouse_id)) + '_' + window_type + '_' + type_of_analysis1 +
               '_acc.txt', go_acc, fmt='%10.5f')
    
    # Perform cross-validation for lick/nolick trials
    lick_acc = cross_validate_clf(split_area_windows_lick, y_lick, cv, scoring, clf, np.unique(y_lick),
                                  np.bincount(y_lick), mouse_id)
    np.savetxt(folder + brain_area + '_' + str(int(mouse_id)) + '_' + window_type + '_' + type_of_analysis2 +
               '_acc.txt', lick_acc, fmt='%10.5f')
    
    peak_go.append(np.max(go_acc))
    peak_lick.append(np.max(lick_acc))
    
    # Plot the decoding results
    plot_decoding_results(go_acc, lick_acc, time, mouse_id)

# Save results to files
arr1 = np.zeros([len(n_lick), len(max(n_lick, key=lambda x: len(x)))], int)
for i, j in enumerate(n_lick):
    arr1[i][0:len(j)] = j

np.savetxt(folder + brain_area + '_' + window_type + '_' + str(time_window * 100) + '_peak_lick.txt', peak_lick,
           fmt='%f')
np.savetxt(folder + brain_area + '_' + window_type + '_' + str(time_window * 100) + '_peak_go.txt', peak_go,
           fmt='%f')
np.savetxt(folder + brain_area + '_ngo.txt', n_go, fmt='%i')
np.savetxt(folder + brain_area + '_nlick.txt', arr1, fmt='%i')
np.savetxt(folder + brain_area + '_nneurons.txt', n_neurons, fmt='%i')
np.savetxt(folder + brain_area + '_ntrials.txt', n_trials, fmt='%i')
np.savetxt(folder + brain_area + '_neuron_list.txt', mask, fmt='%i')
