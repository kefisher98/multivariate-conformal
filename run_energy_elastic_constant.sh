
##################################################################################################
# USER: FILL IN CLUSTER COMMANDS AND SETTINGS 
################################################################################################## 
task_id=1
task_count=1
filepath=data/energy_elastic_constant/ec_energy


##################################################################################################
# compute elastic constant from energy
##################################################################################################
config_size=16
energy_step=0.05
stress_step=0.05
train_n=350
calibrate_n=100


julia src/energy_elastic_constant.jl $task_id  \
                                     $task_count \
                                     $config_size \
                                     $energy_step \
                                     $stress_step \
                                     $train_n \
                                     $calibrate_n


cat ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_conformal.csv >> \
    ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_conformal.csv

cat ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_bonf.csv >> \
    ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_bonf.csv

rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_conformal.csv
rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_bonf.csv


for i in $(seq 2 $task_count);
do
        tail -n +2 ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_conformal.csv >> \
                   ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_conformal.csv

        tail -n +2 ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_bonf.csv >> \
                   ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_bonf.csv

        rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_conformal.csv
        rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_bonf.csv
done



##################################################################################################
# compute elastic constant from energy
##################################################################################################
config_size=16
energy_step=0.01
stress_step=0.01
train_n=350
calibrate_n=100


julia src/energy_elastic_constant.jl $task_id \
                                     $task_count \
                                     $config_size \
                                     $energy_step \
                                     $stress_step \
                                     $train_n \
                                     $calibrate_n


cat ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_conformal.csv >> \
    ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_conformal.csv

cat ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_bonf.csv >> \
    ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_bonf.csv

rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_conformal.csv
rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_bonf.csv

for i in $(seq 2 $task_count);
do
        tail -n +2 ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_conformal.csv >> \
                   ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_conformal.csv

        tail -n +2 ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_bonf.csv >> \
                   ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_bonf.csv

        rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_conformal.csv
        rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_bonf.csv
done


##################################################################################################
# compute elastic constant from energy
##################################################################################################
config_size=54
energy_step=0.05
stress_step=0.05
train_n=350
calibrate_n=100


julia src/energy_elastic_constant.jl $task_id  \
                                     $task_count \
                                     $config_size \
                                     $energy_step \
                                     $stress_step \
                                     $train_n \
                                     $calibrate_n


cat ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_conformal.csv >> \
    ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_conformal.csv

cat ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_bonf.csv >> \
    ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_bonf.csv

rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_conformal.csv
rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_1_bonf.csv

for i in $(seq 2 $task_count);
do
        tail -n +2 ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_conformal.csv >> \
                   ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_conformal.csv

        tail -n +2 ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_bonf.csv >> \
                   ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_bonf.csv

        rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_conformal.csv
        rm ${filepath}_config${config_size}_${energy_step}_${stress_step}_nC${train_n}_nT${calibrate_n}_${i}_bonf.csv
done

###############################################################################################
# make plots
###############################################################################################

julia src/make_elastic_constant_plots.jl


julia src/make_intro_figure.jl
