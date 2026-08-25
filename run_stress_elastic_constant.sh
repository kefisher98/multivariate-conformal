
##################################################################################################
# USER: FILL IN CLUSTER COMMANDS AND SETTINGS 
################################################################################################## 
task_id=1
task_count=1
filepath=data/stress_elastic_constant/ec_stress


##################################################################################################
# compute elastic constant from stress
##################################################################################################
config_size=16
stress_step=0.05
energy_train=300
stress_train=15
calibrate=175


julia src/stress_elastic_constant.jl $task_id \
                                     $task_count \
                                     $config_size \
                                     $stress_step \
                                     $energy_train \
                                     $stress_train \
                                     $calibrate


cat ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_1_conformal.csv >> \
    ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_conformal.csv

cat ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_1_bonf.csv >> \
    ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_bonf.csv

rm ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_1_conformal.csv
rm ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_1_bonf.csv

for i in $(seq 2 $task_count);
do
        tail -n +2 ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_${i}_conformal.csv >> \
                   ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_conformal.csv

        tail -n +2 ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_${i}_bonf.csv >> \
                   ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_bonf.csv

        rm ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_${i}_conformal.csv
        rm ${filepath}_config${config_size}${energy_step}_${stress_step}_nC${stress_train}_nT${calibrate}_${i}_bonf.csv
done


###############################################################################################
# make plots
###############################################################################################

julia src/make_elastic_constant_plots.jl

