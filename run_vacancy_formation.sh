
##################################################################################################
# USER: FILL IN CLUSTER COMMANDS AND SETTINGS 
################################################################################################## 
task_id=1
task_count=1
filepath=data/vacancy_formation/vacancy


##################################################################################################
# compute vacancy formation energy
##################################################################################################
rho=0.55
train_primary=350
train_secondary=150
calibrate_primary=100
calibrate_secondary=40
n_systems=25

julia src/vacancy_formation.jl $task_id \
                               $task_count \
                               $rho        \
                               $train_primary \
                               $train_secondary \
                               $calibrate_primary \
                               $calibrate_secondary \
                               $n_systems


index_size_54=549
index_size_128=659

cat ${filepath}__${index_size_54}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
    ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

rm ${filepath}__${index_size_54}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv

first=$((index_size_54+1))
last=$((index_size_54+n_systems-1))

for i in $(seq $first $last);
do
    tail -n +2 ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
                   ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

    rm ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv
done

first=$index_size_128
last=$((index_size_128+n_systems-1))

for i in $(seq $first $last);
do
        tail -n +2 ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
                   ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

        rm ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv
done


#------------------------------------------------

train_primary=250
train_secondary=150
calibrate_primary=200
calibrate_secondary=50
n_systems=25

julia src/vacancy_formation.jl $task_id \
                               $task_count \
                               $rho        \
                               $train_primary \
                               $train_secondary \
                               $calibrate_primary \
                               $calibrate_secondary \
                               $n_systems

index_size_54=549
index_size_128=659

cat ${filepath}__${index_size_54}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
    ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

rm ${filepath}__${index_size_54}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv

first=$((index_size_54+1))
last=$((index_size_54+n_systems-1))

for i in $(seq $first $last);
do
        tail -n +2 ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
                   ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

        rm ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv
done

first=$((index_size_128))
last=$((index_size_128+n_systems-1))

for i in $(seq $first $last);
do
        tail -n +2 ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
                   ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

        rm ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv
done



#------------------------------------------------

train_primary=150
train_secondary=150
calibrate_primary=300
calibrate_secondary=50
n_systems=25

julia src/vacancy_formation.jl $task_id \
                               $task_count \
                               $rho        \
                               $train_primary \
                               $train_secondary \
                               $calibrate_primary \
                               $calibrate_secondary \
                               $n_systems

index_size_54=549
index_size_128=659

cat ${filepath}__${index_size_54}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
    ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

rm ${filepath}__${index_size_54}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv

first=$((index_size_54+1))
last=$((index_size_54+n_systems-1))

for i in $(seq $first $last);
do
        tail -n +2 ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
                   ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

        rm ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv
done

first=$((index_size_128))
last=$((index_size_128+n_systems-1))

for i in $(seq $first $last);
do
        tail -n +2 ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv >> \
                   ${filepath}_${train_primary}_${train_secondary}_${calibrate_primary}_conformal.csv

        rm ${filepath}__${i}_${train_primary}_${train_secondary}_${calibrate_primary}_${calibrate_secondary}_conformal.csv
done



#----------------------------------------------

julia src/make_vacancy_formation_plots.jl

