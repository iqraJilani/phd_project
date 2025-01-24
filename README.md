# Parameter Tuning of 2D GA

Repository to implement the parameter tuning of 2D GA

## Description of code

## Runing the Code

python <file name>.py --dataset_name "scenario_2_training" --run_name "untuned_scenario2_weights" --prob_crossover 0.6 --prob_mutation 0.4 --w_bl 0.45 --w_pr 0.275 --w_as 0.275 

For example, 

For baseline
python ga_untuned_final.py --dataset_name "scenario_5_testing" --run_name "untuned_scenario5_baseline" --prob_crossover 0.6 --prob_mutation 0.4 --w_bl 0.4 --w_pr 0.3 --w_as 0.3 

For Tuned_settings:
python run_ga.py --dataset_name "scenario_5_testing" --run_name "tuned_scenario5" --prob_crossover 0.59 --prob_mutation 0.89 --w_bl 0.4 --w_pr 0.3 --w_as 0.3 
