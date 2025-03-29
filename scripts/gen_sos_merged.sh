python src/countdown_generate.py --seed 4 --data_dir data/sos_10k_b4/sos/ --min_range 4 --start_range 4 --num_samples 10000 --text_template sos 
python src/countdown_generate.py --seed 4 --data_dir data/sos_10k_b4/sos_react/ --min_range 4 --start_range 4 --num_samples 10000 --text_template sos_react
python scripts/convert_to_sftrainer_format.py data/sos_10k_b4 data/sos_10k_b4_merged


python src/countdown_generate.py --seed 4 --data_dir data/sos_10k_b3/sos/ --min_range 3 --start_range 3 --num_samples 10000 --text_template sos 
python src/countdown_generate.py --seed 4 --data_dir data/sos_10k_b3/sos_react/ --min_range 3 --start_range 3 --num_samples 10000 --text_template sos_react
python scripts/convert_to_sftrainer_format.py data/sos_10k_b3 data/sos_10k_b3_merged 