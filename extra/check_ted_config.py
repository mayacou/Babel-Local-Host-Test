from datasets import get_dataset_config_names

dataset_name = "davidstap/ted_talks"
configs = get_dataset_config_names(dataset_name, trust_remote_code=True)

print("Available TED Talks Configurations:", configs)
