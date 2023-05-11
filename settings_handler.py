import json
import torch
import front_end_utils
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoForCausalLM
import sys
import logging
import warnings
class Settings_Handler:
    """
    Handles/prepares the settings from the config json file for use by the provider classes.
    """

    def __init__(self, json_file_name = None):
        """
        Ctor for the Settings_Handler class. Reads settings from a json file.
        :param json_file_name: The name of the json file. To be put into the designated config folder.
        """
        if json_file_name is not None:
            try:
                config_file = open("config/"+json_file_name, encoding="utf8")
            except Exception as json_exception:
                logging.error("There was an error while trying to open your configuration file. Ensure you have the "
                              "right path?")
                logging.exception(json_exception)
            self.settings = json.load(config_file)
            self.tokenizer = None

            # Preliminary Basic Checks, i.e. absolutely essential settings.
            self.verify_settings_group_existence()
            self.verify_settings_existence()

            # Preparation of Input Settings
            self.concatenate_prompts()

            self.load_tokenizer_settings()
            self.tokenize_bad_words_and_force_ids()

            # Preparation of Experimental Processors
            self.replace_biased_tokens()

            config_file.close()
        else:
            print("No json config file sent!")
            sys.exit(-1)
    def concatenate_prompts(self):
        """
        Concatenates the example conversation alongside the prompt. The prompt in this case would just be the bot's
        description prior to concatenation.
        """
        if 'example_conversation' in self.settings['input_settings'].keys():
            self.settings['input_settings']['prompt'] += self.settings['input_settings']['example_conversation']

    def verify_settings_group_existence(self):
        """
        Verifies if certain essential settings groups exist i.e. settings related to the model, tokenizer, etc..
        """
        vital_settings_group_list = ["model_settings", "tokenizer_settings", "generation_settings", "input_settings", "provider_settings"]
        for vital_settings_group in vital_settings_group_list:
            if not(vital_settings_group in self.settings):
                logging.error(f"No {vital_settings_group} group provided! Check your json config file. This settings group is necessary.")
                sys.exit(-1)
    def verify_settings_existence(self):
        """
        Verifies if certain essential settings exist. I.e. the model, the device, etc..
        """
        vital_settings_list = ["prompt", "tokenizer", "model", "device", "provider_type", "bot_name", "user_name", "max_length"]

        flattened_dict = front_end_utils.flatten_nested_dictionary(self.settings)
        for vital_setting in vital_settings_list:
            if not (vital_setting in flattened_dict):
                logging.error(
                    f"No {vital_setting} setting provided! Check your json config file for this missing key or check "
                    f"if its corrupted. This setting is necessary.")
                sys.exit(-1)
        if self.settings["provider_settings"] == "discord":
            necessary_discord_settings_list = ["token", "main_guild_id"]
            for vital_setting in necessary_discord_settings_list:
                if not (vital_setting in flattened_dict):
                    logging.error(
                        f"No {vital_setting} setting provided! Check your json config file for this missing key or "
                        f"check if its corrupted. This setting is necessary. since you're using discord as a provider type.")
                    sys.exit(-1)
    def translate_tokenizer_and_model(self):
        """
        Translates the 'model' and 'tokenizer' args in settings such that it turns
        into the appropriate settings title. This is mostly done so the user doesn't have to see such an ugly arg name.
        """
        self.settings["model_settings"]["pretrained_model_name_or_path"] = self.settings['model_settings'].pop("model")
        self.settings["tokenizer_settings"]["pretrained_model_name_or_path"] = self.settings['tokenizer_settings'].pop("tokenizer")

    def set_default_settings(self):
        """
        Replace certain missing settings to their default counterparts.
        :acknowledgements: Anthony Mercurio for giving me some default settings that work well, (specifically
        eos_token_id and pad_token_id's default values.)
        """
        if not("pad_token_id" in self.settings['generation_settings']['syntax_settings']):
            print("You didn't set a pad_token_id, so auto setting it to tokenizer.eos_token_id...")
            # Allows for more open-ended generation, supposedly.
            self.settings['model_settings']["pad_token_id"] = self.tokenizer.eos_token_id

        if not ("eos_token_id" in self.settings['generation_settings']['syntax_settings']):
            # Supposedly, 198 is newline in transformers. I'm not entirely sure of this, but this is from the advice
            # of my friend. Given the fact I've had no problems setting it to this value, I believe it'll be fine.
            print("You didn't set a eos_token_id, so auto setting it to 198 (newline)...")
            self.settings['model_settings']["eos_token_id"] = 198

        if not("memory_cycler" in self.settings['netherworld_settings']):
            print("You didn't set the memory cycler, defaulting to none.")
            self.settings["memory_cycler"] = "none"

        if not("device" in self.settings["netherworld_settings"]):
            print("No device set, autosetting...")
            # I would have more options here, but AMD's ROCM is not really supported very well, and I don't even know
            # if it's supported in transformers...
            self.settings["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    def tokenize_bad_words_and_force_ids(self):
        """
        Tokenizes the bad words that are in the settings. Necessary, as the backend expects them to be in a tokenized
        format.
        """
        if 'bad_words_ids' in self.settings['generation_settings']['syntax_settings'].keys():
            self.settings['generation_settings']['syntax_settings']['bad_words_ids'] = \
                self.tokenizer(self.settings['generation_settings']['syntax_settings']['bad_words_ids'], add_special_tokens=False).input_ids
    def check_if_prompt_larger_than_max(self):
        """
        Checks if the prompt is larger than the max_length parameter to avoid
        undefined behavior. Exit if that is the case, as the program is just starting to run.
        """
        prompt_tensor = self.tokenizer.encode(self.settings['input_settings']['prompt'], return_tensors="pt")
        prompt_tensor_size = prompt_tensor.size(dim=1)
        if prompt_tensor_size > self.settings['generation_settings']['syntax_settings']['max_length']:
            print("You cannot make a prompt larger than your maximum limit! Increase max_length or shorten your prompt!")
            exit(-1)
    def retrieve_consolidated_generation_settings(self):
        """
        Returns the merged list of generation settings, including both
        syntax and sampler related settings.
        :return: The two aforementioned sub-dicts, now merged.
        :rtype: dict
        :acknowledgements - adapted from https://www.geeksforgeeks.org/python-merging-two-dictionaries/
        """
        return {**self.settings["generation_settings"]["sampler_settings"], **self.settings["generation_settings"]["syntax_settings"]}

    def load_tokenizer_settings(self):
        """
        Extracts and loads the tokenizer settings from a_settings_handler into the
        memory_handler object.

        :description: This loads the tokenizer, alongside all default values to be replaced by the tokenizer.
        (Well, almost all. Experimental_settings might still need to be refactored to be included). In addition,
        also ensures the prompt is no longer than the maximum length.

        """
        if self.tokenizer is None:
            self.translate_tokenizer_and_model()
            self.tokenizer = AutoTokenizer.from_pretrained(**(self.settings["tokenizer_settings"]))
            self.set_default_settings()
            self.check_if_prompt_larger_than_max()
            print("Tokenizer loaded.")

    def replace_biased_tokens(self):
        """
        Tokenizes the words/tokens that need to be biased in the json config file.

        :description: Prepares the string based words to be biased in the json config file. This is mostly because
        while I could've just told users to search through their model's vocab.json file, that would be tedious for
        many types of common words, puncutation, etc.. as opposed to pad_token_id and eos_token_id which are almost
        always the same, being 50256(<endoftext>) and 198(new line, supposedly). I only made it such that you can only
        have one token at a time as the experimental sampler logit_bias only biases one token at a time.
        However, the process of preparing it is rather difficult since the process involves going through several nestings.
        """
        if 'experimental_settings' in self.settings.keys() and 'experimental_processors' in self.settings['experimental_settings'].keys():
            # checks which type of experimental processor there is
            for processor, processor_values in self.settings['experimental_settings']['experimental_processors'].items():
                if processor == "logit_bias":
                    # Used for indexing. Maybe rename to itr later?
                    count = 0
                    # The original function expects a list of tuples, which are in essence pairs, so hence why
                    # I used keys and values here.
                    for word, bias in dict(processor_values).items():
                        tokenized_word_list = self.tokenizer.encode(word)
                        tokenized_word = 0
                        if len(tokenized_word_list) > 1:
                            raise ValueError("You can only bias one token at a time!")
                        else:
                            tokenized_word = tokenized_word_list[0]
                        # Replace the string with its token. The string again is only one token at most.
                        self.settings_object.settings['experimental_settings']['experimental_processors']['logit_bias'][count] = [tokenized_word, bias]
                        count+=1





