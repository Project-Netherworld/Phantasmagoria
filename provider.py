import sys

import memory_handler
import requests
import logging

import settings_handler


class Provider:
    """
    The Base Abstract class for Providers. This defines basic provider before, like loading settings that both the
    terminal and discord providers have.
    """
    def __init__(self, a_settings_handler: settings_handler.Settings_Handler):
        """
        The ctor of the Provider class, intializes data members from the Settings_Handler class.
        :param a_settings_handler: The Settings_Handler class from which to extract the settings from.
        :type a_settings_handler: Settings_Handler
        """
        self.user_name = a_settings_handler.settings['provider_settings']["user_name"]
        self.bot_name = a_settings_handler.settings['provider_settings']["bot_name"]
        self.backend_url = a_settings_handler.settings["backend_settings"]["url"]
        self.memories = memory_handler.Memory_Handler(a_settings_handler)
        self.generation_args = a_settings_handler.retrieve_consolidated_generation_settings()
        self.tokenizer = a_settings_handler.tokenizer
        load_settings = {
            "device": self.memories.device,
            "model_settings": a_settings_handler.settings['model_settings']
        }
        self.serial_settings = {
            "chat_history": None,
            "experimental_settings": a_settings_handler.settings['experimental_settings'] if 'experimental_settings' in
                                                                                             a_settings_handler.settings.keys() else None,
            "generation_settings": self.generation_args,
            "device": self.memories.device
        }
        self.request_load(load_settings)

    def tokenize_single_token(self, a_str: str):
        """
        Tokenize_single_token- tokenizes a single string
        :description: Tokenizes a single token to be used by the logit_bias experimental sampler. Why there can only
        be one token at a time is simply because discord slashcommands do not accept lists.
        :param a_str: A string to be tokenized.
        :type a_str: str
        :return: The tokenized str.
        :type: int
        """
        tokenized_word_list = self.tokenizer.encode(a_str)
        if len(tokenized_word_list) > 1:
            raise ValueError("Cannot apply logit bias processor with more than 1 token!")
        else:
            tokenized_word = tokenized_word_list[0]
            return tokenized_word

    def detokenize_decoded_message(self, a_list_of_decoded_tokens: [[int]]):
        """
        Detokenize_decoded_message- detokenize a list of tokens, and translates them into a message.

        :param a_list_of_decoded_tokens: The tokens of which to decode.
        :type [[int]]
        :return: str, representing the now decoded tokens turned into a message
        """
        return self.tokenizer.decode(a_list_of_decoded_tokens[0], skip_special_tokens=True)

    def request_load(self, a_load_settings: dict):
        """
        Sends a request to load the model to the back end server.

        :param a_load_settings: The settings relating to loading the model.
        :type: dict
        """
        try:
            requests.post(self.backend_url + "/load", json=a_load_settings)
        except Exception as load_exception:
            logging.error("There was an error in loading the model. This program cannot run without a model. Also, \n"
                          "a common error is forgetting to add http:// to your url, so is forgetting to port forward \n"
                          "or open your port to traffic on your backend server. This only applies to those not "
                          "hosting locally.\n"
                          "Please check the following error log:")
            logging.exception(load_exception)
            sys.exit(-1)
    def request_generation(self, a_serial_settings = None):
        """
        Sends a request to generate text from the model loaded on the backend server. If unsuccessful, return int
        relating to status or an arbitrary error code.
        :param a_serial_settings: The settings relating to serialization.
        :type: None, if passed, typically dict
        :return:
        """
        bot_response = ""
        log_msg = "\n Please check the following log as well as backend logs:"

        # Needed because you can't make default parameters equal to self in Python.
        if a_serial_settings is None:
            a_serial_settings = self.serial_settings

        try:
            bot_response = requests.post(self.backend_url + "/generate", json=a_serial_settings)
            json_bot_response = bot_response.json()
            # https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
            bot_response.raise_for_status()
            return json_bot_response
        except requests.exceptions.HTTPError as http_exception:
            if http_exception.code == 400:
                logging.error("There was an error with the client during generation. "+log_msg)
            elif http_exception.code == 500:
                logging.error("There was an error with the server during generation."+log_msg)
            else:
                logging.error("Unknown HTTP generation error."+log_msg)
            logging.exception(http_exception)
            return http_exception.code
        except requests.exceptions.Timeout as timeout_exception:
            logging.error("The connection timed out."+log_msg)
            logging.exception(timeout_exception)
            return 0
        except requests.exceptions.RequestException as request_exception:
            logging.error("Something catastrophic happened with the request to generate."+log_msg)
            logging.exception(request_exception)
            return -1
        except Exception as generic_exception:
            logging.error("There was a generic exception during generation." + log_msg)
            logging.exception(generic_exception)
            return -2