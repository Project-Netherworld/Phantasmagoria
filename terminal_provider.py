import sys

import front_end_utils
import settings_handler
import memory_handler
import requests
import json
import base64
import provider

class Terminal_Provider(provider.Provider):
    """
    The Provider Class for the terminal. Lacks certain functionality of the Discord Class, but is mainly used for
    debugging purposes, as the terminal does not have any options for clearing parts of a screen (all or nothing).
    """
    def __init__(self, a_settings_handler: settings_handler.Settings_Handler):
        """
        Calls the Provider ctor to initialize its inherited data members from a_settings_Handler
        :param a_settings_handler: The settings from which data members are initialized with.
        :type a_settings_handler: Settings_Handler
        """
        provider.Provider.__init__(self, a_settings_handler)
    def chat(self):
        """
        Simulates the user I/O between the user and the chatbot using the python terminal.

        :description: - The main bread and butter of the user I/O and the chatbot is simulated through this function,
        mostly just through basic string manipulation and sending off user settings.

        :algorithm: 1. Until the user explicitly types !quit, the following is executed:
        2. A prompt for the user's name like such is put in as a prompt: "Octavius: "
        3. If the user doesn't quit immediately, the following is executed:
        4. The typed message is sanitized such that it can be formatted correctly in the chat history and is
           appended to the chat history list.
        5. The encoded chat history, alongside various other settings are put into the dict serial_settings to be sent
           in a server response using the json dump string function.
        6. If the response is successful, decode the bot message and detokenize it.
        7. Print the bot's message and append it to chat history.
        8. Repeat until the user says !quit.
        """
        user_input = ""
        while user_input != "!quit":
            # When the user is prompted for input, it'll look something like such: "Octavius: "
            user_input = input(self.user_name + ": ")
            if user_input != "!quit":
                # overall preparation of settings, including santizing the user's message for the correct conversation
                # format and the encoding of the chat history.
                formatted_user_input = self.user_name + ": " + user_input + "\n"
                self.memories.append_message(formatted_user_input, a_tokenizer=self.tokenizer)
                chat_history_encoded = self.memories.get_encoded_chat_history(self.tokenizer)
                self.serial_settings['chat_history'] = chat_history_encoded
                serial_settings = json.dumps(serial_settings)

                # Send the chat history over to the server
                bot_response = self.request_generation(serial_settings)
                # Since this is more of a debugging provider/synchronous, exit immediately on error.
                if type(bot_response) == int:
                    sys.exit(-1)
                # Get rid of the chat history of the bot saying nothing. If we do not do this, the model will not generate
                # ANYTHING.
                if bot_response is None:
                    print(f"{self.bot_name}: I didn't know what to say, so I cleared my memories.")
                    self.memories.clear_all_memories()
                else:
                    # Decoding and printing of the bot's message. Later it's appended elsewhere.
                    decoded_bot_message = front_end_utils.decode_encoded_tokenized_tensor(bot_response)
                    detokenized_bot_message = self.detokenize_decoded_message(decoded_bot_message)
                    print(detokenized_bot_message, end="")
                    self.memories.append_message(detokenized_bot_message, a_tokenizer=self.tokenizer)
