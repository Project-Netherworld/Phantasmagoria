import json

import settings_handler
import torch
from transformers import AutoTokenizer
import sys
import front_end_utils
import logging

class Memory_Handler:
    def __init__(self, a_settings_handler : settings_handler.Settings_Handler):
        """
        The ctor for the Memory_Handler Class.
        :param a_settings_handler: the settings handler object from which to initialize private class data members to.
        :type a_settings_handler: settings_handler.Settings_Handler
        """
        self.tokenizer = None
        # Why not have it be equal to regular None? Because "none" is an option for memory cycling.
        self.memory_cycler_type = "none"
        self.extra_budget = 0
        self.device = ""
        self.max_length = None
        self.prompt = ""
        self.prompt_tensor = None
        self.chat_history = []
        self.settings_object = a_settings_handler
        self.load_netherworld_settings(a_settings_handler)
        self.load_generation_settings(a_settings_handler)

    def load_netherworld_settings(self, a_settings_handler : settings_handler.Settings_Handler):
        """
        Extracts and loads the unique program parameter settings from a_settings_handler into the
        memory_handler object.
        :param a_settings_handler: A settings_handler object from which to extract settings from.
        :type a_settings_handler: settings_handler.Settings_Handler
        """
        netherworld_dict = a_settings_handler.settings['netherworld_settings']
        # What the program will use as the memory cycler. Aka, what to chop down memory so as not to explode
        # context_length or VRAM.
        self.memory_cycler_type = netherworld_dict['memory_cycler']
        # Extra budget is the 'wiggle' room for extra input that'll shorten down the memory by an extra amount
        # such that the user has extra space to input their text. Only really relevant when using the by_token
        # memory cycler.
        self.extra_budget = netherworld_dict['extra_budget']
        # Not necessarily unique to the program, but still useful to store. Tells the backend what to convert the CPU
        # tensors into.
        self.device = netherworld_dict['device']

    def load_generation_settings(self, a_settings_handler : settings_handler.Settings_Handler):
        """
        Extracts and loads the generation settings from a_settings_handler into the
        memory_handler object.

        :description: Loads the generation settings related to memories into the object. This means this holds nothing
        about samplers, but rather only the prompt and its aforementioned tensor alongside the max_length such that
        memory cycling can be done based on the size of the prompt.

        :param a_settings_handler: A settings_handler object from which to extract settings from.
        :type a_settings_handler: settings_handler.Settings_Handler
        """

        # Due to the fact that the user has the option to not use max_length, set to None.
        self.max_length = a_settings_handler.settings['generation_settings']['syntax_settings'][
            'max_length'] if 'max_length' in a_settings_handler.settings['generation_settings'][
            'syntax_settings'] else None
        # Think of a prompt as a writing prompt. You always remember it when writing a story, and this is the
        # main spawner of the chatbot's personality. This is why it is a setting, because it always stays in the chat
        # history regardless of the cycling done. Otherwise, the bot would not be able to have a consistent personality.
        self.prompt = a_settings_handler.settings['input_settings']['prompt']
        self.prompt_tensor = a_settings_handler.tokenizer.encode(self.prompt, return_tensors="pt")
        self.chat_history.append(self.prompt)

    def dispatch_memory_cycler(self, a_tokenizer : AutoTokenizer):
        """
        Dispatches the memory cyclers either by_sentence or by_token.
        :param: a_tokenizer:
        :type a_tokenizer: AutoTokenizer
        """
        if self.memory_cycler_type == "by_sentence":
            self.memory_cycle_by_sentence(self.max_length, self.chat_history, self.prompt_tensor, a_tokenizer, self.extra_budget, )
        elif self.memory_cycler_type == "by_token":
            self.memory_cycle_by_token(self.max_length, self.chat_history, self.prompt_tensor, a_tokenizer, self.extra_budge)

    def get_encoded_chat_history(self, a_tokenizer : AutoTokenizer):
        """
        Returns the tokenized chat_history in encoded base64 string.
        :description: This function primarily implements the encoding part of a technique utilized by a start-up
        named NovelAI, which tokenizes the words on the frontend and encodes to base64 as a serialization method.
        :param: a_tokenizer
        :type: AutoTokenizer
        :return: The base64 string of the encoded tokenized chat history
        :rtype: str
        """
        # curtosey of Antony Mercurio, who recommended me to utilize this method and giving me this code.
        string_chat_history = ''.join(self.chat_history)
        tokenized_chat_history = a_tokenizer.encode(string_chat_history)
        encoded_str = front_end_utils.get_encoded_str_from_token_list(tokenized_chat_history)
        return encoded_str

    def memory_cycle_by_token(self, a_max_length, a_chat_history, a_prompt_tensor, a_tokenizer, a_extra_budget=0):
        """
        Cycles through the chatbot's memory via sentence based on the size of
        its parameters. This is to perserve VRAM at the expense of the size of saving messages.

        :description: This function primarily cycles through the
        chatbot's memory through the use of removal of words. Does so via subtracting

        :algorithm: 1. Remove the prompt. This is because we need our lengths to be precise excluding the prompt.
        2. Encode a_chat_history into a string and get the size of it after clearing the chat history.
        3. Double check that the prompt size is not larger than the max size. Odd to do here, but necessary as otherwise
        the memory cycling operation is for naught.
        4.  Get the start of where to truncate, i.e. where the prompt ends. This is done by getting the absolute
        value of the difference a_max_length prompt_tensor_size.
        5. Get the distance from which to truncate the chat.
        6. Check if the size of the chat history and prompt is larger than the max amount of length plus the extra budget.
        if so, perform a tensor narrowing operation with the values mentioned in 3) and 4).
        7. Decode the tokenized tensor and then put the truncated chat history into chat history after clearing. This
        also includes the prompt, which is always the 0th element.

        :param a_max_length: The maximum limit of tokens that the chat_history can be contained, see the link below:
        https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.max_length
        :type a_max_length: int
        :param a_chat_history: The  chat history in list form. The original prompt is always the 0th element.
        :type a_chat_history: [str]
        :param a_prompt_tensor: The prompt in tensor form. Needed mostly in such a form for size comparison purposes.
        :type a_prompt_tensor: torch.Tensor, of varying data type.
        :param: a_tokenizer: The tokenizer from which to tokenize the chat prompt.
        :param a_extra_budget: The "wiggle room" given for new tokens to generate. This is an optional safeguard so more
        words can be generated. for instance, while our chat_history might be
        cut down to 199 whereas our max_limit is 200, the chatbot may struggle since it can only generate 1 more
        token.
        :type a_extra_budget: int
        :return The chat history, now truncated.
        :rtype: [str]
        """
        original_prompt = a_chat_history.pop(0)
        # Turn the chat history into a string, as we'll need to encode it using our tokenizer for size comparisons.
        chat_history_str = ''.join(a_chat_history)
        # remove the prompt, as we'll be adding it back later.
        # Gather the size of the chat and prompt tensors for direct comparison.
        chat_history_tensor = a_tokenizer.encode(chat_history_str, return_tensors="pt")
        chat_history_tensor_size = chat_history_tensor.size(dim=1)
        prompt_tensor_size = a_prompt_tensor.size(dim=1)

        # In the case that the prompt is larger than our max_limit, greatly warn the user. There's no way we can
        # fix this, as in essence, the prompt would be the main force in driving a character's personality. To lose
        # it is to lose data crucial to be fed to the model.
        if prompt_tensor_size > a_max_length:
            logging.critical(
                "Your prompt is larger than your largest max length. Please increase your max_length parameter in "
                "your settings!")


        # Does the math for the start and end of what should be the acceptable bounds for the new shortened chat history.
        # Determined mostly by the size of the chat history and the max length. Max length can be analogous to context
        # length in this case.
        start_of_truncated_history = abs(a_max_length - prompt_tensor_size)
        distance_for_truncated_history = chat_history_tensor_size - start_of_truncated_history
        a_chat_history.clear()


        if chat_history_tensor_size + prompt_tensor_size > a_max_length - a_extra_budget:
            # Truncates the chat history utilizing a tensor narrowing operation (i.e. reducing the dimensions),
            # aka getting rid of tokens in the tensor.
            truncated_chat_history_tensor = torch.narrow(chat_history_tensor, 1, start_of_truncated_history,
                                                         distance_for_truncated_history)
            decoded_truncated_messages = a_tokenizer.batch_decode(truncated_chat_history_tensor, skip_special_tokens=True)
            # Necessary as each chat messages is divided by a new line. For instance,
            # The chat will always look something similar to the following:
            # User: Hello. \n Bot: Hi. \n
            # The chat_history list will always look something similar to chat, hence the split.
            a_chat_history.extend(decoded_truncated_messages[0].split('\n'))
        a_chat_history.insert(0, original_prompt)
        return a_chat_history

    def memory_cycle_by_sentence(self, a_max_length, a_chat_history, a_prompt_tensor, a_tokenizer, a_extra_budget=0):
        """
        Cycles through the chatbot's memory via sentence based on the size of
        a_chat_history, a_max_limit, and a_prompt_tensor.

        :description: This function primarily cycles through the
        chatbot's memory through the use of removal of sentences, as seen in earlier functions. It does so by first
        turning a_chat_history into a string, encoding it, and getting the size of the encoded string,
        and continually removing elements of a_chat_history until it is less than or equal to the size of a_max_length
        subtracted from a_extra_budget, and warning the user if the chat_history is completely cleared. In the event
        of the prompt being bigger than the entire max_limit, the program exits. (This is mostly in case the user
        changes the max_limit settings, that the quota of max size is met).

        1. Remove the prompt, as we'll need to make sure to re-add it later, but we don't want the prompt to mess up
        the length calculations.
        2. Encode a_chat_history into a string and get the size of it after clearing the chat history.
        3. Double check that the prompt size is not larger than the max size. Odd to do here, but necessary as otherwise
        the memory cycling operation is for naught.
        4. Get the start of where to truncate, i.e. where the prompt ends. This is done by getting the absolute
        value of the difference a_max_length prompt_tensor_size.
        5. Check if the size of the chat history and prompt is larger than the max amount of length plus the extra budget.
        if so, continually pop from the list (oldest messages go first)
        6. Decode the tokenized tensor and then put the truncated chat history into chat history after clearing.
        7. Retensorize the chat history for checking sizes, as this function will be done continually.
        8. Re-add the prompt.

        :param a_max_length: The maximum limit of
        tokens that the chat_history can be contained,see below for more:
        https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.max_length
        :type a_max_length: int
        :param a_chat_history: The chat history in list form. The original prompt is always the
        0th element.
        :type a_chat_history: [str]
        :param a_prompt_tensor: The prompt in tensor form. Needed mostly in such a form for size
        comparison purposes.
        :type a_prompt_tensor: torch.Tensor, of varying data type.
        :param: a_tokenizer: The tokenizer to encode the prompt.
        :type: AutoTokenizer
        :param a_extra_budget: The "wiggle room" given for new tokens to generate. This is an
        optional safeguard so more words can be generated. for instance, while our chat_history might be cut down to
        199 whereas our max_limit is 200, the chatbot may struggle since it can only generate 1 more token.
        :type a_extra_budget: int
        :return The chat history, now truncated.
        :rtype [str]
        """

        # remove the prompt, as we'll be adding it back later.
        original_prompt = a_chat_history.pop(0)
        # Turn the chat history into a string, as we'll need to encode it using our tokenizer for size comparisons.
        chat_history_str = ''.join(a_chat_history)
        # Gather the size of the chat and prompt tensors for direct comparison.
        chat_history_tensor = a_tokenizer.encode(chat_history_str, return_tensors="pt")
        chat_history_tensor_size = chat_history_tensor.size(dim=1)
        prompt_tensor_size = a_prompt_tensor.size(dim=1)
        # Left behind variable. May need for something else, I just have a gut feeling thus far, so hence why it's
        # not removed just yet... chat_history_excluding_prompt_size = chat_history_tensor_size - prompt_tensor_size

        # In the case that the prompt is larger than our max_limit, greatly warn the user. There's no way we can
        # fix this, as in essence, the prompt would be the main force in driving a character's personality. To lose
        # it is to lose data crucial to be fed to the model.
        if prompt_tensor_size > a_max_length:
            logging.critical(
                "Your prompt is larger than your largest max limit. Please increase your max_limit parameter in your "
                "settings!")

        # Continually loop until the chat history's size is less than difference between the max amount of tokens and
        # extra wiggle room provided for generation. Each time, the sizes are updated, and gradually, each oldest
        # part of the conversation history is removed.
        while chat_history_tensor_size + prompt_tensor_size > a_max_length - a_extra_budget and len(a_chat_history) > 0:
            # As we're using append, the question and response will always be the second last elements of the
            # chat_history list.
            if (len(a_chat_history) == 2):
                print(
                    "WARNING: Your most recent message to the bot has been trimmed in short term memory! Consider "
                    "increasing the max_limit parameter or lowering the extra_budget parameter.")
            elif (len(a_chat_history) == 1):
                print(
                    "WARNING: The chatbot's most recent message has been trimmed in short term memory! Consider "
                    "increasing the max_limit parameter or lowering the extra_budget parameter.")
            a_chat_history.pop(0)
            chat_history_str = ''.join(a_chat_history)
            chat_history_tensor = a_tokenizer.encode(chat_history_str, return_tensors="pt")
            chat_history_tensor_size = chat_history_tensor.size(dim=1)
            prompt_tensor_size = a_prompt_tensor.size(dim=1)
            # Left behind variable. May need for something else, I just have a gut feeling thus far, so hence why
            # it's not removed just yet... chat_history_excluding_prompt_size = chat_history_tensor_size -
            # prompt_tensor_size

        a_chat_history.insert(0, original_prompt)
        return a_chat_history

    def append_message(self, a_message, a_tokenizer, a_cycle_through_memory = True):
        """

        :name append_message: appends a message to the chat history and ensures it does not go over the memory limit.
        :param a_message: the message to append to the chat history.
        :type a_message: str
        :param a_tokenizer: The tokenizer to help encode the prompt.
        :type a_tokenizer: AutoTokenizer
        :param a_cycle_through_memory: Whether to cycle to through the memory.
        :type a_cycle_through_memory: str
        """
        self.chat_history.append(a_message)
        if a_cycle_through_memory:
            self.dispatch_memory_cycler(a_tokenizer)
    def clear_all_memories(self):
        """
        Clears the chat history, thus, wiping out all memories of the chat bot.
        """
        self.chat_history.clear()
        self.chat_history.append(self.prompt)
    def pop_memory(self, a_index = -1):
        """
        A sort of 'proxy' function that calls the pop function of a list specifically for
        self.chat_history. Mostly as the pop syntax looks ugly by itself.
        :param a_index: the index of the chat_history from which to pop.
        :type a_index: int
        :return: The popped message from the chat history.
        :rtype: str
        """
        return self.chat_history.pop(a_index)
