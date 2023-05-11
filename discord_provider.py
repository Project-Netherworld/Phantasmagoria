import logging
from typing import Optional

import discord
from discord import app_commands

import front_end_utils
import settings_handler
import json
import requests
import provider
import discord.app_commands



class Discord_Provider(discord.Client, provider.Provider):
    """
    The Main Provider aka client responsible for User I/O for Discord, the social media site. Inherits from
    discord.Client to overwrite certain event driven functions, like detecting if a message has been sent # or if the
    bot has connected to the discord API.
    """
    def __init__(self, a_settings_handler):
        """
        The ctor for the Discord_Provider object. Initializes all possible data members and
        assigns functions to respond to certain slash commands. Calls both discord.Client and provider.Provider
        super ctor for extra functions and data members.
        """
        # Preparing of bot for being able to read and write through the Discord API.
        intents = discord.Intents.default()
        intents.message_content = True

        # Super ctor class for accessibility to functions and inherited data members
        discord.Client.__init__(self=self, intents=intents)
        provider.Provider.__init__(self=self, a_settings_handler=a_settings_handler)

        self.discord_provider_settings = a_settings_handler.settings['provider_settings']
        # necessary setting, cannot log bot into Discord without token.
        self.token = self.discord_provider_settings['token']
        self.bot_nicknames = self.discord_provider_settings['bot_nicknames']

        self.status_type = self.discord_provider_settings[
            'status_type'] if 'status_type' in self.discord_provider_settings.keys() else None
        self.status_body = self.discord_provider_settings[
            'status_body'] if 'status_body' in self.discord_provider_settings.keys() else None
        # Aka the discord server the bot will be used the most in. Needed for bot command syncing.
        self.main_guild_id = self.discord_provider_settings[
            'main_guild_id'] if 'main_guild_id' in self.discord_provider_settings.keys() else None
        # Aka a way to send messages 'impersonating' a user. Will be updated such that this won't be necessary
        # eventually.
        self.webhook_url = self.discord_provider_settings[
            'webhook_url'] if 'webhook_url' in self.discord_provider_settings.keys() else None
        # Aka if the bot always responds, regardless of what is typed. This does not apply to system messages, however.
        # Discord bots can never reply to system messages.
        self.conditional_response = self.discord_provider_settings[
            'conditional_response'] if 'conditional_response' in self.discord_provider_settings.keys() and \
                                       self.discord_provider_settings[
                                           'conditional_response'] is True else False

        # Initializes all slash commands such that they can be used in what the user defined their server as.
        self.tree = app_commands.CommandTree(self)
        self.assign_slash_commands()

    async def get_or_create_webhook(self, a_interaction: discord.Interaction):
        """
        Gets a pre-existing web-hook from the discord server's channel that the bot is in, or makes a new one
        in that same channel.

        :param a_interaction: The interaction which sprung from the user using a slash command. In this case,
        particularly useful to extract the specific channel the command was sent from to make the webhook.
        :type a_interaction: discord.Interaction
        :return: The pre-existing webhook's url or the new webhook's url.
        :rtype: str
        """
        channel = a_interaction.channel
        webhooks = await channel.webhooks()
        for webhook in webhooks:
            if webhook.name == "Netherworld Webhook":
                return webhook.url
        new_webhook = await channel.create_webhook(name="Netherworld Webhook")
        return new_webhook.url

    def assign_slash_commands(self):
        """
        Umbrella function for assigning functions, mostly since decorators need self
        to assign functions.

        :description: This function is in essence, a compromise between dealing with decorators and events. This is
        mostly as in my attempts to using decorators with OOP, it doesn't work well except when I use a function
        to act as a proxy. Hence, this function is hilariously long, but only due to the nature of the

        """

        @self.tree.command(name="autocomplete", description="Predict what you'll say next!",
                           guild=discord.Object(self.main_guild_id))
        async def autocomplete(a_interaction: discord.Interaction):
            """
            name - autocomplete - Attempts to predict what the user will say next and sends said prediction via
            webhook.

            description: The associated function for a slash command of the same name.
            This function attempts to predict what the user will say next by appending their username into
            the chat history and letting the bot fill in the rest. The function utilizes a
            web hook defined in settings (soon to be removed in favor of just creating webhooks on the fly).
            Note that this function will not work if the setting is invalid. Here is the algorithm -

            1. Defer the response to let discord wait for a response.
            2. Append the username of whoever used the command.
            3. Encode the chat history into base64.
            4. Remove experimental settings for autocompletion, as we're trying to mimic ourselves, not the bot.
               For example, let's say we're trying to get our bot to speak with a certain puncutation mark in mind to
               appear frequently(via logit biases).
               We do not want that same frequency to appear in predicting the user's speech.
            5. Serialize the settings and re-add the experimental settings.
            6. Send a request with the serialized settings excluding the experimental settings.
            7. If our generation is valid, decode and detokenize our message retrieved from the server response.
            8. Remove the autocompleted message from the memory, as otherwise, this message will appear twice in the
               text conversation with the way the bot is set up.
            9. Prepare the webhook payload. This will be our means of impersonating the user, as we can get their
               username and avatar via the fact they used the command.
            10. Send over the payload and check the response, sending a mesage if there are any errors.

            :param a_interaction: The interaction which sprung from the user using a slash command. This will help
            extract the user's username, avatar, to mimic them using a webhook.
            :type a_interaction: discord.Interaction
            """
            # Necessary so as not to show that on the client it won't say "application did not respond."
            await a_interaction.response.defer()

            # Appends the username of whoever sent the command. This is necessary to tell the model "guess what
            # this user will say next."
            user_name = a_interaction.user.name
            self.memories.append_message(user_name + ":", a_tokenizer=self.tokenizer)

            # Preparation of the data to send over through the request.
            self.serial_settings['chat_history'] = self.memories.get_encoded_chat_history(a_tokenizer=self.tokenizer)
            # Necessary to avoid certain logit biased puncutation/tokens appearing in the user's predicted speech.
            experimental_settings_temp_removed = self.serial_settings.pop('experimental_settings')
            serial_settings = json.dumps(self.serial_settings)
            self.serial_settings['experimental_settings'] = experimental_settings_temp_removed

            # Send over serialized settings for text generation, and decode the generated message.
            bot_response = self.request_generation(serial_settings)

            if await self.validate_generation(bot_response, a_interaction=a_interaction):
                decoded_bot_message = front_end_utils.decode_encoded_tokenized_tensor(bot_response)
                detokenized_bot_message = self.detokenize_decoded_message(decoded_bot_message)

                # Necessary so this message does not appear twice in our chat history.
                if len(self.memories.chat_history) > 1:
                    self.memories.pop_memory()

                # Since the interaction has the user's username and avatar, we can use it to simulate their appearance
                # via webhook, though webhooks work like tiny servers which we can respond to, we will need to set a
                # json payload.
                web_hook_payload = {
                    "content": detokenized_bot_message,
                    "username": user_name,
                    "avatar_url": a_interaction.user.avatar.url
                }
                stringified_payload = json.dumps(web_hook_payload)
                webhook_url = await self.get_or_create_webhook(a_interaction)
                response = requests.post(webhook_url, data=stringified_payload,
                                         headers={"Content-Type": "application/json"})
                if response.status_code == 204:
                    # Clears the message such that it's empty. We only need the webhook's message which is sent seperately,
                    # so we delete the message on command.
                    await a_interaction.delete_original_response()
                else:
                    logging.error("The request to the webhook failed.")
                    await a_interaction.followup.send(f"There was an error, please check the console output. "
                                                      f"Status code: {response.status_code}")
        @self.tree.command(name="change_settings", description="Temporarily change settings.",
                           guild=discord.Object(self.main_guild_id))
        @app_commands.describe(
            do_sample="Whether to allow stochastic and multinomial sampling. Necessary for many params.",
            max_length="The maximum allowed amount of tokens.",
            eos_token_id="The id of the token the model should end a sentence with.",
            pad_token_id="The id of the token to pad empty spaces in a sentence with.",
            top_k="A sampler that allows you to sample from the best of k tokens (>=0)",
            top_p="A sampler which only takes the tokens whose probabilities add up to top_p or higher. (0.0-1.0)",
            penalty_alpha="The alpha value for contrastive search. (0.0-1.0) ",
            temperature="A sampler that makes values more randomly distributed at higher temperatures. (>=0.0)",
            repetition_penalty="A sampler which penalizes repetitive tokens. (>=1.0)",
            min_length="The minimum number of tokens to be generated.",
            typical_p="Samples similar to typical sampling in statistics. (0.0-1.0)",
            top_a="Similar to top_p, albeit squares the probability times the threshold. (0.0-1.0)",
            tfs="A complicated sampler. Increase creativity at expense of coherency. (0.0-1.0)",
            short_word_to_bias="Used with word_bias_threshold. The (singular) token to be biased.",
            word_bias_threshold="Used with short_word_to_bias. The (float) threshold with which to bias a token.",
            max_time="The maximum amount of time the model can use for generation.")
        async def change_settings(a_interaction: discord.Interaction, do_sample: bool = None, max_length: int = 1000,
                                  temperature: float = None, top_k: int = None, top_p: float = None,
                                  penalty_alpha: float = None,
                                  repetition_penalty: float = None, typical_p: float = None, eos_token_id: int = 198,
                                  pad_token_id: int = 50256, min_length: int = None, top_a: float = None,
                                  tfs: float = None,
                                  short_word_to_bias: str = None, word_bias_threshold: float = None,
                                  max_time: float = None):
            """
            Changes the settings of the Discord_Provider Class Temporarily. No Changes are saved after shutdown.

            :description: In essence, this function is a way to change settings on the fly. Don't like how your bot is
            generating text? Well, this is an easy way to experiment solutions. A few caveats are the fact that the same
            restrictions apply from the generation command, namely, you can only bias one word at a time since discord
            does not have native list support for slash commands.

            :algorithm: 1. Defer the response so as let Discord know to wait for a response.
            2. Update the class body member's generation settings to match the ones passed through this command.
            3. Create the experimental_settings key.
            4. Create the experimental_warpers key within experimental_settings if top_a or tfs are passed.
            5. Create the experimental_processors key within experimental_settings if short_word_to_bias and word_bias_threshold
            are passed.
            6. Update both the generation and experimental serial settings.

            :param a_interaction: The interaction which sprung from the user using a slash command. In this case,
            helps us get all the parameters for the command to generate text.
            :type a_interaction: discord.Interaction
            :param prompt: The prompt the model should generate text based off of.
            :type prompt: str
            :param do_sample: Whether to allow stochastic and multinominal sampling. Necessary for many params.
            :type do_sample: bool
            :param max_length: The maximum allowed amount of tokens
            :type max_length: float
            :param temperature: A sampler that makes values more randomly distributed at higher temperatures. (>=0.0)
            :type temperature: float
            :param top_k: A sampler that allows you to sample from the best of k tokens (>=0)
            :type top_k: int
            :param top_p: A sampler which only takes the tokens whose probabilities add up to top_p or higher. (0.0-1.0)
            :type top_p: float
            :param penalty_alpha: The hyperparameter alpha for contrastive search. A higher value penalizes homogenous results. (0.0-1.0)
            :type penalty_alpha: float
            :param repetition_penalty:  A sampler which penalizes repetitive tokens. (>=1.0)
            :type repetition_penalty: float
            :param typical_p: Samples similar to typical sampling in statistics. (0.0-1.0)
            :type typical_p: float
            :param eos_token_id: The id of the token the model should end a sentence with.
            :type eos_token_id: int
            :param pad_token_id: The id of the token to pad empty spaces in a sentence with.
            :type pad_token_id: int
            :param min_length: The minimum number of tokens to be generated.
            :type min_length: int
            :param top_a: Similar to top_p, albeit squares the probability times the threshold. (0.0-1.0)
            :type top_a: float
            :param tfs: A complicated sampler. Increase creativity at expense of coherency. (0.0-1.0)
            :type tfs: float
            :param short_word_to_bias: Used with word_bias_threshold. The (singular) token to be biased.
            :type short_word_to_bias: str
            :param word_bias_threshold: Used with short_word_to_bias. The (float) threshold with which to bias a token.
            :type word_bias_threshold: float
            :param max_time: The maximum amount of time the model can use for generation.
            :type max_time: float
            """
            # Necessary so as not to show that on the client it won't say "application did not respond."
            await a_interaction.response.defer()

            self.generation_args = {
                "max_length": max_length,
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
                "top_k": top_k,
                "top_p": top_p,
                "penalty_alpha": penalty_alpha,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "typical_p": typical_p,
                "min_length": min_length,
                "max_time": max_time,
                "do_sample": do_sample,
                "bad_words_ids": self.generation_args[
                    'bad_words_ids'] if 'bad_words_ids' in self.generation_args.keys() else None,
                "forced_words_ids": self.generation_args[
                    'forced_words_ids'] if 'forced_words_ids' in self.generation_args.keys() else None
            }

            self.experimental_args = {
                "experimental_warpers": {
                },
                "experimental_processors": {
                }
            }
            self.memories.max_length = max_length
            if tfs is not None:
                self.experimental_args['experimental_warpers']['tfs'] = tfs
            if top_a is not None:
                self.experimental_args['experimental_warpers']['top_a'] = top_a
            if short_word_to_bias is not None and word_bias_threshold is not None:
                # The doubly nested list is necessary as JSON cannot serialize tuples.
                self.experimental_args['experimental_processors']["logit_bias"] = [
                    [self.tokenize_single_token(short_word_to_bias), word_bias_threshold]]
            # Finally, update the serializable version of the settings. This is only pertinent to the discord provider.
            self.serial_settings['experimental_settings'] = self.experimental_args
            self.serial_settings['generation_settings'] = self.generation_args

            await a_interaction.followup.send("Successfully changed temporary settings.")

        @self.tree.command(name="generate", description="Generate text given a prompt!",
                           guild=discord.Object(self.main_guild_id))
        @app_commands.describe(
        prompt = "The writing prompt for the bot.",
        do_sample = "Whether to allow stochastic and multinomial sampling. Necessary for many params.",
        max_new_tokens = "The maximum amount of tokens in integers to generate.",
        eos_token_id = "The id of the token the model should end a sentence with.",
        pad_token_id = "The id of the token to pad empty spaces in a sentence with.",
        top_k = "A sampler that allows you to sample from the best of k tokens (>=0)",
        top_p = "A sampler which only takes the tokens whose probabilities add up to top_p or higher. (0.0-1.0)",
        penalty_alpha = "The alpha value for contrastive search. (0.0-1.0) ",
        temperature = "A sampler that makes values more randomly distributed at higher temperatures. (>=0.0)",
        repetition_penalty = "A sampler which penalizes repetitive tokens. (>=1.0)",
        min_length = "The minimum number of tokens to be generated.",
        typical_p = "Samples similar to typical sampling in statistics. (0.0-1.0)",
        top_a = "Similar to top_p, albeit squares the probability times the threshold. (0.0-1.0)",
        tfs = "A complicated sampler. Increase creativity at expense of coherency. (0.0-1.0)",
        short_word_to_bias = "Used with word_bias_threshold. The (singular) token to be biased.",
        word_bias_threshold = "Used with short_word_to_bias. The (float) threshold with which to bias a token.",
        max_time = "The maximum amount of time the model can use for generation.")
        async def generation(a_interaction: discord.Interaction, prompt: str, do_sample: bool = None,
                             max_new_tokens: int = 200,
                             temperature: float = None, top_k: int = None, top_p: float = None,
                             penalty_alpha: float = None,
                             repetition_penalty: float = None, typical_p: float = None, eos_token_id: int = 198,
                             pad_token_id: int = 50256, min_length: int = None, top_a: float = None, tfs: float = None,
                             short_word_to_bias: str = None, word_bias_threshold: float = None, max_time: float = None):
            """
            A function that allows the user to bypass settings as well as memories and directly send text generation
            settings to the server.

            :description: This function which responds to a discord slash command of the same name, the generate
            function is quite odd in that it allows the user to bypass any and all pre-defined settings and
            simply ping the server with their own parameters. As a result, this function has quite an amazing amount
            of parameters. Note that none of these generations will be saved in chat history.
            In addition, this was primarily added for a second use case of the models - text generation. Typically,
            GPT models are finetuned on conversations but their base models are trained for text generation. Hence,
            it'd be a shame not to use this feature.

            :algorithm: 1. Defer the response so as let Discord know to wait for a response.
            2. Encode the prompt that was passed.
            3. Prepare both the generation and experimental settings.
            4. Create a generation payload with the generation settings and experimental settings, and serializes it.
            5. Send a response to the server using said payload.
            6. Validate the response. If valid, receive and decode the response.
            7. Send the decoded message back through the bot.

            :param a_interaction: The interaction which sprung from the user using a slash command. In this case,
            helps us get all the parameters for the command to generate text.
            :type a_interaction: discord.Interaction
            :param prompt: The prompt the model should generate text based off of.
            :type prompt: str
            :param do_sample: Whether to allow stochastic and multinominal sampling. Necessary for many params.
            :type do_sample: bool
            :param max_new_tokens: The amount of tokens in integers to generate.
            :type max_new_tokens: float
            :param temperature: A sampler that makes values more randomly distributed at higher temperatures. (>=0.0)
            :type temperature: float
            :param top_k: A sampler that allows you to sample from the best of k tokens (>=0)
            :type top_k: int
            :param top_p: A sampler which only takes the tokens whose probabilities add up to top_p or higher. (0.0-1.0)
            :type top_p: float
            :param penalty_alpha: The hyperparameter alpha for contrastive search. A higher value penalizes homogenous results. (0.0-1.0)
            :type penalty_alpha: float
            :param repetition_penalty:  A sampler which penalizes repetitive tokens. (>=1.0)
            :type repetition_penalty: float
            :param typical_p: Samples similar to typical sampling in statistics. (0.0-1.0)
            :type typical_p: float
            :param eos_token_id: The id of the token the model should end a sentence with.
            :type eos_token_id: int
            :param pad_token_id: The id of the token to pad empty spaces in a sentence with.
            :type pad_token_id: int
            :param min_length: The minimum number of tokens to be generated.
            :type min_length: int
            :param top_a: Similar to top_p, albeit squares the probability times the threshold. (0.0-1.0)
            :type top_a: float
            :param tfs: A complicated sampler. Increase creativity at expense of coherency. (0.0-1.0)
            :type tfs: float
            :param short_word_to_bias: Used with word_bias_threshold. The (singular) token to be biased.
            :type short_word_to_bias: str
            :param word_bias_threshold: Used with short_word_to_bias. The (float) threshold with which to bias a token.
            :type word_bias_threshold: float
            :param max_time: The maximum amount of time the model can use for generation.
            :type max_time: float
            """

            # Necessary so as not to show that on the client it won't say "application did not respond."
            await a_interaction.response.defer()
            # Encode the prompt given into a list of integers.
            prompt_list = self.tokenizer.encode(prompt)
            generation_args = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
                "top_k": top_k,
                "top_p": top_p,
                "penalty_alpha": penalty_alpha,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "typical_p": typical_p,
                "min_length": min_length,
                "max_time": max_time,
                "do_sample": do_sample,
                "bad_words_ids": self.generation_args[
                    'bad_words_ids'] if 'bad_words_ids' in self.generation_args.keys() else None,
                "forced_words_ids": self.generation_args[
                    'forced_words_ids'] if 'forced_words_ids' in self.generation_args.keys() else None
            }

            experimental_args = {
                "experimental_warpers": {
                },
                "experimental_processors": {
                }
            }

            if tfs is not None:
                experimental_args['experimental_warpers']['tfs'] = tfs
            if top_a is not None:
                experimental_args['experimental_warpers']['top_a'] = top_a
            if short_word_to_bias is not None and word_bias_threshold is not None:
                # The doubly nested list is necessary as JSON cannot serialize tuples.
                experimental_args['experimental_processors']["logit_bias"] = [
                    [self.tokenize_single_token(short_word_to_bias), word_bias_threshold]]

            generation_payload = {
                "generation_settings": generation_args,
                "chat_history": front_end_utils.get_encoded_str_from_token_list(prompt_list),
                "device": self.memories.device,
                "experimental_settings": experimental_args
            }

            serial_settings = json.dumps(generation_payload)
            bot_response = self.request_generation(serial_settings)
            if await self.validate_generation(bot_response, a_interaction= a_interaction):
                decoded_bot_message = front_end_utils.decode_encoded_tokenized_tensor(bot_response)
                detokenized_bot_message = self.detokenize_decoded_message(decoded_bot_message)
                embedded_message = discord.Embed(title=f"{prompt}",
                                                 description=detokenized_bot_message)
                await a_interaction.followup.send(embed=embedded_message)

        @self.tree.command(name="display_memories",
                           description="Print the content's of the chat bot's memory (excludes prompt).")
        async def display_memories(a_interaction: discord.Interaction):
            """
            Displays all the memories of the chatbot into a discord embedded that contains
            all the elements of the chat_history list.

            :param a_interaction: The interaction which sprung from the user using a slash command. This will various
            useful information, such as the message, the user who used said command, etc..
            :type a_interaction: discord.Interaction
            """
            # Necessary so as not to show that on the client it won't say "application did not respond."
            await a_interaction.response.defer()
            embedded_message = discord.Embed(title="Current Messages in Memory",
                                             description="Heaven Knows, Earth Knows, I Know, You Know")

            if len(self.memories.chat_history) < 2:
                embedded_message.description = "No memories in chat history. Silence is Golden."
            else:
                for itr in range(1, len(self.memories.chat_history)):
                    embedded_message.add_field(name="memory " + str(itr) + ": ", value=self.memories.chat_history[itr],
                                               inline=False)
            await a_interaction.followup.send(embed=embedded_message)

        @self.tree.command(name="clear_memories",
                           description="Clears all memories and starts conversation from scratch.")
        async def clear_all_memories(a_interaction: discord.Interaction):
            """
            Clears the chat history.

            :param a_interaction: The interaction which sprung from the user using a slash command. This will various
            useful information, such as the message, the user who used said command, etc..
            :type a_interaction: discord.Interaction
            """
            # Necessary so as not to show that on the client it won't say "application did not respond."
            await a_interaction.response.defer()
            self.memories.clear_all_memories()
            await a_interaction.followup.send("Successfully cleared memories.")

        @self.tree.command(name="regenerate_response", description="Regenerates the bot's last response.",
                           guild=discord.Object(self.main_guild_id))
        async def regenerate(a_interaction: discord.Interaction):
            """
            Deletes the bot's most recent response and regenerates a new one under certain conditions.

            :description: Deletes the bot's most recent response and regenerates a new one under certain
            conditions. Responds to a discord command of the same name. The circumstances for regeneration are:
            1. Chat history must not be empty
            2. The person who requested a regeneration must be the same person the bot responded to last
            3. The most recent message must be from the bot
            These restrictions are mostly necessary, otherwise error checking and the use of this command would be
            horribly convoluted due to discord's restrictions on chat history. See further comments below.

            :algorithm: 1. Defers the response to tell Discord to wait for a response longer
            2. Checks if the chat history is populated or not on both the discord server and the front end.
            3. Go through the various checks to ensure all conditions are met.
            4. In the case of checking for if the bot reference is the same, check the discord cache or request the
               reference.
            5. If all conditions are met, delete the last message from the discord client and chat history,
               and regenerate a response once more, following the same procedure as a typical chat.
            :param a_interaction: The interaction which sprung from the user using a slash command. This will various
            useful information, such as the message, the user who used said command, etc..
            :type a_interaction: discord.Interaction
            :acknowledgements: Large chunks of the function are borrowed from the following code below.
            1. Checking for message replies.
            https://stackoverflow.com/questions/66956261/check-if-message-reply-is-a-reply-type-message-discord-py
            2. Checking if a user replied to the bot.
            https://stackoverflow.com/questions/66016979/discord-py-send-a-different-message-if-a-user-replies-to-my-bot
            3. Checking if the user referenced text within our nickname list.
            https://stackoverflow.com/questions/68784024/discord-py-how-to-check-if-a-message-contains-text-from-a-list
            """
            # Necessary so as not to show that on the client it won't say "application did not respond."
            await a_interaction.response.defer()

            # check if chat history in memory is empty
            if len(self.memories.chat_history) < 2:
                await a_interaction.followup.send(
                    "There is no prior chat history. There is nothing to regenerate!")
                return

            message = ""
            history_itr = 0
            # check if chat history in the discord channel is empty
            # note: while this is very ugly, there is no known other way to work with this, as this is an async
            # generator, meaning I have to loop through it manually, not index it, unfortunately. Can't fight the API.
            async for messages in a_interaction.channel.history(limit=2):
                if history_itr == 1:
                    message = messages
                history_itr = history_itr + 1

            # If the same person who requested the regeneration is the same person who the bot's most recent reply was
            # to.
            if message.author.id == self.user.id:
                # https://stackoverflow.com/questions/66956261/check-if-message-reply-is-a-reply-type-message-discord-py
                # https://stackoverflow.com/questions/66016979/discord-py-send-a-different-message-if-a-user-replies-to-my-bot
                if message.reference is not None:
                    original_message = None
                    if message.reference.cached_message is None:
                        # Fetching the message
                        channel = discord.Client.get_channel(message.reference.channel_id)
                        original_message = await channel.fetch_message(message.reference.message_id)
                    else:
                        original_message = message.reference.cached_message
                    if original_message is not None and original_message.author.id == a_interaction.user.id:
                        self.memories.pop_memory()
                        self.memories.pop_memory()
                        await message.delete()
                        await self.send_discord_message(original_message)
                        await a_interaction.delete_original_response()
                        return
                    else:
                        await a_interaction.followup.send(
                            "Last message was a response to someone else. Only regen responses sent to you! ")
                        return
                else:
                    await a_interaction.followup.send(
                        "The most recent message didn't include a reference from the bot.")

            else:
                await a_interaction.followup.send(
                    "The most recent message wasn't a bot message! Only regen the most recent bot message.")
                return

    async def send_discord_message(self, a_message):
        """
        Given a message, come up with a response and append it to chat history.

        :description: Given a discord message that's sent in a channel where the bot is in and can see, the bot
        will come up with a response to this message and append it to chat history. There is some naunce to this,
        however, as for instance, we don't want our reply to be for instance: "Miller: I'm having a good day" as we
        already see the bot's name. Hence, there is a bit of manipulating with the prompt and chat history to trim this.

        :algorithm:1. Sanitize the sent message and format it to be in the proper chat history format and append it.
        2. Append the bot's name such that the message generated will only have the next, not the bot's name.
        3. Encode the chat history and serialize the settings and send it via a request.
        4. Get the response back, pop the bot message with the bot's username, and append the message.
        5. If the generated response is valid,
        Send the message via the discord client, whether be it through a reply or just a message.
        :param a_message: the message to reply to.
        :type: Discord.Message
        :param should_reply: If the bot should reply to the message or instead just send a message without a reply.
        :type: bool

        """

        user_message = "" + a_message.author.name + ": " + a_message.content + "\n"
        self.memories.append_message(user_message, a_tokenizer=self.tokenizer)
        # Necessary such that we only get the bot's response, not the bot's name when the text generates.
        self.memories.append_message(self.bot_name + ":", a_tokenizer=self.tokenizer)

        self.serial_settings['chat_history'] = self.memories.get_encoded_chat_history(a_tokenizer=self.tokenizer)
        # Simulate the bot typing.
        async with a_message.channel.typing():
            serial_settings = json.dumps(self.serial_settings)

            bot_response = self.request_generation(serial_settings)
            if await self.validate_generation(bot_response, a_message=a_message):
                decoded_bot_message = front_end_utils.decode_encoded_tokenized_tensor(bot_response)
                detokenized_bot_message = self.detokenize_decoded_message(decoded_bot_message)
                # Remove the original empty prompt.
                self.memories.pop_memory()
                self.memories.append_message(self.bot_name + ": " + detokenized_bot_message, a_tokenizer=self.tokenizer)
                await a_message.reply(detokenized_bot_message)

    async def conditional_responses(self, a_message: discord.Message):
        """
        Sends a Discord Message only if a bot is replied to, mentioned, or their nickname(s) are in a user's message.

        :param a_message: The Message to potentially reply to.
        :type a_message: discord.Message
        :acknowledgements: Large chunks of the function are borrowed from the following code below.

        1. Checking for message replies.
        https://stackoverflow.com/questions/66956261/check-if-message-reply-is-a-reply-type-message-discord-py
        2. Checking if a user replied to the bot.
        https://stackoverflow.com/questions/66016979/discord-py-send-a-different-message-if-a-user-replies-to-my-bot
        3. Checking if the user referenced text within our nickname list.
        https://stackoverflow.com/questions/68784024/discord-py-how-to-check-if-a-message-contains-text-from-a-list
        """
        if self.user.mentioned_in(a_message):
            await self.send_discord_message(a_message)
            return

        if a_message.reference is not None:
            original_message = None
            if a_message.reference.cached_message is None:
                # Fetching the message
                channel = discord.Client.get_channel(a_message.reference.channel_id)
                original_message = await channel.fetch_message(a_message.reference.message_id)
                return
            else:
                original_message = a_message.reference.cached_message
            if original_message is not None and original_message.author.id == self.user.id:
                await self.send_discord_message(a_message)
                return

        for names in self.bot_nicknames:
            if names in a_message.content:
                await self.send_discord_message(a_message)
                return

    async def on_ready(self):
        """
        An override of the on_ready function from Discord.Client.
        The function that responds to when the discord bot is first connected to the discord API.
        :description: This function is executed when the bot first starts up, and does everything that the bots needs
        to do at start time, including: 1. Connecting to Discord's API, 2. Synching Discord Slash Commands
        3. Setting the Bot's status.
        """
        print(f"Successfully connected to Discord API")
        # Syncs commands with the main guild. I would do global commands, but I heard they take up to an hour to sync.
        self.tree.copy_global_to(guild=discord.Object(self.main_guild_id))
        await self.tree.sync(guild=discord.Object(id=self.main_guild_id))
        print("Succesfully synched with commands.")

        # Set the status of the discord bot. This is mostly for flair and is optional, but I think it's neat.
        if "status_type" and "status_body" and "stream_url" in self.discord_provider_settings.keys() and \
                self.discord_provider_settings['status_type'] == "streaming":
            await discord.Client.change_presence(
                activity=discord.Activity(self, type=discord.ActivityType.streaming, name=self.status_body,
                                          url="stream_url"))
        if "status_type" and "status_body" in self.discord_provider_settings.keys():
            if self.status_type == "playing":
                await discord.Client.change_presence(self, activity=discord.Game(name=self.status_body))
            if self.status_type == "listening":
                await discord.Client.change_presence(self,
                                                     activity=discord.Activity(type=discord.ActivityType.listening,
                                                                               name=self.status_body))
            if self.status_type == "watching":
                await discord.Client.change_presence(self, activity=discord.Activity(type=discord.ActivityType.watching,
                                                                                     name=self.status_body))

    async def on_message(self, a_message):
        """
        Overrides the discord.Client on_message function. Defines behavior for what the bot should
        do when a message pops up.

        :param a_message: the message that the bot has seen be sent in a channel its in and can see.
        :type a_message: discord.Message
        """
        if a_message.author.id == self.user.id:
            return
        if a_message.is_system():
            return
        if self.conditional_response:
            await self.conditional_responses(a_message)
        else:
            await self.send_discord_message(a_message)

    async def validate_generation(self, a_generation, a_interaction: discord.Interaction = None,
                                  a_message: discord.Message = None):
        """
        Validates the passed a_generation. If it isn't valid, print an error message and return false. Else, return true.

        :description: An odd function, primarily because its main purpose is error checking. This is mainly to bridge
        the gap between Provider's request_generation error codes and the client so the user is aware of errors
        while using Discord and not clueless as to why the bot is not responding. This also provides a junction for
        other functions to continue executing or not.

        :algorithm: 1. Check if a_generation is of type integer. This means it has an error code, not generated text.
        2. If so, match the case and change the value of error message accordingly.
        3. Send an error message depending on whether the value passed is of type discord.Interaction or discord.Message
        4. If that is not the case, check if it equals to None.
        5. If a discord.Message is passed, clear the memories. Regardless, send an error message through the
           the corresponding type.

        :param a_generation: The generated text to inspect/validate.
        :param a_interaction: The Interaction if it exists. This will be used to send an error message if it's not None.
        :type a_interaction: discord.Interaction or None
        :param a_message: The Message if it exists. This will be used to send an error message if it's not None.
        :type a_message: discord.Message or None
        :return: Whether the passed generated text is valid or not.
        :rtype: bool
        """
        error_msg = ""
        if type(a_generation) is int:
            match a_generation:
                case 0:
                    error_msg = "There was a timeout error during generation. Please check the console error logs."
                case -1:
                    error_msg = "There was a request error during generation. Please check the console error logs."
                case -2:
                    error_msg = "There was a generic error during generation. Please check the console error logs."
                case _:
                    error_msg = "There was an http error during generation. Please check the following status " \
                                f"code: {a_generation}, and also please check the console error logs."
            if a_interaction is not None:
                await a_interaction.followup.send(error_msg)
            else:
                await a_message.reply(error_msg)

            return False

        # Get rid of the chat history of the bot saying nothing. If we do not do this, the model will not generate
        # ANYTHING. Note that this is only in this function as this problem doesn't seem to plague autocomplete
        # or regeneration of text.
        if a_generation is None or a_generation.strip() == "":
            if a_interaction is not None:
                await a_interaction.followup.send(
                    f"The requested command {a_interaction.command.name} failed to generate anything. "
                    f"Check your parameters, certain models do not like it when do_sample "
                    f"= False is your only parameter. Short prompts may also cause this.")
            else:
                await a_message.reply("I don't know what to say, so I cleared my memories.")
                self.memories.clear_all_memories()
            return False
        return True

