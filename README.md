# Phantasmagoria
The front end for the Project Netherworld Ecosystem. For more information about the Project, see [this homepage](https://github.com/Project-Netherworld).

## Features 
- Efficient Communication via Serialization of Tensors and Chat History(using FastAPI and Uvicorn) to Interface with the [Ayakashi Back End](https://github.com/Project-Netherworld/Ayakashi).
- Two Possible Deployments: The Terminal or the Social Media App Discord.
- Capability for Single-party or Multi-party Conversation with a Conversational Agent (Chatbot).
- Implementation of Short Term Memory and Memory Cycler(by sentence or by token) to handle variable context lengths.
- Use of Discord's UI to allow for convenient model utilities via Slash Commands.
- Conversation-based slash commands for fun or utility such as autocompletion, regenerating text, etc..
- Adjust Settings slash command to adjust settings on the fly to debug without having to restart the Front End and change settings.
- Text Generation slash command to generate short, fictional stories as a quick writing aide.
- Memory-based slash command to either look into or clear memory.
- Option for Discord custom statuses

## Installation 
0. Have Python installed. **This project requires Python 3.10 or higher.**
1. Either clone this repository, download it via zip, or download the release. 
2. Using your favorite CLI (command prompt, bash, etc.), use `cd` to change the directory to where you downloaded the repository.
3.  Run the following command: 
`pip install -r requirements.txt`
4. Alternatively, should you prefer not using pip and want to use conda instead, run the following command: 
`conda install --file requirements.txt`

## Usage
The Front End is the primary attraction of this ecosystem, as it has the most features, namely, it takes advantage of features added in by Discord, called slash commands, which are commands that can link back to functions predefined by the developer, with the advantage of a convenient interface for users.

### Running the Program
0. Have the [Ayakashi Backend](https://github.com/Project-Netherworld/Ayakashi) currently running. 
1. **It is imperative to make sure that the configuration file that you downloaded from the settings builder is in the config folder. Otherwise, the program will not work properly. If you do not have said folder, you will need to make it within the same directory as the Front End.**
2. **To run the program, you will need to pass the configuration file like such:**
`python netherworld_main “your_config_file_here.json”`. Some installations of Python 3 utilize `python3` as a prefix to run Python commands instead, so if this is the case, run the command as such instead:
`python3 netherworld_main “your_config_file_here.json”`
3a. If you are utilizing the discord frontend, you'll be able to verify that the program works if your chatbot is online(marked off by a green dot near their profile picture), and on top of that, has access to slash commands as seen below:![Picture of the settings, including all the slash commands pulled up.](https://github.com/Project-Netherworld/.github/blob/main/images/image30.png?raw=true)

If your bot is not online, it likely is a problem to deal with either the program or your bot’s `token` setting. If slash commands do not show up, there is an issue with your `main_guild_id` setting.

3b. If you are utilizing the terminal front end, you'll be able to verify that the program works if your chatbot responds to one of your messages. 

### Utilizing the Bot, Slash Commands, etc..
As there is quite a lot of ground to cover regarding slash commands and the bot itself on Discord, please check the [wiki](https://github.com/Project-Netherworld/Phantasmagoria/wiki/Usage-Guide#using-the-bot-discord).

As for the terminal, because it lacks slash commands, it is mainly used for a simple conversation interface. To chat, type in the terminal and press enter. To exit the program, press CTRL+C. 
