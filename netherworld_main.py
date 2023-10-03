import sys

import settings_handler
import terminal_provider
import discord_provider
import logging, threading
def main():
    """
    The main driver of the front-end. Organizes Front End Providers alongside config settings.

    :algorithm: 1. Passes the system argument to the settings which will take care of the loading.
    2. Dispatch/assign providers based on the provider type.
    3. If there's too many or too little command line arguments, log errors.
    """
    if len(sys.argv)==2:
        settings_class = settings_handler.Settings_Handler(sys.argv[1])
        if settings_class.settings['provider_settings']['provider_type'] == "terminal":
            terminal = terminal_provider.Terminal_Provider(settings_class)
            terminal.chat()
        elif settings_class.settings['provider_settings']['provider_type'] == "discord":
            discord_frontend = discord_provider.Discord_Provider(settings_class)
            try:
                discord_frontend.run(token=discord_frontend.token)
            except Exception as discord_login_exception:
                logging.error("There was a problem logging into discord. Check your token config?")
                logging.exception(discord_login_exception)
    elif len(sys.argv) > 2:
        logging.info("Launching with experimental multi config file support")
        threads = []
        for config in sys.argv[1:]:
            settings_class = settings_handler.Settings_Handler(config)
            if settings_class.settings['provider_settings']['provider_type'] == "terminal":
                logging.error("Terminal provider does not support multiple config files")
                return
            elif settings_class.settings['provider_settings']['provider_type'] == "discord":
                discord_frontend = discord_provider.Discord_Provider(settings_class)
                threads.append(threading.Thread(target=discord_frontend.run, args=(discord_frontend.token,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        logging.error("Please pass your json config file! You need a config file to run this program")

if __name__ == "__main__":
    main()
