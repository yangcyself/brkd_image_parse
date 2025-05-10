import os
from telethon import TelegramClient
from PIL import Image
import tempfile

ENABLED = True

if ENABLED:
    api_id = os.getenv('Telegram_API_ID')
    api_hash = os.getenv('Telegram_API_HASH')
    my_bot_id = int(os.getenv('Telegram_BOT_ID'))
    bot_token = os.getenv('Telegram_Bot_TOKEN')
    my_client_id = int(os.getenv('Telegram_Client_ID'))

    if api_id is None or api_hash is None or my_bot_id is None or bot_token is None or my_client_id is None:
        raise Exception('Telegram API ID, API HASH, BOT ID, BOT TOKEN, or CLIENT ID not found in environment variables.')

    client = TelegramClient('anon', api_id, api_hash)
    client.start()
    bot = TelegramClient('bot', api_id, api_hash).start(bot_token=bot_token)

    def check_image_type_and_save(image):
        """
        Checks if `image` is a string or a PIL Image. If it is a PIL Image, it saves the image to a temporary
        file and returns the path to the file. If it is a string, it simply returns the string.

        :param image: str or PIL.Image.Image
        :return: str - path to the image file
        """
        if isinstance(image, Image.Image):
            # It's an Image, save to temporary file
            tmp_path = tempfile.mktemp(suffix='.png')
            image.save(tmp_path, 'PNG')
            return tmp_path
        elif isinstance(image, str):
            # It's a string, return as is
            return image
        else:
            raise TypeError("The input must be a PIL Image or a string.")


    async def async_init():
        await client.get_dialogs()

    async def async_bot2client(bot2client_conv, msg=None, image=None):
        if image is not None:
            image = check_image_type_and_save(image)
            await bot2client_conv.send_file(image)
        if msg is not None:
            await bot2client_conv.send_message(msg) # this is my client

    def bot2client(msg=None, image=None):
        with bot.conversation(my_client_id) as bot2client_conv:
            client.loop.run_until_complete(async_bot2client(bot2client_conv, msg, image))
            
    async def async_client2bot(client2bot_conv, msg=None, image=None):
        # await client.start()
        # me = await client.get_me()
        if image is not None:
            image = check_image_type_and_save(image)
            await client2bot_conv.send_file(image)
        if msg is not None:
            await client2bot_conv.send_message(msg)

    def client2bot(msg=None, image=None):
        with client.conversation(my_bot_id) as client2bot_conv:
            client.loop.run_until_complete(async_client2bot(client2bot_conv, msg, image))
            
    client.loop.run_until_complete(async_init())
else:
    def bot2client(msg=None, image=None):
        pass
    def client2bot(msg=None, image=None):
        pass

