from telethon import TelegramClient
import os
import pandas as pd
import re
from dotenv import load_dotenv
import argparse
from datetime import datetime

load_dotenv()

API_ID = os.environ.get('API_ID')
API_HASH = os.environ.get('API_HASH')

channels = [
    'ai_newz',
    'gonzo_ML',
    'denissexy',
    'llm_under_hood',
    'cgevent',
    'sergiobulaev',
    'sburyi',
    'prompt_design',
    'tips_ai'
]

author_tags = [
    "@ai_newz",
    "[@ai_newz](https://t.me/ai_newz)",
    "Ваш, @llm_under_hood 🤗",
    "@cgevent",
    "[👾 подписаться]",
    "(https:\/\/t.me\/sburyi)",
    "[все нейронки]",
    "(https:\/\/boosty.to\/buryi)",
    "@tips_ai",
    "[**Сергей Булаев AI 🤖**]",
    "**не только**",
]

def main():
    parser = argparse.ArgumentParser(description='Scrape Telegram channels for AI/ML content')
    parser.add_argument('--limit', '-l',
                       type=int,
                       default=20,
                       help='Limit number of messages to fetch per channel (default: 20)')
    parser.add_argument('--offset-date', '-d',
                       type=str,
                       default=None,
                       help='Offset date in YYYY-MM-DD format (default: None, fetch all)')
    parser.add_argument('--output', '-o',
                       type=str,
                       default='output/data.json',
                       help='Output JSON file path (default: output/data.json)')
    parser.add_argument('--channels', '-c',
                       type=str,
                       nargs='+',
                       default=channels,
                       help='Channels to scrape (default: all predefined channels)')
    
    args = parser.parse_args()
    
    offset_date = None
    if args.offset_date:
        try:
            offset_date = datetime.strptime(args.offset_date, '%Y-%m-%d')
        except ValueError:
            print("Error: Invalid date format. Use YYYY-MM-DD")
            return 1
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df = pd.DataFrame(columns=['channel', 'date', 'text', 'link'])
    
    client = TelegramClient('llm-tg-agregator-client', API_ID, API_HASH, system_version='4.16.30-vxCUSTOM')
    
    async def scrape_channels(): 
        await client.start()
        regex_author_tags = re.compile("|".join(map(re.escape, author_tags)))
        
        for channel in args.channels:
            try:
                async for msg in client.iter_messages(channel, limit=args.limit, offset_date=offset_date):
                    if msg.text:
                        text = regex_author_tags.sub("", msg.text)
                        link = f"https://t.me/{channel}/{msg.id}"
                        
                        df.loc[len(df)] = [channel, str(msg.date), text, link]
            except Exception as e:
                print(f"Error scraping channel {channel}: {e}")
        
        df.to_json(args.output, orient='records', force_ascii=False, indent=2)
    
    client.loop.run_until_complete(scrape_channels())

if __name__ == "__main__":
    main()
