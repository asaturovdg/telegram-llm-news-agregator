#!/usr/bin/env python3
"""
Пайплайн для обработки данных из Telegram
Фильтрация, дедупликация и суммаризация данных
"""

import argparse
import os
import pandas as pd
from semhash import SemHash
from model2vec import StaticModel
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
from datetime import datetime
import subprocess
import sys

def filter_by_keywords(df, column_name, keywords):
    """Фильтрует DataFrame по ключевым словам и добавляет колонку с найденными словами"""
    keywords_lower = [str(kw).lower() for kw in keywords]
    
    def find_keywords(text):
        text = str(text).lower()
        return [kw for kw in keywords_lower if kw in text]
    
    df['matched_keywords'] = df[column_name].apply(find_keywords)
    filtered_df = df[df['matched_keywords'].notna()].copy()
    
    return filtered_df

def run_scraping(limit=None, offset_date=None, output='output/data.json'):
    """Запускает scraping.py с указанными параметрами"""
    cmd = [sys.executable, 'scraping.py', '--output', output]
    
    if limit:
        cmd.extend(['--limit', str(limit)])
    if offset_date:
        cmd.extend(['--offset-date', offset_date])
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка запуска scraping.py: {e}")
        return False

def create_gigachat_summarizer():
    """Создает суммаризатор на основе GigaChat"""
    load_dotenv()
    GIGACHAT_CREDENTIALS = os.environ.get('GIGACHAT_CREDENTIALS')
    
    if not GIGACHAT_CREDENTIALS:
        raise ValueError("GIGACHAT_CREDENTIALS не найден в переменных окружения")
    
    llm = GigaChat(
        model='GigaChat-2',
        scope='GIGACHAT_API_PERS',
        credentials=GIGACHAT_CREDENTIALS,
        profanity_check=False,
        verify_ssl_certs=False,
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Ты - помощник для анализа текстов. Создай краткое резюме (1-2 предложения) следующего текста, выделив основную мысль. Избегай любых нежелательных тем."), 
        ("user", "{text}")
    ])
    
    parser = StrOutputParser()
    chain = prompt_template | llm | parser
    
    return chain

def create_most_keywords_summary(df, summarizer_chain):
    """Создает сводку топ-10 постов с наибольшим количеством ключевых слов"""
    df_sorted = df.copy()
    df_sorted['keyword_count'] = df_sorted['matched_keywords'].apply(len)
    top_posts = df_sorted.nlargest(10, 'keyword_count')
    
    if top_posts.empty:
        return "Нет постов с ключевыми словами"
    
    summary_parts = []
    for idx, post in top_posts.iterrows():
        try:
            # Summarize the post text
            post_summary = summarizer_chain.invoke({"text": post['text']})
        except Exception as e:
            print(f"Error summarizing post {idx}: {e}")
            post_summary = post['text'][:200] + "..." if len(post['text']) > 200 else post['text']
        
        # Create summary entry - convert timestamp to string
        summary_entry = {
            "channel": post['channel'],
            "date": str(post['date']) if hasattr(post['date'], 'strftime') else post['date'],
            "link": post['link'],
            "keyword_count": post['keyword_count'],
            "keywords": post['matched_keywords'],
            "summary": post_summary
        }
        summary_parts.append(summary_entry)
    
    return {
        "total_posts": len(df),
        "top_posts": summary_parts
    }

def ensure_json_serializable(obj):
    """Конвертирует объекты в JSON-сериализуемый формат"""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif hasattr(obj, 'strftime'):  # datetime/timestamp объекты
        return str(obj)
    elif hasattr(obj, 'isoformat'):  # datetime объекты
        return obj.isoformat()
    else:
        return obj

def add_summaries_to_posts(df, summarizer_chain):
    """Добавляет суммаризацию ко всем постам в DataFrame"""
    print(f"Добавляем суммаризацию к {len(df)} постам...")
    
    for idx, row in df.iterrows():
        try:
            summary = summarizer_chain.invoke({"text": row['text']})
            df.at[idx, 'summary'] = summary
        except Exception as e:
            print(f"Ошибка суммаризации поста {idx}: {e}")
            df.at[idx, 'summary'] = row['text'][:200] + "..." if len(row['text']) > 200 else row['text']
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Process scraped data pipeline')
    parser.add_argument('--limit', '-l',
                       type=int,
                       default=20,
                       help='Limit number of messages to fetch per channel (default: 20)')
    parser.add_argument('--offset-date', '-d',
                       type=str,
                       default=None,
                       help='Offset date in YYYY-MM-DD format (default: None, fetch all)')
    parser.add_argument('--keywords', '-k',
                       type=str,
                       nargs='+',
                       default=['llm', 'chatgpt', 'openai', 'qwen', 'gemini', 'reasoning', 'cot', 'icml', 'neurips', 'paper'],
                       help='Keywords to filter by (default: llm, chatgpt, openai, qwen, gemini, reasoning, cot, icml, neurips, paper)')
    parser.add_argument('--dedup-threshold', '-t',
                       type=float,
                       default=0.9,
                       help='Deduplication threshold (default: 0.9)')
    parser.add_argument('--no-dedup',
                       action='store_true',
                       help='Skip deduplication step')
    parser.add_argument('--no-scraping',
                       action='store_true',
                       help='Skip scraping step, use existing data.json')
    parser.add_argument('--output', '-o', 
                       type=str, 
                       default=None,
                       help='Output JSON file path (default: auto-generated with timestamp)')
    parser.add_argument('--only-most-keywords', '-m',
                       action='store_true',
                       help='Only create most_keywords summary, skip individual post summaries')
    
    args = parser.parse_args()
    
    load_dotenv()
    
    # Запускаем скрапинг если не пропущен
    if not args.no_scraping:
        print("Запускаем скрапинг...")
        if not run_scraping(args.limit, args.offset_date):
            return 1
    
    # Загружаем данные
    input_file = 'output/data.json'
    if not os.path.exists(input_file):
        print(f"Ошибка: Входной файл {input_file} не найден!")
        return 1
    
    df = pd.read_json(input_file, orient='records')
    
    # Фильтруем по ключевым словам
    df_with_keywords = filter_by_keywords(df, 'text', args.keywords)
    
    filtered_count = len(df_with_keywords[df_with_keywords["matched_keywords"].apply(lambda x: bool(len(x)))])
    
    # Дедуплицируем
    if not args.no_dedup:
        model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
        semhash = SemHash.from_records(records=df_with_keywords.to_dict(orient='records'), 
                                      columns=['text'], model=model)
        deduplication_results = semhash.self_deduplicate(threshold=args.dedup_threshold)
        df_with_keywords = pd.DataFrame(deduplication_results.selected)
    
    print("Загружаем GigaChat суммаризатор...")
    try:
        summarizer_chain = create_gigachat_summarizer()
    except Exception as e:
        print(f"Ошибка загрузки GigaChat: {e}")
        print("Пропускаем суммаризацию...")
        summarizer_chain = None
    
    print("Суммаризация постов с максимумом ключевых слов...")
    if summarizer_chain:
        try:
            most_keywords_summary = create_most_keywords_summary(df_with_keywords, summarizer_chain)
        except Exception as e:
            print(f"Ошибка в суммаризации: {e}")
            print("Используем резервную суммаризацию...")
            summarizer_chain = None
    
    if not summarizer_chain:
        # Резервная "суммаризация" без AI
        df_sorted = df_with_keywords.copy()
        df_sorted['keyword_count'] = df_sorted['matched_keywords'].apply(len)
        top_posts = df_sorted.nlargest(10, 'keyword_count')
        
        summary_parts = []
        for idx, post in top_posts.iterrows():
            summary_entry = {
                "channel": post['channel'],
                "date": str(post['date']) if hasattr(post['date'], 'strftime') else post['date'],
                "link": post['link'],
                "keyword_count": post['keyword_count'],
                "keywords": post['matched_keywords'],
                "summary": post['text'][:200] + "..." if len(post['text']) > 200 else post['text']
            }
            summary_parts.append(summary_entry)
        
        most_keywords_summary = {
            "total_posts": len(df_with_keywords),
            "top_posts": summary_parts
        }
    
    # Добавляем суммаризацию ко всем постам (запуск без --only-most-keywords)
    if not args.only_most_keywords:
        print("Добавляем суммаризацию ко всем постам...")
        if summarizer_chain:
            try:
                df_with_keywords = add_summaries_to_posts(df_with_keywords, summarizer_chain)
            except Exception as e:
                print(f"Ошибка добавления суммаризации к постам: {e}")
                print("Продолжаем без суммаризации постов...")
    else:
        print("Пропускаем суммаризацию отдельных постов (используется флаг --only-most-keywords)")
    
    df_with_keywords_dict = df_with_keywords.to_dict(orient='records')
    
    # Генерируем имя выходного файла если не указано
    if args.output is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        args.output = f'pipeline_output/{timestamp}.json'
    
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Создаем финальный выход с метаданными
    final_output = {
        "metadata": {
            "most_keywords_summary": most_keywords_summary,
            "total_posts": len(df_with_keywords),
            "processing_date": datetime.now().isoformat(),
            "keywords_used": args.keywords
        },
        "posts": df_with_keywords_dict
    }
    
    # Обеспечиваем сериализуемость JSON
    final_output = ensure_json_serializable(final_output)
    
    # Сохраняем результаты
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    return 0

if __name__ == "__main__":
    exit(main())