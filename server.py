from fastapi import FastAPI, Request, Form, Query, BackgroundTasks, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Literal
import time
import subprocess
import sys
import os

def format_date(timestamp: str) -> str:
    """Форматирует timestamp в читаемую дату"""
    dt = datetime.fromtimestamp(int(timestamp) / 1000)
    return dt.strftime("%Y-%m-%d, %H:%M:%S")

app = FastAPI(title="JSON Viewer", description="Просмотрщик и агрегатор данных из Telegram")

templates = Jinja2Templates(directory="templates")
templates.env.filters["format_date"] = format_date

app.mount("/static", StaticFiles(directory="static"), name="static")

class DataManager:
    """Менеджер данных приложения"""
    
    def __init__(self):
        self.data: List[Dict] = []
        self.metadata: Dict = {}
        self.current_json_file: str = "28-07-2025.json"
        self.last_scraping_time = 0
        self.scraping_cooldown = 60
        self.scraping_task_status = "idle"
        self.scraping_task_result = None
        self.scraping_task_start_time = 0
    
    def is_scraping_available(self) -> bool:
        """Проверяет доступность скрапинга"""
        if self.scraping_task_status == "running":
            return False
        if self.last_scraping_time > 0:
            time_since_last = time.time() - self.last_scraping_time
            return time_since_last >= self.scraping_cooldown
        return True
    
    def get_cooldown_remaining(self) -> int:
        """Возвращает оставшееся время кулдауна"""
        if self.last_scraping_time > 0:
            time_since_last = time.time() - self.last_scraping_time
            remaining = self.scraping_cooldown - time_since_last
            return max(0, int(remaining))
        return 0
    
    def load_data(self, json_path: str) -> None:
        """Загружает данные из JSON"""
        self.current_json_file = json_path
        
        if Path(json_path).exists():
            with open(json_path, "r", encoding="utf-8") as f:
                json_content = json.load(f)
                
                if isinstance(json_content, dict) and "posts" in json_content:
                    self.data = json_content["posts"]
                    self.metadata = json_content.get("metadata", {})
                else:
                    self.data = json_content
                    self.metadata = {}
                
                for item in self.data:
                    if "is_relevant" not in item:
                        item["is_relevant"] = False
            print(f"Загружено {len(self.data)} записей из {json_path}")
        else:
            print(f"Предупреждение: JSON файл {json_path} не найден. Начинаем с пустыми данными.")
            self.data = []
            self.metadata = {}
    
    def save_data(self) -> None:
        """Сохраняет данные в JSON"""
        if self.metadata:
            output_data = {
                "metadata": self.metadata,
                "posts": self.data
            }
        else:
            output_data = self.data
            
        with open(self.current_json_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    def get_latest_pipeline_file(self) -> str:
        """Получает последний файл пайплайна"""
        pipeline_dir = Path("pipeline_output")
        if not pipeline_dir.exists():
            return None
        
        json_files = list(pipeline_dir.glob("*.json"))
        if not json_files:
            return None
        
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        return str(latest_file)

data_manager = DataManager()

SortKeywordsType = Literal["none", "asc", "desc"]

# Главная страница
@app.get("/", response_class=HTMLResponse)
async def read_root(  
    request: Request,
    page: int = 1,
    per_page: int = 10,
    search: str = "",
    only_relevant: bool = False,
    has_keywords: bool = False,
    sort_keywords: SortKeywordsType = Query("none")
):
    filtered_data = [
        {"item": item, "original_id": i}
        for i, item in enumerate(data_manager.data)
        if (not search or search.lower() in item.get("channel", "").lower() or search.lower() in item.get("text", "").lower())
        and (not only_relevant or item.get("is_relevant", False))
        and (not has_keywords or bool(item.get("matched_keywords", [])))
    ]

    if sort_keywords != "none":
        filtered_data.sort(
            key=lambda x: len(x["item"].get("matched_keywords", [])),
            reverse=(sort_keywords == "desc")
        )

    total = len(filtered_data)
    paginated_data = filtered_data[(page-1)*per_page : page*per_page]
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "items": paginated_data,
            "page": page,
            "per_page": per_page,
            "search": search,
            "only_relevant": only_relevant,
            "has_keywords": has_keywords,
            "sort_keywords": sort_keywords,
            "total": total,
            "total_pages": (total + per_page - 1) // per_page,
            "metadata": data_manager.metadata,
            "current_file": data_manager.current_json_file,
        },
    )

# Страница отдельного поста
@app.get("/item/{item_id}", response_class=HTMLResponse)
async def read_item(  
    request: Request,
    item_id: int,
    search: str = Query(""),
    only_relevant: bool = Query(False),
    has_keywords: bool = Query(False),
    sort_keywords: str = Query("none"),
    page: int = Query(1),
    per_page: int = Query(10)
):
    if 0 <= item_id < len(data_manager.data):
        item = data_manager.data[item_id]
        
        filtered_ids = [
            i for i, item in enumerate(data_manager.data)
            if (not search or search.lower() in item.get("channel", "").lower() or search.lower() in item.get("text", "").lower())
            and (not only_relevant or item.get("is_relevant", False))
            and (not has_keywords or bool(item.get("matched_keywords", [])))
        ]
        
        if sort_keywords != "none":
            filtered_ids.sort(
                key=lambda x: len(data_manager.data[x].get("matched_keywords", [])),
                reverse=(sort_keywords == "desc")
            )
        
        try:
            current_idx = filtered_ids.index(item_id)
            prev_id = filtered_ids[current_idx - 1] if current_idx > 0 else None
            next_id = filtered_ids[current_idx + 1] if current_idx < len(filtered_ids) - 1 else None
        except ValueError:
            prev_id = next_id = None
        
        return templates.TemplateResponse(
            "item.html",
            {
                "request": request,
                "item": item,
                "item_id": item_id,
                "prev_id": prev_id,
                "next_id": next_id,
                "search": search,
                "only_relevant": only_relevant,
                "has_keywords": has_keywords,
                "sort_keywords": sort_keywords,
                "page": page,
                "per_page": per_page,
            },
        )
    return "Запись не найдена"

# Установка релевантности поста
@app.post("/item/{item_id}/set_relevant")
async def set_relevant(  
    item_id: int, 
    is_relevant: str = Form("false")
):
    if 0 <= item_id < len(data_manager.data):
        data_manager.data[item_id]["is_relevant"] = (is_relevant == "on")
        data_manager.save_data()
        return RedirectResponse(url=f"/item/{item_id}", status_code=303)
    return {"status": "error"}

# Запуск пайплайна
@app.post("/run-scraping")
async def run_scraping(background_tasks: BackgroundTasks):  
    if not data_manager.is_scraping_available():
        if data_manager.scraping_task_status == "running":
            return {"status": "already_running"}
        else:
            remaining = data_manager.get_cooldown_remaining()
            return {"status": "cooldown", "remaining": remaining}
    
    def task():
        try:
            data_manager.scraping_task_status = "running"
            print(f"Starting pipeline task...")
            
            result = subprocess.run(
                [sys.executable, "pipeline.py", "--limit", "50", "--only-most-keywords"],
                capture_output=True,
                text=True,
                cwd=".",
                timeout=300
            )
            
            print(f"Pipeline completed with return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            
            latest_file = data_manager.get_latest_pipeline_file()
            if latest_file:
                data_manager.load_data(latest_file)
                print(f"Loaded data from: {latest_file}")
            else:
                print("No pipeline output files found")
            
            data_manager.scraping_task_result = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
            data_manager.scraping_task_status = "completed"
            data_manager.last_scraping_time = time.time()
        except subprocess.TimeoutExpired:
            data_manager.scraping_task_result = {"success": False, "error": "Task timed out after 5 minutes"}
            data_manager.scraping_task_status = "failed"
        except Exception as e:
            print(f"Exception in pipeline task: {e}")
            data_manager.scraping_task_result = {"success": False, "error": str(e)}
            data_manager.scraping_task_status = "failed"

    data_manager.scraping_task_start_time = time.time()
    background_tasks.add_task(task)
    return {"status": "processing"}

# Загрузка JSON файла
@app.post("/load-json")
async def load_json_file(file: UploadFile = File(...)):  
    try:
        content = await file.read()
        json_data = json.loads(content.decode('utf-8'))
        
        if isinstance(json_data, dict) and "posts" in json_data:
            data_manager.data = json_data["posts"]
            data_manager.metadata = json_data.get("metadata", {})
        else:
            data_manager.data = json_data
            data_manager.metadata = {}
        
        for item in data_manager.data:
            if "is_relevant" not in item:
                item["is_relevant"] = False
        
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        filename = file.filename
        filepath = os.path.join(uploads_dir, filename)
        
        if os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
            filename = f"{name}_{timestamp}{ext}"
            filepath = os.path.join(uploads_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        data_manager.current_json_file = filepath
        print(f"Loaded {len(data_manager.data)} items from uploaded file: {filename}")
        
        return {"status": "success", "message": f"Loaded {len(data_manager.data)} items from {file.filename}"}
        
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON file: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Error loading file: {str(e)}"}

# Список доступных файлов
@app.get("/available-files")
async def get_available_files():  
    files = []
    
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.endswith('.json'):
                filepath = os.path.join(uploads_dir, file)
                files.append({
                    "name": file,
                    "path": filepath,
                    "type": "uploaded"
                })
    
    pipeline_dir = "pipeline_output"
    if os.path.exists(pipeline_dir):
        for file in os.listdir(pipeline_dir):
            if file.endswith('.json'):
                filepath = os.path.join(pipeline_dir, file)
                files.append({
                    "name": file,
                    "path": filepath,
                    "type": "pipeline"
                })
    
    for file in os.listdir("."):
        if file.endswith('.json'):
            filepath = os.path.join(".", file)
            files.append({
                "name": file,
                "path": filepath,
                "type": "root"
            })
    
    files.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
    
    return {"files": files}

# Загрузка выбранного файла
@app.post("/load-file")
async def load_file(file_path: str = Form(...)):  
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "message": "File not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        if isinstance(json_data, dict) and "posts" in json_data:
            data_manager.data = json_data["posts"]
            data_manager.metadata = json_data.get("metadata", {})
        else:
            data_manager.data = json_data
            data_manager.metadata = {}
        
        for item in data_manager.data:
            if "is_relevant" not in item:
                item["is_relevant"] = False
        
        data_manager.current_json_file = file_path
        print(f"Loaded {len(data_manager.data)} items from file: {file_path}")
        
        return {"status": "success", "message": f"Loaded {len(data_manager.data)} items from {os.path.basename(file_path)}"}
        
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON file: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Error loading file: {str(e)}"}

# Статус скрапинга
@app.get("/scraping-status")
async def get_scraping_status():  
    if data_manager.scraping_task_status == "idle":
        if not data_manager.is_scraping_available():
            remaining = data_manager.get_cooldown_remaining()
            return {"status": "cooldown", "remaining": remaining}
        return {"status": "idle"}
    elif data_manager.scraping_task_status == "running":
        elapsed = time.time() - data_manager.scraping_task_start_time
        return {"status": "running", "elapsed": int(elapsed)}
    elif data_manager.scraping_task_status == "completed":
        result = data_manager.scraping_task_result
        data_manager.scraping_task_status = "idle"
        return {"status": "completed", "result": result}
    elif data_manager.scraping_task_status == "failed":
        result = data_manager.scraping_task_result
        data_manager.scraping_task_status = "idle"
        return {"status": "failed", "result": result}

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description='JSON Viewer Server')
    parser.add_argument('--json-file', '-j', 
                       type=str, 
                       help='Путь к JSON файлу для загрузки')
    parser.add_argument('--host', 
                       type=str, 
                       default="0.0.0.0",
                       help='Хост для привязки (по умолчанию: 0.0.0.0)')
    parser.add_argument('--port', 
                       type=int, 
                       default=8000,
                       help='Порт для привязки (по умолчанию: 8000)')
    
    args = parser.parse_args()
    
    if args.json_file:
        data_manager.load_data(args.json_file)
    else:
        data_manager.load_data("28-07-2025.json")
    
    print(f"Запускаем сервер на {args.host}:{args.port}")
    print(f"Используем JSON файл: {data_manager.current_json_file}")
    
    uvicorn.run(app, host=args.host, port=args.port)