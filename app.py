import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
import asyncio
from telegram import Bot
from PIL import Image
import json
import base64
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import io

# ================= НАСТРОЙКИ =================
FOTO_DIR = "Foto"
if not os.path.exists(FOTO_DIR):
    os.makedirs(FOTO_DIR)

# Загрузка секретов
TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
SPREADSHEET_ID = st.secrets["google"]["spreadsheet_id"]
DRIVE_FOLDER_ID = st.secrets["google"]["drive_folder_id"]

# ================= ИНИЦИАЛИЗАЦИЯ GOOGLE API =================
@st.cache_resource
def init_google_services():
    """Инициализация сервисов Google"""
    try:
        creds_dict = dict(st.secrets["google_credentials"])
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
        )
        
        gc = gspread.authorize(creds)
        drive_service = build('drive', 'v3', credentials=creds)
        
        return gc, drive_service
    except Exception as e:
        st.error(f"❌ Ошибка инициализации Google API: {e}")
        return None, None

def get_sheet(gc):
    """Получение или создание таблицы"""
    try:
        # Открываем существующую таблицу
        sheet = gc.open_by_key(SPREADSHEET_ID)
        try:
            worksheet = sheet.worksheet("Сотрудники")
        except:
            worksheet = sheet.add_worksheet(title="Сотрудники", rows=1, cols=3)
            worksheet.append_row(['Имя', 'Путь к фото', 'Дата добавления'])
        return worksheet
    except Exception as e:
        st.error(f"❌ Ошибка доступа к таблице: {e}")
        return None

# ================= РАБОТА С ДАННЫМИ =================
def load_employees(worksheet):
    """Загрузка списка сотрудников из Google Sheets"""
    try:
        records = worksheet.get_all_records()
        return pd.DataFrame(records)
    except:
        return pd.DataFrame(columns=['Имя', 'Путь к фото', 'Дата добавления'])

def add_employee(worksheet, drive_service, name, photo_path):
    """Добавление нового сотрудника"""
    try:
        # Загружаем фото на Google Диск
        file_metadata = {
            'name': f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            'parents': [DRIVE_FOLDER_ID]
        }
        media = MediaFileUpload(photo_path, mimetype='image/jpeg')
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()
        
        # Добавляем запись в таблицу
        worksheet.append_row([
            name,
            file.get('webViewLink'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])
        return True
    except Exception as e:
        st.error(f"❌ Ошибка добавления: {e}")
        return False

# ================= ТЕЛЕГРАМ БОТ =================
async def send_telegram_message(name):
    """Отправка сообщения в Telegram"""
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    message = f"✅ Сотрудник {name} прибыл в {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        return True
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def send_message_sync(name):
    """Синхронная отправка"""
    try:
        asyncio.run(send_telegram_message(name))
        return True
    except:
        return False

# ================= ОСНОВНОЙ КЛАСС ПРИЛОЖЕНИЯ =================
class FaceRecognitionApp:
    def __init__(self):
        self.cap = None
        self.recognized_today = set()
        self.last_recognition = {}
        self.cooldown = 60  # Секунд между уведомлениями
        self.gc, self.drive_service = init_google_services()
        self.worksheet = get_sheet(self.gc) if self.gc else None
        
    def start_camera(self):
        """Запуск камеры"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                for i in range(1, 5):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
            if not self.cap.isOpened():
                st.error("❌ Не удалось открыть камеру")
                return False
            return True
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            return False
    
    def release_camera(self):
        """Освобождение камеры"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_known_faces(self):
        """Загрузка известных лиц"""
        if not self.worksheet:
            return {}, []
        
        df = load_employees(self.worksheet)
        known_encodings = {}
        known_names = []
        
        # Пытаемся загрузить кэшированные кодировки
        cache_file = "face_encodings_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    for name, encoding in cache_data.items():
                        known_encodings[name] = np.array(encoding)
                        known_names.append(name)
                return known_encodings, known_names
            except:
                pass
        
        # Если кэша нет, пытаемся загрузить из локальных фото
        for _, row in df.iterrows():
            name = row['Имя']
            local_path = os.path.join(FOTO_DIR, f"{name}.jpg")
            if os.path.exists(local_path):
                try:
                    image = face_recognition.load_image_file(local_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_encodings[name] = encodings[0]
                        known_names.append(name)
                except:
                    pass
        
        # Сохраняем кэш
        if known_encodings:
            cache_data = {name: enc.tolist() for name, enc in known_encodings.items()}
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        
        return known_encodings, known_names
    
    def process_frame(self):
        """Обработка кадра"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Уменьшаем для ускорения
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        # Находим лица
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)
        
        # Получаем известные лица
        known_encodings, known_names = self.get_known_faces()
        
        recognized = []
        
        for encoding in encodings:
            name = "Неизвестный"
            
            if known_encodings:
                for known_name, known_encoding in known_encodings.items():
                    matches = face_recognition.compare_faces(
                        [known_encoding], encoding, tolerance=0.6
                    )
                    if matches[0]:
                        name = known_name
                        break
            
            # Проверяем отправку уведомления
            if name != "Неизвестный":
                current_time = time.time()
                if (name not in self.recognized_today or 
                    current_time - self.last_recognition.get(name, 0) > self.cooldown):
                    self.recognized_today.add(name)
                    self.last_recognition[name] = current_time
                    if send_message_sync(name):
                        st.success(f"✅ Уведомление: {name}")
            
            recognized.append(name)
        
        # Рисуем рамки
        for (top, right, bottom, left), name in zip(locations, recognized):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            color = (0, 255, 0) if name != "Неизвестный" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame

# ================= HTML-КОМПОНЕНТЫ =================
def get_html_header():
    """Возвращает HTML для заголовка"""
    return """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    ">
        <h1>🔐 Система распознавания сотрудников</h1>
        <p>Автоматическая идентификация с уведомлениями в Telegram</p>
    </div>
    """

def get_html_status(status, message):
    """Возвращает HTML для статуса"""
    color = "#4CAF50" if status == "success" else "#f44336"
    return f"""
    <div style="
        padding: 10px;
        border-radius: 5px;
        background-color: {color}20;
        border-left: 4px solid {color};
        margin: 10px 0;
    ">
        {message}
    </div>
    """

# ================= ОСНОВНОЙ ИНТЕРФЕЙС =================
def main():
    st.set_page_config(
        page_title="Система распознавания",
        page_icon="🔐",
        layout="wide"
    )
    
    # HTML заголовок
    st.markdown(get_html_header(), unsafe_allow_html=True)
    
    # Инициализация
    if 'app' not in st.session_state:
        st.session_state.app = FaceRecognitionApp()
    
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    # Сайдбар
    with st.sidebar:
        st.markdown("## 📋 Управление")
        
        # Управление камерой
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Запустить", use_container_width=True):
                if st.session_state.app.start_camera():
                    st.session_state.camera_running = True
                    st.rerun()
        with col2:
            if st.button("⏹️ Остановить", use_container_width=True):
                st.session_state.app.release_camera()
                st.session_state.camera_running = False
                st.rerun()
        
        st.markdown("---")
        
        # Добавление сотрудника
        st.markdown("## ➕ Добавить сотрудника")
        new_name = st.text_input("Имя сотрудника")
        
        if st.button("📸 Сфотографировать и добавить", use_container_width=True):
            if not new_name:
                st.error("❌ Введите имя!")
            elif not st.session_state.camera_running:
                st.error("❌ Запустите камеру!")
            else:
                # Делаем снимок
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Сохраняем локально
                    local_path = os.path.join(FOTO_DIR, f"{new_name}.jpg")
                    cv2.imwrite(local_path, frame)
                    
                    # Находим лицо
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locations = face_recognition.face_locations(rgb)
                    encodings = face_recognition.face_encodings(rgb, locations)
                    
                    if encodings:
                        # Добавляем в Google
                        if add_employee(
                            st.session_state.app.worksheet,
                            st.session_state.app.drive_service,
                            new_name,
                            local_path
                        ):
                            st.success(f"✅ {new_name} добавлен!")
                            st.session_state.app.get_known_faces()
                            st.rerun()
                    else:
                        st.error("⚠️ Лицо не найдено!")
                        os.remove(local_path)
                else:
                    st.error("❌ Ошибка фото!")
        
        st.markdown("---")
        
        # Список сотрудников
        st.markdown("## 👥 Сотрудники")
        if st.session_state.app.worksheet:
            df = load_employees(st.session_state.app.worksheet)
            if len(df) > 0:
                st.dataframe(df[['Имя', 'Дата добавления']], use_container_width=True)
                st.caption(f"Всего: {len(df)}")
            else:
                st.info("Нет сотрудников")
    
    # Основная область
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("## 📷 Видеопоток")
        video_container = st.empty()
        
        if st.session_state.camera_running:
            while st.session_state.camera_running:
                frame = st.session_state.app.process_frame()
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_container.image(frame_rgb, channels="RGB")
                time.sleep(0.05)
        else:
            video_container.info("🔴 Камера не запущена")
    
    with col2:
        st.markdown("## 📊 Статус")
        
        if st.session_state.camera_running:
            st.success("🟢 Камера активна")
        else:
            st.error("🔴 Камера остановлена")
        
        st.markdown("---")
        st.markdown("## 📝 Уведомления")
        
        # Показываем последние уведомления
        if hasattr(st.session_state.app, 'recognized_today'):
            for name in list(st.session_state.app.recognized_today)[-5:]:
                st.write(f"✅ {name} - {datetime.now().strftime('%H:%M')}")

if __name__ == "__main__":
    main()