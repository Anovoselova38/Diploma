from flask import Flask, render_template, jsonify, request, send_file
import threading
import time
import math
import random
from collections import deque
import numpy as np
import os
import json
from datetime import datetime
import csv
import io
import base64
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from functools import lru_cache
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Размеры буферов
MAX_BUFFER_SIZE = 5000
LOOKUP_TABLE_SIZE = 10000
DEFAULT_SAMPLING_RATE = 50.0  # Гц
SLEEP_INTERVAL = 0.01  # секунды (10 мс)

# Параметры сигналов
SQUARE_DUTY_CYCLE = 0.5
TRIANGLE_PHASE_1 = 0.25
TRIANGLE_PHASE_2 = 0.75
SAWTOOTH_MIN = -1
SAWTOOTH_MAX = 1

# Параметры FFT
DEFAULT_WINDOW_SIZE = 256
DEFAULT_OVERLAP_PERCENT = 0.5
MAX_FREQUENCY_LIMIT = 25.0  # Гц
MIN_FFT_SIZE = 1024
MIN_POINTS_FOR_SPECTROGRAM = 128

# Директории
SAVE_DIR = 'saved_signals'
UPLOAD_DIR = 'uploads'

# Лимиты
MAX_SAVED_FILES_DISPLAY = 10
MAX_PEAKS_DISPLAY = 10
MIN_PEAKS_TO_DISPLAY = 5
AUTO_RESET_PAUSE_COUNT = 5
PAUSE_WARNING_THRESHOLD = 3

# Каналы
CHANNELS = ['ch1', 'ch2', 'ch3', 'ch4']

# Типы файлов
CSV_EXTENSION = '.csv'

# Глобальные переменные
data_buffers = {
    'ch1': deque(maxlen=MAX_BUFFER_SIZE),
    'ch2': deque(maxlen=MAX_BUFFER_SIZE),
    'ch3': deque(maxlen=MAX_BUFFER_SIZE),
    'ch4': deque(maxlen=MAX_BUFFER_SIZE),
}
# Кэш для последних отправленных данных (оптимизация API)
last_sent_data = {}
is_generating = False
start_time = time.time()
total_paused_time = 0  # Общее время пауз
last_pause_time = 0    # Время последней паузы
pause_count = 0        # Счётчик пауз для автоматического сброса

config = {
    'ch1': {'type': 'sine', 'frequency': 10.0, 'amplitude': 1.0, 'offset': 0.0, 'noise': 0, 'enabled': True},
    'ch2': {'type': 'square', 'frequency': 2.0, 'amplitude': 3.0, 'offset': 2.0, 'noise': 0, 'enabled': True},
    'ch3': {'type': 'triangle', 'frequency': 2.5, 'amplitude': 4.0, 'offset': -2.0, 'noise': 0, 'enabled': True},
    'ch4': {'type': 'sawtooth', 'frequency': 3.0, 'amplitude': 2.5, 'offset': 1.0, 'noise': 0, 'enabled': True}
}

# Предварительно вычисленные таблицы для быстрых сигналов (lookup tables)
_lookup_tables = {}
def _init_lookup_tables():
    """Инициализация таблиц поиска для оптимизации генерации сигналов"""
    global _lookup_tables
    t = np.linspace(0, 1, LOOKUP_TABLE_SIZE)
    _lookup_tables = {
        'sine': np.sin(2 * np.pi * t),
        'square': np.where((t % 1) < SQUARE_DUTY_CYCLE, 1, -1),
        'triangle': np.array([4 * p if p < TRIANGLE_PHASE_1 else (2 - 4 * p) if p < TRIANGLE_PHASE_2 else (4 * p - 4) for p in t]),
        'sawtooth': 2 * t - 1,
        'noise': np.random.uniform(-1, 1, LOOKUP_TABLE_SIZE)
    }
_init_lookup_tables()

# Цвета каналов
channel_colors = {
    'ch1': '#FFD700',  # Золотой
    'ch2': '#FF6B6B',  # Коралловый
    'ch3': '#4ECDC4',  # Бирюзовый
    'ch4': '#C084FC'   # Фиолетовый
}


def ensure_directories():
    """Создаёт необходимые директории если они не существуют"""
    for directory in [SAVE_DIR, UPLOAD_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)


ensure_directories()

def generate_signal(t, params):
    """Оптимизированная генерация сигнала с использованием lookup таблиц"""
    signal_type = params['type']
    frequency = params['frequency']
    amplitude = params['amplitude']
    offset = params['offset']
    noise_level = params['noise']
    
    # Используем lookup таблицы для основных сигналов (быстрее математических вычислений)
    if signal_type in _lookup_tables and frequency > 0:
        period = 1.0 / frequency
        phase = (t % period) / period
        idx = int(phase * (LOOKUP_TABLE_SIZE - 1)) % LOOKUP_TABLE_SIZE
        value = amplitude * _lookup_tables[signal_type][idx]
    elif signal_type == 'sine':
        value = amplitude * math.sin(2 * math.pi * frequency * t)
    elif signal_type == 'square':
        value = amplitude * (1 if (t * frequency) % 1 < SQUARE_DUTY_CYCLE else -1)
    elif signal_type == 'triangle':
        period = 1.0 / frequency
        phase = (t % period) / period
        if phase < TRIANGLE_PHASE_1:
            value = amplitude * (4 * phase)
        elif phase < TRIANGLE_PHASE_2:
            value = amplitude * (2 - 4 * phase)
        else:
            value = amplitude * (4 * phase - 4)
    elif signal_type == 'sawtooth':
        period = 1.0 / frequency
        phase = (t % period) / period
        value = amplitude * (2 * phase - 1)
    elif signal_type == 'noise':
        value = amplitude * (2 * random.random() - 1)
    elif signal_type == 'composite':
        value = amplitude * math.sin(2 * math.pi * frequency * t)
        value += 0.3 * amplitude * math.sin(2 * math.pi * 3 * frequency * t)
        value += random.gauss(0, noise_level * amplitude)
    else:
        value = amplitude * math.sin(2 * math.pi * frequency * t)

    return value + offset


# Локальные ссылки для ускорения доступа в цикле
_local_config = None
_local_data_buffers = None


def generate_loop():
    """Основной цикл генерации сигналов"""
    global start_time, _local_config, _local_data_buffers
    
    # Кэшируем локальные ссылки для ускорения
    if _local_config is None:
        _local_config = config
        _local_data_buffers = data_buffers
    
    while is_generating:
        # Используем start_time, который уже учитывает все паузы
        t = time.time() - start_time
        
        # Генерируем данные для каждого канала (оптимизированный цикл)
        for channel in CHANNELS:
            if _local_config[channel]['enabled']:
                value = generate_signal(t, _local_config[channel])
                _local_data_buffers[channel].append({'t': round(t, 3), 'value': round(value, 3)})
        
        time.sleep(SLEEP_INTERVAL)  # 50 Гц

@app.route('/')
def index():
    return render_template('index.html', colors=channel_colors)

@app.route('/spectrogram')
def spectrogram_page():
    """Отдельная страница для загрузки файлов и построения спектрограммы"""
    return render_template('spectrogram.html')

@app.route('/api/start', methods=['POST'])
def start_generation():
    """Запускает генерацию сигналов"""
    global is_generating, start_time, total_paused_time, last_pause_time, pause_count
    
    data = request.json
    force_reset = data.get('force_reset', False) if data else False
    
    if not is_generating:
        is_generating = True
        
        # Проверяем, нужно ли сбросить время
        if force_reset or pause_count >= AUTO_RESET_PAUSE_COUNT:
            start_time = time.time()
            total_paused_time = 0
            last_pause_time = 0
            pause_count = 0
            logger.info(f"Автоматический сброс времени после {pause_count} пауз")
        else:
            # Учитываем время последней паузы
            if last_pause_time > 0:
                pause_duration = time.time() - last_pause_time
                total_paused_time += pause_duration
                logger.info(f"Пауза длилась: {pause_duration:.2f} сек, всего пауз: {pause_count}")
            
            # Корректируем start_time с учётом всех пауз
            start_time = time.time() - total_paused_time
        
        last_pause_time = 0
        
        thread = threading.Thread(target=generate_loop)
        thread.daemon = True
        thread.start()
        
        logger.info(f"Генерация запущена. Виртуальное время: {time.time() - start_time:.2f} сек")
        
        return jsonify({
            'success': True,
            'time_reset': force_reset or pause_count >= AUTO_RESET_PAUSE_COUNT,
            'virtual_time': round(time.time() - start_time, 2),
            'pause_count': pause_count
        })
    
    return jsonify({'success': False, 'message': 'Уже запущено'})


@app.route('/api/stop', methods=['POST'])
def stop_generation():
    """Останавливает генерацию сигналов"""
    global is_generating, last_pause_time, pause_count, total_paused_time, start_time
    
    data = request.json
    reset_mode = data.get('reset_mode', 'normal') if data else 'normal'
    
    if is_generating:
        is_generating = False
        current_time = time.time()
        last_pause_time = current_time
        
        # Разные режимы остановки
        if reset_mode == 'full_reset':
            # Полный сброс: время и буфер
            start_time = current_time
            total_paused_time = 0
            last_pause_time = 0
            pause_count = 0
            
            # Очищаем буферы
            clear_all_buffers()
            
            logger.info("ПОЛНЫЙ СБРОС: время и буфер очищены")
            return jsonify({
                'success': True,
                'message': 'Полный сброс выполнен',
                'reset_type': 'full',
                'buffer_cleared': True
            })
            
        elif reset_mode == 'time_reset':
            # Только сброс времени
            start_time = current_time
            total_paused_time = 0
            last_pause_time = 0
            pause_count = 0
            
            logger.info("СБРОС ВРЕМЕНИ: виртуальное время обнулено")
            return jsonify({
                'success': True,
                'message': 'Время сброшено',
                'reset_type': 'time'
            })
            
        else:  # normal
            # Обычная пауза
            pause_count += 1
            logger.info(f"Пауза #{pause_count} в {datetime.now().strftime('%H:%M:%S')}")
            
            # Предупреждение если много пауз
            warning = None
            if pause_count >= PAUSE_WARNING_THRESHOLD:
                warning = f"Сделано {pause_count} пауз. Рекомендуется сбросить время."
            
            return jsonify({
                'success': True,
                'message': 'Пауза',
                'reset_type': 'pause',
                'pause_count': pause_count,
                'warning': warning
            })
    
    return jsonify({'success': False, 'message': 'Уже остановлено'})


def clear_all_buffers():
    """Очищает все буферы данных"""
    for channel in CHANNELS:
        data_buffers[channel].clear()

@app.route('/api/reset_system', methods=['POST'])
def reset_system():
    """Полный сброс системы (время + буфер)"""
    global start_time, total_paused_time, last_pause_time, pause_count, is_generating
    
    data = request.json
    reset_type = data.get('reset_type', 'full') if data else 'full'
    
    was_generating = is_generating
    
    # Останавливаем генерацию если идёт
    if is_generating:
        is_generating = False
        time.sleep(0.1)  # Даём время потоку остановиться
    
    if reset_type == 'full' or reset_type == 'buffer_only':
        # Очищаем буферы
        clear_all_buffers()
    
    if reset_type == 'full' or reset_type == 'time_only':
        # Сбрасываем время
        current_time = time.time()
        start_time = current_time
        total_paused_time = 0
        last_pause_time = 0
        pause_count = 0
    
    # Если генерация была активна, запускаем заново
    if was_generating:
        is_generating = True
        thread = threading.Thread(target=generate_loop)
        thread.daemon = True
        thread.start()
    
    messages = {
        'full': 'Полный сброс системы',
        'time_only': 'Сброс времени',
        'buffer_only': 'Очистка буфера'
    }
    
    return jsonify({
        'success': True,
        'message': messages.get(reset_type, 'Сброс выполнен'),
        'reset_type': reset_type,
        'was_generating': was_generating
    })


@app.route('/api/time_info')
def get_time_info():
    """Информация о времени для отладки"""
    global start_time, total_paused_time, last_pause_time, pause_count
    
    current_time = time.time()
    virtual_time = current_time - start_time if is_generating else last_pause_time - start_time
    
    # Информация о буферах
    buffer_sizes = {ch: len(data_buffers[ch]) for ch in CHANNELS}
    
    return jsonify({
        'is_generating': is_generating,
        'virtual_time': round(virtual_time, 2),
        'total_paused': round(total_paused_time, 2),
        'pause_count': pause_count,
        'start_time': datetime.fromtimestamp(start_time).strftime('%H:%M:%S'),
        'last_pause': datetime.fromtimestamp(last_pause_time).strftime('%H:%M:%S') if last_pause_time > 0 else 'None',
        'buffer_sizes': buffer_sizes,
        'total_points': sum(buffer_sizes.values())
    })


@app.route('/api/data')
def get_signal_data():
    """Возвращает данные буферов с оптимизацией через incremental updates"""
    # Преобразуем deque в list один раз для каждого канала
    return jsonify({
        channel: list(data_buffers[channel]) for channel in CHANNELS
    })


@app.route('/api/data/latest')
def get_latest_signal_data():
    """Возвращает только новые данные с момента последнего запроса (оптимизация трафика)"""
    global last_sent_data
    
    result = {}
    for channel in CHANNELS:
        buffer = data_buffers[channel]
        last_idx = last_sent_data.get(channel, {}).get('last_idx', 0)
        current_len = len(buffer)
        
        if current_len > last_idx:
            # Конвертируем только новые элементы
            buffer_list = list(buffer)
            result[channel] = buffer_list[last_idx:]
            last_sent_data[channel] = {'last_idx': current_len}
        else:
            result[channel] = []
    
    return jsonify(result)

@app.route('/api/save_channel/<channel>', methods=['POST'])
def save_channel(channel):
    """Сохраняет сигнал указанного канала"""
    if channel not in CHANNELS:
        return jsonify({'success': False, 'error': 'Канал не найден'})
    
    data = request.json
    duration = data.get('duration', 10)
    
    filename = save_signal_to_file(channel, duration)
    
    if filename:
        return jsonify({
            'success': True,
            'message': f'Сохранено {duration} секунд',
            'filename': os.path.basename(filename)
        })
    else:
        return jsonify({'success': False, 'error': 'Недостаточно данных'})


def save_signal_to_file(channel, duration=10):
    """Сохраняет сигнал указанного канала в файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIR}/{channel}_{config[channel]['type']}_{timestamp}.csv"
    
    # Получаем данные из буфера
    data = list(data_buffers[channel])
    
    if len(data) < MIN_POINTS_FOR_SPECTROGRAM // 10:
        return None
    
    # Сохраняем в CSV
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Value', 'Type', 'Frequency', 'Amplitude', 'Offset', 'Noise'])
            points_to_save = min(int(duration * DEFAULT_SAMPLING_RATE), len(data))
            for point in data[-points_to_save:]:
                writer.writerow([
                    point['t'],
                    point['value'],
                    config[channel]['type'],
                    config[channel]['frequency'],
                    config[channel]['amplitude'],
                    config[channel]['offset'],
                    config[channel]['noise']
                ])
        
        logger.info(f"Сигнал канала {channel} сохранён в {filename}")
        return filename
    except IOError as e:
        logger.error(f"Ошибка записи файла {filename}: {e}")
        return None


def read_signal_file(filepath, filename):
    """Читает данные сигнала из файла. Возвращает (values, times, sampling_rate)"""
    try:
        if filename.endswith(CSV_EXTENSION):
            file_data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
            if file_data.ndim == 1:
                file_data = file_data.reshape(-1, 1)
            if file_data.shape[1] >= 2:
                values = file_data[:, 1]
                times = file_data[:, 0]
                sampling_rate = 1.0 / np.mean(np.diff(times)) if len(times) > 1 else DEFAULT_SAMPLING_RATE
            else:
                values = file_data[:, 0]
                times = np.arange(len(values)) / DEFAULT_SAMPLING_RATE
                sampling_rate = DEFAULT_SAMPLING_RATE
        else:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            values = np.array([float(line.strip()) for line in lines if line.strip()])
            times = np.arange(len(values)) / DEFAULT_SAMPLING_RATE
            sampling_rate = DEFAULT_SAMPLING_RATE
        
        return values, times, sampling_rate
    except Exception as e:
        logger.error(f"Ошибка чтения файла {filepath}: {e}")
        raise


@app.route('/api/upload_signal', methods=['POST'])
def upload_signal():
    """Загружает файл с сигналом для спектрограммы"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Файл не найден'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Имя файла пустое'})
    
    # Сохраняем файл
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)
    
    # Пытаемся прочитать данные
    try:
        values, times, _ = read_signal_file(filepath, filename)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'points': len(values),
            'duration': float(times[-1]) if len(times) > 0 else 0
        })
    except Exception as e:
        logger.error(f"Ошибка обработки загруженного файла: {e}")
        return jsonify({'success': False, 'error': 'Ошибка обработки файла'})

# Кэш для спектрограмм (избегаем повторных вычислений)
_spectrogram_cache = {}


@app.route('/api/generate_spectrogram', methods=['POST'])
def generate_spectrogram():
    """Генерирует спектрограмму из загруженного файла используя ShortTimeFFT"""
    data = request.json
    filename = data.get('filename')
    window_size = data.get('window_size', DEFAULT_WINDOW_SIZE)
    overlap_percent = data.get('overlap', DEFAULT_OVERLAP_PERCENT)
    
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Читаем данные с помощью универсальной функции
        values, times, sampling_rate = read_signal_file(filepath, filename)
        
        # Убираем среднее
        values = values - np.mean(values)
        
        # Параметры STFT
        nperseg = window_size
        hop = int(nperseg * (1 - overlap_percent))  # Шаг между окнами
        if hop < 1:
            hop = 1
            
        # Создаем окно (массив значений)
        from scipy.signal.windows import tukey
        window = tukey(nperseg, alpha=0.25)  # Окно Тьюки как массив
        
        # Создаем объект ShortTimeFFT (правильные параметры)
        from scipy.signal import ShortTimeFFT
        stft = ShortTimeFFT(
            win=window,
            hop=hop,
            fs=sampling_rate,
            mfft=nperseg * 2,  # Длина БПФ
            scale_to='magnitude'
        )
        
        # Вычисляем STFT
        Sxx = stft.stft(values)
        
        # Получаем частоты и времена
        frequencies = stft.f
        times_spec = stft.t(len(values))
        
        # Обрезаем до полезных частот (до 25 Гц)
        freq_mask = frequencies <= MAX_FREQUENCY_LIMIT
        frequencies = frequencies[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Преобразуем в децибелы
        Sxx_db = 20 * np.log10(np.abs(Sxx) + 1e-10)
        
        # Создаем изображение спектрограммы
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Верхний график - исходный сигнал
        ax1.plot(times, values, color='#3b82f6', linewidth=1)
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Амплитуда')
        ax1.set_title('Исходный сигнал')
        ax1.grid(True, alpha=0.3)
        
        # Нижний график - спектрограмма
        pcm = ax2.pcolormesh(times_spec, frequencies, Sxx_db, 
                             shading='gouraud', cmap='inferno')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Частота (Гц)')
        ax2.set_title('Спектрограмма (ShortTimeFFT)')
        ax2.set_ylim(0, MAX_FREQUENCY_LIMIT)
        
        # Добавляем цветовую шкалу
        plt.colorbar(pcm, ax=ax2, label='Амплитуда (дБ)')
        
        plt.tight_layout()
        
        # Сохраняем в base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'frequencies': frequencies.tolist(),
            'times': times_spec.tolist(),
            'spectrogram': Sxx_db.tolist(),
            'method': 'ShortTimeFFT'
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации спектрограммы: {e}")
        return jsonify({'success': False, 'error': 'Ошибка при генерации спектрограммы'})

@app.route('/api/amplitude_spectrum', methods=['POST'])
def get_amplitude_spectrum():
    """Строит амплитудный спектр сигнала из CSV файла"""
    data = request.json
    filename = data.get('filename')
    
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Читаем данные с помощью универсальной функции
        values, times, sampling_rate = read_signal_file(filepath, filename)
        
        # Убираем среднее (постоянную составляющую)
        values = values - np.mean(values)
        
        # Параметры для БПФ
        n = len(values)
        
        # Если сигнал слишком короткий, дополняем нулями для лучшего разрешения
        n_fft = MIN_FFT_SIZE if n < MIN_FFT_SIZE else n
        
        # Вычисляем БПФ
        fft_vals = np.fft.fft(values, n=n_fft)
        fft_vals = fft_vals[:n_fft//2]  # Только положительные частоты
        
        # Амплитудный спектр
        amplitudes = np.abs(fft_vals) * 2 / n  # Нормировка амплитуды
        
        # Частотная ось
        freqs = np.fft.fftfreq(n_fft, 1/sampling_rate)[:n_fft//2]
        
        # Обрезаем до разумных частот (до 25 Гц)
        mask = freqs <= MAX_FREQUENCY_LIMIT
        freqs_limited = freqs[mask]
        amplitudes_limited = amplitudes[mask]
        
        # Находим основные частоты (пики)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(amplitudes_limited, height=0.1 * np.max(amplitudes_limited))
        peak_freqs = freqs_limited[peaks]
        peak_amps = amplitudes_limited[peaks]
        
        # Сортируем пики по амплитуде
        peak_indices = np.argsort(peak_amps)[::-1]
        peak_freqs = peak_freqs[peak_indices]
        peak_amps = peak_amps[peak_indices]
        
        # Создаем изображение
        fig = plt.figure(figsize=(14, 10))
        
        # 1. Исходный сигнал
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(times, values, color='#3b82f6', linewidth=1)
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Амплитуда')
        ax1.set_title(f'Исходный сигнал (длина: {n} точек, частота дискретизации: {sampling_rate:.1f} Гц)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Амплитудный спектр (полный)
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(freqs_limited, amplitudes_limited, color='#FF6B6B', linewidth=1.5)
        ax2.set_xlabel('Частота (Гц)')
        ax2.set_ylabel('Амплитуда')
        ax2.set_title('Амплитудный спектр сигнала')
        ax2.grid(True, alpha=0.3)
        
        # Отмечаем пики
        for i, (freq, amp) in enumerate(zip(peak_freqs[:MIN_PEAKS_TO_DISPLAY], peak_amps[:MIN_PEAKS_TO_DISPLAY])):
            if amp > 0.01 * np.max(amplitudes_limited):
                ax2.plot(freq, amp, 'ro', markersize=8)
                ax2.annotate(f'{freq:.2f} Гц', (freq, amp), 
                            xytext=(5, 5), textcoords='offset points',
                            color='white', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF6B6B', alpha=0.7))
        
        # 3. Столбчатая диаграмма основных частот
        ax3 = plt.subplot(3, 1, 3)
        if len(peak_freqs) > 0:
            # Берем до 10 самых сильных пиков
            n_peaks = min(MAX_PEAKS_DISPLAY, len(peak_freqs))
            freqs_display = peak_freqs[:n_peaks]
            amps_display = peak_amps[:n_peaks]
            
            bars = ax3.bar(range(n_peaks), amps_display, color='#4ECDC4', alpha=0.8)
            ax3.set_xticks(range(n_peaks))
            ax3.set_xticklabels([f'{f:.2f} Гц' for f in freqs_display], rotation=45, ha='right')
            ax3.set_ylabel('Амплитуда')
            ax3.set_title('Основные частотные компоненты')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Добавляем значения над столбцами
            for i, (bar, amp) in enumerate(zip(bars, amps_display)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(amps_display),
                        f'{amp:.3f}', ha='center', va='bottom', color='white', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Значимые пики не найдены', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        plt.tight_layout()
        
        # Сохраняем в base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Подготавливаем данные для таблицы
        peaks_data = []
        for i, (freq, amp) in enumerate(zip(peak_freqs[:MAX_PEAKS_DISPLAY], peak_amps[:MAX_PEAKS_DISPLAY])):
            if amp > 0.01 * np.max(amplitudes_limited):
                peaks_data.append({
                    'number': i + 1,
                    'frequency': round(freq, 3),
                    'amplitude': round(amp, 3)
                })
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'peaks': peaks_data,
            'fundamental_freq': round(peak_freqs[0], 3) if len(peak_freqs) > 0 else 0,
            'max_amplitude': round(np.max(amplitudes_limited), 3),
            'signal_length': n,
            'sampling_rate': round(sampling_rate, 1)
        })
        
    except Exception as e:
        logger.error(f"Ошибка построения амплитудного спектра: {e}")
        return jsonify({'success': False, 'error': 'Ошибка при построении спектра'})


@app.route('/api/list_saved')
def list_saved_files():
    """Список сохраненных файлов"""
    files = []
    try:
        for f in os.listdir(SAVE_DIR):
            if f.endswith(CSV_EXTENSION):
                file_path = os.path.join(SAVE_DIR, f)
                stats = os.stat(file_path)
                files.append({
                    'name': f,
                    'size': round(stats.st_size / 1024, 1),
                    'modified': datetime.fromtimestamp(stats.st_mtime).strftime("%H:%M:%S"),
                    'date': datetime.fromtimestamp(stats.st_mtime).strftime("%d.%m.%Y")
                })
    except OSError as e:
        logger.error(f"Ошибка чтения директории {SAVE_DIR}: {e}")
    
    return jsonify(sorted(files, key=lambda x: x['modified'], reverse=True)[:MAX_SAVED_FILES_DISPLAY])


@app.route('/api/download/<filename>')
def download_file(filename):
    """Скачать сохраненный файл"""
    return send_file(os.path.join(SAVE_DIR, filename), as_attachment=True)


@app.route('/api/spectrogram/<channel>')
def get_channel_spectrogram(channel):
    """Возвращает данные для спектрограммы указанного канала"""
    if channel not in CHANNELS:
        return jsonify({'error': 'Канал не найден'}), 404
    
    data = list(data_buffers[channel])
    if len(data) < MIN_POINTS_FOR_SPECTROGRAM:
        return jsonify({'frequencies': [], 'magnitudes': []})
    
    n_points = min(1024, len(data))
    values = [point['value'] for point in data[-n_points:]]
    
    fft = np.fft.fft(values)
    fft = np.abs(fft[:n_points//2])
    
    freqs = np.linspace(0, MAX_FREQUENCY_LIMIT, len(fft))
    
    return jsonify({
        'frequencies': freqs.tolist(),
        'magnitudes': fft.tolist()
    })


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    global config
    if request.method == 'POST':
        data = request.json
        if 'channel' in data:
            config[data['channel']].update(data['params'])
        else:
            config.update(data)
    return jsonify(config)


@app.route('/api/channel/<ch>/toggle', methods=['POST'])
def toggle_channel(ch):
    if ch in CHANNELS:
        config[ch]['enabled'] = not config[ch]['enabled']
    return jsonify({'success': True, 'enabled': config[ch]['enabled']})


@app.route('/api/clear_buffer', methods=['POST'])
def clear_buffer():
    """Очищает все буферы данных"""
    clear_all_buffers()
    return jsonify({'success': True})

if __name__ == '__main__':
    print("="*70)
    print("🚀 ЦИФРОВОЙ ОСЦИЛЛОГРАФ DS-2026")
    print("="*70)
    print("\n📊 Доступные страницы:")
    print("  🌐 Главная:        http://localhost:5000/")
    print("  📈 Спектрограмма:  http://localhost:5000/spectrogram")
    print("\n🎛️  Каналы:")
    print("\n💾 Сохранение: до 5000 точек на канал")
    print("="*70)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)