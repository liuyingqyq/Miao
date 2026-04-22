import json
import os
import sys
import html
import shutil
import re
import requests
import tiktoken
import markdown
import importlib.util
import traceback
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, QTextEdit,
                               QPushButton, QSplitter, QMessageBox, QCheckBox,
                               QGroupBox, QStyleFactory, QComboBox,
                               QSpinBox, QFontComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QInputDialog, QListWidget, QListWidgetItem,
                               QTabWidget, QTreeWidget, QTreeWidgetItem, QCheckBox as QCheckBoxWidget)
from PySide6.QtCore import Qt, QThread, Signal, QEvent, QUrl
from PySide6.QtGui import QFont, QPalette, QColor, QFontDatabase, QDesktopServices

# ---------- API 配置 ----------
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_CHAT = "deepseek-chat"
MODEL_REASONER = "deepseek-reasoner"

# ---------- 配置文件路径 ----------
DOCUMENTS_DIR = Path.home() / "Documents" / "DeepSeekClient"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = DOCUMENTS_DIR / "deepseek_config.json"
ARCHIVE_DIR = DOCUMENTS_DIR / "archives"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
# ---------- 字体存储目录 ----------
FONTS_DIR = Path(__file__).parent / "fonts"
FONTS_DIR.mkdir(exist_ok=True)

# ---------- 插件目录 ----------
PLUGIN_DIR = Path(__file__).parent / "mood"
PLUGIN_DIR.mkdir(exist_ok=True)

# ---------- 字符过滤 ----------
def filter_text(text):
    pattern = r'[\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3\u0600-\u06ff]'
    return re.sub(pattern, '', text)

# ---------- Markdown 转 HTML ----------
def markdown_to_html(text):
    md_extensions = ['extra', 'tables', 'fenced_code', 'codehilite']
    html_content = markdown.markdown(text, extensions=md_extensions)
    html_content = html_content.replace('<tr>', '<table class="markdown-table">')
    return html_content

# ---------- Token 估算 ----------
def estimate_tokens(text):
    if not text:
        return 0
    chinese = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    other = len(re.findall(r'[0-9\W_]', text))
    return int(chinese + english_words * 1.3 + other * 0.5)

def count_tokens(messages):
    tokens = 0
    for msg in messages:
        tokens += estimate_tokens(msg.get("content", ""))
    return tokens

# ---------- 插件管理器 ----------
class PluginManager:
    def __init__(self):
        self.plugins = []
        self.load_plugins()

    def load_plugins(self):
        if not PLUGIN_DIR.exists():
            return
        for py_file in PLUGIN_DIR.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                info = module.plugin_info() if hasattr(module, 'plugin_info') else {"name": py_file.stem, "version": "1.0"}
                self.plugins.append({
                    "module": module,
                    "name": info.get("name", py_file.stem),
                    "version": info.get("version", "1.0"),
                    "enabled": True,
                    "file": py_file.name
                })
            except Exception as e:
                print(f"Failed to load plugin {py_file.name}: {e}\n{traceback.format_exc()}")

    def call_hook(self, hook_name, *args, **kwargs):
        results = []
        for plugin in self.plugins:
            if not plugin["enabled"]:
                continue
            if hasattr(plugin["module"], hook_name):
                try:
                    result = getattr(plugin["module"], hook_name)(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"Plugin {plugin['name']} hook {hook_name} error: {e}")
        return results

# ---------- 流式推理工作线程 ----------
class StreamWorker(QThread):
    chunk_received = Signal(str, str)  # (type, content) type: 'reasoning', 'content', 'error'
    finished = Signal()

    def __init__(self, api_key, messages, model):
        super().__init__()
        self.api_key = api_key
        self.messages = messages
        self.model = model

    def run(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": self.messages, "stream": True}
        try:
            response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=300)
            if response.status_code != 200:
                self.chunk_received.emit("error", f"API 错误 {response.status_code}")
                self.finished.emit()
                return
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "reasoning_content" in delta and delta["reasoning_content"]:
                                self.chunk_received.emit("reasoning", delta["reasoning_content"])
                            if "content" in delta and delta["content"]:
                                self.chunk_received.emit("content", delta["content"])
                        except:
                            continue
            self.finished.emit()
        except Exception as e:
            self.chunk_received.emit("error", f"请求失败: {str(e)}")
            self.finished.emit()

# ---------- 推理显示窗口（单例模式，追加显示）----------
class ReasoningDialog(QDialog):
    _instance = None

    def __new__(cls, parent=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, parent=None):
        if hasattr(self, '_initialized'):
            return
        super().__init__(parent)
        self.setWindowTitle("实时推理过程")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("正在等待数据…"))
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.display)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        self.worker = None
        self.current_reasoning = ""
        self.current_content = ""
        self._initialized = True

    def start_stream(self, api_key, messages, model, prompt_name):
        """启动新推理，追加到现有内容"""
        # 添加分隔线和提示信息
        separator = "\n" + "="*60 + f"\n【新对话】提示词: {prompt_name}  时间: {datetime.now().strftime('%H:%M:%S')}\n" + "="*60 + "\n"
        self.display.append(separator)
        # 清空当前累积（新推理）
        self.current_reasoning = ""
        self.current_content = ""
        # 重置标题插入标记
        self._reasoning_title_inserted = False
        self._content_title_inserted = False
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.worker = StreamWorker(api_key, messages, model)
        self.worker.chunk_received.connect(self.on_chunk)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_chunk(self, typ, content):
        if typ == "reasoning":
            self.current_reasoning += content
            self.append_reasoning(content)
        elif typ == "content":
            self.current_content += content
            self.append_content(content)
        elif typ == "error":
            self.display.append(f"<font color='red'>错误: {content}</font>")

    def append_reasoning(self, text):
        if not self._reasoning_title_inserted:
            self.display.append("<b>【推理过程】</b>")
            self._reasoning_title_inserted = True
        self.display.insertPlainText(text)
        self.display.ensureCursorVisible()

    def append_content(self, text):
        if self._reasoning_title_inserted and not self._content_title_inserted:
            self.display.append("")  # 空行分隔
            self.display.append("<b>【最终回复】</b>")
            self._content_title_inserted = True
        elif not self._content_title_inserted:
            self.display.append("<b>【最终回复】</b>")
            self._content_title_inserted = True
        self.display.insertPlainText(text)
        self.display.ensureCursorVisible()

    def on_finished(self):
        if not self.current_content and not self.current_reasoning:
            self.display.append("未接收到内容，请检查网络或API Key。")
        # 添加空行分隔
        self.display.append("\n")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        event.accept()

# ---------- 插件管理对话框 ----------
class PluginManagerDialog(QDialog):
    def __init__(self, plugin_manager, parent=None):
        super().__init__(parent)
        self.plugin_manager = plugin_manager
        self.setWindowTitle("插件管理")
        self.resize(500, 400)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("已加载的插件："))
        self.plugin_list = QListWidget()
        self.checkboxes = []
        for idx, plugin in enumerate(plugin_manager.plugins):
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0,0,0,0)
            cb = QCheckBoxWidget(f"{plugin['name']} v{plugin['version']}")
            cb.setChecked(plugin["enabled"])
            cb.toggled.connect(lambda checked, p=plugin: self.toggle_plugin(p, checked))
            item_layout.addWidget(cb)
            item_layout.addStretch()
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.plugin_list.addItem(list_item)
            self.plugin_list.setItemWidget(list_item, item_widget)
            self.checkboxes.append(cb)
        layout.addWidget(self.plugin_list)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("刷新插件列表")
        refresh_btn.clicked.connect(self.refresh_plugins)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def toggle_plugin(self, plugin, enabled):
        plugin["enabled"] = enabled

    def refresh_plugins(self):
        self.plugin_manager.load_plugins()
        self.plugin_list.clear()
        for idx, plugin in enumerate(self.plugin_manager.plugins):
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            cb = QCheckBoxWidget(f"{plugin['name']} v{plugin['version']}")
            cb.setChecked(plugin["enabled"])
            cb.toggled.connect(lambda checked, p=plugin: self.toggle_plugin(p, checked))
            item_layout.addWidget(cb)
            item_layout.addStretch()
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.plugin_list.addItem(list_item)
            self.plugin_list.setItemWidget(list_item, item_widget)

# ---------- 普通工作线程（非流式）----------
class Worker(QThread):
    finished = Signal(str, dict)
    error = Signal(str)

    def __init__(self, api_key, messages, model):
        super().__init__()
        self.api_key = api_key
        self.messages = messages
        self.model = model

    def run(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": self.messages, "stream": False}
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                self.finished.emit(reply, usage)
            else:
                self.error.emit(f"API 错误 {response.status_code}")
        except Exception as e:
            self.error.emit(f"请求失败: {str(e)}")

# ---------- 多行输入对话框 ----------
class MultiLineInputDialog(QDialog):
    def __init__(self, title, label, text="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(label))
        self.edit = QTextEdit()
        self.edit.setPlainText(text)
        self.edit.setMinimumHeight(150)
        layout.addWidget(self.edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_text(self):
        return self.edit.toPlainText().strip()

# ---------- 搜索对话框 ----------
class SearchDialog(QDialog):
    def __init__(self, parent, messages):
        super().__init__(parent)
        self.setWindowTitle("搜索对话历史")
        self.resize(600, 500)
        self.messages = messages
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("搜索关键词:"))
        self.search_input = QLineEdit()
        self.search_input.returnPressed.connect(self.search)
        layout.addWidget(self.search_input)
        self.result_list = QListWidget()
        self.result_list.itemDoubleClicked.connect(self.jump_to_message)
        layout.addWidget(self.result_list)
        btn_layout = QHBoxLayout()
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.search)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(search_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def search(self):
        keyword = self.search_input.text().strip()
        if not keyword:
            return
        self.result_list.clear()
        for idx, msg in enumerate(self.messages):
            if keyword.lower() in msg["content"].lower():
                item = QListWidgetItem(f"{msg['role']}: {msg['content'][:100]}...")
                item.setData(Qt.UserRole, idx)
                self.result_list.addItem(item)

    def jump_to_message(self, item):
        idx = item.data(Qt.UserRole)
        self.parent().jump_to_message(idx)
        self.close()

# ---------- 主窗口 ----------
class DeepSeekClient(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSeek")
        self.resize(1000, 750)
        self.setMinimumSize(800, 600)

        self.prompts = []
        self.current_prompt_index = 0
        self.config = {}
        self.total_tokens_used = 0
        self.archived_conversations = []

        self.plugin_manager = PluginManager()
        self.plugin_manager.call_hook("on_init", self)

        self.load_config()
        self.setup_ui()
        self.load_current_prompt()
        self.apply_system_theme()
        self.load_custom_fonts()

    # ---------- 插件钩子辅助 ----------
    def trigger_plugin_hook(self, hook_name, *args, **kwargs):
        return self.plugin_manager.call_hook(hook_name, *args, **kwargs)

    # ---------- 自定义字体管理 ----------
    def load_custom_fonts(self):
        if not FONTS_DIR.exists():
            return
        for font_file in list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf")):
            QFontDatabase.addApplicationFont(str(font_file))

    def import_font(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择字体文件", "", "字体文件 (*.ttf *.otf)")
        if not path:
            return
        dest = FONTS_DIR / Path(path).name
        if dest.exists() and QMessageBox.question(self, "覆盖", "字体已存在，是否覆盖？") != QMessageBox.Yes:
            return
        shutil.copy2(path, dest)
        QFontDatabase.addApplicationFont(str(dest))
        QMessageBox.information(self, "成功", "字体已导入，可在下拉框中选择。")

    # ---------- 配置持久化 ----------
    def load_config(self):
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.config = data.get("global", {})
                    self.prompts = data.get("prompts", [])
                    self.archived_conversations = data.get("archives", [])
            else:
                self.config = {}
                self.prompts = []
                self.archived_conversations = []
        except Exception:
            self.config, self.prompts, self.archived_conversations = {}, [], []
        if not self.prompts:
            self.prompts.append({"name": "默认助手", "system_prompt": "You are a helpful assistant.", "messages": [], "model": MODEL_CHAT})
        for p in self.prompts:
            if "model" not in p:
                p["model"] = MODEL_CHAT
        self.current_prompt_index = self.config.get("current_prompt_index", 0)
        if self.current_prompt_index >= len(self.prompts):
            self.current_prompt_index = 0
        self.total_tokens_used = self.config.get("total_tokens_used", 0)

    def save_config(self):
        self.config.update({
            "api_key": self.api_key_edit.text().strip(),
            "username": self.username_edit.text().strip(),
            "font_family": self.font_combo.currentFont().family(),
            "font_size": self.font_size_spin.value(),
            "current_prompt_index": self.current_prompt_index,
            "total_tokens_used": self.total_tokens_used
        })
        data = {"global": self.config, "prompts": self.prompts, "archives": self.archived_conversations}
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    # ---------- 主题跟随 ----------
    def apply_system_theme(self):
        if sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                is_light = winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 1
                winreg.CloseKey(key)
            except:
                is_light = True
        else:
            is_light = True
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        (self.apply_light_theme if is_light else self.apply_dark_theme)()

    def apply_light_theme(self):
        p = QPalette()
        p.setColor(QPalette.Window, QColor(240,240,240))
        p.setColor(QPalette.WindowText, Qt.black)
        p.setColor(QPalette.Base, Qt.white)
        p.setColor(QPalette.Text, Qt.black)
        p.setColor(QPalette.Button, QColor(240,240,240))
        p.setColor(QPalette.ButtonText, Qt.black)
        QApplication.setPalette(p)
        self.setStyleSheet("""
            QGroupBox{border:1px solid #ccc;border-radius:5px;margin-top:0.5em;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}
            QPushButton{background:#e0e0e0;border:1px solid #aaa;border-radius:3px;padding:5px;}
            QPushButton:hover{background:#d0d0d0;}
            QLineEdit,QTextEdit,QComboBox{border:1px solid #ccc;border-radius:3px;padding:3px;}
        """)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background: white;
                border: none;
                font-family: "Microsoft YaHei";
            }
            pre {
                background-color: #f4f4f4;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                overflow-x: auto;
            }
            code {
                background-color: #f4f4f4;
                border-radius: 3px;
                padding: 2px 4px;
                font-family: monospace;
            }
            blockquote {
                border-left: 3px solid #ccc;
                margin: 0;
                padding-left: 10px;
                color: #555;
            }
            table.markdown-table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            table.markdown-table th,
            table.markdown-table td {
                border: 1px solid #ddd;
                padding: 6px;
                text-align: left;
            }
            table.markdown-table th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
        """)
        self.top_role_label.setStyleSheet("font-weight:bold;font-size:14px;background:#e0e0e0;padding:5px;")
        self.token_label_title.setStyleSheet("font-size:10px;color:#555;")
        self.token_label_value.setStyleSheet("font-size:12px;font-weight:bold;color:#555;")
        self.total_token_label.setStyleSheet("font-size:10px;color:#555;")
        self.total_token_value.setStyleSheet("font-size:12px;font-weight:bold;color:#555;")

    def apply_dark_theme(self):
        p = QPalette()
        p.setColor(QPalette.Window, QColor(53,53,53))
        p.setColor(QPalette.WindowText, Qt.white)
        p.setColor(QPalette.Base, QColor(25,25,25))
        p.setColor(QPalette.Text, Qt.white)
        p.setColor(QPalette.Button, QColor(53,53,53))
        p.setColor(QPalette.ButtonText, Qt.white)
        QApplication.setPalette(p)
        self.setStyleSheet("""
            QGroupBox{border:1px solid #555;border-radius:5px;margin-top:0.5em;color:#fff;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;color:#fff;}
            QPushButton{background:#3c3c3c;border:1px solid #555;border-radius:3px;padding:5px;color:#fff;}
            QPushButton:hover{background:#4a4a4a;}
            QLineEdit,QTextEdit,QComboBox{background:#3c3c3c;border:1px solid #555;border-radius:3px;padding:3px;color:#fff;}
            QLabel{color:#fff;}
        """)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                border: none;
                color: #fff;
                font-family: "Microsoft YaHei";
            }
            pre {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                color: #ddd;
                overflow-x: auto;
            }
            code {
                background-color: #2d2d2d;
                border-radius: 3px;
                padding: 2px 4px;
                font-family: monospace;
                color: #ddd;
            }
            blockquote {
                border-left: 3px solid #888;
                margin: 0;
                padding-left: 10px;
                color: #aaa;
            }
            table.markdown-table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            table.markdown-table th,
            table.markdown-table td {
                border: 1px solid #555;
                padding: 6px;
                text-align: left;
                color: #fff;
            }
            table.markdown-table th {
                background-color: #3a3a3a;
                font-weight: bold;
            }
        """)
        self.top_role_label.setStyleSheet("font-weight:bold;font-size:14px;background:#3c3c3c;padding:5px;color:#fff;")
        self.token_label_title.setStyleSheet("font-size:10px;color:#aaa;")
        self.token_label_value.setStyleSheet("font-size:12px;font-weight:bold;color:#aaa;")
        self.total_token_label.setStyleSheet("font-size:10px;color:#aaa;")
        self.total_token_value.setStyleSheet("font-size:12px;font-weight:bold;color:#aaa;")

    # ---------- UI 构建 ----------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 配置区域
        config_group = QGroupBox("基本配置")
        grid = QGridLayout(config_group)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setText(self.config.get("api_key", ""))
        self.api_key_edit.textChanged.connect(self.save_config)
        self.show_key_cb = QCheckBox("显示")
        self.show_key_cb.toggled.connect(lambda checked: self.api_key_edit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password))
        key_layout = QHBoxLayout()
        key_layout.addWidget(self.api_key_edit, 1)
        key_layout.addWidget(self.show_key_cb)
        grid.addWidget(QLabel("API Key:"), 0, 0)
        grid.addLayout(key_layout, 0, 1)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("你的昵称")
        self.username_edit.setText(self.config.get("username", "我"))
        self.username_edit.textChanged.connect(self.save_config)
        grid.addWidget(QLabel("用户昵称:"), 1, 0)
        grid.addWidget(self.username_edit, 1, 1)

        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(self.config.get("font_family", "Microsoft YaHei")))
        self.font_combo.currentFontChanged.connect(self.change_font)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8,20)
        self.font_size_spin.setValue(self.config.get("font_size",10))
        self.font_size_spin.valueChanged.connect(self.change_font)
        import_btn = QPushButton("导入字体")
        import_btn.clicked.connect(self.import_font)
        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self.save_config)
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("字体:"))
        font_layout.addWidget(self.font_combo,2)
        font_layout.addWidget(QLabel("大小:"))
        font_layout.addWidget(self.font_size_spin)
        font_layout.addWidget(import_btn)
        font_layout.addWidget(save_btn)
        grid.addWidget(QLabel("聊天字体:"),2,0)
        grid.addLayout(font_layout,2,1)
        layout.addWidget(config_group)

        # 提示词管理
        prompt_group = QGroupBox("提示词管理")
        pvbox = QVBoxLayout(prompt_group)
        row1 = QHBoxLayout()
        self.prompt_combo = QComboBox()
        self.prompt_combo.currentIndexChanged.connect(self.switch_prompt)
        row1.addWidget(QLabel("当前提示词:"))
        row1.addWidget(self.prompt_combo,1)
        rename_btn = QPushButton("重命名")
        rename_btn.clicked.connect(self.rename_prompt)
        row1.addWidget(rename_btn)
        new_btn = QPushButton("新建")
        new_btn.clicked.connect(self.new_prompt)
        row1.addWidget(new_btn)
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(self.delete_prompt)
        row1.addWidget(del_btn)
        edit_btn = QPushButton("编辑内容")
        edit_btn.clicked.connect(self.edit_prompt)
        row1.addWidget(edit_btn)
        pvbox.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("当前提示词使用的模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("deepseek-chat", MODEL_CHAT)
        self.model_combo.addItem("deepseek-reasoner", MODEL_REASONER)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        row2.addWidget(self.model_combo)
        row2.addStretch()
        pvbox.addLayout(row2)
        layout.addWidget(prompt_group)

        # 对话区域
        chat_group = QGroupBox("对话")
        chat_layout = QVBoxLayout(chat_group)

        top_info_widget = QWidget()
        top_info_layout = QHBoxLayout(top_info_widget)
        self.top_role_label = QLabel()
        self.top_role_label.setAlignment(Qt.AlignCenter)
        top_info_layout.addWidget(self.top_role_label, 1)

        token_widget = QWidget()
        token_grid = QGridLayout(token_widget)
        token_grid.setContentsMargins(0,0,0,0)
        self.token_label_title = QLabel("上下文 token (估算):")
        self.token_label_title.setAlignment(Qt.AlignRight)
        self.token_label_value = QLabel("0")
        self.token_label_value.setAlignment(Qt.AlignRight)
        self.total_token_label = QLabel("累计消耗 token (官方):")
        self.total_token_label.setAlignment(Qt.AlignRight)
        self.total_token_value = QLabel(str(self.total_tokens_used))
        self.total_token_value.setAlignment(Qt.AlignRight)
        token_grid.addWidget(self.token_label_title, 0, 0)
        token_grid.addWidget(self.token_label_value, 0, 1)
        token_grid.addWidget(self.total_token_label, 1, 0)
        token_grid.addWidget(self.total_token_value, 1, 1)
        top_info_layout.addWidget(token_widget)

        chat_layout.addWidget(top_info_widget)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setAcceptRichText(True)
        chat_layout.addWidget(self.chat_display)

        # 输入区域
        input_widget = QWidget()
        input_v = QVBoxLayout(input_widget)
        input_v.setContentsMargins(0,0,0,0)

        # 文件上传
        file_layout = QHBoxLayout()
        self.upload_btn = QPushButton("上传文件")
        self.upload_btn.clicked.connect(self.upload_file)
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("color: gray; font-size: 10px;")
        file_layout.addWidget(self.upload_btn)
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        input_v.addLayout(file_layout)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("输入消息... (Enter 发送, Shift+Enter 换行)")
        self.input_text.setMaximumHeight(100)
        self.input_text.textChanged.connect(self.update_token_display)
        input_v.addWidget(self.input_text)

        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setDefault(True)
        self.reasoning_btn = QPushButton("显示推理")
        self.reasoning_btn.clicked.connect(self.show_reasoning)
        clear_btn = QPushButton("清空当前对话")
        clear_btn.clicked.connect(self.clear_current_conversation)
        export_btn = QPushButton("导出对话")
        export_btn.clicked.connect(self.export_conversation)
        search_btn = QPushButton("搜索历史")
        search_btn.clicked.connect(self.search_history)
        archive_btn = QPushButton("归档当前对话")
        archive_btn.clicked.connect(self.archive_conversation)
        manage_btn = QPushButton("管理归档")
        manage_btn.clicked.connect(self.manage_archives)
        plugin_btn = QPushButton("插件管理")
        plugin_btn.clicked.connect(self.manage_plugins)

        btn_layout.addStretch()
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.reasoning_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(search_btn)
        btn_layout.addWidget(archive_btn)
        btn_layout.addWidget(manage_btn)
        btn_layout.addWidget(plugin_btn)
        input_v.addLayout(btn_layout)

        chat_layout.addWidget(input_widget)
        layout.addWidget(chat_group)

        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)

        self.input_text.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.input_text and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
                self.send_message()
                return True
        return super().eventFilter(obj, event)

    # ---------- 文件上传 ----------
    def upload_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.input_text.append(f"\n[文件内容：{Path(path).name}]\n{content}\n")
            self.file_label.setText(f"已加载：{Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"读取失败: {e}")

    # ---------- 辅助函数 ----------
    def change_font(self):
        font = QFont(self.font_combo.currentFont().family(), self.font_size_spin.value())
        self.chat_display.setFont(font)
        self.save_config()

    def load_current_prompt(self):
        p = self.prompts[self.current_prompt_index]
        self.prompt_combo.blockSignals(True)
        self.prompt_combo.clear()
        for idx, p2 in enumerate(self.prompts):
            self.prompt_combo.addItem(p2["name"])
        self.prompt_combo.setCurrentIndex(self.current_prompt_index)
        self.prompt_combo.blockSignals(False)
        idx = self.model_combo.findData(p.get("model", MODEL_CHAT))
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(idx if idx!=-1 else 0)
        self.model_combo.blockSignals(False)
        self.top_role_label.setText(f"当前角色：{p['name']}")
        self.refresh_chat_display()

    def refresh_chat_display(self):
        self.chat_display.clear()
        for msg in self.prompts[self.current_prompt_index].get("messages", []):
            if msg["role"] == "user":
                self.append_message("user", msg["content"], save=False)
            elif msg["role"] == "assistant":
                self.append_message("assistant", msg["content"], save=False)
        self.update_token_display()

    def switch_prompt(self, idx):
        if idx == self.current_prompt_index:
            return
        self.current_prompt_index = idx
        self.load_current_prompt()
        self.save_config()

    def rename_prompt(self):
        old = self.prompts[self.current_prompt_index]["name"]
        new, ok = QInputDialog.getText(self, "重命名", "新名称:", text=old)
        if ok and new.strip():
            self.prompts[self.current_prompt_index]["name"] = new.strip()
            self.save_config()
            self.load_current_prompt()

    def new_prompt(self):
        dlg = MultiLineInputDialog("新建", "名称:", parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        name = dlg.get_text()
        if not name:
            return
        dlg2 = MultiLineInputDialog("新建", "系统提示词:", text="You are a helpful assistant.", parent=self)
        if dlg2.exec() != QDialog.Accepted:
            return
        self.prompts.append({"name": name, "system_prompt": dlg2.get_text(), "messages": [], "model": MODEL_CHAT})
        self.current_prompt_index = len(self.prompts)-1
        self.save_config()
        self.load_current_prompt()

    def delete_prompt(self):
        if len(self.prompts) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一个提示词")
            return
        if QMessageBox.question(self, "确认", "删除当前提示词？") == QMessageBox.Yes:
            del self.prompts[self.current_prompt_index]
            if self.current_prompt_index >= len(self.prompts):
                self.current_prompt_index = len(self.prompts)-1
            self.save_config()
            self.load_current_prompt()

    def edit_prompt(self):
        p = self.prompts[self.current_prompt_index]
        dlg = MultiLineInputDialog("编辑", "系统提示词:", text=p["system_prompt"], parent=self)
        if dlg.exec() == QDialog.Accepted:
            p["system_prompt"] = dlg.get_text()
            self.save_config()
            QMessageBox.information(self, "完成", "已更新")

    def clear_current_conversation(self):
        self.prompts[self.current_prompt_index]["messages"] = []
        self.refresh_chat_display()
        self.save_config()

    def update_token_display(self):
        p = self.prompts[self.current_prompt_index]
        username = self.username_edit.text().strip() or "我"
        sys_prompt = p["system_prompt"]
        if username and "用户" not in sys_prompt:
            sys_prompt = f"用户的名字是{username}。\n{sys_prompt}"
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.extend(p["messages"])
        cur = self.input_text.toPlainText().strip()
        if cur:
            msgs.append({"role": "user", "content": cur})
        self.token_label_value.setText(str(count_tokens(msgs)))

    def append_message(self, role, content, save=True):
        username = self.username_edit.text().strip() or "我"
        rolename = self.prompts[self.current_prompt_index]["name"]
        if role == "assistant":
            filtered = filter_text(content)
            html_content = markdown_to_html(filtered)
            html_msg = f'<div style="margin-bottom:8px;"><b>【{rolename}】</b><br>{html_content}</div>'
        else:
            safe = html.escape(filter_text(content)).replace('\n', '<br>')
            if role == "user":
                html_msg = f'<div style="margin-bottom:8px;"><b>【{username}】</b>{safe}</div>'
            else:
                html_msg = f'<div style="margin-bottom:8px;"><b>【系统】</b>{safe}</div>'
        self.chat_display.insertHtml(html_msg)
        self.chat_display.insertPlainText("\n")
        self.chat_display.ensureCursorVisible()
        if save and role != "system":
            self.prompts[self.current_prompt_index]["messages"].append({"role": role, "content": content})
            self.save_config()

    # ---------- 对话管理功能 ----------
    def export_conversation(self):
        messages = self.prompts[self.current_prompt_index]["messages"]
        if not messages:
            QMessageBox.information(self, "提示", "当前对话为空，无法导出")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "保存对话", f"对话_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "文本文件 (*.txt)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"对话导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"提示词: {self.prompts[self.current_prompt_index]['name']}\n")
                    f.write(f"模型: {self.prompts[self.current_prompt_index].get('model', MODEL_CHAT)}\n")
                    f.write("-" * 50 + "\n")
                    for msg in messages:
                        role = "用户" if msg["role"] == "user" else "助手"
                        f.write(f"【{role}】\n{msg['content']}\n\n")
                QMessageBox.information(self, "成功", f"对话已导出到 {filename}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", str(e))

    def search_history(self):
        messages = self.prompts[self.current_prompt_index]["messages"]
        if not messages:
            QMessageBox.information(self, "提示", "当前对话历史为空")
            return
        self.search_dialog = SearchDialog(self, messages)
        self.search_dialog.show()

    def jump_to_message(self, index):
        self.refresh_chat_display()
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        self.chat_display.setTextCursor(cursor)
        QMessageBox.information(self, "提示", f"已跳转到第 {index+1} 条消息（请手动向上滚动查看）")

    def archive_conversation(self):
        messages = self.prompts[self.current_prompt_index]["messages"]
        if not messages:
            QMessageBox.information(self, "提示", "当前对话为空，无法归档")
            return
        name, ok = QInputDialog.getText(self, "归档对话", "请输入归档名称:", text=f"归档_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if ok and name.strip():
            archive = {
                "name": name.strip(),
                "messages": messages.copy(),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_name": self.prompts[self.current_prompt_index]["name"],
                "model": self.prompts[self.current_prompt_index].get("model", MODEL_CHAT)
            }
            self.archived_conversations.append(archive)
            self.prompts[self.current_prompt_index]["messages"] = []
            self.refresh_chat_display()
            self.save_config()
            QMessageBox.information(self, "成功", "对话已归档")

    def manage_archives(self):
        if not self.archived_conversations:
            QMessageBox.information(self, "提示", "没有归档的对话")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("管理归档对话")
        dialog.resize(600, 400)
        layout = QVBoxLayout(dialog)

        tree = QTreeWidget()
        tree.setHeaderLabels(["名称", "日期", "原提示词", "消息数"])
        for arch in self.archived_conversations:
            item = QTreeWidgetItem([arch["name"], arch["date"], arch["prompt_name"], str(len(arch["messages"]))])
            item.setData(0, Qt.UserRole, arch)
            tree.addTopLevelItem(item)
        layout.addWidget(tree)

        btn_layout = QHBoxLayout()
        def restore():
            item = tree.currentItem()
            if not item:
                return
            arch = item.data(0, Qt.UserRole)
            reply = QMessageBox.question(dialog, "恢复归档", "选择恢复方式:\n是 - 追加到当前对话\n否 - 新建提示词\n取消 - 取消操作",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                current_messages = self.prompts[self.current_prompt_index]["messages"]
                current_messages.extend(arch["messages"])
                self.refresh_chat_display()
                self.save_config()
                QMessageBox.information(dialog, "成功", "已追加到当前对话")
            elif reply == QMessageBox.No:
                new_name, ok = QInputDialog.getText(dialog, "新建提示词", "请输入新提示词名称:", text=arch["name"])
                if ok and new_name.strip():
                    new_prompt = {
                        "name": new_name.strip(),
                        "system_prompt": self.prompts[self.current_prompt_index]["system_prompt"],
                        "messages": arch["messages"].copy(),
                        "model": arch["model"]
                    }
                    self.prompts.append(new_prompt)
                    self.save_config()
                    QMessageBox.information(dialog, "成功", f"已创建新提示词: {new_name.strip()}")
            dialog.accept()

        def delete_archive():
            item = tree.currentItem()
            if not item:
                return
            if QMessageBox.question(dialog, "确认删除", "确定要删除此归档吗？") == QMessageBox.Yes:
                arch = item.data(0, Qt.UserRole)
                self.archived_conversations.remove(arch)
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
                self.save_config()

        restore_btn = QPushButton("恢复所选归档")
        restore_btn.clicked.connect(restore)
        delete_btn = QPushButton("删除所选归档")
        delete_btn.clicked.connect(delete_archive)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(restore_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()

    # ---------- 插件管理 ----------
    def manage_plugins(self):
        dialog = PluginManagerDialog(self.plugin_manager, self)
        dialog.exec()

    # ---------- 发送消息（自动推理窗口，复用同一窗口）----------
    def send_message(self):
        user_input = self.input_text.toPlainText().strip()
        if not user_input:
            return

        self.trigger_plugin_hook("on_before_send", self, user_input)

        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "警告", "请填写 API Key")
            return
        p = self.prompts[self.current_prompt_index]
        sys_prompt = p["system_prompt"]
        username = self.username_edit.text().strip() or "我"
        if username and "用户" not in sys_prompt:
            sys_prompt = f"用户的名字是{username}。\n{sys_prompt}"
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.extend(p["messages"])
        msgs.append({"role": "user", "content": user_input})

        self.append_message("user", user_input)
        self.input_text.clear()
        self.update_token_display()

        model = p.get("model", MODEL_CHAT)
        self.send_btn.setEnabled(False)
        self.reasoning_btn.setEnabled(False)
        self.status_label.setText(f"请求中... (模型: {model})")

        # 如果是深度思考模式，打开或复用推理窗口
        if model == MODEL_REASONER:
            reasoning_dialog = ReasoningDialog(self)
            reasoning_dialog.start_stream(api_key, msgs, model, p["name"])
            reasoning_dialog.show()
            reasoning_dialog.raise_()
            reasoning_dialog.activateWindow()

        # 非流式请求（用于主聊天区回复）
        self.worker = Worker(api_key, msgs, model)
        self.worker.finished.connect(self.on_normal_finished)
        self.worker.error.connect(self.on_normal_error)
        self.worker.start()

    def on_normal_finished(self, reply, usage):
        self.trigger_plugin_hook("on_after_receive", self, reply)

        self.append_message("assistant", reply)
        if usage and "total_tokens" in usage:
            self.total_tokens_used += usage["total_tokens"]
            self.total_token_value.setText(str(self.total_tokens_used))
            self.save_config()
        self.send_btn.setEnabled(True)
        self.reasoning_btn.setEnabled(True)
        self.status_label.setText("就绪")
        self.update_token_display()

    def on_normal_error(self, err):
        self.append_message("system", f"错误: {err}")
        self.send_btn.setEnabled(True)
        self.reasoning_btn.setEnabled(True)
        self.status_label.setText("错误")

    # ---------- 手动显示推理（复用同一窗口）----------
    def show_reasoning(self):
        user_input = self.input_text.toPlainText().strip()
        if not user_input:
            QMessageBox.warning(self, "提示", "请先输入消息内容")
            return
        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "警告", "请先填写 API Key")
            return
        p = self.prompts[self.current_prompt_index]
        sys_prompt = p["system_prompt"]
        username = self.username_edit.text().strip() or "我"
        model = p.get("model", MODEL_CHAT)
        if model != MODEL_REASONER:
            reply = QMessageBox.question(self, "模型提醒", f"当前模型为 {model}，不提供推理过程。是否仍要继续（仅显示最终回复）？", QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        full_messages = []
        modified_system = sys_prompt
        if username and "用户" not in sys_prompt:
            modified_system = f"用户的名字是{username}。\n{sys_prompt}"
        if modified_system:
            full_messages.append({"role": "system", "content": modified_system})
        full_messages.extend(p["messages"])
        full_messages.append({"role": "user", "content": user_input})

        reasoning_dialog = ReasoningDialog(self)
        reasoning_dialog.start_stream(api_key, full_messages, model, p["name"])
        reasoning_dialog.show()
        reasoning_dialog.raise_()
        reasoning_dialog.activateWindow()

    def on_model_changed(self, idx):
        model = self.model_combo.itemData(idx)
        if model:
            self.prompts[self.current_prompt_index]["model"] = model
            self.save_config()

    def closeEvent(self, event):
        self.save_config()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepSeekClient()
    window.show()
    sys.exit(app.exec())