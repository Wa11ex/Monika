'''
Monika GUI 深色主题（VSCode 风格）
'''

# VSCode 深色主题颜色配置
COLORS = {
    'bg_primary': '#1e1e1e',      # 主背景（深灰）
    'bg_secondary': '#252526',    # 次背景
    'bg_tertiary': '#2d2d30',     # 第三层背景
    'text_primary': '#e0e0e0',    # 主文本
    'text_secondary': '#858585',  # 次文本
    'accent': '#007acc',          # 强调色（蓝）
    'accent_hover': '#0098ff',    # 悬停色
    'border': '#3e3e42',          # 边框
    'success': '#4ec9b0',         # 成功色
    'warning': '#ce9178',         # 警告色
}

def get_stylesheet():
    '''返回完整的深色主题样式表'''
    return f'''
    /* 全局样式 */
    * {{
        margin: 0;
        padding: 0;
        border: none;
    }}

    QMainWindow {{
        background-color: transparent;
        color: {COLORS['text_primary']};
    }}

    QWidget {{
        background-color: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
    }}

    /* 分裂器 */
    QSplitter::handle {{
        background-color: {COLORS['border']};
        border: 1px solid {COLORS['border']};
    }}

    QSplitter::handle:hover {{
        background-color: {COLORS['accent']};
    }}

    /* 文本编辑框 */
    QTextEdit {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px;
        font-size: 11pt;
        selection-background-color: {COLORS['accent']};
    }}

    QTextEdit:focus {{
        border: 1px solid {COLORS['accent']};
    }}

    /* 行编辑框（输入框）*/
    QLineEdit {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 11pt;
        selection-background-color: {COLORS['accent']};
    }}

    QLineEdit:focus {{
        border: 1px solid {COLORS['accent']};
        background-color: {COLORS['bg_tertiary']};
    }}

    QLineEdit::placeholder {{
        color: {COLORS['text_secondary']};
    }}

    /* 按钮 */
    QPushButton {{
        background-color: {COLORS['accent']};
        color: white;
        border: 1px solid {COLORS['accent']};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 11pt;
    }}

    QPushButton:hover {{
        background-color: {COLORS['accent_hover']};
        border: 1px solid {COLORS['accent_hover']};
    }}

    QPushButton:pressed {{
        background-color: #005a9e;
        border: 1px solid #005a9e;
    }}

    QPushButton:disabled {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
    }}

    /* 工具按钮（图标按钮）*/
    QToolButton {{
        background-color: transparent;
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 6px;
        font-size: 14pt;
    }}

    QToolButton:hover {{
        background-color: {COLORS['bg_tertiary']};
        border: 1px solid {COLORS['accent']};
        color: {COLORS['accent_hover']};
    }}

    QToolButton:pressed {{
        background-color: {COLORS['accent']};
        border: 1px solid {COLORS['accent']};
        color: white;
    }}

    /* 标签 */
    QLabel {{
        color: {COLORS['text_primary']};
        background-color: transparent;
    }}

    /* 组合框 */
    QComboBox {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 10pt;
    }}

    QComboBox:focus {{
        border: 1px solid {COLORS['accent']};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}

    QComboBox::down-arrow {{
        image: none;
        color: {COLORS['text_primary']};
    }}

    QComboBox QAbstractItemView {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        selection-background-color: {COLORS['accent']};
        border-radius: 6px;
    }}

    /* 旋转框 */
    QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 6px 12px;
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {COLORS['accent']};
    }}

    /* 滚动条 */
    QScrollBar:vertical {{
        background-color: {COLORS['bg_secondary']};
        width: 12px;
        border: none;
    }}

    QScrollBar::handle:vertical {{
        background-color: #686868;
        border-radius: 6px;
        min-height: 20px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['accent']};
    }}

    QScrollBar:horizontal {{
        background-color: {COLORS['bg_secondary']};
        height: 12px;
        border: none;
    }}

    QScrollBar::handle:horizontal {{
        background-color: #686868;
        border-radius: 6px;
        min-width: 20px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {COLORS['accent']};
    }}

    QScrollBar::add-line, QScrollBar::sub-line {{
        border: none;
        background: none;
    }}

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background-color: {COLORS['bg_secondary']};
    }}

    /* 菜单和对话框 */
    QMenu {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        margin: 2px;
    }}

    QMenu::item:selected {{
        background-color: {COLORS['accent']};
    }}

    QDialog {{
        background-color: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
    }}

    QTabBar::tab {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_secondary']};
        border: none;
        padding: 8px 16px;
        margin-right: 2px;
        border-radius: 4px 4px 0px 0px;
    }}

    QTabBar::tab:selected {{
        background-color: {COLORS['accent']};
        color: white;
    }}

    QTabBar::tab:hover {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_primary']};
    }}
    
    QTabBar::tab:selected:hover {{
        background-color: {COLORS['accent']};
        color: white;
    }}

    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
    }}
    '''
