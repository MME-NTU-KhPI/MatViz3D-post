@echo off
title Запуск обчислень MatViz3D з терміналу PyCharm

:: Змінюємо директорію на ту, де лежить цей .bat файл
cd /d "%~dp0"

:: Додаємо поточну папку, а також батьківські папки на 1, 2 і 3 рівні вище.
:: Це гарантовано захопить вашу "синю папку" (Sources Root)
set "PYTHONPATH=%~dp0;%~dp0..;%~dp0..\..;%~dp0..\..\..;%PYTHONPATH%"

echo Запуск процесу автоматичного перебору методів та картинок...
echo.

:: Використовуємо звичайну команду python
python run_experiments.py

if %errorlevel% neq 0 (
    echo.
    echo Виникла помилка під час виконання Python-скрипта!
    pause
) else (
    echo.
    echo Виконання успішно завершено!
    pause
)