import subprocess
process_name = 'ZenlessZoneZero.exe'
# Завершение процесса
subprocess.run(['taskkill', '/F', '/IM', process_name])
