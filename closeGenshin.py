import subprocess
process_name = 'GenshinImpact.exe'
# Завершение процесса
subprocess.run(['taskkill', '/F', '/IM', process_name])