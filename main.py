
import subprocess

subprocess.run(['python', 'soccer-forecasting-v1/sim_parallel.py'], check=True)

subprocess.run(['python', 'soccer-forecasting-v1/x-api.py'],check=True)