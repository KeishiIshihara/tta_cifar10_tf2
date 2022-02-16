import sys
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
sys.stdout.write(RUN_ID)