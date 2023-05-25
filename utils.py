import os
from datetime import datetime
import pytz


def get_timezone():
    kr_tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz=kr_tz)
    
    return now.strftime('%m/%d-%H:%M:%S')