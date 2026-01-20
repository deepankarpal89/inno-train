from dotenv import load_dotenv
import os
load_dotenv()

print(os.getenv('BASE_URL'))
print(os.getenv('PORT'))

print(type(os.getenv('GPU_SETUP_TIME')))