import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mcp_server.server import get_schedule
import logging
import json

# Setup logging to see inside get_schedule
logging.basicConfig(level=logging.INFO)

codes = ['PEC1008', 'PHI1002', 'HIS1001']
print(f"Testing get_schedule for: {codes}")

result_json = get_schedule(codes)
with open("test/reproduce_output.json", "w", encoding="utf-8") as f:
    f.write(result_json)
print("Result written to test/reproduce_output.json")
