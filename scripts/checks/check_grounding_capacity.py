"""2026-10-27：零更新容量检查入口，服务器必须经Slurm。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.st.grounding_capacity import main

if __name__=='__main__':main()
