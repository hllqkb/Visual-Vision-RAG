import streamlit.web.cli as stcli
import os, sys

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        current_dir = sys._MEIPASS
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "app.py")

    sys.argv = ["streamlit", "run", file_path, 
        "--server.enableCORS=true", "--server.enableXsrfProtection=false", 
        "--global.developmentMode=false", "--client.toolbarMode=minimal"]
    sys.exit(stcli.main())
