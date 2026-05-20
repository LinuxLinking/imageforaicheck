import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from imageforai.web import create_app


app = create_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
