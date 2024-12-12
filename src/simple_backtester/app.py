import os
import traceback
import time
from uuid import uuid4
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for server-side rendering
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from .backtest import Backtester  # Ensure this module is in the PYTHONPATH

# Configuration
UPLOAD_FOLDER = os.path.abspath('./uploads')
PLOT_FOLDER = os.path.abspath('./static/plots')
ALLOWED_EXTENSIONS = {'py', 'yaml'}

# Create necessary folders if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


def clear_uploads_folder():
    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            return

        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        print(f"Error clearing uploads folder: {e}")


@app.route('/upload/<file_type>', methods=['POST'])
def upload_file(file_type):
    try:
        if file_type not in ['strategy', 'config']:
            return jsonify({"error": "Invalid file type. Use 'strategy' or 'config'."}), 400

        if file_type not in request.files:
            return jsonify({"error": f"'{file_type}' file is required."}), 400

        file = request.files[file_type]

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only .py and .yaml files are allowed."}), 400

        filename = f"{uuid4()}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        return jsonify({"message": f"{file_type} file uploaded successfully.", "file_path": file_path}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    try:
        start_time = time.time()

        # Check for uploaded files
        strategy_file = None
        config_file = None

        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.endswith('.py'):
                strategy_file = os.path.join(app.config['UPLOAD_FOLDER'], file)
            elif file.endswith('.yaml'):
                config_file = os.path.join(app.config['UPLOAD_FOLDER'], file)

        if not strategy_file or not config_file:
            return jsonify({"error": "Both strategy and config files must be uploaded before running the backtest."}), 400

        # Run backtest
        backtest = Backtester(strategy_file, config_file)
        config = {
            "data_path": "./src/simple_backtester/data/feature/",
            "features": ["close", "price", "return", "volume"],
        }
        backtest.run(config)
        print(f"Backtest started at {time.time() - start_time} seconds")
        results = backtest.get_results()

        # Generate PnL plot
        cumulative_pnl_list = np.cumsum(results.get('pnl_history'))
        plot_path = os.path.join(app.config['PLOT_FOLDER'], 'pnl_history.png')
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_pnl_list, label="Cumulative PnL")
        plt.title("Cumulative PnL Over Time")
        plt.xlabel("Time Periods")
        plt.ylabel("Cumulative PnL")
        plt.legend()
        plt.grid()
        plt.savefig(plot_path)
        plt.close()

        # Verify plot creation
        if not os.path.exists(plot_path):
            raise FileNotFoundError("PnL plot could not be saved.")

        # Prepare response
        response = {
            "cumulative_pnl": results.get("cumulative_pnl"),
            "Sharpe Ratio": results.get("Sharpe Ratio"),
            "Volatility": results.get("Volatility"),
            "Max Drawdown": results.get("Max Drawdown"),
            "plot_url": f"/static/plots/pnl_history.png?t={int(time.time())}",
        }

        clear_uploads_folder()

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        print("An error occurred, clearing uploads folder.")
        clear_uploads_folder()
        return jsonify({"error": str(e)}), 500


@app.route('/static/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(app.config['PLOT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
