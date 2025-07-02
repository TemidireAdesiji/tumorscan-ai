import os
from flask import Flask, request, render_template
from model import get_prediction, LABELS, ALLOWED_EXTENSIONS
from werkzeug.utils import secure_filename

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Routes
@app.route('/')
def index():
    return render_template('DiseaseDet.html')

@app.route("/uimg", methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('uimg.html')

    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return render_template('error.html', message="Invalid file format."), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    with open(file_path, 'rb') as f:
        img_bytes = f.read()

    class_id, class_name = get_prediction(img_bytes)

    return render_template('pred.html', result=class_name, file=filename)

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', message="Server encountered an error."), 500

# Utility
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
