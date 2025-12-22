import logging
import os
from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
from extensions import db
from dotenv import load_dotenv


# ----------------------
# Environment & Logging
# ----------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Flask setup
# ----------------------

app = Flask(
    __name__,
    template_folder=os.path.join(script_dir, "templates"),
    static_folder=os.path.join(script_dir, "static"),
)
CORS(app, supports_credentials=True, origins=["http://localhost:3003"])
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = SECRET_KEY

# Initialize database
db.init_app(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Load routes
from src.core.routes import routes  # noqa: E402

app.register_blueprint(routes)

# ----------------------
# Run server
# ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
