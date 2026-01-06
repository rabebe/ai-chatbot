from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash


# ----------------------
# User Table
# ----------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    # Set password using hashing
    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    # Check password hash
    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


# ----------------------
# Summary Table
# ----------------------
class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.Text, nullable=False)
    output_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    # Link summaries to user
    user = db.relationship("User", backref=db.backref("summaries", lazy=True))
