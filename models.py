from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash


# ----------------------
# User Table
# ----------------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(512), nullable=False)  # matches SQL

    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    verification_token = db.Column(db.String(255), nullable=True)
    token_expiry = db.Column(db.DateTime, nullable=True)

    daily_summary_count = db.Column(db.Integer, default=0, nullable=False)
    last_summary_date = db.Column(db.Date, nullable=True)

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
    __tablename__ = "summary"
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.Text, nullable=False)
    output_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    critique_text = db.Column(db.Text)
    score = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))

    # Link summaries to user
    user = db.relationship("User", backref=db.backref("summaries", lazy=True))
