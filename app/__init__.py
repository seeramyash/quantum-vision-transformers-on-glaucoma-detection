from flask import Flask

def create_app():
    app = Flask(__name__)

    from backend.app.routes.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app