from flask_apidoc.commands import GenerateApiDoc
from flask_script import Manager

from app import app

manager = Manager(app)
manager.add_command('apidoc', GenerateApiDoc())

# For generate new documentation use:
# python3 manage.py apidoc
if __name__ == "__main__":
    manager.run()
