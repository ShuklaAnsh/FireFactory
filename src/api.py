import logging
from threading import Thread
from flask import Flask, Response, request
from .model import initialize_model

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/heartbeat', methods=["GET"])
def heartbeat():
    """
    Heartbeat endpoint (GET) so that the client can check if API and classifier is up and running
    """
    if initialized:
        return Response(status=200)
    if initialization_failed:
        return Response(status=100)
    return Response(status=101)


@app.route('/fire', methods=["GET"])
def generate_lofi():
    """
    Endpoint (GET) for retrieving fresh fire
    """
    try:
        # fire = model.generate_lofi()
        print("Fire Requested")
        return {'test': 'howdy'}
    except Exception as e:
        print(e)
        return Response(status=500)


def start_server():
    app.run("0.0.0.0", 5000)


if __name__ == '__main__':
    print("Starting Server...")
    server_thread = Thread(name='server_thread', target=start_server)
    server_thread.setDaemon(True)
    server_thread.start()
    print("Server Started")
    classifier_initialized = False
    classifier_failed = False
    print("Initializing Classifier...")
    try:
        model = initialize_model()
        initialized = True
    except Exception as e:
        initialization_failed = True

    while server_thread.is_alive():
        continue
    print("Server shutting down...")
