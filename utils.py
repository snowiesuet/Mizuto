import logging

def configure_logging():
    """
    Configures logging for the application.
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')